"""
Stage 2 Trainer
===============

Custom epoch loop for Adapter + SAM3 Decoder training.
"""

import os

import torch # type: ignore
from pathlib import Path
import torch.nn.functional as F  # type: ignore
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader  # type: ignore
from tqdm import tqdm

from .checkpoint_mgr import CheckpointManager


class Stage2Trainer:
    """
    Stage 2: Train Adapter + SAM3 Decoder with custom epoch loop.
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        config: Dict[str, Any],
        output_dir: Path,
        device: str,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.output_dir = output_dir
        self.device = device

        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []

        self.checkpoint_mgr = CheckpointManager(
            model=model,
            config=config,
            output_dir=output_dir,
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        """
        Full Stage 2 training loop.
        """
        training_config = self.config.get("training", {})
        num_epochs = training_config.get("num_epochs", 100)
        val_interval = training_config.get("val_interval", 1)

        print(f"\n{'='*60}")
        print(f"Starting Stage 2 Training for {num_epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self._train_epoch(train_loader)

            # Validate
            val_metrics = {}
            if val_loader is not None and (epoch + 1) % val_interval == 0:
                val_metrics = self.validate(val_loader)

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Log metrics
            metrics = {
                "epoch": epoch + 1,
                **train_metrics,
                **val_metrics,
                "lr": self.optimizer.param_groups[0]["lr"],
            }

            self._print_metrics(metrics)

            # Early stopping check
            val_loss = val_metrics.get("val_loss", float("inf"))
            if self._check_early_stopping(val_loss):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

            # Save checkpoint
            self.checkpoint_mgr.save(epoch, val_loss)

        # Save final checkpoint
        self.checkpoint_mgr.save("final", float("inf"))
        print(f"\nTraining completed! Results saved to: {self.output_dir}")

    def _train_epoch(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_heatmap_mse = 0.0
        total_gt_prob = 0.0
        total_offset_px = 0.0
        total_rotation_loss = 0.0
        total_scale_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False,
        )

        for _, batch in enumerate(progress_bar):
            batch_loss_tensor, batch_metrics = self._process_batch(batch, training=True)

            # Backward pass
            self.optimizer.zero_grad()
            batch_loss_tensor.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += batch_metrics["total"]
            total_heatmap_mse += batch_metrics["heatmap_mse"]
            total_gt_prob += batch_metrics["gt_prob"]
            total_offset_px += batch_metrics["offset_px"]
            total_rotation_loss += batch_metrics["rotation"]
            total_scale_loss += batch_metrics["scale"]
            num_batches += 1

            progress_bar.set_postfix({"loss": f"{batch_metrics['total']:.4f}"})

        self.train_losses.append(total_loss / num_batches)

        # Effective weights (heatmap fixed, rot/scl dynamic)
        w_hm = self.criterion.heatmap_weight
        w_rot = torch.exp(-self.criterion.log_var_rotation).item()
        w_scl = torch.exp(-self.criterion.log_var_scale).item()

        return {
            "train_loss": total_loss / num_batches,
            "train_heatmap_mse": total_heatmap_mse / num_batches,
            "train_gt_prob": total_gt_prob / num_batches,
            "train_offset_px": total_offset_px / num_batches,
            "train_rotation_loss": total_rotation_loss / num_batches,
            "train_scale_loss": total_scale_loss / num_batches,
            "w_heatmap": w_hm,
            "w_rotation": w_rot,
            "w_scale": w_scl,
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validation loop."""
        self.model.eval()
        os.makedirs("debug/vis", exist_ok=True)
        vis_count = 0

        total_loss = 0.0
        total_heatmap_mse = 0.0
        total_gt_prob = 0.0
        total_offset_px = 0.0
        total_rotation_loss = 0.0
        total_scale_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Validation", leave=False):
            _, batch_metrics = self._process_batch(batch, training=False)

            total_loss += batch_metrics["total"]
            total_heatmap_mse += batch_metrics["heatmap_mse"]
            total_gt_prob += batch_metrics["gt_prob"]
            total_offset_px += batch_metrics["offset_px"]
            total_rotation_loss += batch_metrics["rotation"]
            total_scale_loss += batch_metrics["scale"]
            num_batches += 1

            # 可视化前 5 个 batch
            output = batch.get("output")
            if output is not None and vis_count < 5:
                self._visualize_batch(batch, output, vis_count)
                vis_count += 1

        self.val_losses.append(total_loss / num_batches)

        return {
            "val_loss": total_loss / num_batches,
            "val_heatmap_mse": total_heatmap_mse / num_batches,
            "val_gt_prob": total_gt_prob / num_batches,
            "val_offset_px": total_offset_px / num_batches,
            "val_rotation_loss": total_rotation_loss / num_batches,
            "val_scale_loss": total_scale_loss / num_batches,
        }

    def _process_batch(self, batch, training: bool = True):
        """Process a single batch through the model."""
        plane_images_batch = batch["plane_images"].to(self.device)
        batch_images = batch["images"]
        masks = batch["masks"].to(self.device)
        seg_hidden_batch = batch.get("seg_hidden")
        if seg_hidden_batch is not None:
            seg_hidden_batch = seg_hidden_batch.to(self.device)

        batch_loss_tensor = None
        batch_metrics = {"total": 0.0, "heatmap_mse": 0.0, "gt_prob": 0.0, "offset_px": 0.0, "rotation": 0.0, "scale": 0.0}

        for i in range(len(plane_images_batch)):
            plane_image = plane_images_batch[i]
            sample_images = [img.to(self.device) for img in batch_images[i]]
            text_prompt = batch["text_prompts"][i]
            seg_hidden_i = seg_hidden_batch[i] if seg_hidden_batch is not None else None

            output = self.model(
                plane_image=plane_image,
                text_prompt=text_prompt,
                images=sample_images if seg_hidden_i is None else None,
                seg_hidden=seg_hidden_i,
            )

            # Store output for validation metrics computation
            if not training:
                batch["output"] = output

            gt_rot = batch.get("rotation_6d")
            gt_scl = batch.get("scale")
            loss_dict = self.criterion(
                predicted_heatmaps=output["heatmap"],
                target_heatmaps=masks[i:i+1],
                pred_rotation_6d=output.get("rotation_6d"),
                pred_scale=output.get("scale_relative"),
                gt_rotation_6d=gt_rot[i:i+1] if gt_rot is not None else None,
                gt_scale=gt_scl[i:i+1] if gt_scl is not None else None,
                class_logits=output.get("class_logits"),
            )

            if batch_loss_tensor is None:
                batch_loss_tensor = loss_dict["total"]
            else:
                batch_loss_tensor = batch_loss_tensor + loss_dict["total"]
            batch_metrics["total"] += loss_dict["total"].item()
            batch_metrics["heatmap_mse"] += self._to_float(loss_dict.get("heatmap_mse", 0))
            batch_metrics["gt_prob"] += self._to_float(loss_dict.get("gt_prob", 0))
            batch_metrics["offset_px"] += self._to_float(loss_dict.get("offset_px", 0))
            batch_metrics["rotation"] += self._to_float(loss_dict.get("rotation", 0))
            batch_metrics["scale"] += self._to_float(loss_dict.get("scale", 0))

        # Average over batch size
        batch_size = len(plane_images_batch)
        batch_loss_tensor = batch_loss_tensor / batch_size
        for key in batch_metrics:
            batch_metrics[key] /= batch_size

        return batch_loss_tensor, batch_metrics

    def _to_float(self, v):
        if isinstance(v, torch.Tensor):
            return v.item()
        return v

    def _print_metrics(self, metrics):
        """Print formatted metrics."""
        epoch_num = metrics['epoch']
        # Dynamic weights (effective weight = exp(-log_var))
        w_hm = metrics.get("w_heatmap", 1.0)
        w_rot = metrics.get("w_rotation", 1.0)
        w_scl = metrics.get("w_scale", 1.0)
        print(f"Epoch {epoch_num:3d} | "
              f"train_loss: {metrics.get('train_loss', 0):.4f} | "
              f"hm: {metrics.get('train_heatmap_mse', 0):.4f}(w={w_hm:.2f}) | "
              f"gt_prob: {metrics.get('train_gt_prob', 0):.4f} | "
              f"offset_px: {metrics.get('train_offset_px', 0):.1f} | "
              f"rot: {metrics.get('train_rotation_loss', 0):.4f}(w={w_rot:.2f}) | "
              f"scl: {metrics.get('train_scale_loss', 0):.4f}(w={w_scl:.2f}) | "
              f"val_loss: {metrics.get('val_loss', 0):.4f} | "
              f"lr: {metrics.get('lr', 0):.6f}")

    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check if early stopping should be triggered."""
        training_config = self.config.get("training", {})
        if not training_config.get("early_stopping", False):
            return False

        patience = training_config.get("patience", 20)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.checkpoint_mgr.patience_counter = 0
        else:
            self.checkpoint_mgr.patience_counter += 1
            if self.checkpoint_mgr.patience_counter >= patience:
                return True

        return False

    def _visualize_batch(self, batch, output, batch_idx):
        """Save visualization of predicted and GT heatmaps."""
        import cv2 # type: ignore
        import numpy as np

        pred_heatmap = torch.sigmoid(output["heatmap"])
        gt_heatmap = batch["masks"]

        # Resize pred to GT size if needed
        if pred_heatmap.shape[-2:] != gt_heatmap.shape[-2:]:
            pred_heatmap = F.interpolate(pred_heatmap, size=gt_heatmap.shape[-2:], mode="bilinear", align_corners=False)

        # Get first sample in batch
        pred = pred_heatmap[0, 0].cpu().float().numpy()
        gt = gt_heatmap[0, 0].cpu().numpy()  # [B, 1, H, W] -> [H, W]
        plane_img = batch["plane_images"][0].cpu().numpy().transpose(1, 2, 0)

        # Normalize plane image to [0, 255]
        plane_img = np.clip(plane_img * 255, 0, 255).astype(np.uint8)

        # Convert heatmaps to colored images
        pred_colored = cv2.applyColorMap((np.clip(pred, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        gt_colored = cv2.applyColorMap((np.clip(gt, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Resize heatmaps to match plane image size
        h, w = plane_img.shape[:2]
        pred_colored = cv2.resize(pred_colored, (w, h))
        gt_colored = cv2.resize(gt_colored, (w, h))

        # Overlay on plane image
        overlay_pred = cv2.addWeighted(plane_img, 0.7, pred_colored, 0.3, 0)
        overlay_gt = cv2.addWeighted(plane_img, 0.7, gt_colored, 0.3, 0)

        # Save
        out_path = f"debug/vis/val_batch{batch_idx:02d}.png"
        combined = np.hstack([overlay_pred, overlay_gt])
        cv2.imwrite(out_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
