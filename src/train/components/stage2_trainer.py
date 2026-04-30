"""
Stage 2 Trainer
===============

Custom epoch loop for Adapter + SAM3 Decoder training.
"""

import torch # type: ignore
from pathlib import Path
import torch.nn.functional as F  # type: ignore
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader  # type: ignore
from tqdm import tqdm

from ..metrics import compute_metrics
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
        log_interval = training_config.get("log_interval", 10)

        print(f"\n{'='*60}")
        print(f"Starting Stage 2 Training for {num_epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self._train_epoch(train_loader, log_interval)

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
        log_interval: int = 10,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_bce_loss = 0.0
        total_dice_loss = 0.0
        total_rotation_loss = 0.0
        total_scale_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False,
        )

        for batch_idx, batch in enumerate(progress_bar):
            batch_loss_tensor, batch_metrics = self._process_batch(batch, training=True)

            # Backward pass
            self.optimizer.zero_grad()
            batch_loss_tensor.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += batch_metrics["total"]
            total_bce_loss += batch_metrics["bce"]
            total_dice_loss += batch_metrics["dice"]
            total_rotation_loss += batch_metrics["rotation"]
            total_scale_loss += batch_metrics["scale"]
            num_batches += 1

            progress_bar.set_postfix({"loss": f"{batch_metrics['total']:.4f}"})

            if (batch_idx + 1) % log_interval == 0:
                print(
                    f"  Batch {batch_idx + 1}/{len(dataloader)} - "
                    f"Loss: {batch_metrics['total']:.4f}"
                )

        self.train_losses.append(total_loss / num_batches)

        return {
            "train_loss": total_loss / num_batches,
            "train_bce_loss": total_bce_loss / num_batches,
            "train_dice_loss": total_dice_loss / num_batches,
            "train_rotation_loss": total_rotation_loss / num_batches,
            "train_scale_loss": total_scale_loss / num_batches,
        }

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validation loop."""
        self.model.eval()

        total_loss = 0.0
        total_bce_loss = 0.0
        total_dice_loss = 0.0
        total_rotation_loss = 0.0
        total_scale_loss = 0.0
        num_batches = 0
        all_metrics = {
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

        for batch in tqdm(dataloader, desc="Validation", leave=False):
            _, batch_metrics = self._process_batch(batch, training=False)

            total_loss += batch_metrics["total"]
            total_bce_loss += batch_metrics["bce"]
            total_dice_loss += batch_metrics["dice"]
            total_rotation_loss += batch_metrics["rotation"]
            total_scale_loss += batch_metrics["scale"]
            num_batches += 1

            # Compute IoU metrics
            output = batch.get("output")
            if output is not None:
                metrics = self._compute_iou_metrics(output, batch)
                for key in all_metrics:
                    all_metrics[key] += metrics[key]

        self.val_losses.append(total_loss / num_batches)

        # Average metrics
        for key in all_metrics:
            all_metrics[key] /= num_batches

        return {
            **all_metrics,
            "val_loss": total_loss / num_batches,
            "val_bce_loss": total_bce_loss / num_batches,
            "val_dice_loss": total_dice_loss / num_batches,
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

        batch_loss_tensor = torch.tensor(0.0, device=self.device)
        batch_metrics = {"total": 0.0, "bce": 0.0, "dice": 0.0, "rotation": 0.0, "scale": 0.0}

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
                predicted_masks=output["heatmap"],
                target_masks=masks[i:i+1],
                pred_rotation_6d=output.get("rotation_6d"),
                pred_scale=output.get("scale_relative"),
                gt_rotation_6d=gt_rot[i:i+1] if gt_rot is not None else None,
                gt_scale=gt_scl[i:i+1] if gt_scl is not None else None,
                class_logits=output.get("class_logits"),
            )

            batch_loss_tensor = batch_loss_tensor + loss_dict["total"]
            batch_metrics["total"] += loss_dict["total"].item()
            batch_metrics["bce"] += loss_dict.get("bce", 0).item() if isinstance(loss_dict.get("bce", 0), torch.Tensor) else loss_dict.get("bce", 0)
            batch_metrics["dice"] += loss_dict.get("dice", 0).item() if isinstance(loss_dict.get("dice", 0), torch.Tensor) else loss_dict.get("dice", 0)
            batch_metrics["rotation"] += loss_dict.get("rotation", 0).item() if isinstance(loss_dict.get("rotation", 0), torch.Tensor) else loss_dict.get("rotation", 0)
            batch_metrics["scale"] += loss_dict.get("scale", 0).item() if isinstance(loss_dict.get("scale", 0), torch.Tensor) else loss_dict.get("scale", 0)

        # Average over batch size
        batch_size = len(plane_images_batch)
        batch_loss_tensor = batch_loss_tensor / batch_size
        for key in batch_metrics:
            batch_metrics[key] /= batch_size

        return batch_loss_tensor, batch_metrics

    def _compute_iou_metrics(self, output, batch):
        """Compute IoU, Precision, Recall, F1 metrics."""
        pred_mask = torch.sigmoid(output["heatmap"])

        # Select best query
        class_logits = output.get("class_logits")
        if pred_mask.shape[1] > 1 and class_logits is not None:
            scores = class_logits[-1].squeeze(-1)
            best_idx = scores.argmax(dim=-1)
            pred_mask = pred_mask[torch.arange(pred_mask.shape[0], device=pred_mask.device), best_idx]
            pred_mask = pred_mask.unsqueeze(1)

        # Resize to GT size
        gt_mask = batch["masks"]
        if pred_mask.shape[-2:] != gt_mask.shape[-2:]:
            pred_mask = F.interpolate(pred_mask, size=gt_mask.shape[-2:], mode="bilinear", align_corners=False)

        return compute_metrics(pred_mask, gt_mask)

    def _print_metrics(self, metrics):
        """Print formatted metrics."""
        epoch_num = metrics['epoch']
        print(f"Epoch {epoch_num:3d} | "
              f"train_loss: {metrics.get('train_loss', 0):.4f} | "
              f"bce: {metrics.get('train_bce_loss', 0):.4f} | "
              f"dice: {metrics.get('train_dice_loss', 0):.4f} | "
              f"rot: {metrics.get('train_rotation_loss', 0):.4f} | "
              f"scl: {metrics.get('train_scale_loss', 0):.4f} | "
              f"val_loss: {metrics.get('val_loss', 0):.4f} | "
              f"iou: {metrics.get('iou', 0):.4f} | "
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
