"""
Trainer for SAM-Q
==================

Provides comprehensive training loop with validation, checkpointing, and logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm
import yaml
from datetime import datetime

from ..models import SAMQPlacementModel, PlacementLoss
from .optimizer import create_optimizer, create_scheduler
from .metrics import compute_metrics
from ..utils.config import Config


class Trainer:
    """
    Comprehensive trainer for object placement prediction.
    
    Features:
        - Training loop with progress bar
        - Validation loop
        - Checkpoint management
        - Metrics computation
        - Logging (console and optional W&B)
        - Early stopping
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model: Optional[SAMQPlacementModel] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            model: Model to train (optional, created if None)
            device: Device to run on
        """
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = config.get("experiment", {}).get("name", "experiment")
        self.output_dir = Path(config.get("training", {}).get("save_dir", "outputs"))
        self.output_dir = self.output_dir / f"{exp_name}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = self.output_dir / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        # Initialize model
        self.model = model or self._init_model()
        
        # Initialize loss
        loss_config = config.get("loss", {})
        self.criterion = PlacementLoss(
            dice_weight=loss_config.get("dice_weight", 1.0),
            bce_weight=loss_config.get("bce_weight", 1.0),
        ).to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = create_optimizer(
            self.model,
            config.get("optimizer", {})
        )
        
        training_config = config.get("training", {})
        num_epochs = training_config.get("num_epochs", 100)
        scheduler_config = config.get("scheduler", {})
        warmup_epochs = scheduler_config.get("warmup_epochs", 0)
        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_config,
            num_epochs,
            warmup_epochs,
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        
        # Early stopping
        self.early_stopping = training_config.get("early_stopping", False)
        self.patience = training_config.get("patience", 20)
        self.patience_counter = 0

    def _init_model(self) -> SAMQPlacementModel:
        """Initialize model from configuration."""
        model_config = self.config.get("model", {})
        encoding_mode = model_config.get("encoding_mode", "cross_modal")

        model = SAMQPlacementModel(
            qwen_model_name=model_config.get("qwen", {}).get(
                "model_name", "Qwen/Qwen3-VL-8B-Instruct"
            ),
            sam3_input_dim=model_config.get("sam3", {}).get("input_dim", 256),
            qwen_hidden_dim=model_config.get("qwen", {}).get("hidden_dim", 4096),
            adapter_hidden_dim=model_config.get("adapter", {}).get("hidden_dim", 512),
            device=self.device,
            mode=encoding_mode,
            seg_projector_config=model_config.get("seg_token", {}),
            action_head_config=model_config.get("action_head", {}),
        )
        
        # Freeze components
        if model_config.get("qwen", {}).get("freeze", True):
            model.freeze_qwen()
        if model_config.get("sam3", {}).get("freeze_image_encoder", True):
            model.freeze_sam3_image_encoder()
        
        model.to(self.device)
        return model

    def train_epoch(
        self,
        dataloader: DataLoader,
        log_interval: int = 10,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training DataLoader
            log_interval: Log every N batches
            
        Returns:
            Metrics dictionary
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False,
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            masks = batch["masks"].to(self.device)
            
            # Process each sample
            batch_loss = 0.0
            for i in range(len(batch["plane_images"])):
                plane_image = batch["plane_images"][i]
                object_image = batch["object_images"][i]
                text_prompt = batch["text_prompts"][i]
                
                # Forward pass
                output = self.model(
                    plane_image=plane_image,
                    object_image=object_image,
                    text_prompt=text_prompt,
                )
                
                # Compute loss
                loss_dict = self.criterion(output["masks"], masks[i:i+1])
                batch_loss += loss_dict["total"]
            
            # Average loss
            avg_loss = batch_loss / len(batch["plane_images"])
            
            # Backward pass
            self.optimizer.zero_grad()
            avg_loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += avg_loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{avg_loss.item():.4f}"})
            
            # Log intermediate metrics
            if (batch_idx + 1) % log_interval == 0:
                print(
                    f"  Batch {batch_idx + 1}/{len(dataloader)} - "
                    f"Loss: {avg_loss.item():.4f}"
                )
        
        avg_epoch_loss = total_loss / num_batches
        self.train_losses.append(avg_epoch_loss)
        
        return {"train_loss": avg_epoch_loss}

    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            dataloader: Validation DataLoader
            
        Returns:
            Metrics dictionary
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        all_metrics = {
            "iou": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
        
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            masks = batch["masks"].to(self.device)
            
            batch_loss = 0.0
            for i in range(len(batch["plane_images"])):
                plane_image = batch["plane_images"][i]
                object_image = batch["object_images"][i]
                text_prompt = batch["text_prompts"][i]
                
                output = self.model(
                    plane_image=plane_image,
                    object_image=object_image,
                    text_prompt=text_prompt,
                )
                
                loss_dict = self.criterion(output["masks"], masks[i:i+1])
                batch_loss += loss_dict["total"]
                
                # Compute metrics
                with torch.no_grad():
                    pred_mask = torch.sigmoid(output["masks"])
                    metrics = compute_metrics(pred_mask, masks[i:i+1])
                    for key in all_metrics:
                        all_metrics[key] += metrics[key]
            
            avg_loss = batch_loss / len(batch["plane_images"])
            total_loss += avg_loss.item()
            num_batches += 1
        
        avg_val_loss = total_loss / num_batches
        self.val_losses.append(avg_val_loss)
        
        # Average metrics
        for key in all_metrics:
            all_metrics[key] /= num_batches
        
        all_metrics["val_loss"] = avg_val_loss
        
        return all_metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        """
        Full training loop.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (optional)
        """
        training_config = self.config.get("training", {})
        num_epochs = training_config.get("num_epochs", 100)
        val_interval = training_config.get("val_interval", 1)
        log_interval = training_config.get("log_interval", 10)
        
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader, log_interval)
            
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
            
            # Print metrics
            metrics_str = " | ".join(
                f"{k}: {v:.4f}" for k, v in metrics.items() if k != "epoch"
            )
            print(f"Epoch {metrics['epoch']:3d} | {metrics_str}")
            
            # Save checkpoint
            val_loss = val_metrics.get("val_loss", float("inf"))
            self._save_checkpoint(epoch, val_loss)
            
            # Early stopping
            if self.early_stopping:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        print(f"\nEarly stopping at epoch {epoch + 1}")
                        break
        
        # Save final checkpoint
        self._save_checkpoint("final", float("inf"))
        print(f"\nTraining completed! Results saved to: {self.output_dir}")

    def _save_checkpoint(self, epoch: Any, val_loss: float) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
        """
        is_best = val_loss < self.best_val_loss
        
        if is_best:
            self.best_val_loss = val_loss
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "config": self.config,
        }
        
        # Save epoch checkpoint
        if epoch == "final":
            path = self.output_dir / "checkpoint_final.pt"
        else:
            save_interval = self.config.get("training", {}).get("save_interval", 10)
            if isinstance(epoch, int) and (epoch + 1) % save_interval == 0:
                path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
                torch.save(checkpoint, path)
        
        # Save best checkpoint
        if is_best and self.config.get("training", {}).get("save_best", True):
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
