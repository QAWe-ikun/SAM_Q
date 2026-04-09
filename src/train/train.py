"""
Training Script for SAM3 + Qwen3-VL Object Placement Prediction

Usage:
    python train.py --config configs/config.yaml
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import yaml

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.placement_model import SAM3PlacementModel, PlacementLoss
from data.dataset import ObjectPlacementDataModule


class Trainer:
    """Trainer for object placement prediction model."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        device: Optional[str] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
            device: Device to run training on
        """
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        self.output_dir = Path(config.get("output_dir", "./outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = self._init_model()
        
        # Initialize loss
        self.criterion = PlacementLoss(
            dice_weight=config.get("dice_weight", 1.0),
            bce_weight=config.get("bce_weight", 1.0),
        )
        
        # Initialize optimizer
        self.optimizer = self._init_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._init_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        
    def _init_model(self) -> SAM3PlacementModel:
        """Initialize the placement model."""
        model_config = self.config.get("model", {})
        
        model = SAM3PlacementModel(
            qwen_model_name=model_config.get(
                "qwen_model_name", "Qwen/Qwen3-VL-8B-Instruct"
            ),
            sam3_input_dim=model_config.get("sam3_input_dim", 256),
            qwen_hidden_dim=model_config.get("qwen_hidden_dim", 4096),
            adapter_hidden_dim=model_config.get("adapter_hidden_dim", 512),
            device=self.device,
        )
        
        # Freeze components
        if self.config.get("freeze_qwen", True):
            model.freeze_qwen()
        if self.config.get("freeze_sam3_image_encoder", True):
            model.freeze_sam3_image_encoder()
        
        model.to(self.device)
        return model
    
    def _init_optimizer(self) -> AdamW:
        """Initialize optimizer."""
        opt_config = self.config.get("optimizer", {})
        
        # Get trainable parameters
        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        
        return AdamW(
            trainable_params,
            lr=opt_config.get("lr", 1e-4),
            weight_decay=opt_config.get("weight_decay", 1e-4),
        )
    
    def _init_scheduler(self) -> CosineAnnealingLR:
        """Initialize learning rate scheduler."""
        sched_config = self.config.get("scheduler", {})
        
        return CosineAnnealingLR(
            self.optimizer,
            T_max=sched_config.get("T_max", 100),
            eta_min=sched_config.get("eta_min", 1e-6),
        )
    
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            # Move data to device
            masks = batch["masks"].to(self.device)
            
            # Forward pass (note: actual implementation depends on image loading)
            # For now, this is a simplified version
            losses = {"total": torch.tensor(0.0, device=self.device)}
            
            # Process each sample in batch
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
                losses["total"] = losses["total"] + loss_dict["total"]
            
            # Average loss
            losses["total"] = losses["total"] / len(batch["plane_images"])
            
            # Backward pass
            self.optimizer.zero_grad()
            losses["total"].backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += losses["total"].item()
            num_batches += 1
            
            progress_bar.set_postfix({"loss": losses["total"].item()})
        
        avg_loss = total_loss / num_batches
        
        return {"train_loss": avg_loss}
    
    @torch.no_grad()
    def validate(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Validation"):
            masks = batch["masks"].to(self.device)
            
            losses = {"total": torch.tensor(0.0, device=self.device)}
            
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
                losses["total"] = losses["total"] + loss_dict["total"]
            
            losses["total"] = losses["total"] / len(batch["plane_images"])
            total_loss += losses["total"].item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        return {"val_loss": avg_loss}
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        num_epochs: Optional[int] = None,
    ):
        """
        Full training loop.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (optional)
            num_epochs: Number of epochs to train
        """
        num_epochs = num_epochs or self.config.get("num_epochs", 100)
        assert(num_epochs is not None)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            metrics = {
                "epoch": epoch,
                **train_metrics,
                **val_metrics,
                "lr": self.scheduler.get_last_lr()[0],
            }
            
            print(f"Epoch {epoch}: " + " ".join(
                f"{k}={v:.4f}" for k, v in metrics.items()
            ))
            
            # Save checkpoint
            self._save_checkpoint(epoch, val_metrics.get("val_loss", float("inf")))
        
        # Save final checkpoint
        self._save_checkpoint("final", float("inf"))
    
    def _save_checkpoint(self, epoch: Any, val_loss: float):
        """Save model checkpoint."""
        is_best = val_loss < self.best_val_loss
        
        if is_best:
            self.best_val_loss = val_loss
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        
        # Save checkpoint
        if epoch == "final":
            path = self.output_dir / "checkpoint_final.pt"
        else:
            path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        
        torch.save(checkpoint, path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Train SAM3 + Qwen3-VL for object placement prediction"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line args
    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir
    
    # Initialize data module
    data_module = ObjectPlacementDataModule(
        data_dir=config.get("data_dir", "./data"),
        batch_size=config.get("batch_size", 8),
        num_workers=config.get("num_workers", 4),
    )
    data_module.setup("fit")
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Start training
    trainer.train(
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        num_epochs=config.get("num_epochs", 100),
    )


if __name__ == "__main__":
    main()
