"""
Optimizer and Scheduler Utilities
==================================

Provides factory functions for creating optimizers and schedulers.
"""

import torch
from torch.optim import AdamW, SGD, Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ExponentialLR,
    LambdaLR,
)
from typing import Dict, Any, Optional


def create_optimizer(
    model: torch.nn.Module,
    config: Dict[str, Any],
) -> torch.optim.Optimizer:
    """
    Create optimizer from configuration.
    
    Args:
        model: PyTorch model
        config: Optimizer configuration
        
    Returns:
        Optimizer instance
    """
    opt_type = config.get("type", "AdamW")
    lr = config.get("lr", 1e-4)
    weight_decay = config.get("weight_decay", 1e-4)
    
    # Get trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]
    
    if opt_type == "AdamW":
        return AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=config.get("betas", [0.9, 0.999]),
            eps=config.get("eps", 1e-8),
        )
    elif opt_type == "Adam":
        return Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
        )
    elif opt_type == "SGD":
        return SGD(
            params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=config.get("momentum", 0.9),
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    num_epochs: int,
    warmup_epochs: int = 0,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler from configuration.
    
    Args:
        optimizer: Optimizer instance
        config: Scheduler configuration
        num_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
        
    Returns:
        Scheduler instance
    """
    sched_type = config.get("type", "CosineAnnealingLR")
    
    if sched_type == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.get("T_max", num_epochs),
            eta_min=config.get("eta_min", 1e-6),
        )
    elif sched_type == "StepLR":
        scheduler = StepLR(
            optimizer,
            step_size=config.get("step_size", 30),
            gamma=config.get("gamma", 0.1),
        )
    elif sched_type == "ExponentialLR":
        scheduler = ExponentialLR(
            optimizer,
            gamma=config.get("gamma", 0.95),
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}")
    
    # Wrap with warmup if specified
    if warmup_epochs > 0:
        warmup_lr = config.get("warmup_lr", 1e-5)
        scheduler = WarmupScheduler(
            scheduler,
            warmup_epochs=warmup_epochs,
            warmup_start_lr=warmup_lr,
        )
    
    return scheduler


class WarmupScheduler:
    """
    Learning rate scheduler with warmup phase.
    
    Gradually increases learning rate during warmup,
    then follows the base scheduler.
    """
    
    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        warmup_epochs: int,
        warmup_start_lr: float = 1e-5,
    ):
        """
        Initialize warmup scheduler.
        
        Args:
            scheduler: Base scheduler
            warmup_epochs: Number of warmup epochs
            warmup_start_lr: Starting learning rate for warmup
        """
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lrs = [group["lr"] for group in scheduler.optimizer.param_groups]
    
    def step(self, epoch: Optional[int] = None) -> None:
        """
        Step the scheduler.
        
        Args:
            epoch: Current epoch (optional)
        """
        if epoch is None:
            epoch = self.scheduler.last_epoch
        
        if epoch < self.warmup_epochs:
            # Linear warmup
            alpha = (epoch + 1) / self.warmup_epochs
            warmup_lr = self.warmup_start_lr + (self.base_lrs[0] - self.warmup_start_lr) * alpha
            
            for param_group in self.scheduler.optimizer.param_groups:
                param_group["lr"] = warmup_lr
        else:
            # Follow base scheduler
            self.scheduler.step(epoch - self.warmup_epochs)
    
    def get_last_lr(self):
        """Get last computed learning rate."""
        return [group["lr"] for group in self.scheduler.optimizer.param_groups]
    
    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler state."""
        return {
            "scheduler": self.scheduler.state_dict(),
            "warmup_epochs": self.warmup_epochs,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state."""
        self.scheduler.load_state_dict(state_dict["scheduler"])
