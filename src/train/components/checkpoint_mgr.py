"""
Checkpoint Manager
==================

Manages saving/loading of model checkpoints.
"""

import torch # type: ignore
from pathlib import Path
from typing import Dict, Any


class CheckpointManager:
    """
    Manages model checkpoints with split weights support.
    """

    def __init__(
        self,
        model,
        config: Dict[str, Any],
        output_dir: Path,
    ):
        self.model = model
        self.config = config
        self.output_dir = output_dir
        self.patience_counter = 0

    def save(self, epoch, val_loss: float) -> None:
        """
        Save checkpoint based on epoch and validation loss.

        Args:
            epoch: Current epoch number or "final"
            val_loss: Current validation loss
        """
        training_config = self.config.get("training", {})
        save_best = training_config.get("save_best", True)
        save_epoch = training_config.get("save_epoch", False)
        save_interval = training_config.get("save_interval", 100)

        # Determine suffix
        suffix = None
        if epoch == "final":
            suffix = "_final"
        elif isinstance(epoch, int):
            if save_epoch and (epoch + 1) % save_interval == 0:
                suffix = f"_epoch_{epoch}"

        if save_best and val_loss < getattr(self, 'best_val_loss', float("inf")):
            suffix = "_best"
            self.best_val_loss = val_loss

        if suffix is not None:
            self._save_split_weights(suffix)

    def _save_split_weights(self, suffix: str) -> None:
        """
        Save Adapter and SAM3 weights separately.
        """
        state_dict = self.model.state_dict()

        # Save Adapter weights
        adapter_state = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in ["adapter.", "seg_projector.", "seg_action_head."]):
                adapter_state[k] = v

        if adapter_state:
            path = self.output_dir / f"adapter_checkpoint{suffix}.pt"
            torch.save({"model_state_dict": adapter_state, "config": self.config}, path)
            print(f"  Split checkpoint saved: {path.name}")

        # Save SAM3 weights
        if self.model.sam3_loader is not None and self.model.sam3_loader._loaded:
            sam3_model_state = self.model.sam3_loader.model.state_dict()
            sam3_model_state = {f"detector.{k}": v for k, v in sam3_model_state.items()}

            sam3_ckpt = {"model": sam3_model_state, "config": self.config}
            path = self.output_dir / f"sam3_checkpoint{suffix}.pt"
            torch.save(sam3_ckpt, path)
            print(f"  Split checkpoint saved: {path.name}")
