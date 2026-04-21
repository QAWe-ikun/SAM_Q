"""
SAM3 Loader

This module provides a lazy-loading wrapper for SAM3 models.
Supports loading different SAM3 versions from local checkpoints.

Features:
    - Lazy loading (only loads when actually needed)
    - Version selection via checkpoint path
    - Training/inference mode switching
    - Component extraction (backbone, transformer, seg_head, etc.)
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, Union


class SAM3Loader(nn.Module):
    """
    SAM3 Model Loader with lazy loading and version selection.

    Loads SAM3 image model from local checkpoint,
    provides access to internal components needed by SAM-Q.

    Features:
        - Lazy loading (delays loading until first use)
        - Version selection via checkpoint path
        - Training/inference mode switching
        - Component extraction for SAM-Q pipeline
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize SAM3 Loader.

        Args:
            checkpoint_path: Path to SAM3 checkpoint file (.pt).
                If None, will use default search path.
            device: Device to run model on
            dtype: Data type for model (default: bfloat16 for SAM3)
        """
        super().__init__()

        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        # SAM3 model and components (lazy loaded)
        self.model = None
        self.sam3_vision_backbone = None
        self.sam3_transformer = None
        self.sam3_seg_head = None
        self.sam3_dot_scoring = None

        self._loaded = False

    def load_model(self, checkpoint_path: Optional[str] = None, eval_mode: bool = True):
        """
        Load SAM3 model from checkpoint.

        Args:
            checkpoint_path: Override checkpoint path. If None, uses self.checkpoint_path.
            eval_mode: Whether to set model to eval mode after loading.
        """
        if self._loaded:
            return

        try:
            # Try local SAM3 first (src/sam3)
            import sys
            # loaders/sam3_loader.py → models/ → src/ → src/sam3/
            src_path = os.path.join(os.path.dirname(__file__), "..", "..", "sam3")
            src_path = os.path.normpath(src_path)
            if src_path not in sys.path:
                sys.path.insert(0, src_path)

            from model_builder import build_sam3_image_model
            
        except ImportError:
            raise ImportError(
                "SAM3 not found. Ensure src/sam3/ exists"
            ) from None

        # Resolve checkpoint path
        ckpt_path = checkpoint_path or self.checkpoint_path
        if ckpt_path is None:
            raise ValueError("No checkpoint path provided for SAM3Loader.")

        # Build SAM3 model
        self.model = build_sam3_image_model(
            checkpoint_path=ckpt_path,
            device=self.device,
            eval_mode=eval_mode,
            load_from_HF=False,
        )

        # Debug: Print checkpoint keys to verify format
        import torch
        ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "model" in ckpt_data:
            ckpt_keys = list(ckpt_data["model"].keys())
        elif "model_state_dict" in ckpt_data:
            ckpt_keys = list(ckpt_data["model_state_dict"].keys())
        else:
            ckpt_keys = list(ckpt_data.keys())
        
        # Check if backbone keys exist in checkpoint
        backbone_keys = [k for k in ckpt_keys if "backbone" in k or "vision" in k]
        print(f"  Backbone-related keys: {len(backbone_keys)}")
        if backbone_keys:
            print(f"  Sample backbone keys: {backbone_keys[:5]}")
        print()

        # Extract internal components needed by SAM-Q
        self.sam3_vision_backbone = self.model.backbone.vision_backbone
        self.sam3_transformer = self.model.transformer
        self.sam3_seg_head = self.model.segmentation_head
        self.sam3_dot_scoring = self.model.dot_prod_scoring

        # Set dtype and device for all components
        self.sam3_vision_backbone.to(device=self.device, dtype=self.dtype)
        self.sam3_transformer.to(device=self.device, dtype=self.dtype)
        self.sam3_seg_head.to(device=self.device, dtype=self.dtype)
        self.sam3_dot_scoring.to(device=self.device, dtype=self.dtype)

        if eval_mode:
            self.model.eval()

        self._loaded = True
        print(f"[SAM3Loader] Loaded SAM3 from {ckpt_path}")

    def forward(
        self,
        plane_image: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through SAM3 components.

        Args:
            plane_image: [B, 3, H, W] tensor, preprocessed for SAM3

        Returns:
            dict with SAM3 outputs
        """
        self.load_model()

        # Call the full SAM3 model
        return self.model(plane_image, **kwargs)

    def freeze_vision_backbone(self):
        """Freeze SAM3 vision backbone."""
        self.load_model()
        for param in self.sam3_vision_backbone.parameters():
            param.requires_grad = False
        self.sam3_vision_backbone.eval()

    def freeze_transformer(self):
        """Freeze SAM3 transformer."""
        self.load_model()
        for param in self.sam3_transformer.parameters():
            param.requires_grad = False
        self.sam3_transformer.eval()

    def freeze_seg_head(self):
        """Freeze SAM3 segmentation head."""
        self.load_model()
        for param in self.sam3_seg_head.parameters():
            param.requires_grad = False
        self.sam3_seg_head.eval()

    def freeze_dot_scoring(self):
        """Freeze SAM3 dot product scoring."""
        self.load_model()
        for param in self.sam3_dot_scoring.parameters():
            param.requires_grad = False
        self.sam3_dot_scoring.eval()

    def freeze_all_except(self, trainable_components: list):
        """
        Freeze all SAM3 components except specified ones.

        Args:
            trainable_components: List of component names to keep trainable.
                Options: ["transformer", "seg_head", "dot_scoring"]
        """
        self.load_model()

        # Freeze all by default
        self.freeze_vision_backbone()

        # Unfreeze specified components
        if "transformer" in trainable_components:
            for param in self.sam3_transformer.parameters():
                param.requires_grad = True
            self.sam3_transformer.train()

        if "seg_head" in trainable_components:
            for param in self.sam3_seg_head.parameters():
                param.requires_grad = True
            self.sam3_seg_head.train()

        if "dot_scoring" in trainable_components:
            for param in self.sam3_dot_scoring.parameters():
                param.requires_grad = True
            self.sam3_dot_scoring.train()

    def train(self, mode: bool = True):
        """Set training mode."""
        super().train(mode)
        if self._loaded:
            self.model.train(mode)

    def eval(self):
        """Set eval mode."""
        super().eval()
        if self._loaded:
            self.model.eval()

    @property
    def loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @property
    def output_dim(self) -> int:
        """Get SAM3 output dimension."""
        self.load_model()
        return self.sam3_transformer.d_model
