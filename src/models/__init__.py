"""
SAM-Q Model Architecture
=========================

This package contains all model components for the SAM-Q object placement system.

Structure:
    - encoders/: Encoder modules (Qwen3-VL, etc.)
    - adapters/: Adapter modules for embedding transformation
    - collision/: Collision detection modules (H-MVP)
    - vla/: Vision-Language-Action modules
    - sampling/: Sampling and placement strategies
    
Main Models:
    - SAM3PlacementModel: Main placement prediction model
    - PlacementLoss: Loss functions for training
    - VLALoss: VLA-specific loss functions
"""

from .placement_model import SAM3PlacementModel, PlacementLoss, VLALoss
from .action_head import ActionHead, rotation_6d_to_matrix

__all__ = [
    "SAM3PlacementModel",
    "PlacementLoss",
    "VLALoss",
    "ActionHead",
    "rotation_6d_to_matrix",
]
