"""
Model Loaders for SAM-Q
========================

Lazy-loading wrappers for external models (Qwen3-VL, SAM3, etc.).

Features:
    - Lazy loading (only loads when actually needed)
    - Version selection via checkpoint path
    - Training/inference mode switching
"""

from .sam3_loader import SAM3Loader

__all__ = [
    "SAM3Loader",
]
