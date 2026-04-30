"""
Adapter Modules for SAM-Q
==========================

This package contains adapter modules for embedding space transformation.
"""

from .cross_modal_adapter import CrossModalAdapter
from .seg_token_projector import SegTokenProjector

__all__ = [
    "CrossModalAdapter",
    "SegTokenProjector",
]
