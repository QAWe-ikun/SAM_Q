"""
Adapter Modules for SAM-Q
==========================

This package contains adapter modules for embedding space transformation.
"""

from .base_adapter import Adapter
from .cross_modal_adapter import CrossModalAdapter
from .presence_token_adapter import PresenceTokenAdapter
from .seg_token_projector import SegTokenProjector

__all__ = [
    "Adapter",
    "CrossModalAdapter",
    "PresenceTokenAdapter",
    "SegTokenProjector",
]
