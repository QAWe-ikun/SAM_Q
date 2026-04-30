"""
Pretreatment Components
========================

Components for training data generation.
"""

from .scene_builder import SceneBuilder
from .renderer import SceneRenderer as Renderer
from .heatmap_generator import HeatmapGenerator
from .augmentation import AugmentationProcessor as Augmentation
from .sample_saver import SampleSaver
from .vlm_client import VLMClient

__all__ = [
    "SceneBuilder",
    "Renderer",
    "HeatmapGenerator",
    "Augmentation",
    "SampleSaver",
    "VLMClient",
]
