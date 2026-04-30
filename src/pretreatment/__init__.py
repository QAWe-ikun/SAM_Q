"""
SAM-Q Pretreatment 模块

提供训练数据自动生成功能。
"""

from .data_generator import TrainingDataGenerator
from .models import ObjectInfo
from .components import (
    SceneBuilder,
    Renderer,
    HeatmapGenerator,
    Augmentation,
    SampleSaver,
    VLMClient,
)

__all__ = [
    "TrainingDataGenerator",
    "ObjectInfo",
    "SceneBuilder",
    "Renderer",
    "HeatmapGenerator",
    "Augmentation",
    "SampleSaver",
    "VLMClient",
]
