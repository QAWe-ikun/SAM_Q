"""
SAM-Q Pretreatment 模块

提供训练数据自动生成功能。
"""

from .data_generator import TrainingDataGenerator
from .models import ObjectInfo

__all__ = ["TrainingDataGenerator", "ObjectInfo"]
