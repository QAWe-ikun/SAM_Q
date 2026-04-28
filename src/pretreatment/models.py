"""
SAM-Q 数据模型定义

包含 ObjectInfo 等数据结构。
"""

import trimesh
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class ObjectInfo:
    """物体信息"""
    jid: str
    model_id: str
    desc: str
    pos: List[float]
    rot: List[float]
    size: List[float]
    scale_jid: Tuple[float, float, float]
    mesh: trimesh.Trimesh
    is_on_floor: bool  # 是否在地面上
