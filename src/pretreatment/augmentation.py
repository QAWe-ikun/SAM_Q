"""
SAM-Q 数据增强模块

负责随机移动、旋转、缩放物体，生成增强样本。
"""

import random
import numpy as np
import trimesh
from typing import List, Tuple
from scipy.spatial.transform import Rotation as R

from .models import ObjectInfo


class AugmentationProcessor:
    """数据增强处理器"""

    def __init__(
        self,
        rotation_range: float = 180,
        scale_range: Tuple[float, float] = (0.8, 1.25),
    ):
        self.rotation_range = rotation_range
        self.scale_range = scale_range

    def augmentation_object(
        self,
        obj: ObjectInfo,
        scene: trimesh.Scene,
    ) -> Tuple[ObjectInfo, trimesh.Scene]:
        """
        数据增强：随机移动物体到新位置，设置随机旋转和缩放。

        返回: (增强后的物体信息, 新场景)
        """
        # 1. 随机旋转（Y 轴）
        rot_y = random.uniform(-self.rotation_range, self.rotation_range)
        rot_aug = R.from_euler('y', rot_y, degrees=True).as_quat().tolist()

        # 2. 随机移动到新位置（不与其他物体碰撞）
        bounds = scene.bounds
        if bounds is None:
            return None, None
        x_min, y_min, _ = bounds[0]
        x_max, y_max, _ = bounds[1]
        for _ in range(100):
            new_pos = [
                random.uniform(x_min, x_max),
                random.uniform(y_min, y_max),
                obj.pos[2],  # 保持 Z 值
            ]
            # 检查碰撞
            if not self._check_position_collision(scene, new_pos, obj.mesh):
                break
        else:
            return None, None  # 无法找到合适位置

        # 3. 创建增强后的物体
        aug_mesh = obj.mesh.copy()

        if rot_y != 0:
            R_mat = np.eye(4)
            R_mat[:3, :3] = R.from_quat(rot_aug).as_matrix()
            aug_mesh.apply_transform(R_mat)

        T = np.eye(4)
        T[:3, 3] = new_pos
        aug_mesh.apply_transform(T)

        # 4. 更新场景
        from .scene_builder import SceneBuilder
        new_scene = SceneBuilder.__new__(SceneBuilder)  # 避免循环导入
        new_scene = scene.copy()
        geom_name = f"obj_{obj.jid}"
        if geom_name in new_scene.geometry:
            del new_scene.geometry[geom_name]
        new_scene.add_geometry(aug_mesh, geom_name=geom_name)

        # 5. 创建增强后的物体信息
        aug_obj = ObjectInfo(
            jid=obj.jid,
            model_id=obj.model_id,
            desc=obj.desc,
            pos=new_pos,
            rot=rot_aug,
            size=obj.size,
            scale_jid=(1, 1, 1),
            mesh=obj.mesh,  # 原始 mesh
            is_on_floor=obj.is_on_floor,
        )

        return aug_obj, new_scene

    def _check_position_collision(
        self,
        scene: trimesh.Scene,
        pos: List[float],
        mesh: trimesh.Trimesh,
    ) -> bool:
        """检查指定位置是否与其他物体碰撞（使用 AABB 包围盒）"""
        test_mesh = mesh.copy()
        T = np.eye(4)
        T[:3, 3] = pos
        test_mesh.apply_transform(T)

        test_bounds = test_mesh.bounds

        for geom_name, geom in scene.geometry.items():
            if not isinstance(geom, trimesh.Trimesh):
                continue
            # 只检测其他家具物体，排除地板
            if "floor" in geom_name.lower():
                continue

            geom_bounds = geom.bounds
            if geom_bounds is None:
                continue

            # AABB 重叠检测
            if (test_bounds[0, 0] < geom_bounds[1, 0] and
                test_bounds[1, 0] > geom_bounds[0, 0] and
                test_bounds[0, 1] < geom_bounds[1, 1] and
                test_bounds[1, 1] > geom_bounds[0, 1]):
                return True

        return False
