"""
SAM-Q 热力图生成模块

负责生成 GT 放置热力图。
"""

import numpy as np
from typing import List


class HeatmapGenerator:
    """热力图生成器"""

    def __init__(
        self,
        image_size: int = 1024,
        fov_degrees: float = 45,
        aspect_ratio: float = 1.0,
        top_view_camera_height: float = 1.0,
        sigma: float = 15.0,
    ):
        self.image_size = image_size
        self.fov_degrees = fov_degrees
        self.aspect_ratio = aspect_ratio
        self.top_view_camera_height = top_view_camera_height
        self.sigma = sigma

    @staticmethod
    def build_camera_basis(forward, world_up=None):
        """构建相机坐标系"""
        forward = forward / np.linalg.norm(forward)
        if world_up is None:
            world_up = np.array([0.0, 1.0, 0.0])
        world_up = world_up / np.linalg.norm(world_up)

        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)

        if right_norm < 1e-6:
            candidates = [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([0.0, 1.0, 0.0]),
            ]
            for candidate in candidates:
                if abs(np.dot(candidate, forward)) < 0.99:
                    right = np.cross(forward, candidate)
                    right = right / np.linalg.norm(right)
                    break
        else:
            right = right / right_norm

        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        return right, up, -forward

    def generate(
        self,
        target_pos: List[float],
        bounds_bottom: List[List[float]]
    ) -> np.ndarray:
        """
        生成热力图：将目标点投影到 2D 图像，再生成高斯热力图。
        """
        bounds_bottom = np.array(bounds_bottom)
        min_x = bounds_bottom[:, 0].min()
        min_y = bounds_bottom[:, 2].min()
        max_x = bounds_bottom[:, 0].max()
        max_y = bounds_bottom[:, 2].max()

        scene_center_x = (min_x + max_x) / 2
        scene_center_y = (min_y + max_y) / 2

        # 自适应计算相机高度
        scene_width = max_x - min_x
        scene_depth = max_y - min_y
        diag = np.sqrt(scene_width**2 + scene_depth**2)
        fov_rad = np.radians(self.fov_degrees)
        required_height = (diag / 2.0) / np.tan(fov_rad / 2.0)
        adaptive_height = max(self.top_view_camera_height, required_height) + 0.5

        # 相机
        camera_pos = np.array([scene_center_x, adaptive_height, scene_center_y])
        camera_target = np.array([scene_center_x, bounds_bottom[0][1], scene_center_y])

        forward = camera_target - camera_pos
        right, up, cam_z = self.build_camera_basis(forward)

        # 正向投影：将目标点投影到 2D 图像
        target = np.array(target_pos)
        target_rel = target - camera_pos

        # 投影到相机坐标系
        target_cam_x = np.dot(target_rel, right)
        target_cam_y = np.dot(target_rel, up)
        target_cam_z = np.dot(target_rel, cam_z)  # 深度

        # 透视投影到 2D 图像坐标
        viewport_height = 2.0 * target_cam_z * np.tan(fov_rad / 2.0)
        viewport_width = viewport_height * self.aspect_ratio

        # 图像坐标（注意 X 方向翻转）
        image_x = (0.5 - target_cam_x / viewport_width) * self.image_size
        image_y = (target_cam_y / viewport_height + 0.5) * self.image_size

        # 生成 2D 高斯热力图
        heatmap = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        y_coords, x_coords = np.ogrid[:self.image_size, :self.image_size]
        heatmap = np.exp(
            -((x_coords - image_x)**2 + (y_coords - image_y)**2) / (2 * self.sigma**2)
        )

        # 归一化到 [0, 1]
        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val

        return heatmap
