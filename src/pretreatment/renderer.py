"""
SAM-Q 渲染模块

负责 3D 场景渲染，包括俯视图、物体参考图等。
"""

import os
import logging
import numpy as np
import trimesh
from pathlib import Path
from typing import List, Optional

# WSL 无头环境下强制使用 OSMesa 渲染后端
if not os.environ.get("PYOPENGL_PLATFORM"):
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

logger = logging.getLogger(__name__)


class SceneRenderer:
    """场景渲染器"""

    def __init__(
        self,
        image_size: int = 1024,
        fov_degrees: float = 45,
        aspect_ratio: float = 1.0,
        top_view_camera_height: float = 1.0,
    ):
        self.image_size = image_size
        self.fov_degrees = fov_degrees
        self.aspect_ratio = aspect_ratio
        self.top_view_camera_height = top_view_camera_height

    @staticmethod
    def build_camera_basis(forward, world_up=None):
        """
        构建相机坐标系的 right, up, -forward（右手系）。
        处理 forward 与 world_up 平行的退化情况。
        """
        forward = forward / np.linalg.norm(forward)
        if world_up is None:
            world_up = np.array([0.0, 1.0, 0.0])
        world_up = world_up / np.linalg.norm(world_up)

        # 计算 right = forward × world_up
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)

        if right_norm < 1e-6:
            # 退化：forward 与 world_up 平行
            # 选择一个与 forward 垂直的任意向量
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

        # 重新计算 up，确保严格正交
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)

        return right, up, -forward  # OpenGL 相机 z 轴指向远离场景

    def render_scene(
        self,
        scene: trimesh.Scene,
        camera_position: np.ndarray,
        camera_target: np.ndarray,
        use_light: bool = True,
    ) -> Optional[np.ndarray]:
        """渲染场景为 RGB 图像（使用 pyrender 离屏渲染）"""
        import pyrender  # type: ignore

        try:
            # 计算相机方向
            forward = camera_target - camera_position

            right, up, cam_z = self.build_camera_basis(forward)

            # 构建相机在世界坐标系中的变换矩阵
            cam_transform = np.eye(4)
            cam_transform[:3, 0] = right   # X 轴：右
            cam_transform[:3, 1] = up      # Y 轴：上
            cam_transform[:3, 2] = cam_z   # Z 轴：相机前向（-forward）
            cam_transform[:3, 3] = camera_position

            # 创建 pyrender 场景
            py_scene = pyrender.Scene()

            # 添加 trimesh 几何体
            for geom_name, geom in scene.geometry.items():
                if isinstance(geom, trimesh.Trimesh):
                    try:
                        mesh = pyrender.Mesh.from_trimesh(geom)
                        py_scene.add(mesh, name=geom_name)
                    except Exception as e:
                        logger.debug(f"Skipping mesh {geom_name}: {e}")
                        return None

            # 添加相机
            camera = pyrender.PerspectiveCamera(
                yfov=np.radians(self.fov_degrees),
                aspectRatio=self.aspect_ratio
            )
            py_scene.add(camera, pose=cam_transform)

            # 添加光源
            if use_light:
                light = pyrender.DirectionalLight(
                    color=[1.0, 1.0, 1.0],
                    intensity=3.0
                )
                py_scene.add(light, pose=cam_transform)

            # 离屏渲染
            r = pyrender.OffscreenRenderer(
                viewport_width=self.image_size,
                viewport_height=self.image_size
            )
            color, _ = r.render(py_scene)
            r.delete()

            return color

        except Exception as e:
            raise RuntimeError(f"Pyrender 渲染失败: {e}")

    def render_top_view(
        self,
        scene: trimesh.Scene,
        bounds_bottom: List[List[float]]
    ) -> Optional[np.ndarray]:
        """渲染俯视图（从 Y 轴正方向往下看，投影到 XZ 平面）"""
        bounds_bottom = np.array(bounds_bottom)
        min_x = bounds_bottom[:, 0].min()
        min_y = bounds_bottom[:, 2].min()
        max_x = bounds_bottom[:, 0].max()
        max_y = bounds_bottom[:, 2].max()

        # 计算场景对角线
        scene_width = max_x - min_x
        scene_depth = max_y - min_y
        diag = np.sqrt(scene_width**2 + scene_depth**2)

        # 根据 FOV 自适应计算相机高度
        fov_rad = np.radians(self.fov_degrees)
        required_height = (diag / 2.0) / np.tan(fov_rad / 2.0)

        # 使用计算的高度与配置高度的最大值，并增加一点余量
        adaptive_height = self.top_view_camera_height + required_height

        center = [(min_x + max_x) / 2, (min_y + max_y) / 2, 0]
        camera_pos = np.array([center[0], adaptive_height, center[1]])
        camera_target = np.array([center[0], bounds_bottom[0][1], center[1]])

        return self.render_scene(scene, camera_pos, camera_target)

    def render_object_reference(
        self,
        mesh: trimesh.Trimesh,
        bounds_bottom: List[List[float]]
    ) -> Optional[np.ndarray]:
        """渲染物体参考图（俯视图，正向朝上，保持原始尺寸）"""
        # 复制 mesh 并重置所有变换
        obj_mesh = mesh.copy()

        # 将物体居中
        obj_mesh.apply_translation(-obj_mesh.centroid)

        # 创建只包含该物体的场景
        obj_scene = trimesh.Scene([obj_mesh])

        # 使用与 render_top_view 相同的相机配置
        bounds_bottom = np.array(bounds_bottom)
        min_x = bounds_bottom[:, 0].min()
        min_y = bounds_bottom[:, 2].min()
        max_x = bounds_bottom[:, 0].max()
        max_y = bounds_bottom[:, 2].max()

        scene_width = max_x - min_x
        scene_depth = max_y - min_y
        diag = np.sqrt(scene_width**2 + scene_depth**2)

        fov_rad = np.radians(self.fov_degrees)
        required_height = (diag / 2.0) / np.tan(fov_rad / 2.0)
        adaptive_height = self.top_view_camera_height + required_height

        center = [(min_x + max_x) / 2, (min_y + max_y) / 2, 0]
        camera_pos = np.array([center[0], adaptive_height, center[1]])
        camera_target = np.array([center[0], bounds_bottom[0][1], center[1]])

        return self.render_scene(obj_scene, camera_pos, camera_target)
