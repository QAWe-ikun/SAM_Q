"""
SAM-Q 训练集自动化生成器

从 SSR3D-FRONT 场景 JSON 直接生成训练/验证/测试数据，无需保存中间 GLB。

流程:
1. 加载场景 JSON，构建完整 3D 场景
2. 渲染原始房间俯视图/侧视图
3. 每次随机选择一个目标物体
4. VLM 识别该物体及位置
5. 生成物体参考图（俯视图，正向朝上，随机缩放）
6. 剔除目标物体，渲染剔除后房间图
7. 生成 GT 热力图（原位置最高概率，碰撞区域为 0）
8. 保存样本（annotations.json + 图片）

数据增强:
- 随机移动物体到不重叠位置
- 随机旋转支撑轴
- 预测移动后的位置 + 旋转的倒数
- 生成新的样本
"""

import os
import tqdm
import torch # type: ignore
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any

# WSL 无头环境下强制使用 OSMesa 渲染后端
if not os.environ.get("PYOPENGL_PLATFORM"):
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

# 抑制 trimesh 和 pyrender 的警告
warnings.filterwarnings("ignore", category=UserWarning, module="trimesh")
warnings.filterwarnings("ignore", category=UserWarning, module="pyrender")

# 配置日志系统
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"generate_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logger = logging.getLogger("TrainingDataGenerator")
logger.setLevel(logging.DEBUG)

# 文件处理器（记录所有级别的日志）
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# 控制台处理器（只显示 WARNING 及以上级别）
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

import json
import random
import trimesh
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from scipy.spatial.transform import Rotation as R

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


class TrainingDataGenerator:
    """SAM-Q 训练数据生成器"""

    def __init__(
        self,
        config: Dict[str, Any],
    ):
        # 从配置字典中提取参数
        data_config = config.get("data", {})
        gen_config = config.get("generation", {})
        aug_config = config.get("augmentation", {})
        cam_config = config.get("camera", {})

        self.scene_dir = Path(data_config.get("scene_dir", ""))
        self.model_dir = Path(data_config.get("model_dir", ""))
        self.output_dir = Path(data_config.get("output_dir", ""))
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "val"
        self.test_dir = self.output_dir / "test"

        self.image_size = gen_config.get("image_size", 1024)
        self.heatmap_sigma = gen_config.get("heatmap_sigma", 15.0)
        
        self.augmentation = aug_config.get("enabled", True)
        self.aug_ratio = aug_config.get("aug_ratio", 0.2)
        
        # 更新全局配置
        self.aug_rotation_range = aug_config.get("rotation_range", 180)
        scale_range = aug_config.get("scale_range", {})
        self.scale_range = (
            scale_range.get("min", 0.8),
            scale_range.get("max", 1.25)
        )
        
        self.top_view_camera_height = cam_config.get("top_view_height", 1.0)
        self.side_view_camera_distance = cam_config.get("side_view_distance", 5.0)
        self.fov_degrees = cam_config.get("fov_degrees", 45)
        self.aspect_ratio = cam_config.get("aspect_ratio", 1.0)

        # 最大剔除比例
        self.max_object_nums = gen_config.get("max_object_nums", 2)

        # 数据集划分比例
        split_ratio = gen_config.get("split_ratio", {"train": 0.8, "val": 0.1, "test": 0.1})
        self.train_ratio = split_ratio.get("train", 0.8)
        self.val_ratio = split_ratio.get("val", 0.1)
        self.test_ratio = split_ratio.get("test", 0.1)

        # 输出目录结构：data/train/, data/val/, data/test/
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)

        # 全局样本计数
        self.sample_counter = 0

        # 按 split 收集样本元数据
        self.samples_by_split = {"train": [], "val": [], "test": []}

        # Qwen3-VL 模型（懒加载，参考 qwen3vl_encoder.py 的机制）
        self._qwen_model_name = gen_config.get("qwen_model_name", None)
        self._qwen_model = None
        self._qwen_processor = None

    def find_model_path(self, model_id: str) -> Optional[Path]:
        """查找模型文件"""
        model_path = self.model_dir / model_id
        if (model_path / "normalized_model.glb").exists():
            return model_path / "normalized_model.glb"
        if (model_path / "raw_model.glb").exists():
            return model_path / "raw_model.glb"
        glbs = list(model_path.glob("*.glb"))
        if glbs:
            return glbs[0]
        if (model_path / "raw_model.obj").exists():
            return model_path / "raw_model.obj"
        return None

    def parse_jid(self, jid: str) -> Tuple[str, float, float, float]:
        """解析 jid 获取模型 ID 和缩放"""
        parts = jid.split('-(')
        if len(parts) == 1:
            return jid, 1.0, 1.0, 1.0
        model_id = parts[0]
        try:
            sx = float(parts[1].rstrip(')'))
            sy = float(parts[2].rstrip(')'))
            sz = float(parts[3].rstrip(')'))
        except (ValueError, IndexError):
            return model_id, 1.0, 1.0, 1.0
        return model_id, sx, sy, sz

    def load_mesh(self, model_path: Path) -> Optional[trimesh.Trimesh]:
        """加载网格（让 trimesh 自行处理所有内部数据）"""
        try:
            # 直接加载为单个 mesh，不做任何中间处理
            loaded = trimesh.load(model_path, force='mesh')
            return loaded
        except Exception:
            return None
    
    def _triangulate_polygon(self, n: int, reverse: bool = False) -> np.ndarray:
        """多边形三角化"""
        faces = []
        for i in range(1, n - 1):
            if reverse:
                # 反面：反转顶点顺序
                faces.append([0, i + 1, i])
            else:
                faces.append([0, i, i + 1])
        return np.array(faces)

    def build_scene(self, scene_data: Dict) -> Tuple[trimesh.Scene, List[ObjectInfo]]:
        """构建完整场景，返回场景和物体列表"""
        scene = trimesh.Scene()

        # 添加地板
        bounds_top = scene_data.get('bounds_top', [])
        bounds_bottom = scene_data.get('bounds_bottom', [])
        
        if bounds_top and bounds_bottom:
            bounds_top = np.array(bounds_top)
            bounds_bottom = np.array(bounds_bottom)
            
            # 添加地板
            floor_v = bounds_bottom
            floor_f = self._triangulate_polygon(len(bounds_bottom))
            scene.add_geometry(
                trimesh.Trimesh(vertices=floor_v, faces=floor_f, process=False),
                geom_name="floor"
            )
            
        # 加载所有家具
        objects = []
        for obj_data in scene_data.get('objects', []):
            desc = obj_data.get('desc', '')
            jid = obj_data.get('jid', '')
            pos = obj_data.get('pos', [0, 0, 0])
            rot = obj_data.get('rot', [0, 0, 0, 1])
            size = obj_data.get('size', [1, 1, 1])

            model_id, sx, sy, sz = self.parse_jid(jid)
            model_path = self.find_model_path(model_id)
            if not model_path or not model_path.exists():
                continue

            mesh = self.load_mesh(model_path)
            if mesh is None:
                continue
            
            # 应用缩放
            mesh.apply_scale([sx, sy, sz])

            # 判断是否在地面上
            is_on_floor = abs(pos[1] - bounds_bottom[0][1]) < 0.01
            
            # 创建副本并应用变换
            mesh_transformed = mesh.copy()
            if len(rot) == 4 and not np.allclose(rot, [0, 0, 0, 1]):
                R_mat = np.eye(4)
                R_mat[:3, :3] = R.from_quat(rot).as_matrix()
                mesh_transformed.apply_transform(R_mat)

            T = np.eye(4)
            T[:3, 3] = pos
            mesh_transformed.apply_transform(T)

            scene.add_geometry(mesh_transformed, geom_name=f"obj_{jid}")

            objects.append(ObjectInfo(
                jid=jid,
                model_id=model_id,
                desc=desc,
                pos=pos,
                rot=rot,
                size=size,
                scale_jid=(sx, sy, sz),
                mesh=mesh,
                is_on_floor=is_on_floor,
            ))

        return scene, objects

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

        import pyrender # type: ignore

        try:
            # 计算相机方向
            forward = camera_target - camera_position

            right, up, cam_z = self.build_camera_basis(forward)

            # 构建相机在世界坐标系中的变换矩阵
            # pyrender 的 pose 是相机在世界空间中的位置和朝向
            cam_transform = np.eye(4)
            cam_transform[:3, 0] = right   # X 轴：右
            cam_transform[:3, 1] = up      # Y 轴：上
            cam_transform[:3, 2] = cam_z   # Z 轴：相机前向（-forward）
            cam_transform[:3, 3] = camera_position

            # 创建 pyrender 场景
            py_scene = pyrender.Scene()

            # 添加 trimesh 几何体（直接使用原始网格，不做任何中间处理）
            for geom_name, geom in scene.geometry.items():
                if isinstance(geom, trimesh.Trimesh):
                    try:
                        # 直接传给 pyrender，让 trimesh 自己处理内部数据
                        mesh = pyrender.Mesh.from_trimesh(geom)
                        py_scene.add(mesh, name=geom_name)
                    except Exception as e:
                        # 记录到日志文件（跳过有问题的几何体，如2通道纹理）
                        # 使用 DEBUG 级别，避免在控制台显示
                        logger.debug(f"Skipping mesh {geom_name}: {e}")
                        return None

            # 添加相机
            aspect_ratio = self.aspect_ratio
            yfov = np.radians(self.fov_degrees)
            camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=aspect_ratio)
            py_scene.add(camera, pose=cam_transform)

            # 添加光源（可选）
            if use_light:
                light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
                py_scene.add(light, pose=cam_transform)

            # 离屏渲染
            r = pyrender.OffscreenRenderer(viewport_width=self.image_size, viewport_height=self.image_size)
            color, _ = r.render(py_scene)
            r.delete()
            
            return color
            
        except Exception as e:
            raise RuntimeError("Pyrender 渲染失败 error: " + str(e))

    def render_top_view(self, scene: trimesh.Scene, bounds_bottom: List[List[float]]) -> np.ndarray:
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

        # 根据 FOV 自适应计算相机高度，确保完整覆盖场景
        fov_rad = np.radians(self.fov_degrees)
        # 图像是正方形 (aspect_ratio=1.0)，视口对角线覆盖场景对角线
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
    ) -> np.ndarray:
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

         # 计算场景对角线
        scene_width = max_x - min_x
        scene_depth = max_y - min_y
        diag = np.sqrt(scene_width**2 + scene_depth**2)

        # 根据 FOV 自适应计算相机高度，确保完整覆盖场景
        fov_rad = np.radians(self.fov_degrees)
        # 图像是正方形 (aspect_ratio=1.0)，视口对角线覆盖场景对角线
        required_height = (diag / 2.0) / np.tan(fov_rad / 2.0)

        # 使用计算的高度与配置高度的最大值，并增加一点余量
        adaptive_height = self.top_view_camera_height + required_height

        center = [(min_x + max_x) / 2, (min_y + max_y) / 2, 0]
        camera_pos = np.array([center[0], adaptive_height, center[1]])
        camera_target = np.array([center[0], bounds_bottom[0][1], center[1]])
        
        return self.render_scene(obj_scene, camera_pos, camera_target)

    def generate_heatmap(
        self,
        target_pos: List[float],
        bounds_bottom: List[List[float]]
    ) -> np.ndarray:
        bounds_bottom = np.array(bounds_bottom)
        min_x = bounds_bottom[:, 0].min()
        min_y = bounds_bottom[:, 2].min()
        max_x = bounds_bottom[:, 0].max()
        max_y = bounds_bottom[:, 2].max()

        scene_center_x = (min_x + max_x) / 2
        scene_center_y = (min_y + max_y) / 2

        # 自适应计算相机高度（与 render_top_view 保持一致）
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

        # ========== 核心修改：逐像素反向投影 ==========

        # 1. 生成图像像素网格（中心为原点）
        px = np.arange(self.image_size) + 0.5  # 像素中心
        py = np.arange(self.image_size) + 0.5
        px, py = np.meshgrid(px, py)  # shape: (H, W)

        # 2. 将像素坐标映射到 NDC（[-1, 1]），再映射到相机坐标系中的射线方向
        # 注意：X 方向需要翻转，因为图像坐标系与相机坐标系 X 方向相反
        ndc_x = -(px / self.image_size) * 2.0 + 1.0  # 翻转 X
        ndc_y = (py / self.image_size) * 2.0 - 1.0

        # 3. 计算在目标深度处的视口半宽/半高
        target = np.array(target_pos)
        target_rel = target - camera_pos
        target_depth = np.dot(target_rel, cam_z)  # 目标中心深度

        fov_rad = np.radians(self.fov_degrees)
        tan_half_fov = np.tan(fov_rad / 2.0)

        viewport_height_half = target_depth * tan_half_fov
        viewport_width_half = viewport_height_half * self.aspect_ratio

        # 4. 像素对应的相机坐标系中的射线（在目标深度处）
        # ray_cam = (x_cam, y_cam, z_cam) 其中 z_cam = target_depth
        ray_x = ndc_x * viewport_width_half
        ray_y = ndc_y * viewport_height_half
        ray_z = np.full_like(ray_x, target_depth)

        # 5. 将射线转换回世界坐标系
        # P_world = camera_pos + x*right + y*up + z*cam_z
        world_x = camera_pos[0] + ray_x * right[0] + ray_y * up[0] + ray_z * cam_z[0]
        world_y = camera_pos[1] + ray_x * right[1] + ray_y * up[1] + ray_z * cam_z[1]
        world_z = camera_pos[2] + ray_x * right[2] + ray_y * up[2] + ray_z * cam_z[2]

        # 6. 在世界坐标系中，计算这些点相对于目标中心的偏移
        # 平面在地面上（Y=0），所以用 X 和 Z 计算距离
        dx = world_x - target_pos[0]
        dy = world_y - target_pos[1]
        dz = world_z - target_pos[2]

        # 7. 采样 2D 高斯（在水平面上的距离）
        sigma_3d = self.heatmap_sigma * 0.01  # 将像素单位的 sigma 转换为世界单位（米）
        dist_sq = dx**2 + dz**2 + dy**2  # 计算点到目标中心的欧氏距离平方
        heatmap = np.exp(-dist_sq / (2 * sigma_3d**2))

        # 8. 处理边界和归一化
        # 只保留在视锥体内的（通过 ndc 范围自然限制）
        heatmap = heatmap.astype(np.float32)

        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val

        return heatmap

    def remove_object_from_scene(
        self,
        scene: trimesh.Scene,
        obj: ObjectInfo,
    ) -> trimesh.Scene:
        """从场景中移除物体"""
        # 复制场景
        new_scene = scene.copy()
        # 移除对应几何体
        geom_name = f"obj_{obj.jid}"
        if geom_name in new_scene.geometry:
            del new_scene.geometry[geom_name]
        return new_scene

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
        rot_y = random.uniform(-self.aug_rotation_range, self.aug_rotation_range)
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

        # 4. 创建增强后的物体
        aug_mesh = obj.mesh.copy()

        if rot_y != 0:
            R_mat = np.eye(4)
            R_mat[:3, :3] = R.from_quat(rot_aug).as_matrix()
            aug_mesh.apply_transform(R_mat)

        T = np.eye(4)
        T[:3, 3] = new_pos
        aug_mesh.apply_transform(T)

        # 5. 更新场景
        new_scene = self.remove_object_from_scene(scene, obj)
        new_scene.add_geometry(aug_mesh, geom_name=f"obj_{obj.jid}")

        # 6. 创建增强后的物体信息
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
            if any(skip in geom_name.lower() for skip in "floor"):
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

    def generate_text_prompt(self, original_image: np.ndarray, plane_image: np.ndarray, object_image: np.ndarray, desc: str) -> str:
        """
        生成 text_prompt：使用 Qwen3-VL 对比 original 和 plane 图像，
        自动生成摆放位置描述（如"放在桌子左边"、"整齐摆放在墙角"等）。
        """
        base_prompt = f"物体{desc}的参考图为：<image>\n平面图为：<image>\n两者的尺寸相同。"

        try:
            # 将 numpy 数组转为 PIL Image
            original_pil = Image.fromarray(original_image)
            plane_pil = Image.fromarray(plane_image)
            obj_pil = Image.fromarray(object_image)

            # 让 Qwen3-VL 对比三张图，描述物体应该摆放的位置
            vlm_prompt = self._query_placement_description(original_pil, plane_pil, obj_pil, desc)
        except Exception as e:
            raise RuntimeError(f"Qwen3-VL 生成位置描述失败: {e}")

        prompt = base_prompt + vlm_prompt
        return prompt

    def _load_qwen_model(self):
        """
        懒加载 Qwen3-VL 模型（仅从本地 models/ 目录加载）。
        """
        if self._qwen_model is not None:
            return

        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor # type: ignore

        # 检查配置是否存在
        if not self._qwen_model_name:
            raise RuntimeError(
                f"Qwen3-VL 模型路径未配置，请在 pretreatment.yaml 中设置 generation.qwen_model_name"
            )

        model_path = Path(self._qwen_model_name)
        if not model_path.exists() or not (model_path / "config.json").exists():
            raise RuntimeError(
                f"Qwen3-VL 本地模型未找到: {model_path}"
            )

        # 加载 processor
        self._qwen_processor = AutoProcessor.from_pretrained(
            model_path,
            use_cache=True,
        )

        attn_impl = "eager"
        if torch.cuda.is_available():
            try:
                import flash_attn # type: ignore
                attn_impl = "flash_attention_2"
            except ImportError:
                attn_impl = "sdpa"

        # 加载模型
        self._qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation=attn_impl,
        )
        self._qwen_model.config.use_cache = True
        self._qwen_model.eval()

    def _query_placement_description(self, original_pil: Image.Image, plane_pil: Image.Image, obj_pil: Image.Image, desc: str) -> str:
        """
        使用 Qwen3-VL 对比原始房间图和剔除后房间图，生成摆放位置描述。
        返回类似"放在桌子左侧"、"整齐摆放在墙角"等自然语言描述。
        """
        # 构建对话
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": original_pil,
                    },
                    {
                        "type": "image",
                        "image": plane_pil,
                    },
                    {
                        "type": "image",
                        "image": obj_pil,
                    },
                    {
                        "type": "text",
                        "text": "第一张图是包含所有物体的原始房间图，第二张图是移除了某个物体后的房间图，第三张图是被移除的物体的参考图。"
                              f"请对比这三张图，用简短的中文描述被移除的物体{desc}原来放在什么位置，以及周围参照物的关系。"
                              "以 '请你将[物体名称]摆放在' 开头。"
                    }
                ]
            }
        ]

        # 生成回复
        text = self._qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._qwen_processor(
            text=[text],
            images=[original_pil, plane_pil, obj_pil],
            return_tensors="pt",
        ).to(self._qwen_model.device)

        from transformers import GenerationConfig # type: ignore
        with torch.no_grad():
            outputs = self._qwen_model.generate(
                **inputs,
                generation_config=GenerationConfig(
                    max_new_tokens=512,
                    do_sample=False,
                ),
            )

        input_len = inputs["input_ids"].shape[1]
        response = self._qwen_processor.decode(outputs[0, input_len:], skip_special_tokens=True).strip()

        return response

    def generate_response(self, text_prompt: str, rotation_6d: List[float], scale: float) -> str:
        """
        生成 response：使用 Qwen3-VL 根据 text_prompt 生成自然回复，
        并在末尾硬编码加上绕 Y 轴的旋转角度和缩放比例。
        """
        try:
            if self._qwen_model is None:
                raise RuntimeError("Qwen3-VL 模型未加载")

            rot_y_deg = self._extract_rotation_y(rotation_6d)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"你是一个物体放置助手。用户给出了放置指令，请你用礼貌的语气回复，"
                                   f"并在末尾加上<SEG>标记。"
                                   f"\n指令：{text_prompt}"
                                   f"\n旋转角度：{rot_y_deg:.1f}°（绕Y轴）"
                                   f"\n缩放比例：{scale:.2f}"
                                   f"\n请用'好的，我会...'开头回复，说明放置位置、旋转角度和缩放比例，"
                                   f"并在句末加上<SEG>。"
                        }
                    ]
                }
            ]

            text = self._qwen_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self._qwen_processor(
                text=[text],
                return_tensors="pt",
            ).to(self._qwen_model.device)

            from transformers import GenerationConfig # type: ignore
            with torch.no_grad():
                outputs = self._qwen_model.generate(
                    **inputs,
                    generation_config=GenerationConfig(
                        max_new_tokens=512,
                        do_sample=False,
                    ),
                )

            input_len = inputs["input_ids"].shape[1]
            response = self._qwen_processor.decode(outputs[0, input_len:], skip_special_tokens=True).strip()

            # 确保以 <SEG> 结尾
            if "<SEG>" not in response:
                response += "<SEG>"

            return response
        except Exception as e:
            raise RuntimeError(f"Qwen3-VL 生成 response 失败: {e}")

    def _extract_rotation_y(self, rotation_6d: List[float]) -> float:
        """
        从 6D 旋转表示中提取绕 Y 轴的旋转角度（度数）。

        6D rotation 是旋转矩阵 R 的前两列：
        [r11, r21, r31, r12, r22, r32]

        绕 Y 轴旋转角度 = atan2(-R[2][0], R[2][2])
        """
        import math

        r11, r21, r31, r12, r22, r32 = rotation_6d

        # 计算第三列（叉乘）
        r33 = r11 * r22 - r21 * r12

        # 提取绕 Y 轴的旋转角度
        rot_y = math.atan2(-r31, r33)
        rot_y_deg = math.degrees(rot_y)

        # 规范化到 [-180, 180]
        if rot_y_deg > 180:
            rot_y_deg -= 360
        elif rot_y_deg < -180:
            rot_y_deg += 360

        return rot_y_deg

    def rotation_6d_from_quat(self, quat: List[float]) -> List[float]:
        """四元数转 6D 旋转"""
        r = R.from_quat(quat)
        R_mat = r.as_matrix()
        # 取前两列
        return [
            R_mat[0, 0], R_mat[1, 0], R_mat[2, 0],
            R_mat[0, 1], R_mat[1, 1], R_mat[2, 1],
        ]

    def save_sample(
        self,
        scene_dir: Path,
        obj_id: str,
        plane_image: np.ndarray,
        object_image: np.ndarray,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        text_prompt: str,
        response: str,
        rotation_6d: List[float],
        scale: float,
        split: str = "train",
    ) -> Optional[dict]:
        """保存单个样本到场景目录中"""
        self.sample_counter += 1
        sample_id = f"obj_{obj_id}_{self.sample_counter:06d}"

        # 创建子目录
        plane_dir = scene_dir / "plane_images"
        object_dir = scene_dir / "object_images"
        mask_dir = scene_dir / "masks"
        original_dir = scene_dir / "original_images"
        plane_dir.mkdir(exist_ok=True)
        object_dir.mkdir(exist_ok=True)
        mask_dir.mkdir(exist_ok=True)
        original_dir.mkdir(exist_ok=True)

        # 保存图片
        plane_path = plane_dir / f"{sample_id}.png"
        object_path = object_dir / f"{sample_id}.png"
        mask_path = mask_dir / f"{sample_id}_mask.png"
        original_path = original_dir / f"{sample_id}.png"

        Image.fromarray(plane_image).save(plane_path)
        Image.fromarray(object_image).save(object_path)
        Image.fromarray(original_image).save(original_path)

        # 热力图转为灰度图
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        Image.fromarray(heatmap_uint8).save(mask_path)

        # 返回样本元数据
        metadata = {
            "sample_id": sample_id,
            "split": split,
            "scene_dir": str(scene_dir.relative_to(self.output_dir)),
            "plane_image_path": f"plane_images/{sample_id}.png",
            "images_paths": [
                f"object_images/{sample_id}.png",
                f"plane_images/{sample_id}.png",
            ],
            "mask_path": f"masks/{sample_id}_mask.png",
            "text_prompt": text_prompt,
            "response": response,
            "rotation_6d": rotation_6d,
            "scale": scale,
        }
        # 收集到全局列表
        self.samples_by_split[split].append(metadata)
        return metadata

    def process_scene(
        self,
        json_path: Path,
        split: str = "train",
    ):
        """处理单个场景 JSON, 生成样本"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                scene_data = json.load(f)
        except Exception:
            return

        # 构建场景
        scene, objects = self.build_scene(scene_data)
        if not objects:
            return

        # 渲染原始房间图（用于调试叠加）
        original_image = self.render_top_view(scene, scene_data.get('bounds_bottom', []))
        if original_image is None:
            return

        # 创建场景输出目录（确认有有效数据后再创建）
        scene_name = json_path.stem
        scene_dir = self.output_dir / split / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)

        # 对场景中的每个物体生成一个样本（按 jid 去重，限制最大剔除比例）
        max_objects = min(len(objects), self.max_object_nums)
        objects_processed = 0
        
        random.shuffle(objects)

        for target_obj in objects:
            
            if objects_processed >= max_objects:
                break
            
            if not target_obj.is_on_floor:
                # 只处理在地面上的物体，跳过悬浮物体
                continue

            # 数据增强（仅在 train 阶段，按 aug_ratio 比例决定走增强流程还是普通流程）
            is_aug = split == "train" and self.augmentation and random.random() < self.aug_ratio

            if is_aug:
                # 增强流程
                aug_meta = self._process_augmentation(
                    scene, scene_data, scene_dir, split, target_obj
                )
                if aug_meta is not None:
                    objects_processed += 1
                continue
            
            # 1. 计算旋转和缩放标签（倒数）
            orig_rot = target_obj.rot
            if len(orig_rot) == 4 and not np.allclose(orig_rot, [0, 0, 0, 1]):
                inv_rot = R.from_quat(orig_rot).inv().as_quat().tolist()
            else:
                inv_rot = [0, 0, 0, 1]
            rotation_6d = self.rotation_6d_from_quat(inv_rot)

            orig_scale = random.uniform(*self.scale_range)
            scale = 1.0 / orig_scale
            
            target_obj.mesh.apply_scale(orig_scale)
            
            # 2. 渲染物体参考图
            object_image = self.render_object_reference(target_obj.mesh, scene_data.get('bounds_bottom', []))
            if object_image is None:
                continue

            # 3. 生成 GT 热力图
            heatmap = self.generate_heatmap(
                target_pos=target_obj.pos,
                bounds_bottom=scene_data.get('bounds_bottom', [])
            )
            if heatmap is None:
                continue

            # 4. 剔除物体，渲染剔除后房间图
            scene_without_obj = self.remove_object_from_scene(scene, target_obj)
            plane_image = self.render_top_view(scene_without_obj, scene_data.get('bounds_bottom', []))
            if plane_image is None:
                continue
            
            # 5. 生成文本（失败则跳过该样本）
            try:
                text_prompt = self.generate_text_prompt(
                    original_image, plane_image, object_image, target_obj.desc
                )
                response = self.generate_response(text_prompt, rotation_6d, scale)
            except RuntimeError:
                logger.warning(f"跳过样本：Qwen3-VL 文本生成失败 (obj: {target_obj.jid})")
                continue

            # 6. 保存样本
            sample_meta = self.save_sample(
                scene_dir=scene_dir,
                obj_id=target_obj.jid,
                plane_image=plane_image,
                object_image=object_image,
                original_image=original_image,
                heatmap=heatmap,
                text_prompt=text_prompt,
                response=response,
                rotation_6d=rotation_6d,
                scale=scale,
                split=split,
            )
            if sample_meta is not None:
                objects_processed += 1

        logger.info(f"场景 {scene_name}: 生成 {objects_processed} 个样本")

    def _process_augmentation(
        self,
        scene: trimesh.Scene,
        scene_data: dict,
        scene_dir: Path,
        split: str,
        target_obj: ObjectInfo,
    ) -> Optional[dict]:
        """数据增强：对指定物体进行随机变换，生成增强样本"""
        aug_obj, aug_scene = self.augmentation_object(target_obj, scene)
        if aug_obj is None:
            return None

        # 渲染增强后物体参考图
        aug_object_image = self.render_object_reference(aug_obj.mesh, scene_data.get('bounds_bottom', []))
        
        # 渲染增强后原始房间图（用于调试叠加）
        aug_original_image = self.render_top_view(aug_scene, scene_data.get('bounds_bottom', []))

        # 生成 GT 热力图（基于移动后位置）
        aug_heatmap = self.generate_heatmap(
            target_pos=aug_obj.pos,
            bounds_bottom=scene_data.get('bounds_bottom', [])
        )
        if aug_heatmap is None:
            return None

        # 计算旋转和缩放标签
        aug_rot_6d = self.rotation_6d_from_quat(
            R.from_quat(aug_obj.rot).inv().as_quat().tolist()
        )
        aug_scale = 1.0

        # 渲染剔除后房间图
        aug_plane_image = self.render_top_view(
            self.remove_object_from_scene(aug_scene, aug_obj),
            scene_data.get('bounds_bottom', [])
        )
        if aug_plane_image is None:
            return None

        # 生成文本
        try:
            aug_text_prompt = self.generate_text_prompt(
                aug_original_image, aug_plane_image, aug_object_image, aug_obj.desc
            )
            aug_response = self.generate_response(aug_text_prompt, aug_rot_6d, aug_scale)
        except RuntimeError:
            logger.warning(f"跳过增强样本：Qwen3-VL 文本生成失败")
            return None

        sample_meta = self.save_sample(
            scene_dir=scene_dir,
            obj_id=target_obj.jid,
            plane_image=aug_plane_image,
            object_image=aug_object_image,
            original_image=aug_original_image,
            heatmap=aug_heatmap,
            text_prompt=aug_text_prompt,
            response=aug_response,
            rotation_6d=aug_rot_6d,
            scale=aug_scale,
            split=split,
        )

        return sample_meta

    def run(self):
        """执行数据生成"""
        logger.info(f"扫描场景目录: {self.scene_dir}")
        json_files = sorted(self.scene_dir.glob("*.json"))
        logger.info(f"找到 {len(json_files)} 个场景文件")

        # 随机打乱场景列表
        random.shuffle(json_files)

        # 按 split_ratio 划分
        n = len(json_files)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        n_test = n - n_train - n_val  # 剩余全部分给 test

        splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test

        self._load_qwen_model()

        for idx, json_path in enumerate(tqdm.tqdm(json_files, desc="Processing scenes")):
            self.process_scene(json_path, split=splits[idx])

        # 保存每个 split 的合并 JSON 文件
        for split_name in ["train", "val", "test"]:
            split_samples = self.samples_by_split[split_name]
            if split_samples:
                split_dir = self.output_dir / split_name
                output_path = split_dir / f"{split_name}.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(split_samples, f, ensure_ascii=False, indent=2)
                logger.info(f"保存 {split_name} 数据: {len(split_samples)} 个样本 -> {output_path}")

        logger.info(f"\n生成完成! 总样本数: {self.sample_counter}")
