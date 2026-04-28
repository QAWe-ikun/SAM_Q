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
        heatmap_config = config.get("heatmap", {})

        self.scene_dir = Path(data_config.get("scene_dir", ""))
        self.model_dir = Path(data_config.get("model_dir", ""))
        self.output_dir = Path(data_config.get("output_dir", ""))

        self.image_size = gen_config.get("image_size", 1024)
        self.heatmap_sigma = gen_config.get("heatmap_sigma", 15.0)
        self.num_samples = gen_config.get("num_samples", 1000)
        
        self.augmentation = aug_config.get("enabled", False)
        self.aug_ratio = aug_config.get("aug_ratio", 0.5)
        
        # 更新全局配置
        self.aug_rotation_range = aug_config.get("rotation_range", 180)
        scale_range = aug_config.get("scale_range", {})
        self.aug_scale_range = (
            scale_range.get("min", 0.8),
            scale_range.get("max", 1.5)
        )
        

        self.top_view_camera_height = cam_config.get("top_view_height", 8.0)
        self.side_view_camera_distance = cam_config.get("side_view_distance", 5.0)
        self.fov_degrees = cam_config.get("fov_degrees", 45)
        
        self.heatmap_min_prob = heatmap_config.get("min_prob", 0.01)

        # 输出子目录
        self.plane_dir = self.output_dir / "plane_images"
        self.object_dir = self.output_dir / "object_images"
        self.mask_dir = self.output_dir / "masks"
        self.original_image_dir = self.output_dir / "original_images"
        self.plane_dir.mkdir(parents=True, exist_ok=True)
        self.object_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        self.original_image_dir.mkdir(parents=True, exist_ok=True)

        self.annotations = []
        self.sample_counter = 0

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
                mesh=mesh,  # 原始 mesh（未变换）
                is_on_floor=is_on_floor,
            ))

        return scene, objects

    def render_scene(
        self,
        scene: trimesh.Scene,
        camera_position: List[float],
        camera_target: List[float],
        use_light: bool = True,
    ) -> np.ndarray:
        """渲染场景为 RGB 图像（使用 pyrender 离屏渲染）"""

        import pyrender

        try:
            camera_position = np.array(camera_position, dtype=np.float64)
            camera_target = np.array(camera_target, dtype=np.float64)
            camera_position = camera_position[[0,2,1]]  # XYZ -> XZY
            camera_target = camera_target[[0,2,1]]  # XYZ -> XZY

            # 计算相机方向
            forward = camera_target - camera_position
            forward = forward / np.linalg.norm(forward)

            # 使用 Z 轴作为世界坐标系的上方向（trimesh 坐标系：Z 向上）
            world_up = np.array([0.0, 0.0, 1.0])

            # 计算右向量
            right = np.cross(forward, world_up)
            right = right / np.linalg.norm(right)

            # 重新计算真正的上方向（垂直于 forward 和 right）
            up = np.cross(right, forward)

            # 构建相机在世界坐标系中的变换矩阵
            # pyrender 的 pose 是相机在世界空间中的位置和朝向
            cam_transform = np.eye(4)
            cam_transform[:3, 0] = right   # X 轴：右
            cam_transform[:3, 1] = up      # Y 轴：上
            cam_transform[:3, 2] = -forward  # Z 轴：前（相机朝向）
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
                        continue

            # 添加相机
            aspect_ratio = 1.0
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
            # 回退到 2D 投影
            raise RuntimeError("Pyrender 渲染失败 error: " + str(e))

    def render_top_view(self, scene: trimesh.Scene, bounds_bottom: List[List[float]]) -> np.ndarray:
        """渲染俯视图（从 Z 轴正方向往下看，投影到 XY 平面）"""
        bounds_bottom = np.array(bounds_bottom)
        min_x = bounds_bottom[:, 0].min()
        min_y = bounds_bottom[:, 1].min()
        max_x = bounds_bottom[:, 0].max()
        max_y = bounds_bottom[:, 1].max()
        center = [(min_x + max_x) / 2, (min_y + max_y) / 2, 0]
        camera_pos = [center[0], center[1], self.top_view_camera_height]
        camera_target = [center[0], center[1], bounds_bottom[0][2]]
        return self.render_scene(scene, camera_pos, camera_target)

    def render_object_reference(
        self,
        mesh: trimesh.Trimesh,
    ) -> np.ndarray:
        """渲染物体参考图（俯视图，正向朝上，保持原始尺寸）"""
        # 复制 mesh 并重置所有变换
        obj_mesh = mesh.copy()

        # 将物体居中
        obj_mesh.apply_translation(-obj_mesh.centroid)

        # 创建只包含该物体的场景
        obj_scene = trimesh.Scene([obj_mesh])
        
        # 使用与 render_top_view 相同的相机配置
        bounds = obj_scene.bounds
        if bounds is None:
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        center = bounds.mean(axis=0)
        # 相机位置：在物体正上方
        camera_pos = [center[0], center[1], bounds[1][2] + self.top_view_camera_height]
        # 相机目标：物体中心
        camera_target = [center[0], center[1], bounds[0][2]]
        
        return self.render_scene(obj_scene, camera_pos, camera_target)

    def generate_heatmap(
        self,
        target_pos: List[float],
        bounds_bottom: List[List[float]]
    ) -> np.ndarray:
        """
        生成 GT 热力图（通过渲染高斯颜色平面实现）。

        参数:
            target_pos: 目标物体位置 [x, y, z]
            bounds_bottom: 场景底部边界
        """
        bounds_bottom = np.array(bounds_bottom)
        min_x = bounds_bottom[:, 0].min()
        min_y = bounds_bottom[:, 1].min()
        max_x = bounds_bottom[:, 0].max()
        max_y = bounds_bottom[:, 1].max()
        
        # 在目标位置创建一个小正方形平面
        square_size = 0.5  # 正方形边长
        cx, cy = target_pos[0], target_pos[2]
        cz = target_pos[1]

        # 创建正方形的四个顶点（在 XY 平面）
        vertices = np.array([
            [cx - square_size/2, cz, cy + square_size/2],  # 左上
            [cx + square_size/2, cz, cy + square_size/2],  # 右上
            [cx + square_size/2, cz, cy - square_size/2],  # 右下
            [cx - square_size/2, cz, cy - square_size/2],  # 左下
        ])

        # 创建两个三角形面片
        faces = self._triangulate_polygon(len(vertices))

        # 创建平面 mesh
        heatmap_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        
        heatmap_mesh.fix_normals()   # 确保法向量朝外
        
        # 创建面颜色（使用红色测试）
        # 使用 trimesh.visual.color 设置面颜色（RGBA）
        n_vertices = len(heatmap_mesh.vertices)
        heatmap_mesh.visual.vertex_colors = np.tile(
            [255, 0, 0, 255], 
            (n_vertices, 1)
        ).astype(np.uint8)

        # 确认 visual.kind
        assert heatmap_mesh.visual.kind == 'vertex', f"Expected 'vertex', got {heatmap_mesh.visual.kind}"
        
        # 创建只包含热力图平面的场景
        heatmap_scene = trimesh.Scene([heatmap_mesh])
        
        # 使用与 render_top_view 相同的相机参数渲染
        center = [(min_x + max_x) / 2, (min_y + max_y) / 2, 0]
        camera_pos = [center[0], center[1], self.top_view_camera_height]
        camera_target = [center[0], center[1], bounds_bottom[0][2]]

        # 调试信息
        logger.debug(f"热力图平面 - 中心位置: ({cx}, {cy}, {cz})")
        logger.debug(f"相机位置: {camera_pos}, 目标位置: {camera_target}")

        rendered = self.render_scene(heatmap_scene, camera_pos, camera_target)

        # 调试信息
        logger.debug(f"渲染结果范围: {rendered.min()} - {rendered.max()}")
        logger.debug(f"渲染结果 R 通道范围: {rendered[:, :, 0].min()} - {rendered[:, :, 0].max()}")

        # 提取单通道（灰度图，R 通道即可）
        heatmap = rendered[:, :, 0].astype(np.float32) / 255.0

        logger.debug(f"最终热力图范围: {heatmap.min():.3f} - {heatmap.max():.3f}")

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
        # 1. 随机旋转（Z 轴）
        rot_z = random.uniform(-self.aug_rotation_range, self.aug_rotation_range)
        rot_aug = R.from_euler('z', rot_z, degrees=True).as_quat().tolist()

        # 2. 随机缩放
        scale_aug = random.uniform(*self.aug_scale_range)

        # 3. 随机移动到新位置（不与其他物体碰撞）
        bounds = scene.bounds
        if bounds is None:
            return None, None
        x_min, y_min, z_min = bounds[0]
        x_max, y_max, z_max = bounds[1]
        for _ in range(100):  # 最多尝试 100 次
            new_pos = [
                random.uniform(x_min, x_max),
                random.uniform(y_min, y_max),
                obj.pos[2],  # 保持 Z 值
            ]
            # 检查碰撞
            if not self._check_position_collision(scene, new_pos, obj.mesh, scale_aug):
                break
        else:
            return None, None  # 无法找到合适位置

        # 4. 创建增强后的物体
        aug_mesh = obj.mesh.copy()
        aug_mesh.apply_scale(scale_aug)

        if rot_z != 0:
            R_mat = np.eye(4)
            R_mat[:3, :3] = R.from_quat(rot_aug).as_matrix()
            aug_mesh.apply_transform(R_mat)

        T = np.eye(4)
        T[:3, 3] = new_pos
        aug_mesh.apply_transform(T)

        # 5. 更新场景
        new_scene = self.remove_object_from_scene(scene, obj)
        new_scene.add_geometry(aug_mesh, geom_name=f"obj_{obj.jid}_aug")

        # 6. 创建增强后的物体信息
        aug_obj = ObjectInfo(
            jid=obj.jid,
            model_id=obj.model_id,
            desc=obj.desc,
            pos=new_pos,
            rot=rot_aug,
            size=obj.size,
            scale_jid=(scale_aug, scale_aug, scale_aug),
            mesh=obj.mesh,  # 原始 mesh
            is_on_floor=obj.is_on_floor,
        )

        return aug_obj, new_scene

    def _check_position_collision(
        self,
        scene: trimesh.Scene,
        pos: List[float],
        mesh: trimesh.Trimesh,
        scale: float,
    ) -> bool:
        """检查指定位置是否与其他物体碰撞（使用 AABB 包围盒）"""
        test_mesh = mesh.copy()
        test_mesh.apply_scale(scale)
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

    def generate_text_prompt(self, obj_name: str, is_aug: bool = False) -> str:
        """生成 text_prompt"""
        if is_aug:
            return f"<image>\n<image>\n请把{obj_name}放回原来的位置"
        return f"<image>\n<image>\n请把{obj_name}放回原来的位置"

    def generate_response(self, obj_name: str) -> str:
        """生成 response"""
        return f"好的，我会把{obj_name}放回原来的位置。<SEG>"

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
        plane_image: np.ndarray,
        object_image: np.ndarray,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        text_prompt: str,
        response: str,
        rotation_6d: List[float],
        scale: float,
        split: str = "train",
    ):
        """保存单个样本"""
        self.sample_counter += 1
        scene_id = f"scene_{self.sample_counter:06d}"

        # 保存图片
        plane_path = self.plane_dir / f"{scene_id}.png"
        object_path = self.object_dir / f"{scene_id}.png"
        mask_path = self.mask_dir / f"{scene_id}.png"
        original_image_path = self.original_image_dir / f"{scene_id}.png"

        Image.fromarray(plane_image).save(plane_path)
        Image.fromarray(object_image).save(object_path)
        Image.fromarray(original_image).save(original_image_path)

        # 热力图转为伪彩色（可选）或灰度图
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        Image.fromarray(heatmap_uint8).save(mask_path)

        # 添加到 annotations
        self.annotations.append({
            "scene_id": scene_id,
            "split": split,
            "plane_image_path": f"plane_images/{scene_id}.png",
            "images_path": [
                f"plane_images/{scene_id}.png",
                f"object_images/{scene_id}.png",
            ],
            "mask_path": f"masks/{scene_id}.png",
            "text_prompt": text_prompt,
            "response": response,
            "rotation_6d": rotation_6d,
            "scale": scale,
        })

    def process_scene(
        self,
        json_path: Path,
        split: str = "train",
    ):
        """处理单个场景 JSON，生成样本"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                scene_data = json.load(f)
        except Exception:
            return

        # 构建场景
        scene, objects = self.build_scene(scene_data)
        if not objects:
            return

        # 随机选择目标物体
        target_obj = random.choice(objects)

        # 1. 渲染原始房间图
        original_image = self.render_top_view(scene, scene_data.get('bounds_bottom', []))

        # 2. 渲染物体参考图（俯视图，正向，缩放=1）
        object_image = self.render_object_reference(target_obj.mesh)

        # 3. 生成 GT 热力图
        heatmap = self.generate_heatmap(
            target_pos=target_obj.pos,
            bounds_bottom=scene_data.get('bounds_bottom', [])
        )

        # 4. 生成文本
        text_prompt = self.generate_text_prompt(target_obj.model_id, is_aug=False)
        response = self.generate_response(target_obj.model_id)

        # 5. 计算旋转和缩放标签（倒数）
        # 原始旋转的倒数
        orig_rot = target_obj.rot
        if len(orig_rot) == 4 and not np.allclose(orig_rot, [0, 0, 0, 1]):
            inv_rot = R.from_quat(orig_rot).inv().as_quat().tolist()
        else:
            inv_rot = [0, 0, 0, 1]
        rotation_6d = self.rotation_6d_from_quat(inv_rot)

        # 原始缩放的倒数
        orig_scale = np.mean(target_obj.scale_jid)
        scale = 1.0 / orig_scale if orig_scale > 0 else 1.0

        # 6. 剔除物体，渲染剔除后房间图
        scene_without_obj = self.remove_object_from_scene(scene, target_obj)
        plane_image = self.render_top_view(scene_without_obj, scene_data.get('bounds_bottom', []))

        # 7. 保存样本
        self.save_sample(
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

        # 8. 数据增强（如果启用）
        if self.augmentation:
            aug_obj, aug_scene = self.augmentation_object(target_obj, scene)
            if aug_obj is not None:
                # 渲染增强后物体参考图（仍然正向，缩放=1）
                aug_object_image = self.render_object_reference(aug_obj.mesh)

                # 生成 GT 热力图（基于移动后位置）
                aug_heatmap = self.generate_heatmap(
                    target_pos=aug_obj.pos,
                    bounds_bottom=scene_data.get('bounds_bottom', [])
                )

                # 计算旋转和缩放标签
                # 数据增强：旋转为随机生成的旋转的倒数，缩放为 1
                aug_rot_6d = self.rotation_6d_from_quat(
                    R.from_quat(aug_obj.rot).inv().as_quat().tolist()
                )
                aug_scale = 1.0  # 物体参考图缩放为 1

                # 渲染剔除后房间图
                aug_plane_image = self.render_top_view(
                    self.remove_object_from_scene(aug_scene, aug_obj),
                    scene_data.get('bounds_bottom', [])
                )

                self.save_sample(
                    plane_image=aug_plane_image,
                    object_image=aug_object_image,
                    original_image=original_image,
                    heatmap=aug_heatmap,
                    text_prompt=self.generate_text_prompt(aug_obj.model_id, is_aug=True),
                    response=self.generate_response(aug_obj.model_id),
                    rotation_6d=aug_rot_6d,
                    scale=aug_scale,
                    split=split,
                )

    def save_annotations(self):
        """保存 annotations.json 并按 split 划分"""
        # 简单划分：80% train, 10% val, 10% test
        random.shuffle(self.annotations)
        n = len(self.annotations)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)

        for ann in self.annotations[:n_train]:
            ann["split"] = "train"
        for ann in self.annotations[n_train:n_train + n_val]:
            ann["split"] = "val"
        for ann in self.annotations[n_train + n_val:]:
            ann["split"] = "test"

        output_path = self.output_dir / "annotations.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, ensure_ascii=False, indent=2)
        logger.info(f"Annotations saved to {output_path} ({len(self.annotations)} samples)")

    def run(self):
        """执行数据生成"""
        logger.info(f"扫描场景目录: {self.scene_dir}")
        json_files = sorted(self.scene_dir.glob("*.json"))
        logger.info(f"找到 {len(json_files)} 个场景文件")

        for _, json_path in tqdm.tqdm(enumerate(json_files, 1), total=len(json_files)):
            self.process_scene(json_path)

            if self.sample_counter >= self.num_samples:
                break

        logger.info(f"\n生成完成! 总样本数: {self.sample_counter}")
        self.save_annotations()
