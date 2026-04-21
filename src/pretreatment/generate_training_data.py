"""
SAM-Q 训练集自动化生成器

从 SSR3D-FRONT 场景 JSON 直接生成训练/验证/测试数据，无需保存中间 GLB。

流程:
1. 加载场景 JSON，构建完整 3D 场景
2. 渲染原始房间俯视图/侧视图
3. VLM 识别所有物体及位置（可选）
4. 随机选择一个目标物体
5. 生成物体参考图（俯视图，正向朝上，缩放=1）
6. 剔除目标物体，渲染剔除后房间图
7. 生成 GT 热力图（原位置最高概率，碰撞区域为 0）
8. 保存样本（annotations.json + 图片）

数据增强:
- 随机移动物体到不重叠位置
- 随机旋转 Z 轴，缩放随机变化
- 预测移动后的位置 + 旋转/缩放的倒数

使用方法:
    python generate_training_data.py \\
        --scene_dir d:/3D-Dataset/dataset-ssr3dfront/scenes \\
        --model_dir d:/3D-Dataset/3D-FUTURE-model \\
        --output_dir data/ \\
        --num_samples 1000 \\
        --augmentation \\
        --aug_ratio 0.5
"""

import cv2
import json
import random
import trimesh
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from scipy.spatial.transform import Rotation as R

# ========== 配置路径 ==========
BASE_DIR = Path(r"d:/3D-Dataset")
SCENE_DIR = BASE_DIR / "dataset-ssr3dfront" / "scenes"
MODEL_DIR = BASE_DIR / "3D-FUTURE-model"
OUTPUT_DIR = BASE_DIR / "dataset-ssr3dfront" / "output"
# ==============================

# 渲染相机配置
TOP_VIEW_CAMERA_HEIGHT = 5.0  # 俯视图相机高度
SIDE_VIEW_CAMERA_DISTANCE = 5.0  # 侧视图相机距离
IMAGE_SIZE = 1024
FOV_DEGREES = 45

# 热力图配置
HEATMAP_SIGMA = 15  # 高斯核标准差
HEATMAP_MIN_PROB = 0.01  # 最小概率阈值

# 数据增强配置
AUG_ROTATION_RANGE = 180  # Z 轴旋转范围（度）
AUG_SCALE_RANGE = (0.8, 1.5)  # 缩放范围

# 物体类别到中文映射（示例）
CATEGORY_ZH = {
    "chair": "椅子",
    "table": "桌子",
    "sofa": "沙发",
    "bed": "床",
    "cabinet": "柜子",
    "desk": "书桌",
    "lamp": "台灯",
    "plant": "盆栽",
    "shelf": "架子",
    "stool": "凳子",
}


@dataclass
class ObjectInfo:
    """物体信息"""
    jid: str
    model_id: str
    name: str
    pos: List[float]
    rot: List[float]
    size: List[float]
    scale_jid: Tuple[float, float, float]
    mesh: trimesh.Trimesh
    is_on_floor: bool  # 是否在地面上
    is_on_wall: bool   # 是否在墙面上


class TrainingDataGenerator:
    """SAM-Q 训练数据生成器"""

    def __init__(
        self,
        scene_dir: Path,
        model_dir: Path,
        output_dir: Path,
        image_size: int = IMAGE_SIZE,
        heatmap_sigma: float = HEATMAP_SIGMA,
        augmentation: bool = False,
        aug_ratio: float = 0.5,
        num_samples: int = 1000,
    ):
        self.scene_dir = scene_dir
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.image_size = image_size
        self.heatmap_sigma = heatmap_sigma
        self.augmentation = augmentation
        self.aug_ratio = aug_ratio
        self.num_samples = num_samples

        # 输出子目录
        self.plane_dir = output_dir / "plane_images"
        self.object_dir = output_dir / "object_images"
        self.mask_dir = output_dir / "masks"
        self.plane_dir.mkdir(parents=True, exist_ok=True)
        self.object_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)

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
        """加载并合并网格"""
        try:
            loaded = trimesh.load(model_path, force='scene')
            if isinstance(loaded, trimesh.Scene):
                meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
                if not meshes:
                    return None
                return trimesh.util.concatenate(meshes)
            return loaded
        except Exception:
            return None

    def get_object_name(self, model_id: str) -> str:
        """获取物体类别中文名"""
        for en, zh in CATEGORY_ZH.items():
            if en in model_id.lower():
                return zh
        return "物品"

    def build_scene(self, scene_data: Dict) -> Tuple[trimesh.Scene, List[ObjectInfo]]:
        """构建完整场景，返回场景和物体列表"""
        scene = trimesh.Scene()

        # 添加地板、天花板、墙壁
        bounds_top = scene_data.get('bounds_top', [])
        bounds_bottom = scene_data.get('bounds_bottom', [])
        if bounds_top and bounds_bottom:
            floor_v = np.array(bounds_bottom)
            floor_f = self._triangulate_polygon(len(bounds_bottom))
            scene.add_geometry(
                trimesh.Trimesh(vertices=floor_v, faces=floor_f, process=False),
                geom_name="floor"
            )

        # 加载所有家具
        objects = []
        for obj_data in scene_data.get('objects', []):
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

            # 缩放到期望尺寸
            extents = mesh.extents
            if np.all(extents > 0):
                scale = np.mean(np.array(size) / extents)
                mesh.apply_scale(scale)
                mesh.apply_scale([sx, sy, sz])

            # 判断是否在地面上（Z 值接近 0）
            is_on_floor = abs(pos[2]) < 0.05

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
                name=self.get_object_name(model_id),
                pos=pos,
                rot=rot,
                size=size,
                scale_jid=(sx, sy, sz),
                mesh=mesh,  # 原始 mesh（未变换）
                is_on_floor=is_on_floor,
                is_on_wall=False,  # TODO: 判断是否在墙面
            ))

        return scene, objects

    def _triangulate_polygon(self, n: int) -> np.ndarray:
        """多边形三角化"""
        faces = []
        for i in range(1, n - 1):
            faces.append([0, i, i + 1])
        return np.array(faces)

    def render_scene(
        self,
        scene: trimesh.Scene,
        camera_position: List[float],
        camera_target: List[float],
        resolution: int = IMAGE_SIZE,
    ) -> np.ndarray:
        """渲染场景为 RGB 图像"""
        # 创建相机
        camera = trimesh.scene.cameras.Camera(
            fov=(FOV_DEGREES, FOV_DEGREES),
            resolution=(resolution, resolution),
            z_near=0.01,
            z_far=100.0,
        )

        # 设置相机位置和朝向
        scene.camera = camera
        scene.camera_transform = trimesh.transformations.look_at(
            camera_position, camera_target, [0, 0, 1]
        )

        # 渲染
        png = scene.save_image(
            resolution=(resolution, resolution),
            visible=True,
        )

        # 转换为 numpy 数组
        img = np.frombuffer(png, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def render_top_view(self, scene: trimesh.Scene) -> np.ndarray:
        """渲染俯视图"""
        bounds = scene.bounds
        center = bounds.mean(axis=0)
        camera_pos = [center[0], center[1], bounds[1][2] + TOP_VIEW_CAMERA_HEIGHT]
        camera_target = [center[0], center[1], 0]
        return self.render_scene(scene, camera_pos, camera_target)

    def render_side_view(
        self,
        scene: trimesh.Scene,
        wall_index: int = 0,
    ) -> np.ndarray:
        """渲染侧视图（面向指定墙面）"""
        bounds = scene.bounds
        center = bounds.mean(axis=0)
        # 面向墙面的相机位置
        camera_pos = [center[0] - SIDE_VIEW_CAMERA_DISTANCE, center[1], center[2]]
        camera_target = [center[0], center[1], center[2]]
        return self.render_scene(scene, camera_pos, camera_target)

    def render_object_reference(
        self,
        mesh: trimesh.Trimesh,
        resolution: int = IMAGE_SIZE,
    ) -> np.ndarray:
        """渲染物体参考图（俯视图，正向朝上，缩放=1）"""
        # 复制并重置旋转
        obj_mesh = mesh.copy()
        # 确保正向朝上（绕 Z 轴旋转 0 度）
        # 缩放到单位尺寸
        extents = obj_mesh.extents
        if np.all(extents > 0):
            obj_mesh.apply_scale(1.0 / extents[0])

        # 居中
        obj_mesh.apply_translation(-obj_mesh.centroid)

        scene = trimesh.Scene([obj_mesh])
        camera_pos = [0, 0, 2.0]
        camera_target = [0, 0, 0]
        return self.render_scene(scene, camera_pos, camera_target, resolution)

    def generate_heatmap(
        self,
        scene: trimesh.Scene,
        target_pos: List[float],
        image_size: int = IMAGE_SIZE,
        sigma: float = HEATMAP_SIGMA,
    ) -> np.ndarray:
        """
        生成 GT 热力图。

        规则:
        1. 中心点（target_pos）概率最高
        2. 四周高斯衰减
        3. 碰撞区域概率直接为 0
        """
        # 1. 生成高斯热力图
        heatmap = np.zeros((image_size, image_size), dtype=np.float32)
        center_x, center_y = self._world_to_image(target_pos, scene.bounds, image_size)

        # 高斯核
        y, x = np.ogrid[:image_size, :image_size]
        heatmap = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
        
        # 归一化到 [0, 1]（避免除 0）
        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val

        # 2. 碰撞检测：将碰撞区域置零
        collision_mask = self._compute_collision_mask(scene, image_size)
        heatmap[collision_mask] = 0.0

        # 3. 阈值处理
        heatmap[heatmap < HEATMAP_MIN_PROB] = 0.0

        return heatmap

    def _world_to_image(
        self,
        pos: List[float],
        bounds: np.ndarray,
        image_size: int,
    ) -> Tuple[float, float]:
        """世界坐标转图像坐标（只使用 X, Y）"""
        if bounds is None:
            return image_size / 2, image_size / 2
        # bounds: [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        x_min = bounds[0, 0]
        y_min = bounds[0, 1]
        x_max = bounds[1, 0]
        y_max = bounds[1, 1]
        x_range = x_max - x_min
        y_range = y_max - y_min
        if x_range == 0 or y_range == 0:
            return image_size / 2, image_size / 2
        x = (pos[0] - x_min) / x_range * image_size
        y = (pos[1] - y_min) / y_range * image_size
        return x, y

    def _compute_collision_mask(
        self,
        scene: trimesh.Scene,
        image_size: int,
    ) -> np.ndarray:
        """计算碰撞掩码（占据区域为 True）"""
        mask = np.zeros((image_size, image_size), dtype=bool)
        bounds = scene.bounds
        if bounds is None:
            return mask
        # bounds: [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        x_min = bounds[0, 0]
        y_min = bounds[0, 1]
        x_max = bounds[1, 0]
        y_max = bounds[1, 1]
        x_range = x_max - x_min
        y_range = y_max - y_min
        if x_range == 0 or y_range == 0:
            return mask

        # 碰撞检测：遍历场景中所有几何体的投影
        # 排除地板、天花板、墙壁等基础几何体
        ignore_names = {"floor", "ceiling", "wall"}
        for geom_name, geom in scene.geometry.items():
            if not isinstance(geom, trimesh.Trimesh):
                continue
            # 跳过基础几何体
            if any(ignored in geom_name.lower() for ignored in ignore_names):
                continue
            # 使用顶点和面片生成更完整的投影
            vertices = geom.vertices
            # 对每个面片的顶点进行插值，填充投影区域
            for face in geom.faces:
                # 获取面片的三个顶点
                v0, v1, v2 = vertices[face]
                # 在面片内部进行采样（10x10 网格）
                for i in range(10):
                    for j in range(10 - i):
                        # 重心坐标插值
                        u = i / 9.0
                        v = j / 9.0
                        w = 1.0 - u - v
                        if w < 0:
                            continue
                        px = u * v0[0] + v * v1[0] + w * v2[0]
                        py = u * v0[1] + v * v1[1] + w * v2[1]
                        ix = int((px - x_min) / x_range * image_size)
                        iy = int((py - y_min) / y_range * image_size)
                        if 0 <= ix < image_size and 0 <= iy < image_size:
                            mask[iy, ix] = True

        return mask

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
        rot_z = random.uniform(-AUG_ROTATION_RANGE, AUG_ROTATION_RANGE)
        rot_aug = R.from_euler('z', rot_z, degrees=True).as_quat().tolist()

        # 2. 随机缩放
        scale_aug = random.uniform(*AUG_SCALE_RANGE)

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
            name=obj.name,
            pos=new_pos,
            rot=rot_aug,
            size=obj.size,
            scale_jid=(scale_aug, scale_aug, scale_aug),
            mesh=obj.mesh,  # 原始 mesh
            is_on_floor=obj.is_on_floor,
            is_on_wall=obj.is_on_wall,
        )

        return aug_obj, new_scene

    def _check_position_collision(
        self,
        scene: trimesh.Scene,
        pos: List[float],
        mesh: trimesh.Trimesh,
        scale: float,
    ) -> bool:
        """检查指定位置是否与其他物体碰撞"""
        test_mesh = mesh.copy()
        test_mesh.apply_scale(scale)
        T = np.eye(4)
        T[:3, 3] = pos
        test_mesh.apply_transform(T)

        for geom_name, geom in scene.geometry.items():
            if not isinstance(geom, trimesh.Trimesh):
                continue
            if geom_name.startswith("obj_"):
                if trimesh.intersects.signed_distance(test_mesh, geom).min() < 0:
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
        heatmap: np.ndarray,
        obj: ObjectInfo,
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

        Image.fromarray(plane_image).save(plane_path)
        Image.fromarray(object_image).save(object_path)

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
        if target_obj.is_on_floor:
            room_image = self.render_top_view(scene)
        else:
            room_image = self.render_side_view(scene)

        # 2. 渲染物体参考图（俯视图，正向，缩放=1）
        object_image = self.render_object_reference(target_obj.mesh)

        # 3. 生成 GT 热力图
        heatmap = self.generate_heatmap(scene, target_obj.pos)

        # 4. 生成文本
        text_prompt = self.generate_text_prompt(target_obj.name, is_aug=False)
        response = self.generate_response(target_obj.name)

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
        plane_image = self.render_top_view(scene_without_obj)

        # 7. 保存样本
        self.save_sample(
            plane_image=plane_image,
            object_image=object_image,
            heatmap=heatmap,
            obj=target_obj,
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
                aug_heatmap = self.generate_heatmap(scene_without_obj, aug_obj.pos)

                # 计算旋转和缩放标签
                # 数据增强：旋转为随机生成的旋转的倒数，缩放为 1
                aug_rot_6d = self.rotation_6d_from_quat(
                    R.from_quat(aug_obj.rot).inv().as_quat().tolist()
                )
                aug_scale = 1.0  # 物体参考图缩放为 1

                # 渲染剔除后房间图
                aug_plane_image = self.render_top_view(
                    self.remove_object_from_scene(aug_scene, aug_obj)
                )

                self.save_sample(
                    plane_image=aug_plane_image,
                    object_image=aug_object_image,
                    heatmap=aug_heatmap,
                    obj=aug_obj,
                    text_prompt=self.generate_text_prompt(aug_obj.name, is_aug=True),
                    response=self.generate_response(aug_obj.name),
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
        print(f"Annotations saved to {output_path} ({len(self.annotations)} samples)")

    def run(self):
        """执行数据生成"""
        print(f"扫描场景目录: {self.scene_dir}")
        json_files = sorted(self.scene_dir.glob("*.json"))
        print(f"找到 {len(json_files)} 个场景文件")

        for i, json_path in enumerate(json_files, 1):
            print(f"[{i}/{len(json_files)}] 处理: {json_path.name}", end=" ... ")
            self.process_scene(json_path)
            print(f"成功，当前样本数: {self.sample_counter}")

            if self.sample_counter >= self.num_samples:
                print(f"已达到目标样本数: {self.num_samples}")
                break

        print(f"\n生成完成! 总样本数: {self.sample_counter}")
        self.save_annotations()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SAM-Q 训练数据生成器")
    parser.add_argument(
        "--scene_dir", type=Path, default=SCENE_DIR,
        help="SSR3D-FRONT 场景 JSON 目录"
    )
    parser.add_argument(
        "--model_dir", type=Path, default=MODEL_DIR,
        help="3D-FUTURE 模型目录"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=OUTPUT_DIR,
        help="输出数据目录"
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000,
        help="目标样本数"
    )
    parser.add_argument(
        "--augmentation", action="store_true",
        help="启用数据增强"
    )
    parser.add_argument(
        "--aug_ratio", type=float, default=0.5,
        help="增强样本比例（预留参数）"
    )
    parser.add_argument(
        "--image_size", type=int, default=IMAGE_SIZE,
        help="输出图像分辨率"
    )
    parser.add_argument(
        "--heatmap_sigma", type=float, default=HEATMAP_SIGMA,
        help="热力图高斯核标准差"
    )

    args = parser.parse_args()

    if not args.scene_dir.exists():
        print(f"错误: 场景目录不存在: {args.scene_dir}")
        return
    if not args.model_dir.exists():
        print(f"错误: 模型目录不存在: {args.model_dir}")
        return

    generator = TrainingDataGenerator(
        scene_dir=args.scene_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        heatmap_sigma=args.heatmap_sigma,
        augmentation=args.augmentation,
        aug_ratio=args.aug_ratio,
        num_samples=args.num_samples,
    )
    generator.run()


if __name__ == "__main__":
    main()
