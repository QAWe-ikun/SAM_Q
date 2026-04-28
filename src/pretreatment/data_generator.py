"""
SAM-Q 训练数据生成器主类

编排场景构建、渲染、VLM 推理、样本保存等模块。
"""

import os
import json
import random
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List

# WSL 无头环境下强制使用 OSMesa 渲染后端
if not os.environ.get("PYOPENGL_PLATFORM"):
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"

# 抑制 trimesh 和 pyrender 的警告
warnings.filterwarnings("ignore", category=UserWarning, module="trimesh")
warnings.filterwarnings("ignore", category=UserWarning, module="pyrender")

from .vlm_client import VLMClient
from .renderer import SceneRenderer
from .sample_saver import SampleSaver
from .scene_builder import SceneBuilder
from .heatmap_generator import HeatmapGenerator
from .augmentation import AugmentationProcessor

logger = logging.getLogger(__name__)


class TrainingDataGenerator:
    """SAM-Q 训练数据生成器"""

    def __init__(self, config: Dict[str, Any]):
        # 从配置字典中提取参数
        data_config = config.get("data", {})
        gen_config = config.get("generation", {})
        aug_config = config.get("augmentation", {})
        cam_config = config.get("camera", {})

        self.scene_dir = Path(data_config.get("scene_dir", ""))
        self.model_dir = Path(data_config.get("model_dir", ""))
        self.output_dir = Path(data_config.get("output_dir", ""))

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for split in ["train", "val", "test"]:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)

        # 初始化各模块
        self.scene_builder = SceneBuilder(self.model_dir)

        self.renderer = SceneRenderer(
            image_size=gen_config.get("image_size", 1024),
            fov_degrees=cam_config.get("fov_degrees", 45),
            aspect_ratio=cam_config.get("aspect_ratio", 1.0),
            top_view_camera_height=cam_config.get("top_view_height", 1.0),
        )

        self.heatmap_generator = HeatmapGenerator(
            image_size=gen_config.get("image_size", 1024),
            fov_degrees=cam_config.get("fov_degrees", 45),
            aspect_ratio=cam_config.get("aspect_ratio", 1.0),
            top_view_camera_height=cam_config.get("top_view_height", 1.0),
            sigma=gen_config.get("heatmap_sigma", 15.0),
        )

        self.augmentation = AugmentationProcessor(
            rotation_range=aug_config.get("rotation_range", 180),
            scale_range=(
                aug_config.get("scale_range", {}).get("min", 0.8),
                aug_config.get("scale_range", {}).get("max", 1.25),
            ),
        ) if aug_config.get("enabled", True) else None

        self.aug_ratio = aug_config.get("aug_ratio", 0.2)
        
        # 保存 scale_range 以便在非增强模式下使用
        scale_range = aug_config.get("scale_range", {})
        self.scale_range = (
            scale_range.get("min", 0.8),
            scale_range.get("max", 1.25),
        )

        self.sample_saver = SampleSaver(self.output_dir)

        # VLM 客户端
        self.use_vllm = gen_config.get("use_vllm", False)
        self.vlm_client = None
        if gen_config.get("qwen_model_name"):
            self.vlm_client = VLMClient(gen_config["qwen_model_name"])

        # 数据划分
        split_ratio = gen_config.get("split_ratio", {"train": 0.8, "val": 0.1, "test": 0.1})
        self.train_ratio = split_ratio.get("train", 0.8)
        self.val_ratio = split_ratio.get("val", 0.1)
        self.test_ratio = split_ratio.get("test", 0.1)

        # 最大剔除数量
        self.max_object_nums = gen_config.get("max_object_nums", 2)

        # VLM 批量处理大小
        self.vlm_batch_size = gen_config.get("vlm_batch_size", 4)

    def rotation_6d_from_quat(self, quat: List[float]) -> List[float]:
        """四元数转 6D 旋转"""
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat(quat)
        R_mat = r.as_matrix()
        return [
            R_mat[0, 0], R_mat[1, 0], R_mat[2, 0],
            R_mat[0, 1], R_mat[1, 1], R_mat[2, 1],
        ]

    def _collect_scene_samples(
        self,
        json_path: Path,
        split: str,
    ) -> List[dict]:
        """
        收集单个场景的所有样本数据（仅渲染，不调用 VLM）。
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                scene_data = json.load(f)
        except Exception:
            return []

        # 构建场景
        scene, objects = self.scene_builder.build_scene(scene_data)
        if not objects:
            return []

        # 渲染原始房间图
        original_image = self.renderer.render_top_view(
            scene, scene_data.get('bounds_bottom', [])
        )
        if original_image is None:
            return []

        # 创建场景输出目录
        scene_name = json_path.stem
        scene_dir = self.output_dir / split / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)

        max_objects = min(len(objects), self.max_object_nums)
        random.shuffle(objects)

        collected = []

        for target_obj in objects:
            if len(collected) >= max_objects:
                break

            if not target_obj.is_on_floor:
                continue

            # 数据增强（仅在 train 阶段，按 aug_ratio 比例决定）
            is_aug = (
                split == "train"
                and self.augmentation is not None
                and random.random() < self.aug_ratio
            )

            if is_aug:
                # 增强流程：直接处理并保存
                aug_obj, aug_scene = self.augmentation.augmentation_object(
                    target_obj, scene
                )
                if aug_obj is not None:
                    aug_object_image = self.renderer.render_object_reference(
                        aug_obj.mesh, scene_data.get('bounds_bottom', [])
                    )
                    aug_heatmap = self.heatmap_generator.generate(
                        aug_obj.pos, scene_data.get('bounds_bottom', [])
                    )
                    aug_rot_6d = self.rotation_6d_from_quat(aug_obj.rot)

                    if aug_object_image is not None and aug_heatmap is not None:
                        # 生成文本
                        try:
                            aug_text_prompt = (
                                f"物体{aug_obj.desc}的参考图为：<image>\n"
                                f"平面图为：<image>\n两者的尺寸相同。"
                            )
                            aug_response = self.vlm_client.generate_response(
                                aug_text_prompt, aug_rot_6d, 1.0
                            )
                        except Exception:
                            logger.warning("跳过增强样本：VLM 生成失败")
                            continue

                        # 保存
                        aug_meta = self.sample_saver.save_sample(
                            scene_dir=scene_dir,
                            obj_id=f"aug_{target_obj.jid}",
                            plane_image=aug_object_image,
                            object_image=aug_object_image,
                            heatmap=aug_heatmap,
                            text_prompt=aug_text_prompt,
                            response=aug_response,
                            rotation_6d=aug_rot_6d,
                            scale=1.0,
                            split=split,
                        )
                        if aug_meta is not None:
                            collected.append({"is_aug": True})
                continue

            # 计算旋转和缩放标签
            orig_rot = target_obj.rot
            if len(orig_rot) == 4 and not allclose(orig_rot, [0, 0, 0, 1]):
                inv_rot = self._invert_quat(orig_rot)
            else:
                inv_rot = [0, 0, 0, 1]
            rotation_6d = self.rotation_6d_from_quat(inv_rot)

            orig_scale = random.uniform(*self.scale_range)
            scale = 1.0 / orig_scale

            target_obj.mesh.apply_scale(orig_scale)

            # 渲染物体参考图
            object_image = self.renderer.render_object_reference(
                target_obj.mesh, scene_data.get('bounds_bottom', [])
            )
            if object_image is None:
                continue

            # 生成 GT 热力图
            heatmap = self.heatmap_generator.generate(
                target_obj.pos, scene_data.get('bounds_bottom', [])
            )
            if heatmap is None:
                continue

            # 剔除物体，渲染剔除后房间图
            scene_without_obj = self._remove_object_from_scene(scene, target_obj)
            plane_image = self.renderer.render_top_view(
                scene_without_obj, scene_data.get('bounds_bottom', [])
            )
            if plane_image is None:
                continue

            # 收集样本数据
            collected.append({
                "is_aug": False,
                "target_obj": target_obj,
                "original_image": original_image,
                "plane_image": plane_image,
                "object_image": object_image,
                "heatmap": heatmap,
                "rotation_6d": rotation_6d,
                "scale": scale,
                "split": split,
                "scene_dir": scene_dir,
                "scene_name": scene_name,
            })

        return collected

    @staticmethod
    def _invert_quat(quat: List[float]) -> List[float]:
        """四元数求逆"""
        from scipy.spatial.transform import Rotation as R
        return R.from_quat(quat).inv().as_quat().tolist()

    @staticmethod
    def _remove_object_from_scene(
        scene,
        obj,
    ):
        """从场景中移除物体"""
        new_scene = scene.copy()
        geom_name = f"obj_{obj.jid}"
        if geom_name in new_scene.geometry:
            del new_scene.geometry[geom_name]
        return new_scene

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
        n_test = n - n_train - n_val

        splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test

        # 加载 VLM 模型
        if self.vlm_client:
            self.vlm_client.load_model()

        # 按 epoch 分批处理
        epoch_size = 1000
        n_epochs = (n + epoch_size - 1) // epoch_size

        total_processed = 0

        for epoch in range(n_epochs):
            start_idx = epoch * epoch_size
            end_idx = min(start_idx + epoch_size, n)

            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch+1}/{n_epochs} (场景 {start_idx+1}-{end_idx})")
            logger.info(f"{'='*60}")

            # Phase 1: 收集所有场景的样本数据
            all_collected = []

            for idx in range(start_idx, end_idx):
                json_path = json_files[idx]
                split = splits[idx]
                logger.info(f"  [{idx+1}/{n}] 收集场景数据: {json_path.stem} ({split})")
                scene_samples = self._collect_scene_samples(json_path, split)
                all_collected.extend(scene_samples)

            # 过滤出需要 VLM 处理的样本
            vlm_samples = [s for s in all_collected if not s.get("is_aug", False)]
            logger.info(f"\n  Phase 1 完成: 收集 {len(vlm_samples)} 个样本需 VLM 处理")

            # Phase 2: 批量 VLM 推理
            if vlm_samples and self.vlm_client:
                # 批量生成 placement description
                try:
                    placement_descs = []
                    for sample in vlm_samples:
                        desc = self.vlm_client.generate_placement_description(
                            sample["original_image"],
                            sample["plane_image"],
                            sample["object_image"],
                            sample["target_obj"].desc,
                        )
                        placement_descs.append(desc)
                except Exception as e:
                    logger.error(f"批量生成 placement description 失败: {e}")
                    placement_descs = [None] * len(vlm_samples)

                # 构建 text prompts
                text_prompts = []
                rotation_6d_list = []
                scale_list = []
                valid_indices = []

                for i, sample in enumerate(vlm_samples):
                    if placement_descs[i] is not None:
                        base_prompt = (
                            f"物体{sample['target_obj'].desc}的参考图为：<image>\n"
                            f"平面图为：<image>\n两者的尺寸相同，"
                        )
                        text_prompts.append(base_prompt + placement_descs[i])
                        rotation_6d_list.append(sample["rotation_6d"])
                        scale_list.append(sample["scale"])
                        valid_indices.append(i)

                # 批量生成 responses
                try:
                    responses = []
                    for j in range(len(text_prompts)):
                        response = self.vlm_client.generate_response(
                            text_prompts[j],
                            rotation_6d_list[j],
                            scale_list[j],
                        )
                        responses.append(response)
                except Exception as e:
                    logger.error(f"批量生成 responses 失败: {e}")
                    responses = [None] * len(text_prompts)

                # Phase 3: 保存所有样本
                for i, sample_idx in enumerate(valid_indices):
                    sample = vlm_samples[sample_idx]
                    text_prompt = text_prompts[i]
                    response = responses[i] if i < len(responses) else None

                    if response is None:
                        continue

                    sample_meta = self.sample_saver.save_sample(
                        scene_dir=sample["scene_dir"],
                        obj_id=sample["target_obj"].jid,
                        plane_image=sample["plane_image"],
                        object_image=sample["object_image"],
                        heatmap=sample["heatmap"],
                        text_prompt=text_prompt,
                        response=response,
                        rotation_6d=sample["rotation_6d"],
                        scale=sample["scale"],
                        split=sample["split"],
                    )
                    if sample_meta is not None:
                        total_processed += 1

            # 每个 epoch 结束后保存数据
            for split_name in ["train", "val", "test"]:
                self.sample_saver.save_split_json(split_name)
                self.sample_saver.clear_split_samples(split_name)

        logger.info(f"\n{'='*60}")
        logger.info(f"生成完成! 总样本数: {self.sample_saver.sample_counter} (VLM 处理: {total_processed})")
        logger.info(f"{'='*60}")


def allclose(a, b, tol=1e-5):
    """简单的数组比较"""
    return all(abs(x - y) < tol for x, y in zip(a, b))
