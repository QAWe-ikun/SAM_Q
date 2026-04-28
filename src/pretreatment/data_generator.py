"""
SAM-Q 训练数据生成器主类

编排场景构建、渲染、VLM 推理、样本保存等模块。
"""

import os
import json
import tqdm
import random
import logging
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

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

from .vlm_client import VLMClient
from .renderer import SceneRenderer
from .sample_saver import SampleSaver
from .scene_builder import SceneBuilder
from .heatmap_generator import HeatmapGenerator
from .augmentation import AugmentationProcessor


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
            self.vlm_client = VLMClient(
                model_path=gen_config["qwen_model_name"],
                use_vllm=self.use_vllm,
            )

        # 数据划分
        split_ratio = gen_config.get("split_ratio", {"train": 0.8, "val": 0.1, "test": 0.1})
        self.train_ratio = split_ratio.get("train", 0.8)
        self.val_ratio = split_ratio.get("val", 0.1)
        self.test_ratio = split_ratio.get("test", 0.1)

        # 最大剔除数量
        self.max_object_nums = gen_config.get("max_object_nums", 2)

        # VLM 批量处理大小
        self.vlm_batch_size = gen_config.get("vlm_batch_size", 4)
        self.epoch_size = gen_config.get("epoch_size", 1000)

        # 两步流程控制
        self.step = gen_config.get("step", "all")  # "render_only", "vlm_only", "all"
        self.rendered_dir = Path(gen_config.get("rendered_dir", str(self.output_dir / "_rendered")))

        # 创建渲染输出目录
        if self.step in ["render_only", "all"]:
            self.rendered_dir.mkdir(parents=True, exist_ok=True)
            for split in ["train", "val", "test"]:
                (self.rendered_dir / split).mkdir(parents=True, exist_ok=True)

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

        if len(objects) <= 2:
            return []
        
        random.shuffle(objects)

        collected = []

        for target_obj in objects:
            if len(collected) >= self.max_object_nums:
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
                # 增强流程：收集到 collected 列表中，与其他样本一起批量处理
                aug_obj, _ = self.augmentation.augmentation_object(
                    target_obj, scene
                )
                if aug_obj is None:
                    continue

                aug_object_image = self.renderer.render_object_reference(
                    aug_obj.mesh, scene_data.get('bounds_bottom', [])
                )
                if aug_object_image is None:
                    continue

                aug_heatmap = self.heatmap_generator.generate(
                    aug_obj.pos, scene_data.get('bounds_bottom', [])
                )
                if aug_heatmap is None:
                    continue

                aug_rot_6d = self.rotation_6d_from_quat(aug_obj.rot)

                # 收集增强样本数据，稍后批量 VLM 处理
                collected.append({
                    "target_obj": aug_obj,
                    "original_image": original_image,
                    "plane_image": aug_object_image,
                    "object_image": aug_object_image,
                    "heatmap": aug_heatmap,
                    "rotation_6d": aug_rot_6d,
                    "scale": 1.0,
                    "split": split,
                    "scene_dir": scene_dir,
                    "scene_name": scene_name,
                    "is_augmentation": True,  # 标记这是增强样本
                })
                continue

            # 计算旋转和缩放标签
            orig_rot = target_obj.rot
            if len(orig_rot) == 4 and not np.allclose(orig_rot, [0, 0, 0, 1]):
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

    def _render_and_save_scene(
        self,
        json_path: Path,
        split: str,
    ) -> int:
        """
        渲染单个场景并保存为 npz + meta.json（第一步）。
        
        Returns:
            保存的样本数量
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                scene_data = json.load(f)
        except Exception:
            return 0

        # 构建场景
        scene, objects = self.scene_builder.build_scene(scene_data)
        if not objects:
            return 0

        # 渲染原始房间图
        original_image = self.renderer.render_top_view(
            scene, scene_data.get('bounds_bottom', [])
        )
        if original_image is None:
            return 0

        # 创建场景输出目录（渲染结果）
        scene_name = json_path.stem
        scene_rendered_dir = self.rendered_dir / split / scene_name
        scene_rendered_dir.mkdir(parents=True, exist_ok=True)

        if len(objects) <= 2:
            return 0

        random.shuffle(objects)

        saved_count = 0

        for target_obj in objects:
            if saved_count >= self.max_object_nums:
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
                # 增强流程
                aug_obj, _ = self.augmentation.augmentation_object(
                    target_obj, scene
                )
                if aug_obj is None:
                    continue

                aug_object_image = self.renderer.render_object_reference(
                    aug_obj.mesh, scene_data.get('bounds_bottom', [])
                )
                if aug_object_image is None:
                    continue

                aug_heatmap = self.heatmap_generator.generate(
                    aug_obj.pos, scene_data.get('bounds_bottom', [])
                )
                if aug_heatmap is None:
                    continue

                aug_rot_6d = self.rotation_6d_from_quat(aug_obj.rot)

                # 保存增强样本
                sample_id = f"aug_{target_obj.jid}"
                np.savez(
                    scene_rendered_dir / f"{sample_id}.npz",
                    original_image=original_image,
                    plane_image=aug_object_image,
                    object_image=aug_object_image,
                    heatmap=aug_heatmap,
                )
                meta = {
                    "sample_id": sample_id,
                    "scene_name": scene_name,
                    "split": split,
                    "rotation_6d": aug_rot_6d,
                    "scale": 1.0,
                    "is_augmentation": True,
                }
                with open(scene_rendered_dir / f"{sample_id}_meta.json", 'w') as f:
                    json.dump(meta, f, indent=2)
                saved_count += 1
                continue

            # 计算旋转和缩放标签
            orig_rot = target_obj.rot
            if len(orig_rot) == 4 and not np.allclose(orig_rot, [0, 0, 0, 1]):
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

            # 保存样本
            sample_id = target_obj.jid
            np.savez(
                scene_rendered_dir / f"{sample_id}.npz",
                original_image=original_image,
                plane_image=plane_image,
                object_image=object_image,
                heatmap=heatmap,
            )
            meta = {
                "sample_id": sample_id,
                "scene_name": scene_name,
                "split": split,
                "rotation_6d": rotation_6d,
                "scale": scale,
                "is_augmentation": False,
            }
            with open(scene_rendered_dir / f"{sample_id}_meta.json", 'w') as f:
                json.dump(meta, f, indent=2)
            saved_count += 1

        logger.info(f"  场景 {scene_name}: 渲染并保存 {saved_count} 个样本")
        return saved_count

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

    def _get_splits(self, n: int) -> List[str]:
        """根据 split_ratio 生成划分标签"""
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        return ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)

    def run(self):
        """执行数据生成"""
        if self.step == "render_only":
            self._run_render_only()
        elif self.step == "vlm_only":
            self._run_vlm_only()
        else:  # "all"
            self._run_all()

    def _run_render_only(self):
        """第一步：只渲染场景并保存为 npz + meta.json"""
        logger.info("=" * 60)
        logger.info("第一步：渲染场景并保存为 npz + meta.json")
        logger.info(f"渲染结果保存目录: {self.rendered_dir}")
        logger.info("=" * 60)

        json_files = sorted(self.scene_dir.glob("*.json"))
        random.shuffle(json_files)

        n = len(json_files)
        splits = self._get_splits(n)

        total_saved = 0

        for idx, json_path in enumerate(tqdm.tqdm(json_files)):
            split = splits[idx]
            saved = self._render_and_save_scene(json_path, split)
            total_saved += saved

        logger.info(f"\n{'='*60}")
        logger.info(f"渲染完成! 共保存 {total_saved} 个样本")
        logger.info(f"{'='*60}")

    def _run_vlm_only(self):
        """第二步：读取渲染结果做 VLM 推理"""
        logger.info("=" * 60)
        logger.info("第二步：读取渲染结果做 VLM 推理")
        logger.info(f"渲染结果目录: {self.rendered_dir}")
        logger.info(f"最终输出目录: {self.output_dir}")
        logger.info("=" * 60)

        if not self.vlm_client:
            logger.warning("VLM 客户端未配置，跳过 VLM 推理")
            return

        self.vlm_client.load_model()

        total_processed = 0

        for split in ["train", "val", "test"]:
            split_dir = self.rendered_dir / split
            if not split_dir.exists():
                continue

            scene_dirs = sorted(split_dir.iterdir())
            if not scene_dirs:
                continue

            logger.info(f"\n处理 {split} 数据集: {len(scene_dirs)} 个场景")

            # 收集所有样本
            all_samples = []
            for scene_dir in tqdm.tqdm(scene_dirs):
                for meta_file in scene_dir.glob("*_meta.json"):
                    meta_path = meta_file
                    npz_path = meta_path.parent / meta_path.name.replace("_meta.json", ".npz")
                    
                    if not npz_path.exists():
                        continue

                    with open(meta_path, 'r') as f:
                        meta = json.load(f)

                    data = np.load(npz_path)
                    all_samples.append({
                        "meta": meta,
                        "data": data,
                    })

            if not all_samples:
                continue

            # 批量 VLM 推理
            batch_size = self.vlm_batch_size
            for batch_start in range(0, len(all_samples), batch_size):
                batch_end = min(batch_start + batch_size, len(all_samples))
                batch = all_samples[batch_start:batch_end]

                try:
                    # 生成 placement descriptions
                    placement_descs = []
                    for item in batch:
                        meta = item["meta"]
                        data = item["data"]
                        desc = self.vlm_client.generate_placement_description(
                            data["original_image"],
                            data["plane_image"],
                            data["object_image"],
                            meta["sample_id"],
                        )
                        placement_descs.append(desc)

                    # 构建 text prompts
                    text_prompts = []
                    rotation_6d_list = []
                    scale_list = []
                    valid_indices = []

                    for i, item in enumerate(batch):
                        meta = item["meta"]
                        if placement_descs[i] is None:
                            continue

                        base_prompt = (
                            f"物体{meta['sample_id']}的参考图为：<image>\n"
                            f"平面图为：<image>\n两者的尺寸相同，"
                        )
                        text_prompts.append(base_prompt + placement_descs[i])
                        rotation_6d_list.append(meta["rotation_6d"])
                        scale_list.append(meta["scale"])
                        valid_indices.append(i)

                    # 批量生成 responses
                    responses = self.vlm_client.generate_responses_batch(
                        text_prompts,
                        rotation_6d_list,
                        scale_list,
                    )

                    # 保存最终样本
                    for i, idx in enumerate(valid_indices):
                        item = batch[idx]
                        meta = item["meta"]
                        data = item["data"]

                        self.sample_saver.save_sample(
                            scene_dir=self.output_dir / meta["split"],
                            obj_id=meta["sample_id"],
                            plane_image=data["plane_image"],
                            object_image=data["object_image"],
                            heatmap=data["heatmap"],
                            text_prompt=text_prompts[i],
                            response=responses[i] if i < len(responses) else None,
                            rotation_6d=rotation_6d_list[i],
                            scale=scale_list[i],
                            split=meta["split"],
                        )
                        total_processed += 1

                except Exception as e:
                    logger.error(f"批量 VLM 推理失败: {e}")
                    continue

        # 保存划分信息
        for split_name in ["train", "val", "test"]:
            self.sample_saver.save_split_json(split_name)

        logger.info(f"\n{'='*60}")
        logger.info(f"VLM 推理完成! 共处理 {total_processed} 个样本")
        logger.info(f"{'='*60}")

    def _run_all(self):
        """完整流程：先渲染再 VLM 推理"""
        logger.info("=" * 60)
        logger.info("完整流程：先渲染场景再 VLM 推理")
        logger.info("=" * 60)

        json_files = sorted(self.scene_dir.glob("*.json"))
        random.shuffle(json_files)

        n = len(json_files)
        splits = self._get_splits(n)

        # 加载 VLM 模型
        if self.vlm_client:
            self.vlm_client.load_model()

        # 按 epoch 分批处理
        n_epochs = (n + self.epoch_size - 1) // self.epoch_size
        total_processed = 0

        for epoch in range(n_epochs):
            start_idx = epoch * self.epoch_size
            end_idx = min(start_idx + self.epoch_size, n)

            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch+1}/{n_epochs} (场景 {start_idx+1}-{end_idx})")
            logger.info(f"{'='*60}")

            # Phase 1: 收集所有场景的样本数据
            vlm_samples = []

            for idx in tqdm.tqdm(range(start_idx, end_idx)):
                json_path = json_files[idx]
                split = splits[idx]
                logger.info(f"  [{idx+1}/{n}] 收集场景数据: {json_path.stem} ({split})")
                scene_samples = self._collect_scene_samples(json_path, split)
                vlm_samples.extend(scene_samples)

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
                    continue

                # 构建 text prompts
                scale_list = []
                text_prompts = []
                valid_indices = []
                rotation_6d_list = []

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
                    responses = self.vlm_client.generate_responses_batch(
                        text_prompts,
                        rotation_6d_list,
                        scale_list,
                    )
                except Exception as e:
                    logger.error(f"批量生成 responses 失败: {e}")
                    responses = [None] * len(text_prompts)

                # Phase 3: 保存所有样本
                for i, sample_idx in enumerate(tqdm.tqdm(valid_indices)):
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
