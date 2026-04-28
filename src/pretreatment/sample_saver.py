"""
SAM-Q 样本保存模块

负责保存样本图片和生成元数据。
"""

import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class SampleSaver:
    """样本保存器"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.sample_counter = 0
        self.samples_by_split = {"train": [], "val": [], "test": []}

    def save_sample(
        self,
        scene_dir: Path,
        obj_id: str,
        plane_image: np.ndarray,
        object_image: np.ndarray,
        heatmap: np.ndarray,
        text_prompt: str,
        response: str,
        rotation_6d: List[float],
        scale: float,
        split: str = "train",
    ) -> Optional[dict]:
        """
        保存单个样本到场景目录中。

        Returns:
            样本元数据字典
        """
        self.sample_counter += 1
        sample_id = f"obj_{obj_id}"

        # 创建子目录
        plane_dir = scene_dir / "plane_images"
        object_dir = scene_dir / "object_images"
        mask_dir = scene_dir / "masks"
        plane_dir.mkdir(exist_ok=True)
        object_dir.mkdir(exist_ok=True)
        mask_dir.mkdir(exist_ok=True)

        # 保存图片
        plane_path = plane_dir / f"{sample_id}.png"
        object_path = object_dir / f"{sample_id}.png"
        mask_path = mask_dir / f"{sample_id}_mask.png"

        Image.fromarray(plane_image).save(plane_path)
        Image.fromarray(object_image).save(object_path)

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

    def save_split_json(self, split_name: str) -> int:
        """
        保存指定 split 的样本数据为 JSON 文件。

        Returns:
            保存的样本数量
        """
        split_samples = self.samples_by_split[split_name]
        if not split_samples:
            return 0

        split_dir = self.output_dir / split_name
        output_path = split_dir / f"{split_name}.json"

        if output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            existing.extend(split_samples)
            total_count = len(existing)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
        else:
            total_count = len(split_samples)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(split_samples, f, ensure_ascii=False, indent=2)

        logger.info(
            f"保存 {split_name} 数据: {len(split_samples)} 个样本 (总计 {total_count})"
        )

        return total_count

    def clear_split_samples(self, split_name: str):
        """清空指定 split 的样本列表"""
        self.samples_by_split[split_name] = []

    def clear_all_samples(self):
        """清空所有样本列表"""
        self.samples_by_split = {"train": [], "val": [], "test": []}
