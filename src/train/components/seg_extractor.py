"""
SEG Feature Extractor
=====================

Extracts <SEG> token hidden states from Qwen3-VL after Stage 1 training.
"""

import torch # type: ignore
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import DataLoader  # type: ignore


class SegFeatureExtractor:
    """
    Extracts <SEG> hidden states for Stage 2 training.
    """

    def __init__(
        self,
        model,
        config: Dict[str, Any],
    ):
        self.model = model
        self.config = config

    def extract(self, dataloader: DataLoader, output_dir: Path) -> None:
        """
        Extract <SEG> features and save to disk.

        Args:
            dataloader: DataLoader containing samples
            output_dir: Directory to save extracted features
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.eval()
        self.model.qwen_encoder.load_model()

        print(f"\n{'=' * 60}")
        print(f"提取 <SEG> features → {output_dir}")
        print(f"{'=' * 60}")

        count = 0
        dataset = dataloader.dataset
        num_seg = self.config.get("model", {}).get("num_seg_tokens", 1)

        for idx in tqdm(range(len(dataset)), desc="提取 <SEG>", leave=False):
            sample = dataset[idx]
            ann = dataset.annotations[idx]
            sample_id = ann.get("id", ann.get("scene_id", f"{dataset.split}_{idx:06d}"))

            out_path = output_dir / f"{sample_id}.pt"
            if out_path.exists():
                count += 1
                continue

            text_prompt = sample["text_prompt"]
            images = sample["images"]

            seg_hidden, _ = self.model.qwen_encoder.generate_with_seg(
                text_prompt=text_prompt,
                images=images,
                force_only=True,
                num_seg=num_seg,
            )

            seg_hidden = seg_hidden.squeeze(0).cpu().float()
            torch.save({"seg_hidden": seg_hidden, "sample_id": sample_id}, out_path)
            count += 1

        print(f"完成: {count} 个特征已保存到 {output_dir}")
