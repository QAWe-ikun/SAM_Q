"""
SEG Feature Extractor
=====================

Extracts <SEG> token hidden states from Qwen3-VL after Stage 1 training.
"""

import gc
import torch # type: ignore
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import DataLoader  # type: ignore


class SegFeatureExtractor:
    """
    Extracts <SEG> hidden states for Stage 2 training.
    Uses batching and memory cleanup to prevent OOM.
    """

    def __init__(
        self,
        model,
        config: Dict[str, Any],
        batch_size: int = 4,
    ):
        self.model = model
        self.config = config
        self.batch_size = batch_size

    def extract(self, dataloader: DataLoader, output_dir: Path) -> None:
        """
        Extract <SEG> features and save to disk.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.eval()
        self.model.qwen_encoder.load_model()

        print(f"\n{'=' * 60}")
        print(f"提取 <SEG> features → {output_dir}")
        print(f"{'=' * 60}")

        dataset = dataloader.dataset
        # 如果是 Subset（测试模式），需要获取原始数据集以访问 annotations
        original_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
        
        # 获取 Subset 的真实索引
        if hasattr(dataset, 'indices'):
            indices = dataset.indices
        else:
            indices = range(len(original_dataset.samples))
        
        num_seg = self.config.get("model", {}).get("num_seg_tokens", 1)
        count = 0
        device = self.model.qwen_encoder.device

        for start_idx in tqdm(range(0, len(indices), self.batch_size), desc="提取 <SEG>"):
            end_idx = min(start_idx + self.batch_size, len(indices))
            
            for subset_idx in range(start_idx, end_idx):
                idx = indices[subset_idx]
                sample = dataset[subset_idx]
                ann = original_dataset.samples[idx]
                sample_id = ann.get("sample_id")
                if sample_id is None:
                    sample_id = ann.get("id", f"sample_{idx:06d}")

                out_path = output_dir / f"{sample_id}.pt"
                if out_path.exists():
                    count += 1
                    continue

                try:
                    text_prompt = sample["text_prompt"]
                    images = [img.to(device) for img in sample["images"]]

                    with torch.no_grad():
                        seg_hidden, _ = self.model.qwen_encoder.generate_with_seg(
                            text_prompt=text_prompt,
                            images=images,
                            force_only=True,
                            num_seg=num_seg,
                        )

                    seg_hidden = seg_hidden.squeeze(0).cpu().float()
                    torch.save({"seg_hidden": seg_hidden, "sample_id": sample_id}, out_path)
                    count += 1
                except Exception as e:
                    print(f"\n[Warning] 提取样本 {sample_id} 失败: {e}")
                    continue

            # 清理显存
            torch.cuda.empty_cache()
            gc.collect()

        print(f"完成: {count} 个特征已保存到 {output_dir}")
