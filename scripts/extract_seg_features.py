#!/usr/bin/env python3
"""
预提取 [SEG] hidden states

Stage 1 训练完成后运行此脚本，将每个样本的 [SEG] hidden state 保存为 .pt 文件。
Stage 2 训练时直接读取，不再加载 Qwen3-VL（省 ~16GB 显存）。

Usage:
    python scripts/extract_seg_features.py \
        --config configs/stage1_qwen_lora.yaml \
        --output_dir data/seg_features/
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from tqdm import tqdm

from src.utils.config import Config
from src.data.dataset import ObjectPlacementDataset
from src.models.encoders.qwen3vl_encoder import Qwen3VLEncoder


def main():
    parser = argparse.ArgumentParser(description="预提取 [SEG] hidden states")
    parser.add_argument("--config", type=str, required=True, help="Config YAML 路径")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Stage 1 LoRA checkpoint 路径")
    parser.add_argument("--output_dir", type=str, default="data/seg_features/", help="输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"], help="要提取的 split")
    args = parser.parse_args()

    config = Config(args.config).to_dict()
    data_config = config.get("data", {})
    model_config = config.get("model", {})
    qwen_config = model_config.get("qwen", {})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载 Qwen3-VL
    print("=" * 60)
    print("加载 Qwen3-VL...")
    print("=" * 60)

    encoder = Qwen3VLEncoder(
        model_name=qwen_config.get("model_name", "Qwen/Qwen3-VL-8B-Instruct"),
        device=args.device,
        num_seg_tokens=model_config.get("num_seg_tokens", 1),
    )
    encoder.load_model()

    # 加载 Stage 1 LoRA（如果有）
    lora_ckpt = args.lora_checkpoint or qwen_config.get("lora_checkpoint")
    if lora_ckpt and Path(lora_ckpt).exists():
        try:
            from peft import PeftModel
            print(f"加载 LoRA checkpoint: {lora_ckpt}")
            encoder.model = PeftModel.from_pretrained(encoder.model, lora_ckpt)
            print("LoRA 加载成功")
        except Exception as e:
            print(f"警告: LoRA 加载失败: {e}")

    encoder.model.eval()

    # 遍历每个 split
    for split in args.splits:
        print(f"\n{'=' * 60}")
        print(f"提取 {split} split...")
        print(f"{'=' * 60}")

        try:
            dataset = ObjectPlacementDataset(
                data_dir=data_config.get("root_dir", "data/"),
                ann_file=data_config.get("ann_file", "annotations.json"),
                plane_image_size=tuple(data_config.get("plane_image_size", [1024, 1024])),
                object_image_size=tuple(data_config.get("object_image_size", [1024, 1024])),
                split=split,
            )
        except FileNotFoundError:
            print(f"跳过 {split}: 数据集不存在")
            continue

        if len(dataset) == 0:
            print(f"跳过 {split}: 无样本")
            continue

        print(f"样本数: {len(dataset)}")
        count = 0

        for idx in tqdm(range(len(dataset)), desc=f"[{split}]"):
            sample = dataset[idx]
            ann = dataset.annotations[idx]
            sample_id = ann.get("id", ann.get("scene_id", f"{split}_{idx:06d}"))

            out_path = output_dir / f"{sample_id}.pt"
            if out_path.exists():
                count += 1
                continue  # 跳过已提取的

            plane_img = sample["plane_image"]   # tensor [3, H, W]
            obj_img = sample["object_image"]    # tensor [3, H, W]
            text_prompt = sample["text_prompt"]

            with torch.no_grad():
                seg_hidden, _ = encoder.generate_with_seg(
                    text_prompt=text_prompt,
                    images=[plane_img, obj_img],
                    force_only=True,
                    num_seg=model_config.get("num_seg_tokens", 1),
                )

            # seg_hidden: [1, hidden_dim] → [hidden_dim]
            seg_hidden = seg_hidden.squeeze(0).cpu().float()

            torch.save({
                "seg_hidden": seg_hidden,
                "sample_id": sample_id,
            }, out_path)
            count += 1

        print(f"[{split}] 完成: {count}/{len(dataset)} 个特征已保存到 {output_dir}")

    print(f"\n预提取完成！特征目录: {output_dir}")
    print("Stage 2 训练时在 config 中设置:")
    print(f'  data.seg_feature_dir: "{output_dir}"')


if __name__ == "__main__":
    main()
