#!/usr/bin/env python3
"""
生成假的 seg_features 用于测试 Stage 2 pipeline。
真实训练时应使用 scripts/extract_seg_features.py 生成。

Usage:
    python scripts/gen_dummy_seg_features.py
"""

import torch
from pathlib import Path

output_dir = Path("data/seg_features")
output_dir.mkdir(parents=True, exist_ok=True)

# 和 annotations.json 中的 scene_id 对应
sample_ids = ["scene_001", "scene_002", "scene_003"]
hidden_dim = 4096

for sid in sample_ids:
    seg_hidden = torch.randn(hidden_dim)
    torch.save({"seg_hidden": seg_hidden, "sample_id": sid}, output_dir / f"{sid}.pt")
    print(f"已生成: {output_dir / f'{sid}.pt'}")

print(f"\n完成！共 {len(sample_ids)} 个假特征，目录: {output_dir}")
