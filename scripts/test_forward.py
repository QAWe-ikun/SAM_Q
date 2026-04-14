#!/usr/bin/env python3
"""
End-to-end forward pass test for SAM-Q.

Usage (from project root):
    python scripts/test_forward.py
    python scripts/test_forward.py --device cuda
"""

import sys
import argparse
import warnings
from pathlib import Path

# Suppress noisy third-party warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from PIL import Image


def make_dummy_image(size):
    """Create a random RGB PIL image."""
    import numpy as np
    arr = (torch.rand(size[1], size[0], 3) * 255).byte().numpy()
    return Image.fromarray(arr, mode="RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_seg_tokens", default=1, type=int)
    args = parser.parse_args()

    print(f"Device:         {args.device}")
    print(f"num_seg_tokens: {args.num_seg_tokens}")
    print()

    # --- Build model ---
    print("Building SAMQPlacementModel...")
    from models import SAMQPlacementModel

    model = SAMQPlacementModel(
        qwen_model_name="./models/qwen3_vl",
        sam3_input_dim=256,
        qwen_hidden_dim=4096,
        adapter_hidden_dim=512,
        num_seg_tokens=args.num_seg_tokens,
        device=args.device,
        action_head_config={
            "heatmap_size": 64,
        },
    )
    model.freeze_all_except_adapter_and_detector()
    model.eval()
    print("Model built OK\n")

    # --- Dummy inputs ---
    # Note: All input images use the same size (1024x1024)
    img_size = 1024
    plane_image  = make_dummy_image((img_size, img_size))
    object_image = make_dummy_image((img_size, img_size))
    text_prompt  = "这是房间<image>，把椅子<image>放在合适的位置"
    images       = [plane_image, object_image]

    # --- Forward pass ---
    print("Running forward()...")
    with torch.no_grad():
        fwd_output = model.forward(
            plane_image=plane_image,
            text_prompt=text_prompt,
            images=images,
        )

    print("\n=== Forward output ===")
    for k, v in fwd_output.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {list(v.shape)}")

    print("\nRunning predict()...")
    with torch.no_grad():
        output = model.predict(
            plane_image=plane_image,
            text_prompt=text_prompt,
            images=images,
            threshold=0.5,
        )

    print("\n=== Predict output ===")
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {list(v.shape)}")

    print("\nForward pass completed successfully!")


if __name__ == "__main__":
    main()
