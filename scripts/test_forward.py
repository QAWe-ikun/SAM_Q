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
    parser.add_argument("--mode", default="cross_modal", choices=["cross_modal", "seg_token"])
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Mode:   {args.mode}")
    print()

    # --- Build model ---
    print("Building SAM3PlacementModel...")
    from models import SAM3PlacementModel

    model = SAM3PlacementModel(
        qwen_model_name="./models/qwen3_vl",
        sam3_input_dim=256,
        qwen_hidden_dim=4096,
        adapter_hidden_dim=512,
        device=args.device,
        mode=args.mode,
    )
    model.freeze_all_except_adapter_and_detector()
    model.eval()
    print("Model built OK\n")

    # --- Dummy inputs ---
    plane_image  = make_dummy_image((1024, 1024))
    object_image = make_dummy_image((512, 512))
    text_prompt  = "Place the chair near the window"

    # --- Forward pass ---
    print("Running forward pass...")
    with torch.no_grad():
        output = model(
            plane_image=plane_image,
            object_image=object_image,
            text_prompt=text_prompt,
        )

    print("\n=== Output shapes ===")
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {list(v.shape)}")
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, torch.Tensor):
                    print(f"  {k}.{kk}: {list(vv.shape)}")
        else:
            print(f"  {k}: {type(v).__name__}")

    print("\nForward pass completed successfully!")


if __name__ == "__main__":
    main()
