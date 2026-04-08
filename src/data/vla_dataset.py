"""
VLA Dataset for Object Placement with Action Instructions
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Dict, Any
import json
import numpy as np


class VLADataset(Dataset):
    """
    VLA Dataset extending ObjectPlacementDataset with action instructions.
    """

    def __init__(
        self,
        data_dir: str,
        plane_image_size: tuple = (1024, 1024),
        object_image_size: tuple = (512, 512),
        split: str = "train",
    ):
        self.data_dir = Path(data_dir)
        self.plane_image_size = plane_image_size
        self.object_image_size = object_image_size
        self.split = split

        self.annotations = self._load_annotations()

    def _load_annotations(self):
        annotations_path = self.data_dir / "annotations.json"

        with open(annotations_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "split" in data[0]:
            data = [item for item in data if item.get("split") == self.split]

        return data

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ann = self.annotations[idx]

        plane_image = self._load_image(
            self.data_dir / ann["plane_image_path"],
            self.plane_image_size,
        )

        object_image = self._load_image(
            self.data_dir / ann["object_image_path"],
            self.object_image_size,
        )

        mask = self._load_mask(self.data_dir / ann["mask_path"])

        return {
            "plane_image": plane_image,
            "object_image": object_image,
            "text_prompt": ann.get("text_prompt", ""),
            "action_instruction": ann.get("action_instruction", ""),
            "mask": mask,
        }

    def _load_image(self, path: Path, size: tuple) -> Image.Image:
        image = Image.open(path).convert("RGB")
        image = image.resize(size, Image.Resampling.LANCZOS)
        return image

    def _load_mask(self, path: Path) -> torch.Tensor:
        mask = Image.open(path).convert("L")
        mask = mask.resize(self.plane_image_size, Image.Resampling.NEAREST)
        mask_array = np.array(mask, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        return mask_tensor

