"""
Dataset for Object Placement Prediction

Handles loading and preprocessing of:
    - Plane/room top-down images
    - Object top-down images
    - Text prompts
    - Ground truth placement masks
"""

import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any


class ObjectPlacementDataset(Dataset):
    """
    Dataset for object placement prediction task.
    
    Expected directory structure:
        data/
        ├── plane_images/
        │   ├── scene_001.png
        │   └── ...
        ├── object_images/
        │   ├── obj_001.png
        │   └── ...
        ├── masks/
        │   ├── scene_001_mask.png
        │   └── ...
        └── annotations.json
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[Any] = None,
        split: str = "train",
        ann_file: str = "annotations.json",
        seg_feature_dir: Optional[str] = None,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Root directory containing the dataset
            transform: Optional transforms to apply
            split: Dataset split ('train', 'val', 'test')
            ann_file: Annotation filename
            seg_feature_dir: Directory containing pre-extracted <SEG> hidden states (for Stage 2)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        self.ann_file = ann_file
        self.seg_feature_dir = Path(seg_feature_dir) if seg_feature_dir else None

        # Load annotations
        self.annotations = self._load_annotations()
        
    def _load_annotations(self) -> List[Dict]:
        """Load annotations from JSON file."""
        annotations_path = self.data_dir / self.ann_file

        if not annotations_path.exists():
            raise FileNotFoundError(
                f"Annotations file not found at {annotations_path}"
            )

        with open(annotations_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Filter by split if applicable
        if data and "split" in data[0]:
            data = [item for item in data if item.get("split") == self.split]

        return data

    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Returns:
            sample: Dictionary containing:
                - plane_image: Plane/room top-down view (Tensor) [3, H, W]
                - images: List of image tensors [3, H, W] (order matches <image> in prompt)
                - text_prompt: Placement instruction (str)
                - mask: Ground truth placement mask (Tensor) [1, H, W]
                - metadata: Additional info
        """
        ann = self.annotations[idx]

        # Load plane_image (for SAM3 input)
        plane_image = self._load_image(self.data_dir / ann["plane_image_path"])
        plane_tensor = torch.from_numpy(np.array(plane_image, dtype=np.float32) / 255.0).permute(2, 0, 1)

        # Load all images (for Qwen3-VL input, order matches <image> placeholders)
        images = []
        for img_path in ann["images_path"]:
            pil_img = self._load_image(self.data_dir / img_path)
            tensor = torch.from_numpy(np.array(pil_img, dtype=np.float32) / 255.0).permute(2, 0, 1)
            images.append(tensor)

        # Load mask
        mask = self._load_mask(self.data_dir / ann["mask_path"])

        # Get text prompt and optional stage1 response
        text_prompt = ann.get("text_prompt", "Place the object here.")
        # stage1 conversation response (contains <SEG> token)
        response = ann.get("response", None)

        # Stage 2 GT: 6D rotation [6] + scale [1] (optional)
        rotation_6d = torch.tensor(ann["rotation_6d"], dtype=torch.float32) if "rotation_6d" in ann else None
        scale = torch.tensor([ann["scale"]], dtype=torch.float32) if "scale" in ann else None

        # Pre-extracted <SEG> hidden state (for Stage 2)
        seg_hidden = None
        if self.seg_feature_dir is not None:
            sample_id = ann.get("id", ann.get("scene_id", f"{self.split}_{idx:06d}"))
            seg_path = self.seg_feature_dir / f"{sample_id}.pt"
            if seg_path.exists():
                seg_data = torch.load(seg_path, map_location="cpu", weights_only=True)
                seg_hidden = seg_data["seg_hidden"]  # [hidden_dim]

        return {
            "plane_image": plane_tensor,              # [3, H, W] for SAM3
            "images": images,                         # List of [3, H, W] for Qwen3-VL
            "text_prompt": text_prompt,
            "response": response,
            "mask": mask,
            "rotation_6d": rotation_6d,
            "scale": scale,
            "seg_hidden": seg_hidden,                 # None or [hidden_dim]
            "metadata": {
                "scene_id": ann.get("scene_id", idx),
                "object_id": ann.get("object_id", None),
            },
        }
    
    def _load_image(self, path: Path) -> Image.Image:
        """Load an image. Assumes uniform size."""
        return Image.open(path).convert("RGB")

    def _load_mask(self, path: Path) -> torch.Tensor:
        """Load mask and convert to tensor. Assumes uniform size."""
        mask = Image.open(path).convert("L")

        # Convert to tensor (H, W) -> (1, H, W)
        mask_array = np.array(mask, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

        return mask_tensor

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """Collate function for DataLoader."""
        # plane_images: [B, 3, H, W] for SAM3
        plane_images = torch.stack([item["plane_image"] for item in batch])
        # images: List[List[Tensor]] for Qwen3-VL
        batch_images = [item["images"] for item in batch]

        text_prompts = [item["text_prompt"] for item in batch]
        responses = [item.get("response") for item in batch]
        masks = torch.stack([item["mask"] for item in batch])
        metadata = [item["metadata"] for item in batch]

        # rotation_6d / scale: stack if GT exists, else None
        rot_list = [item.get("rotation_6d") for item in batch]
        scale_list = [item.get("scale") for item in batch]
        rotation_6d = torch.stack(rot_list) if all(r is not None for r in rot_list) else None
        scale = torch.stack(scale_list) if all(s is not None for s in scale_list) else None

        # seg_hidden: stack if pre-extracted, else None
        seg_list = [item.get("seg_hidden") for item in batch]
        seg_hidden = torch.stack(seg_list) if all(s is not None for s in seg_list) else None

        return {
            "plane_images": plane_images,               # [B, 3, H, W]
            "images": batch_images,                     # List[List[Tensor]]
            "text_prompts": text_prompts,
            "responses": responses,
            "masks": masks,
            "rotation_6d": rotation_6d,
            "scale": scale,
            "seg_hidden": seg_hidden,                   # [B, hidden_dim] or None
            "metadata": metadata,
        }
