"""
Dataset for Object Placement Prediction

Handles loading and preprocessing of:
    - Plane/room top-down images
    - Object top-down images
    - Text prompts
    - Ground truth placement masks
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import json
import numpy as np


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
        plane_image_size: Tuple[int, int] = (1024, 1024),
        object_image_size: Tuple[int, int] = (512, 512),
        transform: Optional[Any] = None,
        split: str = "train",
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory containing the dataset
            plane_image_size: Target size for plane images
            object_image_size: Target size for object images
            transform: Optional transforms to apply
            split: Dataset split ('train', 'val', 'test')
        """
        self.data_dir = Path(data_dir)
        self.plane_image_size = plane_image_size
        self.object_image_size = object_image_size
        self.transform = transform
        self.split = split
        
        # Load annotations
        self.annotations = self._load_annotations()
        
    def _load_annotations(self) -> List[Dict]:
        """Load annotations from JSON file."""
        annotations_path = self.data_dir / "annotations.json"
        
        if not annotations_path.exists():
            raise FileNotFoundError(
                f"Annotations file not found at {annotations_path}"
            )
        
        with open(annotations_path, "r") as f:
            data = json.load(f)
        
        # Filter by split if applicable
        if "split" in data[0]:
            data = [item for item in data if item.get("split") == self.split]
        
        return data
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            sample: Dictionary containing:
                - plane_image: Plane/room top-down view (PIL Image)
                - object_image: Object top-down view (PIL Image)
                - text_prompt: Placement instruction (str)
                - mask: Ground truth placement mask (Tensor)
                - metadata: Additional info
        """
        ann = self.annotations[idx]
        
        # Load images
        plane_image = self._load_image(
            self.data_dir / ann["plane_image_path"],
            self.plane_image_size,
        )
        
        object_image = self._load_image(
            self.data_dir / ann["object_image_path"],
            self.object_image_size,
        )
        
        # Load mask
        mask = self._load_mask(self.data_dir / ann["mask_path"])
        
        # Get text prompt
        text_prompt = ann.get("text_prompt", "Place the object here.")
        
        # Apply transforms if specified
        if self.transform:
            plane_image = self.transform(plane_image)
            object_image = self.transform(object_image)
        
        return {
            "plane_image": plane_image,
            "object_image": object_image,
            "text_prompt": text_prompt,
            "mask": mask,
            "metadata": {
                "scene_id": ann.get("scene_id", idx),
                "object_id": ann.get("object_id", None),
            },
        }
    
    def _load_image(self, path: Path, size: Tuple[int, int]) -> Image.Image:
        """Load and resize an image."""
        image = Image.open(path).convert("RGB")
        image = image.resize(size, Image.Resampling.LANCZOS)
        return image
    
    def _load_mask(self, path: Path) -> torch.Tensor:
        """Load mask and convert to tensor."""
        mask = Image.open(path).convert("L")
        mask = mask.resize(self.plane_image_size, Image.Resampling.NEAREST)

        # Convert to tensor (H, W) -> (1, H, W)
        mask_array = np.array(mask, dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

        return mask_tensor


class ObjectPlacementDataModule:
    """
    DataModule for object placement prediction.
    
    Handles train/val/test splits and DataLoader creation.
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 8,
        num_workers: int = 4,
        plane_image_size: Tuple[int, int] = (1024, 1024),
        object_image_size: Tuple[int, int] = (512, 512),
    ):
        """
        Initialize the DataModule.
        
        Args:
            data_dir: Root directory containing the dataset
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            plane_image_size: Target size for plane images
            object_image_size: Target size for object images
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.plane_image_size = plane_image_size
        self.object_image_size = object_image_size
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        if stage == "fit" or stage is None:
            self.train_dataset = ObjectPlacementDataset(
                data_dir=self.data_dir,
                plane_image_size=self.plane_image_size,
                object_image_size=self.object_image_size,
                split="train",
            )
            
            self.val_dataset = ObjectPlacementDataset(
                data_dir=self.data_dir,
                plane_image_size=self.plane_image_size,
                object_image_size=self.object_image_size,
                split="val",
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = ObjectPlacementDataset(
                data_dir=self.data_dir,
                plane_image_size=self.plane_image_size,
                object_image_size=self.object_image_size,
                split="test",
            )
    
    def train_dataloader(self) -> DataLoader:
        """Get training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict[str, Any]:
        """Collate function for DataLoader."""
        plane_images = [item["plane_image"] for item in batch]
        object_images = [item["object_image"] for item in batch]
        text_prompts = [item["text_prompt"] for item in batch]
        masks = torch.stack([item["mask"] for item in batch])
        metadata = [item["metadata"] for item in batch]
        
        return {
            "plane_images": plane_images,
            "object_images": object_images,
            "text_prompts": text_prompts,
            "masks": masks,
            "metadata": metadata,
        }
