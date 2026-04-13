"""
Data Transforms for SAM-Q
==========================

Provides data augmentation and preprocessing transforms.
"""

import torch
import torchvision.transforms.functional as F
from PIL import Image
from typing import Tuple, Optional, Dict, Any
import random


class Compose:
    """Compose multiple transforms together."""
    
    def __init__(self, transforms):
        """
        Args:
            transforms: List of transform callables
        """
        self.transforms = transforms
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class ResizeImages:
    """Resize plane and object images to target sizes."""
    
    def __init__(
        self,
        plane_size: Tuple[int, int] = (1024, 1024),
        object_size: Tuple[int, int] = (512, 512),
    ):
        """
        Args:
            plane_size: Target size for plane images (H, W)
            object_size: Target size for object images (H, W)
        """
        self.plane_size = plane_size
        self.object_size = object_size
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "plane_image" in sample:
            sample["plane_image"] = sample["plane_image"].resize(
                (self.plane_size[1], self.plane_size[0]),
                Image.Resampling.LANCZOS
            )
        
        if "object_image" in sample:
            sample["object_image"] = sample["object_image"].resize(
                (self.object_size[1], self.object_size[0]),
                Image.Resampling.LANCZOS
            )
        
        return sample


class RandomHorizontalFlip:
    """Randomly flip images horizontally with given probability."""
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of flipping
        """
        self.p = p
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() < self.p:
            if "plane_image" in sample:
                sample["plane_image"] = F.hflip(sample["plane_image"])
            
            if "object_image" in sample:
                sample["object_image"] = F.hflip(sample["object_image"])
            
            if "mask" in sample:
                sample["mask"] = F.hflip(sample["mask"])
        
        return sample


class NormalizeImages:
    """Normalize images with mean and std."""
    
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Args:
            mean: Mean values for normalization
            std: Standard deviation values
        """
        self.mean = mean
        self.std = std
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # Note: This expects tensors, not PIL images
        # Should be used after ToTensor transform
        if "plane_image" in sample and isinstance(sample["plane_image"], torch.Tensor):
            sample["plane_image"] = F.normalize(
                sample["plane_image"], mean=self.mean, std=self.std
            )
        
        return sample


class ToTensor:
    """Convert PIL images to tensors."""
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if "plane_image" in sample:
            img = sample["plane_image"]
            if isinstance(img, Image.Image):
                sample["plane_image"] = F.to_tensor(img)
        
        if "object_image" in sample:
            img = sample["object_image"]
            if isinstance(img, Image.Image):
                sample["object_image"] = F.to_tensor(img)
        
        return sample


def get_default_transforms() -> Compose:
    """
    Get default transform pipeline.
    
    Returns:
        Compose: Default transforms
    """
    return Compose([
        ResizeImages(),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
    ])
