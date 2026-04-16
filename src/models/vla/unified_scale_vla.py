"""
Unified Scale VLA (Vision-Language-Action) Module
===================================================

Key insight: unified pixel-meter encoding lets VLM naturally understand
physical scale through image resolution, without explicit size injection.

- All images use the same pixels_per_meter ratio
- Object scaled to actual physical size in pixels
- Scene scaled similarly
- VLM sees: "object is X pixels, scene is Y pixels" → natural scale understanding

Uses <SEG> token for dual purpose:
  - Segmentation: triggers SAM3 mask generation
  - Action output: position (heatmap) + rotation + scale
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Union
from PIL import Image


class UnifiedScalePreprocessor(nn.Module):
    """
    Preprocess images with unified pixel-meter encoding.

    All images are scaled to the same pixels_per_meter ratio,
    so VLM naturally understands physical scale through resolution.
    """

    def __init__(self, pixels_per_meter: int = 512, model_input_size: int = 1024):
        super().__init__()
        self.pixels_per_meter = pixels_per_meter
        self.model_input_size = model_input_size

    def forward(
        self,
        obj_image: Image.Image,
        obj_size_meters: float,
        scene_image: Image.Image,
        scene_size_meters: float,
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Scale both object and scene images to unified pixel-meter encoding.

        Args:
            obj_image: Object top-down view (PIL)
            obj_size_meters: Object physical size in meters (e.g., 0.5 for 50cm chair)
            scene_image: Scene top-down view (PIL)
            scene_size_meters: Scene physical size in meters (e.g., 4.0 for 4m room)

        Returns:
            obj_scaled: Object image scaled to physical size in pixels
            scene_scaled: Scene image scaled to physical size in pixels
        """
        # Calculate target pixel dimensions
        obj_target = int(obj_size_meters * self.pixels_per_meter)
        scene_target = int(scene_size_meters * self.pixels_per_meter)

        # Scale to physical size, then resize to model input
        # Key: maintain aspect ratio during resize
        obj_scaled = obj_image.resize(
            (obj_target, obj_target), Image.Resampling.LANCZOS
        ).resize(
            (self.model_input_size, self.model_input_size), Image.Resampling.LANCZOS
        )

        scene_scaled = scene_image.resize(
            (scene_target, scene_target), Image.Resampling.LANCZOS
        ).resize(
            (self.model_input_size, model_input_size), Image.Resampling.LANCZOS
        )

        return obj_scaled, scene_scaled


def rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Uses Gram-Schmidt orthogonalization (Zhou et al. 2019).

    Args:
        rot_6d: [B, 6] first two columns of rotation matrix (a1, a2)

    Returns:
        matrix: [B, 3, 3] proper rotation matrix (orthonormal, det=+1)
    """
    import torch.nn.functional as F
    a1 = rot_6d[:, 0:3]
    a2 = rot_6d[:, 3:6]

    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)


class SEGActionHead(nn.Module):
    """
    Action head that decodes <SEG> token hidden state into:
      - Coarse placement heatmap (position)
      - 6D rotation representation (Zhou et al. 2019)
      - Relative scale (1.0 = original size, 0.5-2.0 range)
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        heatmap_size: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heatmap_size = heatmap_size

        # Project <SEG> hidden to action features
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )

        # Position: coarse heatmap
        self.heatmap_head = nn.Sequential(
            nn.Linear(512, heatmap_size * heatmap_size),
        )

        # Rotation: 6D representation (first two columns of rotation matrix)
        self.rotation_head = nn.Linear(512, 6)

        # Scale: 0.5x to 2.0x
        self.scale_head = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid(),  # [0, 1] → scale to [0.5, 2.0]
        )

    def forward(
        self, seg_hidden: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            seg_hidden: [B, hidden_dim] hidden state at <SEG> position

        Returns:
            Dict with heatmap, rotation_6d, rotation_matrix, scale
        """
        features = self.proj(seg_hidden)  # [B, 512]

        # Heatmap: [B, heatmap_size, heatmap_size]
        heatmap = self.heatmap_head(features).view(
            -1, self.heatmap_size, self.heatmap_size
        )

        # Rotation: 6D representation [B, 6]
        rotation_6d = self.rotation_head(features)

        # Convert to rotation matrix [B, 3, 3]
        rotation_matrix = rotation_6d_to_matrix(rotation_6d)

        # Scale: [B] in range [0.5, 2.0]
        scale = self.scale_head(features).squeeze(-1) * 1.5 + 0.5

        return {
            "heatmap": heatmap,
            "rotation_6d": rotation_6d,
            "rotation_matrix": rotation_matrix,
            "scale": scale,
        }

    def load_from_checkpoint(
        self, 
        path: Union[str, Path], 
        device: str = "cpu",
        prefix: str = ""
    ):
        """
        Load trained weights from a checkpoint file.

        Args:
            path: Path to checkpoint (.pt)
            device: Device to load weights to
            prefix: Key prefix in checkpoint (e.g., "seg_action_head.") if loading from full model state_dict
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SEGActionHead checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)

        # Filter keys if loading from a full model checkpoint
        if prefix:
            filtered_state_dict = {
                k[len(prefix):]: v 
                for k, v in state_dict.items() 
                if k.startswith(prefix)
            }
        else:
            filtered_state_dict = state_dict

        missing, unexpected = self.load_state_dict(filtered_state_dict, strict=False)
        
        if unexpected:
            print(f"[SEGActionHead] Unexpected keys: {unexpected}")
        
        self.to(device)
        self.eval()
        print(f"[SEGActionHead] Loaded weights from {path}")