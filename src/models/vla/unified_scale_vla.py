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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
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


class UnifiedScaleVLA(nn.Module):
    """
    Unified Scale Vision-Language-Action model.

    Architecture:
        Object (scaled to physical size) ──┐
                                           ├→ Qwen3-VL → <SEG> token → Action Head
        Scene  (scaled to physical size) ──┘

    The <SEG> token's hidden state contains the full reasoning about:
      - Where to place (decoded to heatmap)
      - How to rotate (decoded to angle)
      - How to scale (decoded to relative size)

    Supports iterative refinement: feed back the placed scene for adjustment.
    """

    def __init__(
        self,
        pixels_per_meter: int = 512,
        model_input_size: int = 1024,
        hidden_dim: int = 4096,
        heatmap_size: int = 64,
    ):
        super().__init__()

        self.preprocessor = UnifiedScalePreprocessor(
            pixels_per_meter=pixels_per_meter,
            model_input_size=model_input_size,
        )

        self.action_head = SEGActionHead(
            hidden_dim=hidden_dim,
            heatmap_size=heatmap_size,
        )

    def forward(
        self,
        obj_image: Image.Image,
        obj_size_meters: float,
        scene_image: Image.Image,
        scene_size_meters: float,
        text_prompt: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with unified scale encoding.

        Args:
            obj_image: Object top-down view
            obj_size_meters: Object physical size in meters
            scene_image: Scene top-down view
            scene_size_meters: Scene physical size in meters
            text_prompt: Placement instruction

        Returns:
            Dict with heatmap, rotation, scale, and encoder outputs
        """
        # Preprocess with unified pixel-meter encoding
        obj_scaled, scene_scaled = self.preprocessor(
            obj_image, obj_size_meters, scene_image, scene_size_meters
        )

        # Build messages for Qwen3-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": obj_scaled},
                    {"type": "image", "image": scene_scaled},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        return {
            "messages": messages,
            "images": [obj_scaled, scene_scaled],
            "obj_size_meters": obj_size_meters,
            "scene_size_meters": scene_size_meters,
        }

    def decode_action(
        self, seg_hidden: torch.Tensor, scene_size_meters: float
    ) -> Dict[str, torch.Tensor]:
        """
        Decode <SEG> hidden state into physical actions.

        Args:
            seg_hidden: [B, hidden_dim] from Qwen3-VL at <SEG> position
            scene_size_meters: Scene physical size in meters

        Returns:
            Dict with position_meters, rotation_deg, scale_relative
        """
        action = self.action_head(seg_hidden)

        # Convert heatmap to position (normalized [0,1])
        heatmap = action["heatmap"]  # [B, H, W]
        y_norm, x_norm = self._soft_argmax2d(heatmap)

        # Convert to physical coordinates (meters)
        x_meters = x_norm * scene_size_meters
        y_meters = y_norm * scene_size_meters

        return {
            "position_meters": torch.stack([x_meters, y_meters], dim=-1),
            "position_norm": torch.stack([x_norm, y_norm], dim=-1),
            "rotation_deg": action["rotation"],
            "scale_relative": action["scale"],
            "heatmap": heatmap,
        }

    def _soft_argmax2d(self, heatmap: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute soft argmax of 2D heatmap.

        Args:
            heatmap: [B, H, W]

        Returns:
            y_norm, x_norm: Normalized coordinates [0, 1]
        """
        B, H, W = heatmap.shape
        device = heatmap.device

        # Softmax over spatial dimensions
        probs = heatmap.view(B, -1).softmax(dim=-1).view(B, H, W)

        # Create normalized coordinate grids
        y_grid = torch.linspace(0, 1, H, device=device)
        x_grid = torch.linspace(0, 1, W, device=device)
        y_grid = y_grid.view(1, H, 1).expand(B, -1, W)
        x_grid = x_grid.view(1, 1, W).expand(B, H, -1)

        # Weighted average
        y_norm = (probs * y_grid).sum(dim=[1, 2])
        x_norm = (probs * x_grid).sum(dim=[1, 2])

        return y_norm, x_norm
