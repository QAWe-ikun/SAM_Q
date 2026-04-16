"""
Placement Predictor for Inference
==================================

Provides easy-to-use predictor for object placement inference.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

from ..models import SAMQPlacementModel


class PlacementPredictor:
    """
    High-level predictor for object placement inference.

    Example:
        >>> predictor = PlacementPredictor("outputs/stage2_full/checkpoint_best.pt")
        >>> results = predictor.predict(
        ...     plane_image=Image.open("room.png"),
        ...     object_image=Image.open("chair.png"),
        ...     text_prompt="把椅子放在桌子旁边"
        ... )
        >>> print(f"Heatmap shape: {results['heatmap'].shape}")
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
        threshold: float = 0.5,
    ):
        """
        Initialize predictor.

        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            device: Device to run on (default: auto-detect)
            threshold: Confidence threshold for predictions
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint_path)
        self.threshold = threshold

        # Load checkpoint
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        print(f"[PlacementPredictor] Loading checkpoint from {self.checkpoint_path}")
        self.checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        # Initialize model
        self.model = self._init_model()
        self.model.eval()

        print(f"[PlacementPredictor] Model initialized on {self.device}")

    def _init_model(self) -> SAMQPlacementModel:
        """Initialize model from checkpoint."""
        config = self.checkpoint.get("config", {})
        model_config = config.get("model", {})
        num_seg_tokens = model_config.get("num_seg_tokens", 1)

        # Get LoRA path if available (for inference with fine-tuned Qwen)
        qwen_lora_path = model_config.get("qwen", {}).get("lora_path")

        # Get SAM3 checkpoint path
        sam3_ckpt = model_config.get("sam3", {}).get("checkpoint_path")

        model = SAMQPlacementModel(
            sam3_checkpoint_path=sam3_ckpt,
            qwen_model_name=model_config.get("qwen", {}).get(
                "model_name", "Qwen/Qwen3-VL-8B-Instruct"
            ),
            qwen_lora_path=qwen_lora_path,
            sam3_input_dim=model_config.get("sam3", {}).get("input_dim", 256),
            qwen_hidden_dim=model_config.get("qwen", {}).get("hidden_dim", 4096),
            adapter_hidden_dim=model_config.get("adapter", {}).get("hidden_dim", 512),
            num_seg_tokens=num_seg_tokens,
            device=self.device,
            seg_token_config=model_config.get("seg_token", {}),
            action_head_config=model_config.get("action_head", {}),
        )

        # Load weights from checkpoint
        print(f"[PlacementPredictor] Loading model weights...")
        model.load_state_dict(self.checkpoint["model_state_dict"], strict=False)
        model.to(self.device)

        # Load all components (Qwen, LoRA, SAM3) for inference
        model.load_all(eval_mode=True)

        return model

    def predict(
        self,
        plane_image: Union[Image.Image, str, Path],
        object_image: Union[Image.Image, str, Path],
        text_prompt: str,
        threshold: Optional[float] = None,
        return_heatmap_raw: bool = False,
    ) -> Dict[str, Any]:
        """
        Run placement prediction.

        Args:
            plane_image: Plane/room top-down view (PIL Image or path)
            object_image: Object top-down view (PIL Image or path)
            text_prompt: Placement instruction (with optional <image> placeholders)
            threshold: Confidence threshold for heatmap binarization (default: 0.5)
            return_heatmap_raw: If True, return raw heatmap (0~1); if False, return binary mask

        Returns:
            results: Dictionary with:
                - heatmap: [H, W] placement heatmap (0~1 float) or binary mask
                - mask: [H, W] binary placement mask (thresholded)
                - rotation_deg: Predicted rotation angle in degrees
                - scale_relative: Predicted relative scale factor
                - image_size: Original plane image size (W, H)
        """
        th = threshold if threshold is not None else self.threshold

        # Load images if paths are provided
        if isinstance(plane_image, (str, Path)):
            plane_image = Image.open(plane_image).convert("RGB")
        if isinstance(object_image, (str, Path)):
            object_image = Image.open(object_image).convert("RGB")

        original_size = plane_image.size  # (W, H)

        # Run prediction
        with torch.no_grad():
            # Build input for model
            images = [plane_image, object_image]
            
            output = self.model(
                plane_image=plane_image,
                text_prompt=text_prompt,
                images=images,
            )

        # Extract outputs
        heatmap = output["heatmap"]  # [B, 1, H, W]
        rotation_deg = output.get("rotation_deg")  # [B] or scalar
        scale_relative = output.get("scale_relative")  # [B] or scalar

        # Postprocess
        results = self._postprocess(
            heatmap=heatmap,
            rotation_deg=rotation_deg,
            scale_relative=scale_relative,
            original_size=original_size,
            threshold=th,
            return_heatmap_raw=return_heatmap_raw,
        )

        return results

    def predict_with_seg_hidden(
        self,
        plane_image: Union[Image.Image, str, Path],
        seg_hidden: torch.Tensor,
        text_prompt: str = "",
        threshold: Optional[float] = None,
        return_heatmap_raw: bool = False,
    ) -> Dict[str, Any]:
        """
        Run placement prediction with pre-extracted <SEG> hidden state.

        This is faster as it skips Qwen3-VL inference.

        Args:
            plane_image: Plane/room top-down view (PIL Image or path)
            seg_hidden: Pre-extracted <SEG> hidden state [4096] or [B, 4096]
            text_prompt: Placement instruction (optional, for logging)
            threshold: Confidence threshold (default: 0.5)
            return_heatmap_raw: If True, return raw heatmap (0~1)

        Returns:
            results: Same as predict()
        """
        th = threshold if threshold is not None else self.threshold

        # Load image if path is provided
        if isinstance(plane_image, (str, Path)):
            plane_image = Image.open(plane_image).convert("RGB")

        original_size = plane_image.size  # (W, H)

        # Run prediction with pre-extracted hidden state
        with torch.no_grad():
            output = self.model(
                plane_image=plane_image,
                text_prompt=text_prompt,
                seg_hidden=seg_hidden,
            )

        # Extract outputs
        heatmap = output["heatmap"]
        rotation_deg = output.get("rotation_deg")
        scale_relative = output.get("scale_relative")

        # Postprocess
        results = self._postprocess(
            heatmap=heatmap,
            rotation_deg=rotation_deg,
            scale_relative=scale_relative,
            original_size=original_size,
            threshold=th,
            return_heatmap_raw=return_heatmap_raw,
        )

        return results

    def _postprocess(
        self,
        heatmap: torch.Tensor,
        rotation_deg: Optional[torch.Tensor],
        scale_relative: Optional[torch.Tensor],
        original_size: tuple,
        threshold: float = 0.5,
        return_heatmap_raw: bool = False,
    ) -> Dict[str, Any]:
        """
        Postprocess model outputs.

        Args:
            heatmap: [B, 1, H, W] placement heatmap (logits or sigmoid)
            rotation_deg: [B] predicted rotation angle
            scale_relative: [B] predicted relative scale
            original_size: Original plane image size (W, H)
            threshold: Threshold for binarization
            return_heatmap_raw: If True, keep raw heatmap values

        Returns:
            Processed results dictionary
        """
        # Take first batch item
        heat = heatmap[0, 0]  # [H, W]

        # Apply sigmoid if logits (values outside 0~1 range)
        if heat.min() < 0 or heat.max() > 1:
            heat = torch.sigmoid(heat)

        # Resize to original image size
        orig_w, orig_h = original_size
        heat = F.interpolate(
            heat.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)  # [H, W]

        # Convert to numpy
        heatmap_np = heat.cpu().numpy()  # [H, W], 0~1

        # Create binary mask
        mask_np = (heatmap_np >= threshold).astype(np.uint8)

        # Extract rotation and scale
        rot_deg = rotation_deg[0].item() if rotation_deg is not None else 0.0
        scale = scale_relative[0].item() if scale_relative is not None else 1.0

        return {
            "heatmap": heatmap_np if return_heatmap_raw else (heatmap_np * 255).astype(np.uint8),
            "mask": mask_np,
            "rotation_deg": rot_deg,
            "scale_relative": scale,
            "image_size": original_size,
        }

    def visualize(
        self,
        plane_image: Image.Image,
        results: Dict[str, Any],
        alpha: float = 0.5,
    ) -> Image.Image:
        """
        Overlay heatmap/mask on the original plane image for visualization.

        Args:
            plane_image: Original plane image
            results: Output from predict() or predict_with_seg_hidden()
            alpha: Overlay transparency (0~1)

        Returns:
            Overlay image (PIL Image)
        """
        # Resize mask to match image size
        mask = results["mask"]
        img = plane_image.resize((mask.shape[1], mask.shape[0]), Image.Resampling.LANCZOS)

        # Create colored overlay (red)
        overlay = Image.new("RGBA", img.size, (255, 0, 0, 0))
        overlay_np = np.array(overlay)

        # Apply mask to overlay
        mask_3ch = np.stack([mask, mask, mask], axis=-1)  # [H, W, 3]
        overlay_np[:, :, :3] = np.where(mask_3ch > 0, [255, 0, 0], [0, 0, 0])
        overlay_np[:, :, 3] = mask * int(alpha * 255)

        # Composite
        overlay_pil = Image.fromarray(overlay_np, "RGBA")
        result = Image.alpha_composite(img.convert("RGBA"), overlay_pil)

        return result.convert("RGB")
