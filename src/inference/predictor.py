"""
Placement Predictor for Inference
==================================

Provides easy-to-use predictor for object placement inference.
"""

import torch
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union

from ..models import SAMQPlacementModel


class PlacementPredictor:
    """
    High-level predictor for object placement inference.
    
    Example:
        >>> predictor = PlacementPredictor("checkpoints/checkpoint_best.pt")
        >>> results = predictor.predict(
        ...     plane_image=Image.open("room.png"),
        ...     object_image=Image.open("chair.png"),
        ...     text_prompt="Place the chair near the table"
        ... )
        >>> print(f"Found {len(results['boxes'])} placements")
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
            checkpoint_path: Path to model checkpoint
            device: Device to run on (default: auto-detect)
            threshold: Confidence threshold for predictions
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint_path)
        self.threshold = threshold
        
        # Load checkpoint
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        self.checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,  # Required for older PyTorch versions
        )
        
        # Initialize model
        self.model = self._init_model()
        self.model.eval()

    def _init_model(self) -> SAMQPlacementModel:
        """Initialize model from checkpoint."""
        config = self.checkpoint.get("config", {})
        model_config = config.get("model", {})
        num_seg_tokens = model_config.get("num_seg_tokens", 1)

        model = SAMQPlacementModel(
            qwen_model_name=model_config.get(
                "qwen", {}
            ).get("model_name", "Qwen/Qwen3-VL-8B-Instruct"),
            sam3_input_dim=model_config.get("sam3", {}).get("input_dim", 256),
            qwen_hidden_dim=model_config.get("qwen", {}).get("hidden_dim", 4096),
            adapter_hidden_dim=model_config.get("adapter", {}).get("hidden_dim", 512),
            num_seg_tokens=num_seg_tokens,
            device=self.device,
            seg_token_config=model_config.get("seg_token", {}),
            action_head_config=model_config.get("action_head", {}),
        )
        
        # Load weights
        model.load_state_dict(self.checkpoint["model_state_dict"])
        model.to(self.device)
        
        return model

    def predict(
        self,
        plane_image: Union[Image.Image, str, Path],
        object_image: Union[Image.Image, str, Path],
        text_prompt: str,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run placement prediction.

        Args:
            plane_image: Plane/room top-down view (PIL Image or path)
            object_image: Object top-down view (PIL Image or path)
            text_prompt: Placement instruction
            threshold: Confidence threshold (optional)

        Returns:
            results: Dictionary with:
                - mask: Predicted placement mask (numpy array)
                - heatmap: Visualization heatmap (numpy array)
                - boxes: Bounding boxes (numpy array)
                - scores: Confidence scores (numpy array)
                - image_size: Original image size (tuple)
        """
        th = threshold if threshold is not None else self.threshold
        
        # Load images if paths are provided
        if isinstance(plane_image, (str, Path)):
            plane_image = Image.open(plane_image).convert("RGB")
        if isinstance(object_image, (str, Path)):
            object_image = Image.open(object_image).convert("RGB")
        
        # Resize to expected sizes
        plane_image = plane_image.resize((1024, 1024), Image.Resampling.LANCZOS)
        object_image = object_image.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Run prediction
        with torch.no_grad():
            output = self.model.predict(
                plane_image=plane_image,
                object_image=object_image,
                text_prompt=text_prompt,
                threshold=th,
            )
        
        # Postprocess
        results = self._postprocess(output, plane_image.size)
        
        return results

    def _postprocess(
        self,
        output: Dict[str, torch.Tensor],
        image_size: tuple,
    ) -> Dict[str, Any]:
        """
        Postprocess model outputs.
        
        Args:
            output: Model output dictionary
            image_size: Original image size
            
        Returns:
            Processed results dictionary
        """
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
        
        # Convert masks to numpy
        mask_np = masks[0].cpu().numpy()
        
        # Create heatmap visualization
        heatmap = self._create_heatmap(mask_np)
        
        # Format boxes
        boxes_np = boxes[0].cpu().numpy() if len(boxes) > 0 else np.zeros((0, 4))
        scores_np = scores[0].cpu().numpy() if len(scores) > 0 else np.zeros(0)
        
        return {
            "mask": mask_np,
            "heatmap": heatmap,
            "boxes": boxes_np,
            "scores": scores_np,
            "image_size": image_size,
        }

    def _create_heatmap(self, mask: np.ndarray) -> np.ndarray:
        """
        Create heatmap from mask for visualization.
        
        Args:
            mask: Binary mask array
            
        Returns:
            Heatmap image (H, W, 3) uint8
        """
        # Normalize mask to [0, 255]
        heatmap = (mask * 255).astype(np.uint8)
        
        # Apply colormap (simple jet-like)
        heatmap_colored = np.zeros((heatmap.shape[0], heatmap.shape[1], 3), dtype=np.uint8)
        heatmap_colored[:, :, 0] = heatmap  # Red channel
        heatmap_colored[:, :, 1] = heatmap // 2  # Green channel
        
        return heatmap_colored
