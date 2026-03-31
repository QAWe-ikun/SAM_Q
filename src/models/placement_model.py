"""
Placement Model - SAM3 with Qwen3-VL for Object Placement Prediction

This module integrates Qwen3-VL as Text Encoder with SAM3 for
predicting optimal object placement positions.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List
from PIL import Image

from .qwen3vl_encoder import Qwen3VLEncoder
from .adapter import Adapter, CrossModalAdapter


class SAM3PlacementModel(nn.Module):
    """
    Object Placement Prediction Model using SAM3 + Qwen3-VL.
    
    Architecture:
        - Image Encoder: SAM3 Image Encoder (frozen)
        - Text Encoder: Qwen3-VL (frozen) + Adapter (trainable)
        - Decoder: SAM3 Detector/Decoder (trainable)
    """
    
    def __init__(
        self,
        sam3_model: Optional[nn.Module] = None,
        qwen_model_name: str = "Qwen/Qwen3-VL-7B-Instruct",
        sam3_input_dim: int = 256,
        qwen_hidden_dim: int = 3584,  # Qwen3-VL-7B hidden size
        adapter_hidden_dim: int = 512,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize the placement model.
        
        Args:
            sam3_model: Pre-loaded SAM3 model (if None, will be loaded)
            qwen_model_name: HuggingFace model name for Qwen3-VL
            sam3_input_dim: SAM3 Detector input dimension
            qwen_hidden_dim: Qwen3-VL hidden dimension
            adapter_hidden_dim: Adapter hidden dimension
            device: Device to run model on
            dtype: Data type for model
        """
        super().__init__()
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.sam3_input_dim = sam3_input_dim
        self.qwen_hidden_dim = qwen_hidden_dim
        
        # Qwen3-VL Encoder (frozen during training)
        self.qwen_encoder = Qwen3VLEncoder(
            model_name=qwen_model_name,
            device=self.device,
            dtype=self.dtype,
        )
        
        # Adapter to project Qwen embeddings to SAM3 space
        self.adapter = CrossModalAdapter(
            qwen_dim=qwen_hidden_dim,
            sam3_dim=sam3_input_dim,
            hidden_dim=adapter_hidden_dim,
        )
        
        # SAM3 components (will be loaded lazily)
        self.sam3_image_encoder = None
        self.sam3_detector = None
        self.sam3_model = sam3_model
        
        # Placeholder for SAM3 dimensions
        self._sam3_loaded = False
        
    def _load_sam3(self):
        """Lazy load SAM3 model components."""
        if self._sam3_loaded:
            return
            
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            # Load SAM3 model
            if self.sam3_model is None:
                self.sam3_model = build_sam3_image_model()
            
            # Extract components
            self.sam3_image_encoder = self.sam3_model.image_encoder
            self.sam3_detector = self.sam3_model.detector
            
            # Move to device
            self.sam3_image_encoder.to(self.device)
            self.sam3_detector.to(self.device)
            
            self._sam3_loaded = True
            
        except ImportError as e:
            raise ImportError(
                "Please install SAM3: pip install git+https://github.com/facebookresearch/segment-anything-3.git"
            ) from e
    
    def freeze_qwen(self):
        """Freeze Qwen3-VL parameters."""
        for param in self.qwen_encoder.parameters():
            param.requires_grad = False
        self.qwen_encoder.eval()
        
    def freeze_sam3_image_encoder(self):
        """Freeze SAM3 Image Encoder parameters."""
        self._load_sam3()
        for param in self.sam3_image_encoder.parameters():
            param.requires_grad = False
        self.sam3_image_encoder.eval()
        
    def freeze_all_except_adapter_and_detector(self):
        """Freeze all components except adapter and detector."""
        self.freeze_qwen()
        self.freeze_sam3_image_encoder()
        
        # Make adapter and detector trainable
        for param in self.adapter.parameters():
            param.requires_grad = True
        for param in self.sam3_detector.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        plane_image: Image.Image,
        object_image: Image.Image,
        text_prompt: str,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for placement prediction.
        
        Args:
            plane_image: Top-down view of the plane/room (for SAM3 Image Encoder)
            object_image: Top-down view of the object (for Qwen3-VL)
            text_prompt: Text instruction for placement
            **kwargs: Additional arguments
            
        Returns:
            output: Dictionary containing:
                - masks: Predicted placement masks
                - embeddings: Intermediate embeddings
        """
        self._load_sam3()
        
        # Encode plane image with SAM3 Image Encoder
        plane_embeddings = self._encode_plane_image(plane_image)
        
        # Encode object + text with Qwen3-VL + Adapter
        text_embeddings = self._encode_object_and_text(object_image, text_prompt)
        
        # Combine embeddings for detector
        combined_embeddings = self._combine_embeddings(
            plane_embeddings, text_embeddings
        )
        
        # Run detector to get placement masks
        masks = self.sam3_detector(
            image_embeddings=plane_embeddings,
            text_embeddings=text_embeddings,
        )
        
        return {
            "masks": masks,
            "plane_embeddings": plane_embeddings,
            "text_embeddings": text_embeddings,
        }
    
    def _encode_plane_image(self, image: Image.Image) -> torch.Tensor:
        """Encode plane image using SAM3 Image Encoder."""
        # Preprocess image for SAM3
        # Note: Actual preprocessing depends on SAM3's expected input format
        self.sam3_image_encoder.eval()
        
        with torch.no_grad():
            embeddings = self.sam3_image_encoder(image)
        
        return embeddings
    
    def _encode_object_and_text(
        self,
        object_image: Image.Image,
        text: str,
    ) -> torch.Tensor:
        """Encode object image and text using Qwen3-VL + Adapter."""
        # Get Qwen3-VL embeddings
        qwen_embeddings = self.qwen_encoder(
            object_image=object_image,
            text_prompt=text,
        )
        
        # Project to SAM3 space using adapter
        projected = self.adapter(qwen_embeddings)
        
        return projected
    
    def _combine_embeddings(
        self,
        plane_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Combine plane and text embeddings for detector."""
        # Concatenate along sequence dimension
        # Shape: (batch, plane_seq + text_seq, dim)
        combined = torch.cat([plane_embeddings, text_embeddings], dim=1)
        return combined
    
    def predict(
        self,
        plane_image: Image.Image,
        object_image: Image.Image,
        text_prompt: str,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Inference method for placement prediction.
        
        Args:
            plane_image: Plane/room top-down view
            object_image: Object top-down view
            text_prompt: Placement instruction
            threshold: Confidence threshold for masks
            
        Returns:
            results: Dictionary with masks, boxes, scores
        """
        self.eval()
        
        with torch.no_grad():
            output = self.forward(plane_image, object_image, text_prompt)
        
        masks = output["masks"]
        
        # Apply threshold
        binary_masks = (masks > threshold).float()
        
        # Extract boxes and scores (implementation depends on SAM3 output format)
        boxes = self._masks_to_boxes(binary_masks)
        scores = masks.amax(dim=(2, 3))  # Simple confidence score
        
        return {
            "masks": binary_masks,
            "boxes": boxes,
            "scores": scores,
        }
    
    def _masks_to_boxes(self, masks: torch.Tensor) -> torch.Tensor:
        """Convert masks to bounding boxes."""
        # Find bounding boxes from binary masks
        # Shape: (batch, num_masks, 4) - (x1, y1, x2, y2)
        boxes = []
        for mask in masks:
            coords = torch.nonzero(mask)
            if len(coords) > 0:
                y_min, x_min = coords.min(dim=0).values
                y_max, x_max = coords.max(dim=0).values
                boxes.append(torch.stack([x_min, y_min, x_max, y_max]))
            else:
                boxes.append(torch.zeros(4, device=masks.device))
        
        return torch.stack(boxes) if boxes else torch.zeros(
            masks.size(0), 4, device=masks.device
        )


class PlacementLoss(nn.Module):
    """
    Loss function for object placement prediction.
    
    Combines:
        - Segmentation loss (Dice + BCE)
        - Position regularization
    """
    
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(
        self,
        predicted_masks: torch.Tensor,
        target_masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute placement prediction loss.
        
        Args:
            predicted_masks: Predicted masks (logits)
            target_masks: Ground truth masks
            
        Returns:
            losses: Dictionary with total and component losses
        """
        # BCE loss
        bce_loss = self.bce_loss(predicted_masks, target_masks)
        
        # Dice loss
        dice_loss = self._dice_loss(predicted_masks, target_masks)
        
        # Total loss
        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        
        return {
            "total": total_loss,
            "dice": dice_loss,
            "bce": bce_loss,
        }
    
    def _dice_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1.0,
    ) -> torch.Tensor:
        """Compute Dice loss."""
        predicted = torch.sigmoid(predicted)
        
        intersection = (predicted * target).sum()
        union = predicted.sum() + target.sum()
        
        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        
        return 1.0 - dice_score


class VLALoss(nn.Module):
    """
    VLA Loss combining mask segmentation and text generation.
    """

    def __init__(self, mask_weight: float = 1.0, text_weight: float = 0.5):
        super().__init__()
        self.mask_weight = mask_weight
        self.text_weight = text_weight
        self.mask_loss_fn = PlacementLoss()
        self.text_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
        pred_text_logits: Optional[torch.Tensor] = None,
        target_text_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        losses = self.mask_loss_fn(pred_masks, target_masks)

        if pred_text_logits is not None and target_text_ids is not None:
            text_loss = self.text_loss_fn(
                pred_text_logits.view(-1, pred_text_logits.size(-1)),
                target_text_ids.view(-1)
            )
            losses["text"] = text_loss
            losses["total"] = self.mask_weight * losses["total"] + self.text_weight * text_loss

        return losses

