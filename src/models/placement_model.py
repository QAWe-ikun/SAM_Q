"""
Placement Model - SAM3 with Qwen3-VL for Object Placement Prediction

This module integrates Qwen3-VL as Text Encoder with SAM3 for
predicting optimal object placement positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
from PIL import Image

from .encoders.qwen3vl_encoder import Qwen3VLEncoder
from .adapters import Adapter, CrossModalAdapter, SegTokenProjector
from .action_head import ActionHead

# SAM3 image normalization constants
_SAM3_MEAN = (0.5, 0.5, 0.5)
_SAM3_STD  = (0.5, 0.5, 0.5)
_SAM3_SIZE = 1008  # SAM3 ViT input resolution


class SAMQPlacementModel(nn.Module):
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
        qwen_model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        sam3_input_dim: int = 256,
        qwen_hidden_dim: int = 4096,  # Qwen3-VL-8B hidden size
        adapter_hidden_dim: int = 512,
        num_seg_tokens: int = 1,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        seg_token_config: Optional[Dict[str, Any]] = None,
        action_head_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the placement model.

        Args:
            sam3_model: Pre-loaded SAM3 model (if None, will be loaded)
            qwen_model_name: HuggingFace model name for Qwen3-VL
            sam3_input_dim: SAM3 Detector input dimension
            qwen_hidden_dim: Qwen3-VL hidden dimension
            adapter_hidden_dim: Adapter hidden dimension
            num_seg_tokens: Number of [SEG] tokens. 1=single placement, >1=multi-placement.
            device: Device to run model on
            dtype: Data type for model
            seg_token_config: Config for SegTokenProjector (when num_seg_tokens > 1)
            action_head_config: Config dict for ActionHead (None or {"enabled": false} to disable)
        """
        super().__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.sam3_input_dim = sam3_input_dim
        self.qwen_hidden_dim = qwen_hidden_dim
        self.num_seg_tokens = num_seg_tokens

        # Qwen3-VL Encoder (frozen during training)
        self.qwen_encoder = Qwen3VLEncoder(
            model_name=qwen_model_name,
            device=self.device,
            dtype=self.dtype,
            num_seg_tokens=num_seg_tokens,
        )

        # Adapter: CrossModalAdapter works for both single and multi-SEG
        self.adapter = CrossModalAdapter(
            qwen_dim=qwen_hidden_dim,
            sam3_dim=sam3_input_dim,
            hidden_dim=adapter_hidden_dim,
        )

        # SegTokenProjector: optional, used when num_seg_tokens > 1
        self.seg_projector = None
        if num_seg_tokens > 1:
            seg_cfg = seg_token_config or {}
            self.seg_projector = SegTokenProjector(
                qwen_dim=qwen_hidden_dim,
                sam3_dim=sam3_input_dim,
                num_output_tokens=seg_cfg.get("num_output_tokens", 64),
                hidden_dim=seg_cfg.get("hidden_dim", adapter_hidden_dim),
                dropout=seg_cfg.get("dropout", 0.1),
            )

        # SAM3 components (will be loaded lazily)
        self.sam3_vision_backbone = None   # Sam3DualViTDetNeck
        self.sam3_transformer = None       # TransformerWrapper (encoder + decoder)
        self.sam3_seg_head = None          # UniversalSegmentationHead
        self.sam3_dot_scoring = None       # DotProductScoring
        self.sam3_model = sam3_model

        # Placeholder for SAM3 dimensions
        self._sam3_loaded = False

        # [SEG] token config for inference
        self._seg_force_only = seg_token_config.get("force_only_in_training", True) if seg_token_config else True
        self._seg_max_tokens = seg_token_config.get("max_generate_tokens", 128) if seg_token_config else 128

        # Action head (optional, for VLA pose prediction)
        ah_cfg = action_head_config or {}
        if ah_cfg.get("enabled", False):
            self.action_head = ActionHead(
                input_dim=sam3_input_dim,
                hidden_dim=ah_cfg.get("hidden_dim", 512),
            )
        else:
            self.action_head = None

        # Move trainable components to device
        self.adapter.to(self.device)
        if self.seg_projector is not None:
            self.seg_projector.to(self.device)
        if self.action_head is not None:
            self.action_head.to(self.device)
        
    def _load_sam3(self):
        """Lazy load SAM3 model and extract components."""
        if self._sam3_loaded:
            return

        try:
            from sam3.model_builder import build_sam3_image_model
        except ImportError as e:
            raise ImportError(
                "Please install SAM3: pip install git+https://github.com/facebookresearch/sam3.git"
            ) from e

        if self.sam3_model is None:
            import os
            # Look for local checkpoint relative to project root (cwd)
            local_ckpt = os.path.join(os.getcwd(), "models", "sam3", "sam3.pt")
            if not os.path.exists(local_ckpt):
                raise FileNotFoundError(
                    f"SAM3 checkpoint not found at {local_ckpt}. "
                    "Run: python scripts/download_models.py --only-sam3"
                )
            self.sam3_model = build_sam3_image_model(
                checkpoint_path=local_ckpt,
                device=self.device,
                eval_mode=True,
                load_from_HF=False,
            )

        # Extract the components we need
        self.sam3_vision_backbone = self.sam3_model.backbone.vision_backbone
        self.sam3_transformer     = self.sam3_model.transformer
        self.sam3_seg_head        = self.sam3_model.segmentation_head
        self.sam3_dot_scoring     = self.sam3_model.dot_prod_scoring

        # Keep SAM3 in bfloat16 (checkpoint dtype); convert any stray float32 params too
        self.sam3_vision_backbone.to(device=self.device, dtype=torch.bfloat16)
        self.sam3_transformer.to(device=self.device, dtype=torch.bfloat16)
        self.sam3_seg_head.to(device=self.device, dtype=torch.bfloat16)
        self.sam3_dot_scoring.to(device=self.device, dtype=torch.bfloat16)

        self._sam3_loaded = True
    
    def freeze_qwen(self):
        """Freeze Qwen3-VL parameters."""
        for param in self.qwen_encoder.parameters():
            param.requires_grad = False
        self.qwen_encoder.eval()
        
    def freeze_sam3_image_encoder(self):
        """Freeze SAM3 vision backbone parameters."""
        self._load_sam3()
        for param in self.sam3_vision_backbone.parameters():
            param.requires_grad = False
        self.sam3_vision_backbone.eval()
        
    def freeze_all_except_adapter_and_detector(self):
        """Freeze all components except adapter and detector."""
        self.freeze_qwen()
        self.freeze_sam3_image_encoder()

        # Make adapter and detector trainable
        for param in self.adapter.parameters():
            param.requires_grad = True
        if self.seg_projector is not None:
            for param in self.seg_projector.parameters():
                param.requires_grad = True
        for param in self.sam3_transformer.parameters():
            param.requires_grad = True
        for param in self.sam3_seg_head.parameters():
            param.requires_grad = True
        for param in self.sam3_dot_scoring.parameters():
            param.requires_grad = True
        if self.action_head is not None:
            for param in self.action_head.parameters():
                param.requires_grad = True
    
    def forward(
        self,
        plane_image: Image.Image,
        text_prompt: str,
        images: Optional[List[Image.Image]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for placement prediction.

        Args:
            plane_image: Top-down view of the plane/room (for SAM3)
            text_prompt: Text instruction with optional <image> placeholders
            images: List of PIL images for Qwen3-VL (e.g., [room_image, object_image])
            **kwargs: Additional arguments (ignored, for compatibility)

        Returns:
            output dict with masks, text_embeddings, and optional action outputs
        """
        self._load_sam3()

        # 1. Encode plane image with SAM3 vision backbone
        backbone_out = self._encode_plane_image(plane_image)

        # 2. Encode images + text with Qwen3-VL + Adapter/Projector → [B, num_queries, 256]
        text_embeddings = self._encode_qwen_multimodal(text_prompt, images)

        # 3. Prepare prompt for SAM3 encoder (seq-first: [seq, B, 256], bfloat16)
        prompt = text_embeddings.permute(1, 0, 2).to(dtype=torch.bfloat16)
        prompt_mask = torch.zeros(
            text_embeddings.size(0), text_embeddings.size(1),
            dtype=torch.bool, device=self.device,
        )

        # 4–6: Run SAM3 transformer + segmentation under autocast
        device_type = self.device.split(":")[0] if ":" in self.device else self.device
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            # 4. SAM3 transformer encoder (num_feature_levels=1, use last scale)
            last_feat = backbone_out["img_feats"][-1]
            last_pos  = backbone_out["img_pos_embeds"][-1]
            img_feats_seq = [last_feat.flatten(2).permute(2, 0, 1)]  # [H*W, B, C]
            img_pos_seq   = [last_pos.flatten(2).permute(2, 0, 1)]
            feat_sizes    = [backbone_out["vis_feat_sizes"][-1]]

            prompt_pos = torch.zeros_like(prompt)
            memory = self.sam3_transformer.encoder(
                src=img_feats_seq,
                src_pos=img_pos_seq,
                prompt=prompt,
                prompt_pos=prompt_pos,
                prompt_key_padding_mask=prompt_mask,
                feat_sizes=feat_sizes,
            )

            # 5. SAM3 transformer decoder
            B = text_embeddings.size(0)
            query_embed = self.sam3_transformer.decoder.query_embed.weight
            tgt = query_embed.unsqueeze(1).repeat(1, B, 1)

            hs, reference_boxes, _, _ = self.sam3_transformer.decoder(
                tgt=tgt,
                memory=memory["memory"],
                memory_key_padding_mask=memory["padding_mask"],
                pos=memory["pos_embed"],
                level_start_index=memory["level_start_index"],
                spatial_shapes=memory["spatial_shapes"],
                valid_ratios=memory["valid_ratios"],
                memory_text=prompt,
                text_attention_mask=prompt_mask,
                apply_dac=False,
            )

            # 6. Segmentation head → masks
            # hs: [num_layers, num_queries, B, d] — dot_scoring needs [num_layers, B, num_queries, d]
            hs_4d = hs.permute(0, 2, 1, 3)  # [num_layers, B, num_queries, d]
            class_logits = self.sam3_dot_scoring(hs_4d, prompt, prompt_mask)
            masks = self.sam3_seg_head(
                backbone_feats=backbone_out["img_feats"],
                obj_queries=hs_4d,
                image_ids=torch.arange(B, device=self.device),
                encoder_hidden_states=memory["memory"],
                prompt=prompt,
                prompt_mask=prompt_mask,
            )

        output = {
            "masks": masks,
            "class_logits": class_logits,
            "text_embeddings": text_embeddings,
        }

        # 7. Action head (optional)
        if self.action_head is not None:
            action_output = self.action_head(text_embeddings)
            output.update({
                "scale": action_output["scale"],
                "rotation_6d": action_output["rotation_6d"],
                "rotation_matrix": action_output["rotation_matrix"],
            })

        return output

    def _encode_plane_image(self, image: Image.Image) -> Dict:
        """Encode plane image using SAM3 vision backbone."""
        import torchvision.transforms.functional as TF

        img = image.convert("RGB").resize((_SAM3_SIZE, _SAM3_SIZE), Image.Resampling.BILINEAR)
        tensor = TF.to_tensor(img).to(self.device, dtype=torch.float32)
        mean = torch.tensor(_SAM3_MEAN, device=self.device).view(3, 1, 1)
        std  = torch.tensor(_SAM3_STD,  device=self.device).view(3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.unsqueeze(0)  # [1, 3, H, W]

        # Cast input to match backbone checkpoint dtype (bfloat16)
        backbone_dtype = next(self.sam3_vision_backbone.parameters()).dtype
        tensor = tensor.to(dtype=backbone_dtype)

        with torch.no_grad():
            img_feats, img_pos_embeds, _, _ = self.sam3_vision_backbone(tensor)

        # Construct feat_sizes from spatial dims (neck doesn't return them)
        vis_feat_sizes = [(f.shape[2], f.shape[3]) for f in img_feats]

        return {
            "img_feats": img_feats,
            "img_pos_embeds": img_pos_embeds,
            "vis_feat_sizes": vis_feat_sizes,
        }

    def _encode_qwen_multimodal(
        self,
        text: str,
        images: Optional[List[Image.Image]] = None,
    ) -> torch.Tensor:
        """Encode image(s) and text using Qwen3-VL [SEG] token + Adapter.

        Qwen3-VL first reasons/generates text, then outputs [SEG].
        The [SEG] hidden state contains the full reasoning via self-attention.
        Then we project it to SAM3 prompt space.

        num_seg_tokens=1:  CrossModalAdapter [B, 1, 4096] -> [B, 64, 256]
        num_seg_tokens>1:  CrossModalAdapter [B, n, 4096] -> [B, n, 256]
                          (optional SegTokenProjector for expansion)
        """
        force_only = self._seg_force_only if self.training else False
        seg_hidden, _ = self.qwen_encoder.generate_with_seg(
            text_prompt=text,
            images=images,
            max_new_tokens=self._seg_max_tokens,
            force_only=force_only,
            num_seg=self.num_seg_tokens,
        )

        if self.num_seg_tokens == 1:
            # [B, 4096] -> [B, 1, 4096] -> CrossModalAdapter -> [B, 64, 256]
            seg_3d = seg_hidden.unsqueeze(1)
            return self.adapter(seg_3d.to(device=self.device, dtype=torch.float32))

        # Multi-SEG: [B, n, 4096] -> Adapter/Projector -> [B, n, 256]
        seg_hidden_fp32 = seg_hidden.to(device=self.device, dtype=torch.float32)
        if self.seg_projector is not None:
            return self.seg_projector(seg_hidden_fp32)
        
        return self.adapter(seg_hidden_fp32)

    def predict(
        self,
        plane_image: Image.Image,
        text_prompt: str,
        images: Optional[List[Image.Image]] = None,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Inference method for placement prediction.

        Args:
            plane_image: Plane/room top-down view (for SAM3)
            text_prompt: Placement instruction with optional <image> placeholders
            images: List of PIL images for Qwen3-VL
            threshold: Confidence threshold for masks

        Returns:
            results: Dictionary with masks, boxes, scores
        """
        self.eval()

        with torch.no_grad():
            output = self.forward(plane_image, text_prompt, images=images)
        
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
    VLA Loss for SAM-Q-HMVP system
    Combines placement heatmap loss, collision loss, and semantic alignment
    """

    def __init__(self, heatmap_weight: float = 1.0, collision_weight: float = 0.5, semantic_weight: float = 0.3):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.collision_weight = collision_weight
        self.semantic_weight = semantic_weight
        self.heatmap_loss_fn = nn.MSELoss()
        self.collision_loss_fn = nn.BCEWithLogitsLoss()
        self.semantic_loss_fn = nn.CosineEmbeddingLoss()

    def forward(
        self,
        pred_heatmaps: torch.Tensor,        # [B, 1, H, W] predicted placement heatmaps
        target_heatmaps: torch.Tensor,      # [B, 1, H, W] target placement heatmaps
        pred_collision: torch.Tensor,       # [B] predicted collision probabilities
        target_collision: torch.Tensor,     # [B] target collision (0=free, 1=collision)
        pred_poses: torch.Tensor,           # [B, 7] predicted poses (x,y,z,qx,qy,qz,qw)
        target_poses: torch.Tensor,         # [B, 7] target poses
        semantic_features: torch.Tensor,    # [B, D] semantic features from Qwen3-VL
        target_semantic: torch.Tensor       # [B, D] target semantic features
    ) -> Dict[str, torch.Tensor]:
        losses = {"total": torch.tensor(0.0)}
        
        # Heatmap loss (placement location accuracy)
        heatmap_loss = self.heatmap_loss_fn(pred_heatmaps, target_heatmaps)
        losses["heatmap"] = heatmap_loss
        losses["total"] = losses["total"] + self.heatmap_weight * heatmap_loss
        
        # Collision loss (avoid collisions)
        collision_loss = self.collision_loss_fn(pred_collision, target_collision)
        losses["collision"] = collision_loss
        losses["total"] = losses["total"] + self.collision_weight * collision_loss
        
        # Semantic alignment loss (ensure placement matches text instruction)
        # Use cosine similarity between semantic features
        target_sim = torch.ones(pred_poses.size(0), device=pred_poses.device)  # Same class
        semantic_loss = self.semantic_loss_fn(
            semantic_features, target_semantic, target_sim
        )
        losses["semantic_alignment"] = semantic_loss
        losses["total"] = losses["total"] + self.semantic_weight * semantic_loss

        return losses

