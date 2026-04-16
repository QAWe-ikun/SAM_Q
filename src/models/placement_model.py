"""
Placement Model - SAM3 with Qwen3-VL for Object Placement Prediction

Architecture:
    Qwen3-VL → <SEG> token → 并行输出:
        ├→ SAM3 Decoder → 摆放位置热力图
        └→ SEGActionHead → 旋转角度 + 缩放比例
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List
from PIL import Image

from .encoders.qwen3vl_encoder import Qwen3VLEncoder
from .adapters import CrossModalAdapter, SegTokenProjector
from .vla import SEGActionHead
from .loaders import SAM3Loader

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
        sam3_checkpoint_path: Optional[str] = None,
        qwen_model_name: Optional[str] = None,
        qwen_lora_path: Optional[str] = None,
        sam3_input_dim: int = 256,
        qwen_hidden_dim: int = 4096,
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
            sam3_checkpoint_path: Path to SAM3 checkpoint file (.pt).
                           Can be pretrained weights OR a trained model checkpoint.
            qwen_model_name: HuggingFace model name or local path for Qwen3-VL.
                           Set None to skip Qwen3-VL initialization (Stage 2 with seg_features).
            qwen_lora_path: Path to Qwen3-VL LoRA adapter directory (Stage 1 fine-tuned weights).
                           If provided, the LoRA adapter will be loaded for inference.
            sam3_input_dim: SAM3 Detector input dimension.
            qwen_hidden_dim: Qwen3-VL hidden dimension
            adapter_hidden_dim: Adapter hidden dimension.
            num_seg_tokens: Number of <SEG> tokens. 1=single placement, >1=multi-placement.
            device: Device to run model on
            dtype: Data type for model
            seg_token_config: Config for <SEG> token.
            action_head_config: Config for SEGActionHead.
        """
        super().__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.sam3_input_dim = sam3_input_dim
        self.qwen_hidden_dim = qwen_hidden_dim
        self.num_seg_tokens = num_seg_tokens

        # Qwen3-VL Encoder (only if qwen_model_name is provided)
        self.qwen_encoder = None
        if qwen_model_name is not None:
            self.qwen_encoder = Qwen3VLEncoder(
                model_name=qwen_model_name,
                device=self.device,
                dtype=self.dtype,
                num_seg_tokens=num_seg_tokens,
            )
        
        self.qwen_lora_path = qwen_lora_path  # Store LoRA path for loading during forward pass

        # SAM3-related components (only initialized if checkpoint path is provided)
        self.sam3_loader = None
        self.adapter = None
        self.seg_projector = None
        self.seg_action_head = None

        if sam3_checkpoint_path is not None:
            # SAM3 Loader (lazy loading, version selection via checkpoint path)
            self.sam3_loader = SAM3Loader(
                checkpoint_path=sam3_checkpoint_path,
                device=self.device,
                dtype=torch.bfloat16,  # SAM3 uses bfloat16
            )
            
            # Adapter: <SEG> hidden state → SAM3 prompt embeddings
            self.adapter = CrossModalAdapter(
                qwen_dim=qwen_hidden_dim,
                sam3_dim=sam3_input_dim,
                hidden_dim=adapter_hidden_dim,
            )
            
            # SegTokenProjector: optional, used when num_seg_tokens > 1
            if num_seg_tokens > 1:
                seg_cfg = seg_token_config or {}
                self.seg_projector = SegTokenProjector(
                    qwen_dim=qwen_hidden_dim,
                    sam3_dim=sam3_input_dim,
                    num_output_tokens=seg_cfg.get("num_output_tokens", 64),
                    hidden_dim=seg_cfg.get("hidden_dim", adapter_hidden_dim),
                    dropout=seg_cfg.get("dropout", 0.1),
                )

            # SEGActionHead: <SEG> hidden → rotation + scale (parallel to SAM3)
            ah_cfg = action_head_config or {}
            heatmap_size = ah_cfg.get("heatmap_size", 64)
            self.seg_action_head = SEGActionHead(
                hidden_dim=qwen_hidden_dim,
                heatmap_size=heatmap_size,
            )

            # <SEG> token config
            self._seg_force_only = seg_token_config.get("force_only_in_training", True) if seg_token_config else True
            self._seg_max_tokens = seg_token_config.get("max_generate_tokens", 128) if seg_token_config else 128

            # Move trainable components to device
            self.adapter.to(self.device)
            self.seg_action_head.to(self.device)
            if self.seg_projector is not None:
                self.seg_projector.to(self.device)
        else:
            # Stage 1 (LM) mode: no SAM3/Adapter components needed
            self._seg_force_only = seg_token_config.get("force_only_in_training", True) if seg_token_config else True
            self._seg_max_tokens = seg_token_config.get("max_generate_tokens", 128) if seg_token_config else 128
        
    def freeze_qwen(self):
        """Freeze Qwen3-VL parameters."""
        if self.qwen_encoder is None:
            return
        for param in self.qwen_encoder.parameters():
            param.requires_grad = False
        self.qwen_encoder.eval()
        
    def freeze_sam3_image_encoder(self):
        """Freeze SAM3 vision backbone parameters."""
        if self.sam3_loader is None:
            return
        self.sam3_loader.freeze_vision_backbone()
        
    def freeze_all_except_adapter_and_detector(self):
        """Freeze all components except adapter, action head, and detector."""
        self.freeze_qwen()
        self.freeze_sam3_image_encoder()

        # Make adapter, action head, and detector trainable
        for param in self.adapter.parameters():
            param.requires_grad = True
        for param in self.seg_action_head.parameters():
            param.requires_grad = True
        if self.seg_projector is not None:
            for param in self.seg_projector.parameters():
                param.requires_grad = True

        # Unfreeze SAM3 transformer, seg_head, dot_scoring
        self.sam3_loader.freeze_all_except(trainable_components=["transformer", "seg_head", "dot_scoring"])

    def _encode_plane_image(self, image) -> Dict:
        """Encode plane image using SAM3 vision backbone.

        Args:
            image: PIL.Image 或 torch.Tensor [C, H, W] / [B, C, H, W]
        """
        import torchvision.transforms.functional as TF

        if isinstance(image, torch.Tensor):
            # 已经是 tensor，确保形状正确
            tensor = image.to(self.device, dtype=torch.float32)
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)  # [C,H,W] → [1,C,H,W]
            # resize 到 SAM3 尺寸
            tensor = F.interpolate(tensor, size=(_SAM3_SIZE, _SAM3_SIZE), mode="bilinear", align_corners=False)
            # 归一化
            mean = torch.tensor(_SAM3_MEAN, device=self.device).view(1, 3, 1, 1)
            std  = torch.tensor(_SAM3_STD,  device=self.device).view(1, 3, 1, 1)
            tensor = (tensor - mean) / std
        else:
            # PIL.Image 路径
            img = image.convert("RGB").resize((_SAM3_SIZE, _SAM3_SIZE), Image.Resampling.BILINEAR)
            tensor = TF.to_tensor(img).to(self.device, dtype=torch.float32)
            mean = torch.tensor(_SAM3_MEAN, device=self.device).view(3, 1, 1)
            std  = torch.tensor(_SAM3_STD,  device=self.device).view(3, 1, 1)
            tensor = (tensor - mean) / std
            tensor = tensor.unsqueeze(0)  # [1, 3, H, W]

        # Cast input to match backbone checkpoint dtype (bfloat16)
        backbone_dtype = next(self.sam3_loader.sam3_vision_backbone.parameters()).dtype
        tensor = tensor.to(dtype=backbone_dtype)

        with torch.no_grad():
            img_feats, img_pos_embeds, _, _ = self.sam3_loader.sam3_vision_backbone(tensor)

        # Construct feat_sizes from spatial dims (neck doesn't return them)
        vis_feat_sizes = [(f.shape[2], f.shape[3]) for f in img_feats]

        return {
            "img_feats": img_feats,
            "img_pos_embeds": img_pos_embeds,
            "vis_feat_sizes": vis_feat_sizes,
        }

    def load_all(self, eval_mode: bool = True):
        """
        Load all model components (Qwen3-VL, SAM3, LoRA adapter).

        This is useful for inference mode after training is complete.

        Args:
            eval_mode: Whether to set all models to eval mode.
        """

        # 1. Load Qwen3-VL (if initialized)
        if self.qwen_encoder is not None:
            if not self.qwen_encoder.model:
                self.qwen_encoder.load_model(use_cache=eval_mode)

            # 2. Load LoRA adapter (Stage 1 fine-tuned weights)
            if self.qwen_lora_path is not None:
                self.qwen_encoder.load_lora_adapter(self.qwen_lora_path)

        # 3. Load SAM3
        if self.sam3_loader is not None:
            if not self.sam3_loader.loaded:
                self.sam3_loader.load_model(eval_mode=eval_mode)

        # Set eval mode if requested
        if eval_mode:
            self.eval()
    
    def forward(
        self,
        plane_image,
        text_prompt: Optional[str] = None,
        images: Optional[List] = None,
        seg_hidden: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: <SEG> token → parallel outputs.

        Qwen3-VL → <SEG> token
                            ├→ SAM3 Decoder → placement heatmap
                            └→ SEGActionHead → rotation + scale

        Args:
            plane_image: Top-down view of the room (for SAM3)
            text_prompt: Text instruction with optional <image> placeholders
            images: List of PIL images for Qwen3-VL (e.g., [room_image, object_image])
            seg_hidden: Optional pre-extracted <SEG> hidden state (skip Qwen3-VL if provided)

        Returns:
            output dict with:
                - heatmap: [B, 1, H_heatmap, W_heatmap] placement heatmap
                - rotation_deg: [B] rotation angle (-180 to 180)
                - scale_relative: [B] relative scale (0.5 to 2.0)
                - seg_hidden: [B, 4096] <SEG> token hidden state
        """
        self.load_all()
        # 1. Encode plane image with SAM3 vision backbone
        backbone_out = self._encode_plane_image(plane_image)

        # 2. <SEG> hidden state: 预提取 or Qwen3-VL 在线推理
        if seg_hidden is None:
            if self.qwen_encoder is None:
                raise ValueError(
                    "qwen_model_name is None but seg_hidden is also None. "
                    "Either provide seg_hidden or set qwen_model_name."
                )
            seg_hidden, _ = self.qwen_encoder.generate_with_seg(
                text_prompt=text_prompt,
                images=images,
                max_new_tokens=self._seg_max_tokens,
                force_only=self._seg_force_only if self.training else False,
                num_seg=self.num_seg_tokens,
            )
        else:
            # 预提取的 seg_hidden 可能是 [hidden_dim]，需要确保 [B, hidden_dim]
            if seg_hidden.dim() == 1:
                seg_hidden = seg_hidden.unsqueeze(0)
            seg_hidden = seg_hidden.to(self.device)

        # === Parallel Branch 1: SAM3 → Heatmap ===
        # Project <SEG> to SAM3 prompt space
        seg_3d = seg_hidden.unsqueeze(1)  # [B, 1, 4096]
        text_embeddings = self.adapter(seg_3d.to(device=self.device, dtype=torch.float32))

        prompt = text_embeddings.permute(1, 0, 2).to(dtype=torch.bfloat16)
        prompt_mask = torch.zeros(
            text_embeddings.size(0), text_embeddings.size(1),
            dtype=torch.bool, device=self.device,
        )

        device_type = self.device.split(":")[0] if ":" in self.device else self.device
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            # SAM3 transformer encoder
            last_feat = backbone_out["img_feats"][-1]
            last_pos  = backbone_out["img_pos_embeds"][-1]
            img_feats_seq = [last_feat.flatten(2).permute(2, 0, 1)]
            img_pos_seq   = [last_pos.flatten(2).permute(2, 0, 1)]
            feat_sizes    = [backbone_out["vis_feat_sizes"][-1]]

            prompt_pos = torch.zeros_like(prompt)
            memory = self.sam3_loader.sam3_transformer.encoder(
                src=img_feats_seq,
                src_pos=img_pos_seq,
                prompt=prompt,
                prompt_pos=prompt_pos,
                prompt_key_padding_mask=prompt_mask,
                feat_sizes=feat_sizes,
            )

            # SAM3 transformer decoder
            B = text_embeddings.size(0)
            query_embed = self.sam3_loader.sam3_transformer.decoder.query_embed.weight
            tgt = query_embed.unsqueeze(1).repeat(1, B, 1)

            hs, _, _, _ = self.sam3_loader.sam3_transformer.decoder(
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

            # Segmentation head → heatmap
            hs_4d = hs.permute(0, 2, 1, 3)  # [num_layers, B, num_queries, d]
            class_logits = self.sam3_loader.sam3_dot_scoring(hs_4d, prompt, prompt_mask)
            masks = self.sam3_loader.sam3_seg_head(
                backbone_feats=backbone_out["img_feats"],
                obj_queries=hs_4d,
                image_ids=torch.arange(B, device=self.device),
                encoder_hidden_states=memory["memory"],
                prompt=prompt,
                prompt_mask=prompt_mask,
            )

        # Extract heatmap from masks dict
        if isinstance(masks, dict):
            heatmap = masks.get("pred_masks", masks.get("semantic_seg"))
        else:
            heatmap = masks

        # === Parallel Branch 2: SEGActionHead → rotation + scale ===
        # Convert to float32 for SEGActionHead (Qwen outputs float16)
        seg_hidden_fp32 = seg_hidden.to(dtype=torch.float32)
        action_output = self.seg_action_head(seg_hidden_fp32)

        return {
            "heatmap": heatmap,            # [B, num_candidates, H, W] or [B, 1, H, W]
            "rotation_6d": action_output["rotation_6d"],   # [B, 6]
            "rotation_matrix": action_output["rotation_matrix"],  # [B, 3, 3]
            "scale_relative": action_output["scale"],    # [B]
            "seg_hidden": seg_hidden,      # [B, 4096]
            "class_logits": class_logits,  # [num_layers, B, num_queries, 1]
        }

    def predict(
        self,
        plane_image: Image.Image,
        text_prompt: str,
        images: Optional[List[Image.Image]] = None,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Inference: placement heatmap + rotation + scale.

        Note: Exact position is determined by H-MVP from the heatmap.

        Args:
            plane_image: Plane/room top-down view
            text_prompt: Placement instruction with optional <image> placeholders
            images: List of PIL images for Qwen3-VL
            threshold: Confidence threshold for heatmap

        Returns:
            results: Dictionary with heatmap, rotation_deg, scale_relative
        """
        self.eval()

        with torch.no_grad():
            output = self.forward(plane_image, text_prompt, images=images)

        heatmap = output["heatmap"]  # [B, num_candidates, H, W]
        class_logits = output.get("class_logits")  # [num_layers, B, num_candidates, 1]

        # Apply sigmoid if logits
        if heatmap.min() < 0 or heatmap.max() > 1:
            heatmap = torch.sigmoid(heatmap)

        # Select best candidate using class_logits (SAM3 objectness score)
        best_idx = 0
        if class_logits is not None:
            scores = class_logits[-1, 0].squeeze(-1)  # [num_candidates]
            best_idx = scores.argmax().item()
        else:
            # Fallback: use mean probability
            candidate_scores = heatmap[0].flatten(1).mean(dim=1)
            best_idx = candidate_scores.argmax().item()

        # Extract best candidate's heatmap
        best_heatmap = heatmap[0, best_idx]  # [H, W]

        # Upsample heatmap to original plane_image size
        orig_size = plane_image.size[::-1]  # (H, W)
        if best_heatmap.shape[-2:] != orig_size:
            best_heatmap = F.interpolate(
                best_heatmap.unsqueeze(0).unsqueeze(0),
                size=orig_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        # Apply threshold
        binary_heatmap = (best_heatmap > threshold).float()

        # Generate Qwen response
        qwen_response = None
        if self.qwen_encoder is not None and self.qwen_encoder.model is not None:
            try:
                qwen_response = self.qwen_encoder.generate_response(
                    text_prompt=text_prompt,
                    images=images,
                    max_new_tokens=128,
                )
            except Exception:
                pass

        return {
            "heatmap": best_heatmap,
            "binary_heatmap": binary_heatmap,
            "rotation_6d": output["rotation_6d"],
            "rotation_matrix": output["rotation_matrix"],
            "scale_relative": output["scale_relative"],
            "best_candidate_idx": best_idx,
            "qwen_response": qwen_response,
        }


class PlacementLoss(nn.Module):
    """
    Loss function for object placement prediction.
    
    Combines:
        - Segmentation loss (Dice + BCE)
        - Position regularization
    """
    
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0,
                 rotation_weight: float = 0.5, scale_weight: float = 0.3):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.rotation_weight = rotation_weight
        self.scale_weight = scale_weight

        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        predicted_masks: torch.Tensor,
        target_masks: torch.Tensor,
        pred_rotation_6d: Optional[torch.Tensor] = None,
        pred_scale: Optional[torch.Tensor] = None,
        class_logits: Optional[torch.Tensor] = None,
        gt_rotation_6d: Optional[torch.Tensor] = None,
        gt_scale: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute placement prediction loss.

        Args:
            predicted_masks: Predicted masks (logits) [B, num_candidates, H, W]
            target_masks: Ground truth masks [B, 1, H, W]
            pred_rotation_6d: Predicted 6D rotation [B, 6] (optional)
            pred_scale: Predicted scale [B] (optional)
            class_logits: [num_layers, B, num_candidates, 1] (optional)
            gt_rotation_6d: GT 6D rotation [B, 6] (optional, 有监督)
            gt_scale: GT scale [B, 1] (optional, 有监督)

        Returns:
            losses: Dictionary with total and component losses
        """
        losses = {"total": torch.tensor(0.0, device=predicted_masks.device, dtype=predicted_masks.dtype)}

        # 如果有多个候选 mask，用 class_logits 选最佳
        if predicted_masks.shape[1] > 1 and class_logits is not None:
            # class_logits: [num_layers, B, num_candidates, 1] → 取最后一层
            scores = class_logits[-1].squeeze(-1)  # [B, num_candidates]
            best_idx = scores.argmax(dim=-1)        # [B]
            # 取每个 batch 的最佳 mask
            B = predicted_masks.shape[0]
            predicted_masks = predicted_masks[torch.arange(B, device=predicted_masks.device), best_idx]  # [B, H, W]
            predicted_masks = predicted_masks.unsqueeze(1)  # [B, 1, H, W]

        # resize 到 target 尺寸
        if predicted_masks.shape[-2:] != target_masks.shape[-2:]:
            predicted_masks = F.interpolate(
                predicted_masks.float(), size=target_masks.shape[-2:],
                mode="bilinear", align_corners=False,
            ).to(predicted_masks.dtype)

        # BCE loss
        bce_loss = self.bce_loss(predicted_masks.float(), target_masks.float())
        losses["bce"] = bce_loss
        losses["total"] = losses["total"] + self.bce_weight * bce_loss

        # Dice loss
        dice_loss = self._dice_loss(predicted_masks, target_masks)
        losses["dice"] = dice_loss
        losses["total"] = losses["total"] + self.dice_weight * dice_loss

        # Rotation loss
        if pred_rotation_6d is not None:
            if gt_rotation_6d is not None:
                # 有监督: L1 loss between predicted and GT 6D rotation
                rotation_loss = F.l1_loss(pred_rotation_6d, gt_rotation_6d.to(pred_rotation_6d.device))
            else:
                # 无监督: L2 regularization (趋近单位旋转)
                rotation_loss = pred_rotation_6d.pow(2).mean()
            losses["rotation"] = rotation_loss
            losses["total"] = losses["total"] + self.rotation_weight * rotation_loss

        # Scale loss
        if pred_scale is not None:
            if gt_scale is not None:
                # 有监督: L1 loss between predicted and GT scale
                gt_s = gt_scale.to(pred_scale.device).squeeze(-1)  # [B]
                scale_loss = F.l1_loss(pred_scale, gt_s)
            else:
                # 无监督: 正则化 (趋近 1.0)
                scale_loss = (pred_scale - 1.0).pow(2).mean()
            losses["scale"] = scale_loss
            losses["total"] = losses["total"] + self.scale_weight * scale_loss

        return losses
    
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

