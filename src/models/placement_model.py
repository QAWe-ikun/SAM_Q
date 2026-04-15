"""
Placement Model - SAM3 with Qwen3-VL for Object Placement Prediction

Architecture:
    Qwen3-VL → [SEG] token → 并行输出:
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
from .vla import SEGActionHead, VLAIterativeRefinement

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
            sam3_model: Pre-loaded SAM3 model (if None, will be loaded)
            qwen_model_name: HuggingFace model name for Qwen3-VL
            sam3_input_dim: SAM3 Detector input dimension
            qwen_hidden_dim: Qwen3-VL hidden dimension
            adapter_hidden_dim: Adapter hidden dimension
            num_seg_tokens: Number of [SEG] tokens. 1=single placement, >1=multi-placement.
            device: Device to run model on
            dtype: Data type for model
            seg_token_config: Config for [SEG] token (heatmap, rotation, scale)
            action_head_config: Config for SEGActionHead (rotation + scale)
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

        # Adapter: [SEG] hidden state → SAM3 prompt embeddings
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

        # SEGActionHead: [SEG] hidden → rotation + scale (parallel to SAM3)
        ah_cfg = action_head_config or {}
        heatmap_size = ah_cfg.get("heatmap_size", 64)
        self.seg_action_head = SEGActionHead(
            hidden_dim=qwen_hidden_dim,
            heatmap_size=heatmap_size,
        )

        # SAM3 components (will be loaded lazily)
        self.sam3_vision_backbone = None
        self.sam3_transformer = None
        self.sam3_seg_head = None
        self.sam3_dot_scoring = None
        self.sam3_model = sam3_model

        self._sam3_loaded = False

        # [SEG] token config
        self._seg_force_only = seg_token_config.get("force_only_in_training", True) if seg_token_config else True
        self._seg_max_tokens = seg_token_config.get("max_generate_tokens", 128) if seg_token_config else 128

        # Optional iterative refinement
        self.vla_refinement = None
        if ah_cfg.get("use_iterative_refinement", False):
            # Note: refinement uses same SEGActionHead, not separate module
            self.vla_refinement = VLAIterativeRefinement(
                vla_model=self,
                max_iterations=ah_cfg.get("max_iterations", 3),
                adjustment_threshold=ah_cfg.get("adjustment_threshold", 0.01),
            )

        # Move trainable components to device
        self.adapter.to(self.device)
        self.seg_action_head.to(self.device)
        if self.seg_projector is not None:
            self.seg_projector.to(self.device)
        
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
        for param in self.sam3_transformer.parameters():
            param.requires_grad = True
        for param in self.sam3_seg_head.parameters():
            param.requires_grad = True
        for param in self.sam3_dot_scoring.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        plane_image,
        text_prompt: str = "",
        images: Optional[List] = None,
        seg_hidden: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: [SEG] token → parallel outputs.

        Args:
            plane_image: 房间俯视图（PIL.Image 或 Tensor）
            text_prompt: 文本指令
            images: Qwen3-VL 图片列表（seg_hidden 为 None 时需要）
            seg_hidden: 预提取的 [SEG] hidden state [B, hidden_dim]（有值时跳过 Qwen3-VL）

        Qwen3-VL → [SEG] token
            ├→ SAM3 Decoder → placement heatmap
            └→ SEGActionHead → rotation + scale

        Args:
            plane_image: Top-down view of the room (for SAM3)
            text_prompt: Text instruction with optional <image> placeholders
            images: List of PIL images for Qwen3-VL (e.g., [room_image, object_image])

        Returns:
            output dict with:
                - heatmap: [B, 1, H_heatmap, W_heatmap] placement heatmap
                - rotation_deg: [B] rotation angle (-180 to 180)
                - scale_relative: [B] relative scale (0.5 to 2.0)
                - seg_hidden: [B, 4096] [SEG] token hidden state
        """
        self._load_sam3()

        # 1. Encode plane image with SAM3 vision backbone
        backbone_out = self._encode_plane_image(plane_image)

        # 2. [SEG] hidden state: 预提取 or Qwen3-VL 在线推理
        if seg_hidden is None:
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
        # Project [SEG] to SAM3 prompt space
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
            memory = self.sam3_transformer.encoder(
                src=img_feats_seq,
                src_pos=img_pos_seq,
                prompt=prompt,
                prompt_pos=prompt_pos,
                prompt_key_padding_mask=prompt_mask,
                feat_sizes=feat_sizes,
            )

            # SAM3 transformer decoder
            B = text_embeddings.size(0)
            query_embed = self.sam3_transformer.decoder.query_embed.weight
            tgt = query_embed.unsqueeze(1).repeat(1, B, 1)

            hs, _, _, _ = self.sam3_transformer.decoder(
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
            class_logits = self.sam3_dot_scoring(hs_4d, prompt, prompt_mask)
            masks = self.sam3_seg_head(
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

        heatmap = output["heatmap"]

        # Upsample heatmap to original plane_image size
        orig_size = plane_image.size[::-1]  # (H, W)
        if heatmap.shape[-2:] != orig_size:
            heatmap = F.interpolate(
                heatmap,
                size=orig_size,
                mode="bilinear",
                align_corners=False,
            )

        # Apply threshold
        binary_heatmap = (heatmap > threshold).float()

        return {
            "heatmap": heatmap,
            "binary_heatmap": binary_heatmap,
            "rotation_6d": output["rotation_6d"],
            "rotation_matrix": output["rotation_matrix"],
            "scale_relative": output["scale_relative"],
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

