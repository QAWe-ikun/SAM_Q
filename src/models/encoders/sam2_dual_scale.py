"""
SAM Dual-Scale Encoder: High-resolution Detail + Low-resolution Context
=======================================================================

Implementation of dual-scale SAM3 encoding for efficient 2D perception.
Combines high-resolution detail extraction with low-resolution contextual understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class SAM2DualScaleEncoder(nn.Module):
    """
    SAM: Dual-Scale SAM3 Encoder
    
    Combines high-resolution detail extraction with low-resolution contextual understanding
    using cross-scale attention mechanisms.
    """
    
    def __init__(self,
                 sam3_model_path: str = "facebook/sam3-hiera-large",
                 high_res: int = 1024,
                 low_res: int = 256,
                 feature_dims: List[int] = None,
                 output_dim: int = 512):
        super().__init__()
        
        self.high_res = high_res
        self.low_res = low_res
        self.feature_dims = feature_dims or [256, 256, 256, 256]
        self.output_dim = output_dim
        
        # High-resolution branch (detail extraction)
        self.sam_hq = self._build_sam3_branch(input_resolution=high_res)
        
        # Low-resolution branch (context understanding) 
        self.sam_lq = self._build_sam3_branch(input_resolution=low_res)
        
        # Cross-scale feature fusion modules
        self.scale_fusion = CrossScaleFusion(
            hq_dims=self.feature_dims,
            lq_dims=self.feature_dims[:3],  # Assuming fewer levels for low-res
            output_dim=output_dim
        )
        
        # Feature pyramid alignment
        self.pyramid_aligners = nn.ModuleList([
            nn.Conv2d(fd, output_dim, 1) for fd in self.feature_dims
        ])
        
    def _build_sam3_branch(self, input_resolution: int):
        """Build a SAM3 encoder branch"""
        # Load the actual SAM3 model from the official release
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            return SAM3EncoderWrapper(
                input_resolution=input_resolution,
                feature_dims=self.feature_dims
            )
        except ImportError:
            # Fallback to simulated encoder if SAM3 not installed
            print("SAM3 not found, using simulated encoder. Install with: pip install git+https://github.com/facebookresearch/sam3.git")
            return HierarchicalFeatureExtractor(
                input_resolution=input_resolution,
                feature_dims=self.feature_dims
            )
    
    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Dual-scale forward pass
        
        Args:
            image: [B, 3, H, W] Input RGB image
            
        Returns:
            Dict containing multi-scale features and pyramid
        """
        B = image.size(0)
        
        # Prepare inputs for different scales
        if image.size(-1) > self.high_res:
            image_hq = F.interpolate(
                image, 
                size=(self.high_res, self.high_res),
                mode='bilinear', align_corners=False
            )
        else:
            image_hq = image
        
        image_lq = F.interpolate(
            image,
            size=(self.low_res, self.low_res),
            mode='bilinear', align_corners=False
        )
        
        # Process through both branches in parallel
        features_hq = self.sam_hq(image_hq)  # List of [B, C_i, H_i, W_i]
        features_lq = self.sam_lq(image_lq)  # List of [B, C_j, H_j, W_j]
        
        # Cross-scale fusion
        fused_pyramid = self.scale_fusion(features_hq, features_lq)
        
        # Align to consistent output dimension
        aligned_pyramid = [
            aligner(feat) for feat, aligner in zip(fused_pyramid, self.pyramid_aligners)
        ]
        
        return {
            'pyramid': aligned_pyramid,           # Aligned multi-scale features
            'hq_features': features_hq,           # Raw high-res features  
            'lq_features': features_lq,           # Raw low-res features
            'hq_resolution': self.high_res,
            'lq_resolution': self.low_res,
            'global_features': self._extract_global_features(aligned_pyramid)
        }
    
    def _extract_global_features(self, pyramid: List[torch.Tensor]) -> torch.Tensor:
        """Extract global features from pyramid"""
        # Global average pooling on finest level
        finest = pyramid[0]  # Highest resolution
        global_feat = F.adaptive_avg_pool2d(finest, 1).view(finest.size(0), -1)
        return global_feat


class HierarchicalFeatureExtractor(nn.Module):
    """
    Simulated hierarchical feature extractor mimicking SAM3 behavior
    """
    
    def __init__(self, input_resolution: int, feature_dims: List[int]):
        super().__init__()
        self.input_resolution = input_resolution
        self.feature_dims = feature_dims
        
        # Simulate hierarchical encoder with different receptive fields
        self.levels = nn.ModuleList()
        for i, dim in enumerate(feature_dims):
            # Each level processes at different scale
            scale_factor = 2 ** (i + 2)  # 4, 8, 16, 32 (typical SAM3 strides)
            self.levels.append(
                nn.Sequential(
                    nn.Conv2d(3 if i == 0 else feature_dims[i-1], dim, 3, 
                             stride=2, padding=1),  # Downsample
                    nn.GroupNorm(8, dim),
                    nn.ReLU(),
                    # Additional conv layers for feature extraction
                    nn.Conv2d(dim, dim, 3, padding=1),
                    nn.GroupNorm(8, dim),
                    nn.ReLU()
                )
            )
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returning feature pyramid"""
        features = []
        current = x
        
        for level in self.levels:
            current = level(current)
            features.append(current)
        
        return features


class SAM3EncoderWrapper(nn.Module):
    """
    Wrapper for the official SAM3 encoder to match our interface
    """
    
    def __init__(self, input_resolution: int, feature_dims: List[int]):
        super().__init__()
        self.input_resolution = input_resolution
        self.feature_dims = feature_dims
        
        # Try to load the official SAM3 model
        try:
            from sam3.model_builder import build_sam3_image_model
            self.sam3_model = build_sam3_image_model()
            self.image_encoder = self.sam3_model.image_encoder
            self.loaded_successfully = True
        except ImportError:
            print("Warning: Could not load SAM3. Using fallback.")
            self.loaded_successfully = False
            # Create a dummy model with similar interface
            self.image_encoder = HierarchicalFeatureExtractor(
                input_resolution=input_resolution,
                feature_dims=feature_dims
            )
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through SAM3 encoder
        """
        if self.loaded_successfully:
            # Process with actual SAM3 model
            # This assumes SAM3 returns multi-level features
            try:
                # Actual SAM3 processing
                features = self.image_encoder(x)
                # Convert to list format expected by our system
                if isinstance(features, dict):
                    return list(features.values())
                elif isinstance(features, (list, tuple)):
                    return list(features)
                else:
                    # If it's a single tensor, return as list
                    return [features]
            except Exception as e:
                print(f"SAM3 processing failed: {e}, using fallback")
                return self.image_encoder(x)
        else:
            # Use fallback encoder
            return self.image_encoder(x)


class CrossScaleFusion(nn.Module):
    """
    Cross-scale feature fusion: inject high-res details into low-res context
    """
    
    def __init__(self, hq_dims: List[int], lq_dims: List[int], output_dim: int):
        super().__init__()
        
        self.num_levels = len(hq_dims)
        self.output_dim = output_dim
        
        # Cross-scale attention for each level
        self.cross_attention = nn.ModuleList([
            CrossScaleAttention(
                hq_dim=hq_dims[i],
                lq_dim=lq_dims[min(i, len(lq_dims)-1)],
                output_dim=output_dim
            ) for i in range(self.num_levels)
        ])
        
        # Upsampling modules for detail injection
        self.upsamplers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(output_dim, output_dim, 3, padding=1),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ) if i < len(hq_dims) - 1 else nn.Identity()
            for i in range(len(hq_dims) - 1)
        ])
        
    def forward(self, hq_features: List[torch.Tensor], 
                lq_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Fuse high-resolution details with low-resolution context
        
        Strategy: Top-down fusion, injecting details from higher levels
        """
        fused = []
        
        for i in range(len(hq_features)):
            # Get corresponding features from both scales
            hq_feat = hq_features[i]
            lq_idx = min(i, len(lq_features) - 1)
            lq_feat = lq_features[lq_idx]
            
            # Upsample low-res features to match high-res resolution if needed
            if lq_feat.shape[-2:] != hq_feat.shape[-2:]:
                lq_feat = F.interpolate(
                    lq_feat, 
                    size=hq_feat.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Cross-scale attention fusion
            fused_feat = self.cross_attention[i](hq_feat, lq_feat)
            
            # If we have finer details from previous level, inject them
            if i > 0 and i-1 < len(fused):
                # Upsample previous fused feature and add
                prev_up = self.upsamplers[i-1](fused[i-1])
                if prev_up.shape[-2:] == fused_feat.shape[-2:]:
                    fused_feat = fused_feat + prev_up
            
            fused.append(fused_feat)
        
        return fused


class CrossScaleAttention(nn.Module):
    """
    Cross-scale attention: high-res details as queries, low-res context as keys/values
    """
    
    def __init__(self, hq_dim: int, lq_dim: int, output_dim: int):
        super().__init__()
        
        self.query_proj = nn.Conv2d(hq_dim, output_dim, 1)
        self.key_proj = nn.Conv2d(lq_dim, output_dim, 1) 
        self.value_proj = nn.Conv2d(lq_dim, output_dim, 1)
        self.out_proj = nn.Conv2d(output_dim, output_dim, 1)
        
        self.scale = output_dim ** -0.5
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self, hq: torch.Tensor, lq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hq: [B, C_hq, H, W] - High-resolution features (details)
            lq: [B, C_lq, H, W] - Low-resolution features (context)
            
        Returns:
            [B, output_dim, H, W] - Fused features
        """
        B, C_hq, H, W = hq.shape
        
        # Project to attention space
        Q = self.query_proj(hq)  # [B, output_dim, H, W]
        K = self.key_proj(lq)    # [B, output_dim, H, W] 
        V = self.value_proj(lq)  # [B, output_dim, H, W]
        
        # Reshape for attention: [B, H*W, output_dim]
        Q_flat = Q.view(B, -1, H * W).permute(0, 2, 1)  # [B, H*W, output_dim]
        K_flat = K.view(B, -1, H * W)                     # [B, output_dim, H*W]
        V_flat = V.view(B, -1, H * W).permute(0, 2, 1)   # [B, H*W, output_dim]
        
        # Attention: Q @ K^T, then attend to V
        attn_scores = torch.bmm(Q_flat, K_flat) * self.scale  # [B, H*W, H*W]
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        attended = torch.bmm(attn_weights, V_flat)  # [B, H*W, output_dim]
        attended = attended.permute(0, 2, 1).view(B, -1, H, W)  # [B, output_dim, H, W]
        
        # Output projection and residual connection
        output = self.out_proj(attended)
        
        # Add residual from high-res features
        hq_aligned = F.interpolate(hq, size=(H, W), mode='bilinear', align_corners=False)
        hq_proj = self.query_proj(hq_aligned)
        
        return output + hq_proj


class FeaturePyramidAligner(nn.Module):
    """
    Align feature pyramid to consistent dimensions
    """
    
    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__()
        
        self.aligners = nn.ModuleList([
            nn.Conv2d(dim, output_dim, 1) for dim in input_dims
        ])
    
    def forward(self, pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
        return [aligner(feat) for feat, aligner in zip(pyramid, self.aligners)]


def test_sam2_encoder():
    """Test the SAM encoder"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = SAM2DualScaleEncoder().to(device)
    
    # Test with sample image
    batch_size = 2
    height, width = 512, 512
    image = torch.randn(batch_size, 3, height, width).to(device)
    
    print(f"Input image shape: {image.shape}")
    
    result = encoder(image)
    
    print(f"Output pyramid levels: {len(result['pyramid'])}")
    for i, feat in enumerate(result['pyramid']):
        print(f"  Level {i}: {feat.shape}")
    
    print(f"HQ features: {[f.shape for f in result['hq_features']]}")
    print(f"LQ features: {[f.shape for f in result['lq_features']]}")
    print(f"Global features: {result['global_features'].shape}")
    
    return encoder, result


if __name__ == "__main__":
    model, result = test_sam2_encoder()
    print("SAM Dual-Scale Encoder test completed successfully!")