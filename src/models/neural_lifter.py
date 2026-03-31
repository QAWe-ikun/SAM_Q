"""
Neural Lifting Module: 2D Features to 3D Depth Representation
==============================================================

Converts 2D perception features to hierarchical 3D depth representations
using pixel-aligned neural networks. Maintains efficiency while enabling
3D-aware operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class PixelAlignedNeuralLifter(nn.Module):
    """
    Pixel-Aligned Neural Lifter: Converts 2D features to hierarchical 3D depth
    
    For each 2D pixel, predicts depth intervals [near, far] at multiple resolutions
    to enable efficient 3D operations while maintaining 2D computational efficiency.
    """
    
    def __init__(self,
                 feature_dim: int = 512,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 num_depth_layers: int = 5,  # Corresponds to H-MVP levels
                 num_views: int = 6,        # Six orthographic projections
                 use_semantic_guidance: bool = True):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_depth_layers = num_depth_layers
        self.num_views = num_views
        self.use_semantic_guidance = use_semantic_guidance
        
        # Feature processing backbone
        self.feature_processor = self._build_feature_processor()
        
        # Hierarchical depth predictors (one per resolution level)
        self.depth_predictors = nn.ModuleList([
            self._build_depth_predictor(level) 
            for level in range(num_depth_layers)
        ])
        
        # View direction embeddings
        self.view_embeddings = nn.Parameter(
            torch.randn(num_views, feature_dim) * 0.02
        )
        
        # Resolution embeddings
        self.resolution_embeddings = nn.Parameter(
            torch.randn(num_depth_layers, feature_dim) * 0.02
        )
        
        # Semantic guidance fusion (if using Qwen3-VL features)
        if use_semantic_guidance:
            self.semantic_fusion = SemanticGuidanceFusion(
                feature_dim=feature_dim,
                semantic_dim=3584  # Qwen3-VL hidden size
            )
    
    def _build_feature_processor(self) -> nn.Module:
        """Build feature processing backbone"""
        layers = []
        in_dim = self.feature_dim
        
        for i in range(self.num_layers):
            layers.extend([
                nn.Conv2d(in_dim, self.hidden_dim, 3, padding=1),
                nn.GroupNorm(8, self.hidden_dim),
                nn.ReLU(),
                nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
                nn.GroupNorm(8, self.hidden_dim),
                nn.ReLU()
            ])
            in_dim = self.hidden_dim
            
            # Add skip connection every 2 layers
            if i > 0 and (i + 1) % 2 == 0:
                layers.append(nn.Conv2d(self.hidden_dim, self.hidden_dim, 1))
        
        return nn.Sequential(*layers)
    
    def _build_depth_predictor(self, level: int) -> nn.Module:
        """Build depth predictor for specific resolution level"""
        # Higher levels (finer resolution) need more capacity
        channels = self.hidden_dim + (level * 32)
        
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.ReLU(),
            
            # Predict depth intervals for all views
            nn.Conv2d(channels, self.num_views * 2, 1),  # 2 for near/far per view
            nn.Sigmoid()  # Normalize to [0,1] range
        )
    
    def forward(self,
                features: List[torch.Tensor],    # Multi-scale features from SAM²
                semantic_features: Optional[torch.Tensor] = None,  # From Qwen3-VL
                grounding_map: Optional[torch.Tensor] = None,      # Semantic grounding
                mode: str = 'object') -> Dict[int, torch.Tensor]:
        """
        Lift 2D features to hierarchical 3D depth representation
        
        Args:
            features: Multi-scale features from SAM² encoder [P2, P3, P4, P5]
            semantic_features: Optional semantic features from Qwen3-VL
            grounding_map: Optional semantic grounding heatmaps
            mode: 'object' or 'scene' (may affect prediction strategy)
            
        Returns:
            Dict mapping level -> [B, num_views, 2, H, W] depth intervals
        """
        # Use finest resolution features for depth prediction
        finest_features = features[0]  # [B, feature_dim, H, W]
        B, C, H, W = finest_features.shape
        
        # Process features
        processed_features = self.feature_processor(finest_features)
        
        # Apply semantic guidance if provided
        if self.use_semantic_guidance and semantic_features is not None:
            processed_features = self.semantic_fusion(
                processed_features, semantic_features, grounding_map
            )
        
        # Predict depth intervals at multiple levels
        depth_pyramid = {}
        
        for level in range(self.num_depth_layers):
            # Get target resolution for this level
            target_res = min(H, W) // (2 ** (self.num_depth_layers - 1 - level))
            
            # Adjust features to target resolution
            if processed_features.shape[-2:] != (target_res, target_res):
                level_features = F.interpolate(
                    processed_features,
                    size=(target_res, target_res),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                level_features = processed_features
            
            # Add resolution-specific embedding
            res_embed = self.resolution_embeddings[level].view(1, -1, 1, 1)
            level_features = level_features + res_embed.expand_as(level_features)
            
            # Predict depth intervals
            depth_pred = self.depth_predictors[level](level_features)  # [B, num_views*2, H, W]
            
            # Reshape to [B, num_views, 2, H, W]
            depth_pred = depth_pred.view(B, self.num_views, 2, target_res, target_res)
            
            # Separate near and far, ensure ordering
            near_pred = depth_pred[:, :, 0]  # [B, num_views, H, W]
            far_pred = depth_pred[:, :, 1]   # [B, num_views, H, W]
            
            # Ensure near < far (conservative approach)
            near_final = torch.min(near_pred, far_pred)
            far_final = torch.max(near_pred, far_pred)
            
            # Add small margin to avoid degenerate intervals
            far_final = far_final + 0.01
            
            depth_pyramid[level] = torch.stack([near_final, far_final], dim=2)
        
        return depth_pyramid


class SemanticGuidanceFusion(nn.Module):
    """
    Fuse semantic features with spatial features for guided depth prediction
    """
    
    def __init__(self, feature_dim: int, semantic_dim: int):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.semantic_dim = semantic_dim
        
        # Project semantic features to spatial attention
        self.semantic_to_attention = nn.Sequential(
            nn.Linear(semantic_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()  # Attention weights
        )
        
        # Project grounding map to spatial features
        self.grounding_projection = nn.Conv2d(1, feature_dim, 1)
        
        # Cross-attention fusion
        self.cross_attention = CrossModalAttention(
            spatial_dim=feature_dim,
            semantic_dim=semantic_dim
        )
        
    def forward(self,
                spatial_features: torch.Tensor,    # [B, feature_dim, H, W]
                semantic_features: torch.Tensor,   # [B, semantic_dim]
                grounding_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Fuse semantic guidance with spatial features
        """
        B, C, H, W = spatial_features.shape
        
        # Generate spatial attention from semantic features
        semantic_attention = self.semantic_to_attention(semantic_features)  # [B, feature_dim]
        semantic_attention = semantic_attention.view(B, C, 1, 1).expand(-1, -1, H, W)
        
        # Apply grounding map if provided
        grounding_features = torch.zeros_like(spatial_features)
        if grounding_map is not None:
            # Ensure grounding map matches spatial feature resolution
            if grounding_map.shape[-2:] != (H, W):
                grounding_map = F.interpolate(
                    grounding_map,
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )
            grounding_features = self.grounding_projection(grounding_map.unsqueeze(1))
        
        # Cross-modal attention fusion
        attended_features = self.cross_attention(
            spatial_features, semantic_features
        )
        
        # Combine all modalities
        fused_features = (
            spatial_features * semantic_attention +
            attended_features +
            grounding_features
        )
        
        return fused_features


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between spatial and semantic features
    """
    
    def __init__(self, spatial_dim: int, semantic_dim: int):
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.semantic_dim = semantic_dim
        
        # Query from spatial features
        self.spatial_query = nn.Conv2d(spatial_dim, spatial_dim, 1)
        
        # Key and Value from semantic features
        self.semantic_kv = nn.Linear(semantic_dim, spatial_dim * 2)
        
        # Output projection
        self.out_proj = nn.Conv2d(spatial_dim, spatial_dim, 1)
        
    def forward(self, 
                spatial: torch.Tensor,     # [B, spatial_dim, H, W]
                semantic: torch.Tensor):   # [B, semantic_dim]
        """
        Apply cross-modal attention
        """
        B, C_spatial, H, W = spatial.shape
        B_semantic, C_semantic = semantic.shape
        
        # Generate query from spatial features
        Q = self.spatial_query(spatial)  # [B, C_spatial, H, W]
        Q_flat = Q.view(B, C_spatial, H * W).permute(0, 2, 1)  # [B, H*W, C_spatial]
        
        # Generate key and value from semantic features
        kv = self.semantic_kv(semantic)  # [B, C_spatial * 2]
        K, V = kv.chunk(2, dim=-1)      # [B, C_spatial]
        
        # Expand K and V to spatial dimensions
        K_expanded = K.unsqueeze(1).expand(-1, H * W, -1)  # [B, H*W, C_spatial]
        V_expanded = V.unsqueeze(1).expand(-1, H * W, -1)  # [B, H*W, C_spatial]
        
        # Compute attention weights
        attn_weights = torch.softmax(
            torch.sum(Q_flat * K_expanded, dim=-1, keepdim=True),  # [B, H*W, 1]
            dim=1
        )
        
        # Apply attention to values
        attended = attn_weights * V_expanded  # [B, H*W, C_spatial]
        attended = attended.permute(0, 2, 1).view(B, C_spatial, H, W)  # [B, C_spatial, H, W]
        
        return self.out_proj(attended)


class DepthConsistencyEnforcer(nn.Module):
    """
    Enforce consistency across different resolution levels
    """
    
    def __init__(self, num_levels: int):
        super().__init__()
        self.num_levels = num_levels
        
        # Learnable consistency weights
        self.consistency_weights = nn.Parameter(torch.ones(num_levels))
        
    def forward(self, depth_pyramid: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Enforce consistency across resolution levels
        """
        consistent_pyramid = {}
        
        for level in range(self.num_levels):
            current_depth = depth_pyramid[level]
            
            # If not the finest level, enforce consistency with finer levels
            if level < self.num_levels - 1:
                # Get next finer level
                finer_depth = depth_pyramid[level + 1]
                
                # Upsample finer level to current resolution
                upsampled_finer = F.interpolate(
                    finer_depth,
                    size=current_depth.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                
                # Blend with consistency weight
                consistency_weight = torch.sigmoid(self.consistency_weights[level])
                consistent_depth = (
                    consistency_weight * current_depth + 
                    (1 - consistency_weight) * upsampled_finer
                )
                
                consistent_pyramid[level] = consistent_depth
            else:
                consistent_pyramid[level] = current_depth
        
        return consistent_pyramid


class MultiViewDepthRefiner(nn.Module):
    """
    Refine depth predictions across multiple orthographic views
    """
    
    def __init__(self, num_views: int = 6):
        super().__init__()
        self.num_views = num_views
        
        # Cross-view consistency network
        self.consistency_network = nn.Sequential(
            nn.Conv3d(num_views * 2, 64, 1),  # near/far for all views
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv3d(64, 32, 1),
            nn.GroupNorm(8, 32),
            nn.ReLU(),
            nn.Conv3d(32, num_views * 2, 1)
        )
        
    def forward(self, depth_maps: torch.Tensor) -> torch.Tensor:
        """
        depth_maps: [B, num_views, 2, H, W] - near/far for each view
        """
        B, num_views, _, H, W = depth_maps.shape
        
        # Apply cross-view consistency refinement
        refined = self.consistency_network(depth_maps)
        
        # Add residual connection
        refined = refined + depth_maps
        
        # Ensure depth ordering
        near = refined[:, :, 0]
        far = refined[:, :, 1]
        near_final = torch.min(near, far)
        far_final = torch.max(near, far) + 0.01
        
        return torch.stack([near_final, far_final], dim=2)


def test_neural_lifter():
    """Test the neural lifter module"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    lifter = PixelAlignedNeuralLifter().to(device)
    
    # Create sample multi-scale features (simulating SAM² output)
    batch_size = 2
    feature_pyramid = []
    
    # Simulate features at different scales: P2, P3, P4, P5
    for scale_factor in [4, 8, 16, 32]:  # Typical SAM3 strides
        H, W = 64 // scale_factor, 64 // scale_factor
        features = torch.randn(batch_size, 512, H, W).to(device)
        feature_pyramid.append(features)
    
    # Create sample semantic features
    semantic_features = torch.randn(batch_size, 3584).to(device)
    
    # Create sample grounding map
    grounding_map = torch.rand(batch_size, 64, 64).to(device)
    
    print(f"Input feature pyramid: {[f.shape for f in feature_pyramid]}")
    print(f"Semantic features: {semantic_features.shape}")
    print(f"Grounding map: {grounding_map.shape}")
    
    # Test neural lifting
    depth_pyramid = lifter(
        features=feature_pyramid,
        semantic_features=semantic_features,
        grounding_map=grounding_map,
        mode='object'
    )
    
    print(f"Output depth pyramid levels: {len(depth_pyramid)}")
    for level, depth_map in depth_pyramid.items():
        print(f"  Level {level}: {depth_map.shape}")
        print(f"    Near range: [{depth_map[:, :, 0].min():.3f}, {depth_map[:, :, 0].max():.3f}]")
        print(f"    Far range: [{depth_map[:, :, 1].min():.3f}, {depth_map[:, :, 1].max():.3f}]")
    
    return lifter, depth_pyramid


if __name__ == "__main__":
    lifter, pyramid = test_neural_lifter()
    print("Neural Lifter test completed successfully!")