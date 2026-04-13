"""
H-MVP: Hierarchical Multi-View Projection Collision Detection
=============================================================

Efficient 3D collision detection using hierarchical 2D depth maps,
similar to Mipmap but for collision detection. Provides early-out optimization
and adaptive subdivision for complex geometries.
Standard implementation aligned with common HMVP practices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class HierarchicalDepthMap(nn.Module):
    """
    Hierarchical Depth Map: Mipmap-style multi-resolution depth representation
    
    Standard HMVP implementation storing depth intervals [near, far] at multiple resolutions 
    for efficient collision checking across multiple orthographic projections.
    """
    
    def __init__(self, 
                 max_level: int = 4, 
                 base_resolution: int = 8,
                 num_views: int = 6):
        super().__init__()
        
        self.max_level = max_level
        self.base_resolution = base_resolution
        self.num_views = num_views  # Standard 6 orthographic views: +X, -X, +Y, -Y, +Z, -Z
        
        # Calculate resolutions for each level following standard HMVP practices
        self.level_resolutions = [
            base_resolution * (2 ** i) for i in range(max_level + 1)
        ]
        
        # Standard Gaussian kernels for downsampling
        self.downsample_kernels = self._create_gaussian_kernels()
        
    def _create_gaussian_kernels(self) -> List[torch.Tensor]:
        """Create Gaussian kernels for multi-level downsampling"""
        kernels = []
        for level in range(self.max_level):
            kernel_size = 5
            sigma = 1.0 * (level + 1)  # Increase blur with level
            x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
            gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
            kernel_1d = gauss / gauss.sum()
            kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
            kernels.append(kernel_2d)
        return kernels
    
    def build_pyramid(self, fine_depth: torch.Tensor) -> List[torch.Tensor]:
        """
        Build hierarchical depth pyramid from fine resolution
        
        Args:
            fine_depth: [B, num_views, 2, H, W] - depth intervals (near, far) for each view
            
        Returns:
            List of [B, num_views, 2, res, res] at different resolutions
        """
        B, num_views, _, H, W = fine_depth.shape
        device = fine_depth.device
        
        pyramid = [fine_depth]  # Start with finest level
        current = fine_depth
        
        # Build coarser levels
        for level in range(self.max_level - 1, -1, -1):
            level_depths = []
            
            for view_idx in range(num_views):
                # Get depth for this view
                view_depth = current[:, view_idx]  # [B, 2, H, W]
                
                # Apply Gaussian blur
                blurred = self._gaussian_blur(view_depth, level)
                
                # Downsample by 2x
                downsampled = F.avg_pool2d(blurred, kernel_size=2, stride=2)
                
                level_depths.append(downsampled)
            
            # Stack all views
            level_result = torch.stack(level_depths, dim=1)  # [B, num_views, 2, H//2, W//2]
            pyramid.insert(0, level_result)  # Insert at beginning (coarsest first)
            current = level_result
        
        return pyramid  # [level_0, level_1, ..., level_max] (coarse to fine)
    
    def _gaussian_blur(self, x: torch.Tensor, level: int) -> torch.Tensor:
        """Apply Gaussian blur using precomputed kernel"""
        kernel = self.downsample_kernels[level].to(x.device)
        kernel = kernel.view(1, 1, 5, 5).expand(x.size(1), -1, -1, -1)
        
        # Pad to maintain size
        padding = 2
        return F.conv2d(x, kernel, padding=padding, groups=x.size(1))
    
    def conservative_depth_merge(self, depth_pair: torch.Tensor) -> torch.Tensor:
        """
        Conservatively merge depth intervals to avoid false negatives
        
        Args:
            depth_pair: [B, 2, H, W] - [near, far] intervals
            
        Returns:
            [B, 2, H, W] - Conservative merged intervals
        """
        near, far = depth_pair[:, 0:1], depth_pair[:, 1:2]
        
        # Conservative: extend intervals to ensure no collisions are missed
        # near: take minimum (closest to camera)
        # far: take maximum (farthest from camera)
        expanded_near = F.max_pool2d(-near, kernel_size=3, stride=1, padding=1)
        expanded_near = -expanded_near  # Negate back
        
        expanded_far = F.max_pool2d(far, kernel_size=3, stride=1, padding=1)
        
        return torch.cat([expanded_near, expanded_far], dim=1)


class HMVPCollisionDetector(nn.Module):
    """
    Hierarchical Multi-View Projection Collision Detector
    
    Standard implementation performing collision detection from coarse to fine levels 
    with early-out optimization. Follows common HMVP practices for efficiency.
    Aligned with LLM-3D distance map approach for collision detection.
    """
    
    def __init__(self, 
                 max_level: int = 4,
                 base_resolution: int = 8,
                 early_out_threshold: float = 0.1,
                 adaptive_subdivision: bool = True,
                 view_directions: List[str] | None = None):
        super().__init__()
        
        self.max_level = max_level
        self.base_resolution = base_resolution
        self.early_out_threshold = early_out_threshold
        self.adaptive_subdivision = adaptive_subdivision
        self.view_directions = view_directions or ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
        
        # Hierarchical depth map builder (standard HMVP component)
        self.hdm_builder = HierarchicalDepthMap(max_level, base_resolution)
        
        # Standard level-specific collision detectors (lightweight CNNs)
        self.level_detectors = nn.ModuleList([
            self._build_level_detector(level) 
            for level in range(max_level + 1)
        ])
        
        # Standard adaptive subdivision controller
        if adaptive_subdivision:
            self.subdivision_controller = AdaptiveSubdivisionController()
        
        # Distance map based collision detection (aligned with LLM-3D approach)
        self.distance_map_collision = DistanceMapBasedCollision()
    
    def forward(self,
                obj_depths: torch.Tensor,      # [B, 6, 2, 128, 128]
                scene_depths: torch.Tensor,    # [B, 6, 2, 128, 128]
                early_out: bool = True) -> Dict:
        """
        Hierarchical collision detection from coarse to fine
        
        Args:
            obj_depths: Object depth intervals [B, num_views, 2, H, W]
            scene_depths: Scene depth intervals [B, num_views, 2, H, W]
            early_out: Whether to use early-out optimization
            
        Returns:
            Dict with collision probabilities and level details
        """
        B = obj_depths.size(0)
        device = obj_depths.device
        
        # Build hierarchical pyramids
        obj_pyramid = self.hdm_builder.build_pyramid(obj_depths)
        scene_pyramid = self.hdm_builder.build_pyramid(scene_depths)
        
        # Track active samples (those still being processed)
        active_mask = torch.ones(B, dtype=torch.bool, device=device)
        level_results = []
        total_cost = torch.zeros(B, device=device)
        
        # Process from coarse to fine
        for level in range(self.max_level + 1):
            if not active_mask.any():
                break  # All samples have been processed
            
            # Get current level depths for active samples
            obj_current = obj_pyramid[level][active_mask]    # [B_active, 6, 2, H, W]
            scene_current = scene_pyramid[level][active_mask]
            
            # Perform collision detection at this level
            level_collision, level_subdivide = self._detect_level_collision(
                obj_current, scene_current, level
            )
            
            # Expand results back to full batch
            full_collision = torch.zeros(B, device=device)
            full_subdivide = torch.zeros(B, device=device)
            full_collision[active_mask] = level_collision
            full_subdivide[active_mask] = level_subdivide
            
            # Store level result
            level_results.append({
                'collision_prob': full_collision,
                'subdivide_flag': full_subdivide,
                'resolution': self.hdm_builder.level_resolutions[level],
                'active_ratio': active_mask.float().mean().item()
            })
            
            # Accumulate computational cost
            cost_this_level = active_mask.float() * (level + 1)  # Higher levels cost more
            total_cost += cost_this_level
            
            # Early-out optimization
            if early_out and level < self.max_level:
                # Determine which samples should continue to next level
                continue_mask = self._determine_next_level(
                    full_collision, full_subdivide, level
                )
                
                # Update active mask
                active_mask = active_mask & continue_mask
        
        # Final collision probability (from finest level or aggregated)
        final_prob = level_results[-1]['collision_prob'] if level_results else torch.zeros(B, device=device)
        
        return {
            'collision_prob': final_prob,
            'level_details': level_results,
            'computation_cost': total_cost,
            'early_out_ratio': 1.0 - (active_mask.float().mean().item() 
                                     if level < self.max_level else 0.0),
            'pyramid': obj_pyramid  # Return for debugging/visualization
        }
    
    def _build_level_detector(self, level: int) -> nn.Module:
        """Build collision detector for specific level"""
        # Higher levels need more capacity due to finer details
        base_channels = 16 * (level + 1)
        
        return nn.Sequential(
            # Input: [obj_near, obj_far, scene_near, scene_far] = 4 channels
            nn.Conv2d(4, base_channels, 3, padding=1),
            nn.GroupNorm(min(8, base_channels // 4), base_channels),
            nn.ReLU(),
            
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(min(8, base_channels // 4), base_channels),
            nn.ReLU(),
            
            # Output: collision probability + subdivision suggestion
            nn.Conv2d(base_channels, 2, 1)  # [collision_prob, subdivide_flag]
        )
    
    def _detect_level_collision(self,
                              obj_depths: torch.Tensor,
                              scene_depths: torch.Tensor,
                              level: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect collision at specific level
        
        Args:
            obj_depths: [B_active, 6, 2, H, W] - Object depth intervals
            scene_depths: [B_active, 6, 2, H, W] - Scene depth intervals
            level: Current level index
            
        Returns:
            (collision_prob, subdivide_flag) for each sample
        """
        B_active = obj_depths.size(0)
        num_views = obj_depths.size(1)
        
        # Process each view separately and aggregate
        view_collisions = []
        view_subdivides = []
        
        for view_idx in range(num_views):
            # Extract depth intervals for this view
            obj_near = obj_depths[:, view_idx, 0]  # [B_active, H, W]
            obj_far = obj_depths[:, view_idx, 1]
            scene_near = scene_depths[:, view_idx, 0]
            scene_far = scene_depths[:, view_idx, 1]
            
            # Check for depth interval overlap (collision condition)
            # Two intervals [a,b] and [c,d] overlap iff a <= d and c <= b
            overlap_mask = (obj_near <= scene_far) & (scene_near <= obj_far)
            
            # Pass to level-specific detector
            collision_input = torch.stack([
                obj_near, obj_far, scene_near, scene_far
            ], dim=1)  # [B_active, 4, H, W]
            
            detection_output = self.level_detectors[level](collision_input)
            
            collision_map = torch.sigmoid(detection_output[:, 0])  # [B_active, H, W]
            subdivide_map = torch.sigmoid(detection_output[:, 1])  # [B_active, H, W]
            
            # Aggregate spatially (mean for collision, max for subdivision)
            view_collision = collision_map.mean(dim=[1, 2])      # [B_active]
            view_subdivide = subdivide_map.max(dim=2)[0].max(dim=1)[0]  # [B_active]
            
            view_collisions.append(view_collision)
            view_subdivides.append(view_subdivide)
        
        # Aggregate across views (max for collision - any view collision = overall collision)
        level_collision = torch.stack(view_collisions, dim=1).max(dim=1)[0]  # [B_active]
        level_subdivide = torch.stack(view_subdivides, dim=1).max(dim=1)[0]  # [B_active]
        
        return level_collision, level_subdivide
    
    def _determine_next_level(self,
                            collision_prob: torch.Tensor,
                            subdivide_flag: torch.Tensor,
                            current_level: int) -> torch.Tensor:
        """
        Determine which samples should proceed to next level
        
        Args:
            collision_prob: [B] Collision probabilities
            subdivide_flag: [B] Subdivision suggestions
            current_level: Current level index
            
        Returns:
            [B] Boolean mask for samples continuing to next level
        """
        # Continue if: high collision probability OR subdivision suggested
        continue_mask = (collision_prob > 0.3) | (subdivide_flag > 0.5)
        
        # But stop if separation confidence is very high (early-out)
        separation_conf = 1.0 - collision_prob
        early_out_mask = separation_conf > (1.0 - self.early_out_threshold)
        
        # Final decision: continue if suggested AND not early-out
        return continue_mask & (~early_out_mask)


class AdaptiveSubdivisionController(nn.Module):
    """
    Adaptive subdivision controller: determine which regions need finer resolution
    """
    
    def __init__(self):
        super().__init__()
        
        # Spatial attention network to identify regions needing subdivision
        self.attention_net = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),  # Combined depth intervals
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, obj_depths: torch.Tensor, scene_depths: torch.Tensor) -> torch.Tensor:
        """
        Predict subdivision regions
        
        Args:
            obj_depths: [B, 2, H, W] - Object depth intervals
            scene_depths: [B, 2, H, W] - Scene depth intervals
            
        Returns:
            [B, H, W] - Subdivision probability for each spatial location
        """
        # Combine depth intervals
        combined = torch.cat([obj_depths, scene_depths], dim=1)  # [B, 4, H, W]
        
        # Predict subdivision attention
        attention = self.attention_net(combined)  # [B, 1, H, W]
        
        return attention.squeeze(1)  # [B, H, W]


class DifferentiableHMVPOperations:
    """
    Differentiable operations for H-MVP to ensure gradient flow
    """
    
    @staticmethod
    def soft_depth_interval_overlap(obj_near: torch.Tensor, 
                                  obj_far: torch.Tensor,
                                  scene_near: torch.Tensor, 
                                  scene_far: torch.Tensor,
                                  temperature: float = 1.0) -> torch.Tensor:
        """
        Differentiable depth interval overlap calculation
        
        Args:
            obj_near, obj_far: Object depth intervals
            scene_near, scene_far: Scene depth intervals
            temperature: Softness parameter
            
        Returns:
            Overlap probability (differentiable)
        """
        # Calculate overlap bounds
        left_bound = torch.max(obj_near, scene_near)
        right_bound = torch.min(obj_far, scene_far)
        
        # Calculate overlap length (can be negative if no overlap)
        overlap_length = torch.relu(right_bound - left_bound)
        
        # Soft version of "is overlapping"
        # As temperature -> 0, approaches hard overlap check
        # As temperature -> inf, approaches uniform probability
        normalized_overlap = overlap_length / (torch.abs(obj_far - obj_near) + 
                                             torch.abs(scene_far - scene_near) + 1e-6)
        
        return torch.sigmoid(normalized_overlap * temperature)
    
    @staticmethod
    def soft_max_pool2d(x: torch.Tensor, kernel_size: int, stride: int | None = None, 
                       padding: int = 0, temperature: float = 10.0) -> torch.Tensor:
        """
        Soft version of max pooling for differentiability
        """
        if stride is None:
            stride = kernel_size
            
        # Unfold the tensor
        x_unfold = F.unfold(x, kernel_size=kernel_size, stride=stride, padding=padding)
        x_unfold = x_unfold.view(x.size(0), x.size(1), -1, -1)  # [B, C, kernel_size^2, n_patches]
        
        # Softmax over the kernel dimension (approximates max)
        weights = F.softmax(x_unfold * temperature, dim=2)
        
        # Weighted average (approaches max as temperature increases)
        soft_max = (x_unfold * weights).sum(dim=2)
        
        # Reshape back to spatial format
        out_h = (x.size(2) + 2 * padding - kernel_size) // stride + 1
        out_w = (x.size(3) + 2 * padding - kernel_size) // stride + 1
        
        return soft_max.view(x.size(0), x.size(1), out_h, out_w)


def test_hmvp_detector():
    """Test the H-MVP collision detector"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    detector = HMVPCollisionDetector().to(device)
    
    # Create sample depth maps
    batch_size = 4
    num_views = 6
    resolution = 128
    
    # Random depth intervals [near, far] for object and scene
    obj_depths = torch.rand(batch_size, num_views, 2, resolution, resolution).to(device)
    obj_depths[:, :, 1] = obj_depths[:, :, 0] + torch.rand_like(obj_depths[:, :, 0]) * 0.5  # far > near
    
    scene_depths = torch.rand(batch_size, num_views, 2, resolution, resolution).to(device)
    scene_depths[:, :, 1] = scene_depths[:, :, 0] + torch.rand_like(scene_depths[:, :, 0]) * 0.5
    
    print(f"Object depths shape: {obj_depths.shape}")
    print(f"Scene depths shape: {scene_depths.shape}")
    
    # Test collision detection
    result = detector(obj_depths, scene_depths, early_out=True)
    
    print(f"Collision probabilities: {result['collision_prob']}")
    print(f"Computation cost: {result['computation_cost']}")
    print(f"Early-out ratio: {result['early_out_ratio']:.2%}")
    print(f"Number of levels processed: {len(result['level_details'])}")
    
    for i, level_detail in enumerate(result['level_details']):
        print(f"  Level {i} ({level_detail['resolution']}x{level_detail['resolution']}): "
              f"collision={level_detail['collision_prob'].mean():.3f}, "
              f"active={level_detail['active_ratio']:.1%}")
    
    return detector, result


if __name__ == "__main__":
    detector, result = test_hmvp_detector()
    print("H-MVP Collision Detector test completed successfully!")


class DistanceMapBasedCollision(nn.Module):
    """
    Distance map based collision detection aligned with LLM-3D approach.
    Uses distance maps to detect collisions similar to how LLM-3D does it.
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, obj_distances: torch.Tensor, scene_distances: torch.Tensor) -> torch.Tensor:
        """
        Detect collision using distance maps approach.
        
        Args:
            obj_distances: [B, 6, H, W] - Distance maps for object in 6 views
            scene_distances: [B, 6, H, W] - Distance maps for scene in 6 views
            
        Returns:
            collision_prob: [B] - Collision probability for each sample
        """
        B, num_views, H, W = obj_distances.shape
        
        # Calculate collision based on distance map overlap
        # Similar to LLM-3D approach: if distance difference is negative, there's collision
        distance_diff = scene_distances - obj_distances  # [B, 6, H, W]
        
        # Count negative differences (indicating collision)
        collision_mask = (distance_diff < 0).float()  # [B, 6, H, W]
        
        # Calculate collision probability as ratio of collision pixels
        collision_pixels = collision_mask.sum(dim=[1, 2, 3])  # [B]
        total_pixels = num_views * H * W
        collision_prob = collision_pixels / total_pixels  # [B]
        
        return collision_prob  
