"""
Asset Database and Related Components for Known Geometry Mode
==============================================================

Components for managing pre-computed 3D assets and retrieving HMVP representations
when geometry is known a priori.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pickle
import json
import hashlib


class HeatmapGuidedPlacer:
    """
    Heatmap-guided object placer using Qwen3-VL grounding maps
    Replaces asset database with direct heatmap-to-placement approach
    """
    
    def __init__(self):
        # No asset database needed - using direct heatmap approach
        # This simplifies the system and removes the neural fallback requirement
        pass
        
    def _load_asset_index(self) -> Dict[str, Dict[str, Any]]:
        """Load asset index mapping object IDs to HMVP files"""
        index_file = self.asset_dir / "asset_index.json"
        
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Create default index
            return {
                "objects": {},
                "scenes": {}
            }
    
    def store_object_hmvp(self, 
                         object_id: str, 
                         hmvp_representation: torch.Tensor,
                         metadata: Dict[str, Any] = None):
        """Store pre-computed object HMVP representation"""
        # Save HMVP representation
        asset_path = self.asset_dir / "objects" / f"{object_id}_hmvp.pt"
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'hmvp': hmvp_representation,
            'metadata': metadata or {}
        }, asset_path)
        
        # Update index
        self.asset_index['objects'][object_id] = {
            'path': str(asset_path.relative_to(self.asset_dir)),
            'hash': self._compute_tensor_hash(hmvp_representation),
            'metadata': metadata or {}
        }
        
        # Save index
        with open(self.asset_dir / "asset_index.json", 'w', encoding='utf-8') as f:
            json.dump(self.asset_index, f, ensure_ascii=False, indent=2)
    
    def retrieve_object_hmvp(self, 
                           object_id: str,
                           text_instruction: Optional[str] = None) -> torch.Tensor:
        """Retrieve pre-computed object HMVP representation"""
        if object_id not in self.asset_index['objects']:
            # Fallback: return zeros or trigger neural lifting
            print(f"Warning: Object {object_id} not found in asset database")
            # Return a default tensor that will trigger neural lifting
            return None
        
        asset_path = self.asset_dir / self.asset_index['objects'][object_id]['path']
        data = torch.load(asset_path)
        
        hmvp = data['hmvp']
        
        # Apply text-guided adjustments if instruction provided
        if text_instruction:
            # This would apply learned transformations based on text
            # For now, return as-is
            pass
        
        return hmvp
    
    def store_scene_hmvp(self, 
                        scene_id: str, 
                        hmvp_representation: torch.Tensor,
                        metadata: Dict[str, Any] = None):
        """Store pre-computed scene HMVP representation"""
        # Save HMVP representation
        asset_path = self.asset_dir / "scenes" / f"{scene_id}_hmvp.pt"
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'hmvp': hmvp_representation,
            'metadata': metadata or {}
        }, asset_path)
        
        # Update index
        self.asset_index['scenes'][scene_id] = {
            'path': str(asset_path.relative_to(self.asset_dir)),
            'hash': self._compute_tensor_hash(hmvp_representation),
            'metadata': metadata or {}
        }
        
        # Save index
        with open(self.asset_dir / "asset_index.json", 'w', encoding='utf-8') as f:
            json.dump(self.asset_index, f, ensure_ascii=False, indent=2)
    
    def retrieve_scene_hmvp(self, scene_id: str) -> torch.Tensor:
        """Retrieve pre-computed scene HMVP representation"""
        if scene_id not in self.asset_index['scenes']:
            # Fallback: return zeros or trigger neural lifting
            print(f"Warning: Scene {scene_id} not found in asset database")
            return None
        
        asset_path = self.asset_dir / self.asset_index['scenes'][scene_id]['path']
        data = torch.load(asset_path)
        
        return data['hmvp']
    
    def _compute_tensor_hash(self, tensor: torch.Tensor) -> str:
        """Compute hash for tensor to detect changes"""
        # Convert tensor to bytes and hash
        tensor_bytes = tensor.detach().cpu().numpy().tobytes()
        return hashlib.md5(tensor_bytes).hexdigest()
    
    def list_available_objects(self) -> List[str]:
        """List all available object assets"""
        return list(self.asset_index['objects'].keys())
    
    def list_available_scenes(self) -> List[str]:
        """List all available scene assets"""
        return list(self.asset_index['scenes'].keys())


class ObjectIdentifier(nn.Module):
    """
    Identify object from image features and text to retrieve from asset database
    """
    
    def __init__(self, image_feature_dim: int, text_feature_dim: int):
        super().__init__()
        
        # Combine image and text features to identify object
        self.id_classifier = nn.Sequential(
            nn.Linear(image_feature_dim + text_feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),  # Reduced dimension for object ID space
        )
        
        # Object ID embeddings (learned mappings to asset IDs)
        self.object_id_embeddings = nn.Embedding(10000, 256)  # 10k possible objects
        
    def forward(self, 
                image_features: torch.Tensor,  # [B, img_dim]
                text_features: torch.Tensor,   # [B, text_dim]
                text_query: str) -> List[str]:
        """
        Identify object IDs from features
        """
        B = image_features.size(0)
        
        # Combine features
        combined_features = torch.cat([image_features, text_features], dim=1)
        
        # Get object ID embeddings
        id_projections = self.id_classifier(combined_features)  # [B, 256]
        
        # Compute similarities with known object embeddings
        all_embeddings = self.object_id_embeddings.weight  # [10000, 256]
        
        # Cosine similarity
        similarities = F.cosine_similarity(
            id_projections.unsqueeze(1),  # [B, 1, 256]
            all_embeddings.unsqueeze(0),  # [1, 10000, 256]
            dim=2  # [B, 10000]
        )
        
        # Get top-k most similar object IDs
        top_k = min(5, all_embeddings.size(0))
        _, top_indices = torch.topk(similarities, k=top_k, dim=1)
        
        # Convert indices to object IDs (in a real system, these would map to asset names)
        object_ids = []
        for i in range(B):
            # For now, return a string representation of the top ID
            # In practice, this would map to actual asset names
            top_id = f"obj_{top_indices[i, 0].item()}"
            object_ids.append(top_id)
        
        return object_ids


class SceneIdentifier(nn.Module):
    """
    Identify scene from image features to retrieve from asset database
    """
    
    def __init__(self, image_feature_dim: int):
        super().__init__()
        
        # Scene classification from features
        self.scene_classifier = nn.Sequential(
            nn.Linear(image_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),  # Reduced dimension for scene ID space
        )
        
        # Scene ID embeddings
        self.scene_id_embeddings = nn.Embedding(1000, 128)  # 1k possible scenes
        
    def forward(self, image_features: torch.Tensor) -> List[str]:
        """
        Identify scene IDs from features
        """
        B = image_features.size(0)
        
        # Get scene ID embeddings
        id_projections = self.scene_classifier(image_features)  # [B, 128]
        
        # Compute similarities with known scene embeddings
        all_embeddings = self.scene_id_embeddings.weight  # [1000, 128]
        
        # Cosine similarity
        similarities = F.cosine_similarity(
            id_projections.unsqueeze(1),  # [B, 1, 128]
            all_embeddings.unsqueeze(0),  # [1, 1000, 128]
            dim=2  # [B, 1000]
        )
        
        # Get top scene ID
        _, top_indices = torch.max(similarities, dim=1)
        
        # Convert to scene IDs
        scene_ids = [f"scene_{idx.item()}" for idx in top_indices]
        
        return scene_ids


class TextGuidedHMVPTransformer(nn.Module):
    """
    Apply text-guided transformations to known HMVP representations
    """
    
    def __init__(self, feature_dim: int, text_dim: int):
        super().__init__()
        
        # Cross-attention between text and HMVP features
        self.text_hmvp_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Transformation network
        self.transformer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 2),  # near/far adjustment
            nn.Tanh()  # Normalize to [-1, 1]
        )
        
    def forward(self, 
                hmvp_representation: torch.Tensor,  # [B, 6, 2, H, W]
                text_queries: List[str],
                grounding_maps: torch.Tensor) -> torch.Tensor:
        """
        Apply text-guided transformations to HMVP representation
        """
        B, num_views, _, H, W = hmvp_representation.shape
        
        # Process each sample independently
        adjusted_representations = []
        
        for i in range(B):
            # Flatten HMVP representation for attention
            # [6*H*W, feature_dim] - treating each depth value as a "token"
            flat_hmvp = hmvp_representation[i].permute(1, 2, 3, 0).reshape(-1, num_views)  # [2*H*W, 6]
            
            # In a real implementation, we would use text features here
            # For now, we'll use a simplified approach
            text_adjustment = torch.randn(2, H, W, device=hmvp_representation.device) * 0.1  # Small random adjustment
            
            # Apply adjustment to near and far values
            adjusted_hmvp = hmvp_representation[i].clone()
            adjusted_hmvp[:, 0] = adjusted_hmvp[:, 0] + text_adjustment[0]  # Adjust near
            adjusted_hmvp[:, 1] = adjusted_hmvp[:, 1] + text_adjustment[1]  # Adjust far
            
            # Ensure near < far
            near = torch.min(adjusted_hmvp[:, 0], adjusted_hmvp[:, 1])
            far = torch.max(adjusted_hmvp[:, 0], adjusted_hmvp[:, 1])
            adjusted_hmvp = torch.stack([near, far], dim=1)
            
            adjusted_representations.append(adjusted_hmvp)
        
        return torch.stack(adjusted_representations, dim=0)


def create_default_assets():
    """
    Create default assets for common objects/scenes if database is empty
    """
    db = AssetDatabase()
    
    # Create default object assets (these would be pre-computed from 3D models)
    default_objects = [
        "chair", "table", "sofa", "lamp", "plant", "tv", "bed", "desk"
    ]
    
    for obj_name in default_objects:
        # Create a dummy HMVP representation for the object
        # In reality, this would come from a 3D model
        dummy_hmvp = torch.randn(1, 6, 2, 64, 64)  # [B, views, near/far, H, W]
        dummy_hmvp[:, :, 1] = dummy_hmvp[:, :, 0] + 0.1  # Ensure far > near
        
        db.store_object_hmvp(
            object_id=obj_name,
            hmvp_representation=dummy_hmvp,
            metadata={
                "type": "furniture",
                "category": "common",
                "dimensions": [1.0, 1.0, 1.0]  # x, y, z
            }
        )
    
    # Create default scene assets
    default_scenes = [
        "living_room", "bedroom", "office", "kitchen", "dining_room"
    ]
    
    for scene_name in default_scenes:
        # Create a dummy HMVP representation for the scene
        dummy_hmvp = torch.randn(1, 6, 2, 128, 128)  # Higher res for scenes
        dummy_hmvp[:, :, 1] = dummy_hmvp[:, :, 0] + 0.5  # Larger depth range for scenes
        
        db.store_scene_hmvp(
            scene_id=scene_name,
            hmvp_representation=dummy_hmvp,
            metadata={
                "type": "indoor",
                "category": "common",
                "dimensions": [5.0, 4.0, 3.0]  # x, y, z in meters
            }
        )
    
    print(f"Created default assets for {len(default_objects)} objects and {len(default_scenes)} scenes")
    return db