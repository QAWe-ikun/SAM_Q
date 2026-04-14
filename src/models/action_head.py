"""
Action Head with 6D Rotation Prediction
=========================================

Predicts object placement pose (position + rotation) from text embeddings.
Uses the 6D rotation representation (Zhou et al. 2019) for stable regression,
converted to rotation matrix / quaternion at output.

Sits parallel to SAM3 detector — both consume the same text_embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


def rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Uses Gram-Schmidt orthogonalization (Zhou et al. 2019).

    Args:
        rot_6d: [B, 6] first two columns of rotation matrix (a1, a2)

    Returns:
        matrix: [B, 3, 3] proper rotation matrix (orthonormal, det=+1)
    """
    a1 = rot_6d[:, 0:3]
    a2 = rot_6d[:, 3:6]

    # Normalize first column
    b1 = F.normalize(a1, dim=-1)

    # Second column: subtract projection onto b1, then normalize
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)

    # Third column: cross product
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)  # [B, 3, 3]


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert 3x3 rotation matrix to quaternion (qx, qy, qz, qw).
    Uses Shepperd's method for numerical stability.

    Args:
        matrix: [B, 3, 3] rotation matrix

    Returns:
        quat: [B, 4] quaternion in (qx, qy, qz, qw) format
    """
    batch_size = matrix.shape[0]
    m = matrix

    # Trace and diagonal elements
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]

    quat = torch.zeros(batch_size, 4, device=matrix.device, dtype=matrix.dtype)

    # Case 1: trace > 0
    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-10)) * 2  # s = 4*qw
    mask = trace > 0
    quat[mask, 3] = 0.25 * s[mask]
    quat[mask, 0] = (m[mask, 2, 1] - m[mask, 1, 2]) / s[mask]
    quat[mask, 1] = (m[mask, 0, 2] - m[mask, 2, 0]) / s[mask]
    quat[mask, 2] = (m[mask, 1, 0] - m[mask, 0, 1]) / s[mask]

    # Case 2: m00 is largest diagonal
    mask2 = (~mask) & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    s2 = torch.sqrt(torch.clamp(1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], min=1e-10)) * 2
    quat[mask2, 0] = 0.25 * s2[mask2]
    quat[mask2, 1] = (m[mask2, 0, 1] + m[mask2, 1, 0]) / s2[mask2]
    quat[mask2, 2] = (m[mask2, 0, 2] + m[mask2, 2, 0]) / s2[mask2]
    quat[mask2, 3] = (m[mask2, 2, 1] - m[mask2, 1, 2]) / s2[mask2]

    # Case 3: m11 is largest diagonal
    mask3 = (~mask) & (~mask2) & (m[:, 1, 1] > m[:, 2, 2])
    s3 = torch.sqrt(torch.clamp(1.0 + m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2], min=1e-10)) * 2
    quat[mask3, 0] = (m[mask3, 0, 1] + m[mask3, 1, 0]) / s3[mask3]
    quat[mask3, 1] = 0.25 * s3[mask3]
    quat[mask3, 2] = (m[mask3, 1, 2] + m[mask3, 2, 1]) / s3[mask3]
    quat[mask3, 3] = (m[mask3, 0, 2] - m[mask3, 2, 0]) / s3[mask3]

    # Case 4: m22 is largest diagonal
    mask4 = (~mask) & (~mask2) & (~mask3)
    s4 = torch.sqrt(torch.clamp(1.0 + m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1], min=1e-10)) * 2
    quat[mask4, 0] = (m[mask4, 0, 2] + m[mask4, 2, 0]) / s4[mask4]
    quat[mask4, 1] = (m[mask4, 1, 2] + m[mask4, 2, 1]) / s4[mask4]
    quat[mask4, 2] = 0.25 * s4[mask4]
    quat[mask4, 3] = (m[mask4, 1, 0] - m[mask4, 0, 1]) / s4[mask4]

    # Normalize
    quat = F.normalize(quat, dim=-1)
    return quat


class ActionHead(nn.Module):
    """
    Predicts placement scale and rotation from text embeddings (parallel to SAM3 detector).
    Position comes from heatmap + HMVP; this head only predicts scale and orientation.

    Input:  text_embeddings [B, num_queries, sam3_dim]  (same as SAM3 detector)
    Output: scale [B, 1], rotation_6d [B, 6], rotation_matrix [B, 3, 3]
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 512,
    ):
        super().__init__()

        # Attention-weighted pooling over query tokens
        self.attn_pool = nn.Sequential(
            nn.Linear(input_dim, 1),
        )

        # Shared feature extraction
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Scale head: predict relative scale factor
        # exp output allows both shrink (<1) and enlarge (>1)
        self.scale_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # Rotation head: predict 6D rotation (first two columns of rotation matrix)
        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 6),
        )

    def forward(self, text_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            text_embeddings: [B, num_queries, input_dim] from adapter/projector

        Returns:
            dict with scale, rotation_6d, rotation_matrix
        """
        # Attention-weighted pooling: [B, num_queries, dim] -> [B, dim]
        attn_weights = self.attn_pool(text_embeddings)  # [B, N, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = (text_embeddings * attn_weights).sum(dim=1)  # [B, dim]

        # Shared features
        features = self.shared_mlp(pooled)  # [B, hidden_dim]

        # Predict scale and rotation
        scale = torch.exp(self.scale_head(features))   # [B, 1], always positive, >1 = enlarge, <1 = shrink
        rotation_6d = self.rotation_head(features)     # [B, 6]

        # Convert 6D -> rotation matrix
        rotation_matrix = rotation_6d_to_matrix(rotation_6d)  # [B, 3, 3]

        return {
            "scale": scale,
            "rotation_6d": rotation_6d,
            "rotation_matrix": rotation_matrix,
        }
