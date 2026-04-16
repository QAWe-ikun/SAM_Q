"""
Cross-Modal Adapter Module
============================

Cross-attention based adapter for multimodal embedding fusion.
"""

import torch
import torch.nn as nn
from typing import Union
from pathlib import Path


class CrossModalAdapter(nn.Module):
    """
    Cross-modal adapter that uses cross-attention to project embeddings.

    Architecture:
        Input → Linear → LayerNorm
        ↓
        Learnable Queries cross-attend to Input
        ↓
        Output Projection → SAM3 space

    Produces fixed-length output (num_queries) regardless of input sequence length.
    """

    def __init__(
        self,
        qwen_dim: int,
        sam3_dim: int,
        num_queries: int = 64,
        hidden_dim: int = 512,
    ):
        """
        Initialize cross-modal adapter.

        Args:
            qwen_dim: Source embedding dimension (e.g., Qwen3-VL: 3584)
            sam3_dim: Target embedding dimension (e.g., SAM3: 256)
            num_queries: Number of output queries for DETR-style detector
            hidden_dim: Hidden dimension for transformation
        """
        super().__init__()

        self.num_queries = num_queries

        # Input projection
        self.input_proj = nn.Linear(qwen_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        # Learnable query embeddings
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Cross-attention layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, sam3_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with cross-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, qwen_dim)

        Returns:
            output: Output tensor of shape (batch_size, num_queries, sam3_dim)
        """
        # Project input to hidden space
        x = self.input_proj(x)
        x = self.norm(x)

        # Expand learnable queries
        queries = self.query_embed.weight.unsqueeze(0).expand(x.size(0), -1, -1)

        # Cross-attention: queries attend to input embeddings
        attended, _ = self.cross_attn(
            query=queries,
            key=x,
            value=x,
        )

        # Project to output space
        output = self.output_proj(attended)

        return output

    def load_from_checkpoint(
        self, 
        path: Union[str, Path], 
        device: str = "cpu",
        prefix: str = ""
    ):
        """
        Load trained weights from a checkpoint file.

        Args:
            path: Path to checkpoint (.pt)
            device: Device to load weights to
            prefix: Key prefix in checkpoint (e.g., "adapter.") if loading from full model state_dict
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Adapter checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)

        # Filter keys if loading from a full model checkpoint
        if prefix:
            filtered_state_dict = {
                k[len(prefix):]: v 
                for k, v in state_dict.items() 
                if k.startswith(prefix)
            }
        else:
            filtered_state_dict = state_dict

        missing, unexpected = self.load_state_dict(filtered_state_dict, strict=False)
        
        if unexpected:
            print(f"[CrossModalAdapter] Unexpected keys: {unexpected}")
        
        self.to(device)
        self.eval()
        print(f"[CrossModalAdapter] Loaded weights from {path}")

