"""
Adapter Module for Qwen3-VL to SAM3 Integration

Projects Qwen3-VL embeddings to SAM3 Detector input space.
"""

import torch
import torch.nn as nn
from typing import Optional


class Adapter(nn.Module):
    """
    Lightweight adapter for projecting Qwen3-VL embeddings
    to SAM3's expected input format.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        layer_norm: bool = True,
        residual: bool = False,
    ):
        """
        Initialize the adapter.
        
        Args:
            input_dim: Input embedding dimension (Qwen3-VL hidden size)
            output_dim: Output dimension (SAM3 Detector input size)
            hidden_dim: Hidden layer dimension (default: input_dim)
            num_layers: Number of MLP layers
            dropout: Dropout rate
            layer_norm: Whether to use LayerNorm
            residual: Whether to use residual connection (requires input_dim == output_dim)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.residual = residual
        
        if residual and input_dim != output_dim:
            raise ValueError(
                f"Residual connection requires input_dim ({input_dim}) == output_dim ({output_dim})"
            )
        
        hidden_dim = hidden_dim or input_dim
        
        # Build MLP layers
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Optional output layer norm
        self.output_norm = nn.LayerNorm(output_dim) if layer_norm else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the adapter.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            output: Output tensor of shape (batch_size, seq_len, output_dim)
        """
        projected = self.mlp(x)
        projected = self.output_norm(projected)
        
        if self.residual:
            projected = projected + x
        
        return projected


class CrossModalAdapter(nn.Module):
    """
    Cross-modal adapter that handles both image and text embeddings
    from Qwen3-VL and projects them to SAM3 space.
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
            qwen_dim: Qwen3-VL embedding dimension
            sam3_dim: SAM3 expected input dimension
            num_queries: Number of output queries for DETR-style detector
            hidden_dim: Hidden dimension for transformation
        """
        super().__init__()
        
        self.num_queries = num_queries
        
        # Main projection
        self.input_proj = nn.Linear(qwen_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Query transformer for fixed-length output
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
        # Project input
        x = self.input_proj(x)
        x = self.norm(x)
        
        # Get query embeddings
        queries = self.query_embed.weight.unsqueeze(0).expand(x.size(0), -1, -1)
        
        # Cross-attention
        attended, _ = self.cross_attn(
            query=queries,
            key=x,
            value=x,
        )
        
        # Output projection
        output = self.output_proj(attended)
        
        return output


class PresenceTokenAdapter(nn.Module):
    """
    Adapter that incorporates SAM3's presence token mechanism.
    
    Presence tokens help distinguish similar text prompts
    by adding learnable token embeddings.
    """
    
    def __init__(
        self,
        qwen_dim: int,
        sam3_dim: int,
        num_presence_tokens: int = 4,
        hidden_dim: Optional[int] = None,
    ):
        """
        Initialize presence token adapter.
        
        Args:
            qwen_dim: Qwen3-VL embedding dimension
            sam3_dim: SAM3 expected input dimension
            num_presence_tokens: Number of learnable presence tokens
            hidden_dim: Hidden dimension (default: qwen_dim)
        """
        super().__init__()
        
        hidden_dim = hidden_dim or qwen_dim
        
        # Learnable presence tokens
        self.presence_tokens = nn.Embedding(num_presence_tokens, hidden_dim)
        
        # Main adapter
        self.adapter = Adapter(
            input_dim=qwen_dim,
            output_dim=sam3_dim,
            hidden_dim=hidden_dim,
        )
        
        # Merge presence tokens with embeddings
        self.merge_proj = nn.Linear(sam3_dim + hidden_dim, sam3_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        presence_token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with presence tokens.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, qwen_dim)
            presence_token_ids: Optional token IDs for presence tokens
            
        Returns:
            output: Output tensor of shape (batch_size, seq_len, sam3_dim)
        """
        # Get base adaptation
        adapted = self.adapter(x)
        
        # Add presence tokens
        batch_size = x.size(0)
        presence_tokens = self.presence_tokens.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        
        # Concatenate and project
        combined = torch.cat([
            adapted,
            presence_tokens[:, :adapted.size(1)],  # Match sequence length
        ], dim=-1)
        
        output = self.merge_proj(combined)
        
        return output
