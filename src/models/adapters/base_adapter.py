"""
Base Adapter Module
====================

Simple MLP-based adapter for embedding space transformation.
"""

import torch
import torch.nn as nn
from typing import Optional


class Adapter(nn.Module):
    """
    Lightweight adapter for projecting embeddings from one space to another.
    
    Architecture:
        Input → Linear → LayerNorm → GELU → Dropout → ... → Linear → LayerNorm
    
    Supports optional residual connections when input_dim == output_dim.
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
            input_dim: Input embedding dimension
            output_dim: Output embedding dimension
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
