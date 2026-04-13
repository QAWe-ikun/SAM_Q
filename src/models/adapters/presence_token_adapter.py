"""
Presence Token Adapter Module
===============================

Adapter incorporating SAM3's presence token mechanism.
"""

import torch
import torch.nn as nn
from typing import Optional

from .base_adapter import Adapter


class PresenceTokenAdapter(nn.Module):
    """
    Adapter that incorporates SAM3's presence token mechanism.
    
    Presence tokens are learnable embeddings that help distinguish
    similar text prompts by adding contextual information.
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
            qwen_dim: Source embedding dimension
            sam3_dim: Target embedding dimension
            num_presence_tokens: Number of learnable presence tokens
            hidden_dim: Hidden dimension (default: qwen_dim)
        """
        super().__init__()

        hidden_dim = hidden_dim or qwen_dim

        # Learnable presence tokens
        self.presence_tokens = nn.Embedding(num_presence_tokens, hidden_dim)

        # Base adapter
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
        # Note: This assumes adapted and presence_tokens have compatible shapes
        seq_len = min(adapted.size(1), presence_tokens.size(1))
        combined = torch.cat([
            adapted[:, :seq_len, :],
            presence_tokens[:, :seq_len, :],
        ], dim=-1)

        output = self.merge_proj(combined)

        return output
