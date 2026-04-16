"""
SEG Token Projector Module
============================

SA2VA-inspired projector that maps the <SEG> token hidden state
from Qwen3-VL to SAM3 prompt embedding space.

The <SEG> token's hidden state encodes the full semantic understanding
of the placement instruction (after the LLM has attended to all prior context).
This projector expands it into multiple query tokens for the SAM3 detector.
"""

import torch 
import torch.nn as nn 


class SegTokenProjector(nn.Module):
    """
    Projects <SEG> token hidden state to SAM3 prompt tokens.

    Architecture:
        <SEG> hidden state [B, qwen_dim]
            -> Input projection + LayerNorm + GELU
        Learnable queries [num_output_tokens, hidden_dim]
            -> Cross-attention (queries attend to projected <SEG> state)
            -> Output projection -> [B, num_output_tokens, sam3_dim]

    Output shape matches CrossModalAdapter so SAM3 detector interface is unchanged.
    """

    def __init__(
        self,
        qwen_dim: int = 4096,
        sam3_dim: int = 256,
        num_output_tokens: int = 64,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_output_tokens = num_output_tokens

        # Project <SEG> hidden state to hidden space
        self.input_proj = nn.Sequential(
            nn.Linear(qwen_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Learnable output queries (expand single token to multiple)
        self.query_embed = nn.Embedding(num_output_tokens, hidden_dim)

        # Cross-attention: queries attend to projected <SEG> state
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        # Project to SAM3 space
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, sam3_dim),
        )

    def forward(self, seg_hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seg_hidden_state: [B, qwen_dim] hidden state at <SEG> position

        Returns:
            output: [B, num_output_tokens, sam3_dim]
        """
        # Project to hidden space: [B, qwen_dim] -> [B, 1, hidden_dim]
        projected = self.input_proj(seg_hidden_state).unsqueeze(1)

        # Expand queries: [num_output_tokens, hidden_dim] -> [B, num_output_tokens, hidden_dim]
        batch_size = seg_hidden_state.size(0)
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attention: queries attend to single projected <SEG> token
        attended, _ = self.cross_attn(
            query=queries,
            key=projected,
            value=projected,
        )

        # Project to SAM3 space
        output = self.output_proj(attended)

        return output
