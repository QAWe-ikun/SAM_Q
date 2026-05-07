"""
Cross-Modal Adapter Module
============================

Deep MLP adapter for multimodal embedding fusion.
"""

import torch  # type: ignore
import torch.nn as nn  # type: ignore
from typing import Union
from pathlib import Path


class CrossModalAdapter(nn.Module):
    """
    MLP adapter that maps <SEG> hidden state to SAM3 prompt embeddings.

    Architecture:
        <SEG> [B,1,4096] → Linear(4096→512) → GELU → LayerNorm → Dropout
                           → Linear(512→512)  → GELU → LayerNorm → Dropout
                           → Linear(512→256)
    """

    def __init__(
        self,
        qwen_dim: int,
        sam3_dim: int,
        num_queries: int = 1,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.num_queries = num_queries

        self.proj = nn.Sequential(
            nn.Linear(qwen_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, sam3_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, qwen_dim]

        Returns:
            output: [B, num_queries, sam3_dim]
        """
        # Mean pool over sequence dim → [B, hidden_dim]
        pooled = x.mean(dim=1)  # [B, qwen_dim]
        out = self.proj(pooled)  # [B, sam3_dim]
        return out.unsqueeze(1)  # [B, 1, sam3_dim]

    def load_from_checkpoint(
        self,
        path: Union[str, Path],
        device: str = "cpu",
        prefix: str = ""
    ):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Adapter checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)

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
