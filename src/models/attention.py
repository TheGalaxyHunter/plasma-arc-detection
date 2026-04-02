"""Self-attention module for temporal feature selection.

Implements scaled dot-product self-attention with a learned query vector
for pooling variable-length temporal sequences into a fixed-size context.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSelfAttention(nn.Module):
    """Multi-head self-attention with learnable query for sequence pooling.

    Instead of using the standard encoder self-attention where every position
    attends to every other position, this module uses a learned global query
    to attend over all temporal positions. This produces a single context
    vector per sequence, suitable for downstream classification.

    Args:
        embed_dim: Dimension of input features (must match LSTM output).
        num_heads: Number of attention heads.
        dropout: Dropout rate on attention weights.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-weighted context vector from temporal features.

        Args:
            x: Temporal features of shape (batch, seq_len, embed_dim).
            mask: Optional boolean mask of shape (batch, seq_len). True values
                are positions to ignore (padding).

        Returns:
            Tuple of:
                - context: Pooled representation, shape (batch, embed_dim).
                - weights: Attention weights, shape (batch, seq_len).
        """
        batch_size, seq_len, _ = x.shape

        # Expand learned query across the batch
        q = self.query.expand(batch_size, -1, -1)  # (batch, 1, embed_dim)
        k = self.key_proj(x)  # (batch, seq_len, embed_dim)
        v = self.value_proj(x)  # (batch, seq_len, embed_dim)

        # Reshape for multi-head attention
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # q: (batch, heads, 1, head_dim)
        # k, v: (batch, heads, seq_len, head_dim)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # attn_scores: (batch, heads, 1, seq_len)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        context = torch.matmul(attn_weights, v)  # (batch, heads, 1, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, self.embed_dim)

        context = self.out_proj(context)
        context = self.layer_norm(context)

        # Average attention weights across heads for interpretability
        avg_weights = attn_weights.mean(dim=1).squeeze(1)  # (batch, seq_len)

        return context, avg_weights
