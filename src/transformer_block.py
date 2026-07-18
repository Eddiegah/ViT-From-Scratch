"""
transformer_block.py
====================
A single Transformer Encoder Block — the repeating unit stacked N_layers times
inside the ViT encoder.

Architecture (Pre-LN variant, used in the ViT paper)
-----------------------------------------------------

    ┌─────────────────────────────────────────────┐
    │   x  ──► LayerNorm ──► MHSA ──► (+) ──► x' │  residual stream
    │                                  ▲           │
    │                                  x           │
    └─────────────────────────────────────────────┘
    ┌─────────────────────────────────────────────┐
    │  x' ──► LayerNorm ──► MLP  ──► (+) ──► x'' │  residual stream
    │                                  ▲           │
    │                                  x'          │
    └─────────────────────────────────────────────┘

Key design choices explained
-----------------------------
Pre-LayerNorm ("Pre-LN"):
  LayerNorm is applied *before* the attention/MLP sub-layers, not after.
  This stabilises training at the cost of slightly lower final performance on
  some tasks — but it converges more reliably, which matters when training
  from scratch with limited data.

Residual connections:
  Each sub-layer wraps as:  output = x + SubLayer(LayerNorm(x))
  Residuals allow gradients to flow directly back through the network without
  passing through MHSA/MLP at every layer — essential for training deep models.

MLP block:
  A two-layer feed-forward network with GELU activation.
  Hidden dimension is typically 4× the embedding dim (following the original
  Transformer paper). This is where most of the model's "memory" lives —
  attention decides what to mix, the MLP transforms what was mixed.

GELU activation:
  GELU (Gaussian Error Linear Unit) is standard in modern transformers.
  It's smoother than ReLU and empirically works better for transformers.
"""

import torch
import torch.nn as nn

from src.attention import MultiHeadSelfAttention


class MLP(nn.Module):
    """
    Two-layer feed-forward network used inside each Transformer block.

    Architecture:
        Linear(D, mlp_dim) → GELU → Dropout → Linear(mlp_dim, D) → Dropout

    Args:
        embed_dim   : int   — input/output dimension D
        mlp_dim     : int   — hidden dimension (typically 4 * embed_dim)
        dropout     : float — dropout probability
    """

    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One Transformer Encoder Block (Pre-LN variant).

    Args:
        embed_dim    : int   — embedding dimension D
        num_heads    : int   — number of attention heads
        mlp_dim      : int   — MLP hidden dimension
        attn_dropout : float — dropout on attention weights
        mlp_dropout  : float — dropout inside MLP
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()

        # Layer normalisations (applied BEFORE each sub-layer — "Pre-LN")
        self.norm1 = nn.LayerNorm(embed_dim)  # before MHSA
        self.norm2 = nn.LayerNorm(embed_dim)  # before MLP

        # Multi-head self-attention (our manual implementation)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
        )

        # Feed-forward MLP
        self.mlp = MLP(embed_dim=embed_dim, mlp_dim=mlp_dim, dropout=mlp_dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Args:
            x               : Tensor (B, N, D)
            return_attention : bool — pass through to MHSA for visualisation

        Returns:
            out  : Tensor (B, N, D)
            attn : Tensor (B, h, N, N) — only returned if return_attention=True
        """
        # ── Sub-layer 1: Multi-Head Self-Attention with residual ──────────────
        # Pre-LN: normalise first, attend, then add residual
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + attn_out
        else:
            x = x + self.attn(self.norm1(x))

        # ── Sub-layer 2: MLP with residual ────────────────────────────────────
        x = x + self.mlp(self.norm2(x))

        if return_attention:
            return x, attn_weights
        return x
