"""
attention.py
============
Manual implementation of Multi-Head Self-Attention (MHSA) from scratch.

This file is intentionally the most heavily commented in the project — the
goal is that every line of code maps directly to a named piece of the math.

Reference papers
----------------
- "Attention Is All You Need" (Vaswani et al., 2017) — introduced scaled
  dot-product attention and multi-head attention.
- "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020) — applied the
  same mechanism to image patches.

══════════════════════════════════════════════════════════════════════════════
  THE MATH (scaled dot-product attention)
══════════════════════════════════════════════════════════════════════════════

Given an input sequence X ∈ R^(N × D), we produce three linear projections:

    Q = X W_Q    (Queries,  shape N × d_k)
    K = X W_K    (Keys,     shape N × d_k)
    V = X W_V    (Values,   shape N × d_v)

where W_Q, W_K ∈ R^(D × d_k) and W_V ∈ R^(D × d_v) are learned weight matrices.

The attention output is:

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

  - Q K^T  produces an (N × N) matrix of raw "compatibility scores" between
    every query token and every key token.
  - Dividing by sqrt(d_k) prevents the dot products from growing too large
    (which would push softmax into near-zero gradient regions).
  - softmax(·) normalises each row to a probability distribution — row i
    tells us how much token i should "attend to" each other token.
  - Multiplying by V produces a weighted mixture of value vectors, one per
    query token.

══════════════════════════════════════════════════════════════════════════════
  MULTI-HEAD ATTENTION
══════════════════════════════════════════════════════════════════════════════

Instead of a single attention function with d_k = D, we run h attention
"heads" in parallel, each on a lower-dimensional subspace (d_k = D / h):

    head_i = Attention(X W_Q^i, X W_K^i, X W_V^i)

    MultiHead(X) = Concat(head_1, ..., head_h) W_O

where W_O ∈ R^(h*d_v × D) is the output projection.

The intuition: different heads can learn to attend to different kinds of
relationships (e.g. one head captures local adjacency, another long-range
semantic similarity). Concatenating and projecting merges these views.

In practice we implement this efficiently by keeping Q, K, V as single
tensors of shape (B, N, D) and then reshaping/transposing to expose the head
dimension, rather than using h separate projection matrices.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention block, implemented from the raw math.

    No nn.MultiheadAttention is used — all projections and the attention
    computation are done manually so every step is explicit.

    Args:
        embed_dim   : int   — total embedding dimension D
        num_heads   : int   — number of attention heads h
        attn_dropout: float — dropout probability on attention weights
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.embed_dim = embed_dim   # D
        self.num_heads = num_heads   # h
        self.head_dim = embed_dim // num_heads  # d_k = d_v = D / h

        # ── Linear projections for Q, K, V ────────────────────────────────────
        # We use a single Linear(D, D) for each of Q, K, V.
        # Conceptually this is equivalent to h separate Linear(D, d_k) matrices
        # stacked together — we just split the output later.
        #
        # W_Q, W_K, W_V each have shape (D, D) with a bias of shape (D,).
        self.W_q = nn.Linear(embed_dim, embed_dim)  # projects to h*d_k = D
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # ── Output projection W_O ──────────────────────────────────────────────
        # After concatenating all heads: (B, N, h*d_v) = (B, N, D)
        # W_O projects this back to (B, N, D).
        self.W_o = nn.Linear(embed_dim, embed_dim)

        # Optional dropout on attention weights (regularises which tokens
        # the model attends to)
        self.attn_drop = nn.Dropout(attn_dropout)

        # Scale factor: 1 / sqrt(d_k)
        # Pre-computing avoids recomputing it every forward pass.
        self.scale = math.sqrt(self.head_dim)

    # ──────────────────────────────────────────────────────────────────────────
    def _split_heads(self, t: torch.Tensor, B: int, N: int) -> torch.Tensor:
        """
        Reshape (B, N, D) → (B, h, N, d_k) so we can apply attention
        independently per head.

        The transformation:
          (B, N, D)
          → (B, N, h, d_k)   via .view()
          → (B, h, N, d_k)   via .transpose(1, 2)
        """
        return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, t: torch.Tensor, B: int, N: int) -> torch.Tensor:
        """
        Reverse of _split_heads: (B, h, N, d_k) → (B, N, D).

        .contiguous() is needed before .view() because transpose creates a
        non-contiguous tensor in memory.
        """
        return t.transpose(1, 2).contiguous().view(B, N, self.embed_dim)

    # ──────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Args:
            x               : Tensor (B, N, D) — input token sequence
            return_attention : bool — if True, also return the attention weight
                               matrix (B, h, N, N), useful for visualisation

        Returns:
            out  : Tensor (B, N, D) — attended output
            attn : Tensor (B, h, N, N) — attention weights (only if return_attention=True)
        """
        B, N, D = x.shape

        # ── Step 1: Compute Q, K, V projections ───────────────────────────────
        # Each of Q, K, V: (B, N, D)
        Q = self.W_q(x)   # Q = X W_Q
        K = self.W_k(x)   # K = X W_K
        V = self.W_v(x)   # V = X W_V

        # ── Step 2: Split into heads ───────────────────────────────────────────
        # (B, N, D) → (B, h, N, d_k)
        Q = self._split_heads(Q, B, N)
        K = self._split_heads(K, B, N)
        V = self._split_heads(V, B, N)

        # ── Step 3: Scaled dot-product attention scores ────────────────────────
        # Q: (B, h, N, d_k)
        # K: (B, h, N, d_k) → K^T: (B, h, d_k, N)   via matmul broadcasting
        #
        # scores = Q K^T / sqrt(d_k)
        # shape:  (B, h, N, N)  — every token vs every token, per head
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores[b, h, i, j] = how much token i queries for token j's key

        # ── Step 4: Softmax — normalise rows to a probability distribution ─────
        # attn_weights[b, h, i, :] sums to 1.0
        # Row i tells us what fraction of each value token j contributes to
        # the attended output for token i.
        attn_weights = F.softmax(scores, dim=-1)  # (B, h, N, N)
        attn_weights = self.attn_drop(attn_weights)

        # ── Step 5: Weighted sum of Values ────────────────────────────────────
        # attn_weights: (B, h, N, N)
        # V:            (B, h, N, d_k)
        # out:          (B, h, N, d_k)
        #
        # out[b, h, i, :] = sum_j attn_weights[b, h, i, j] * V[b, h, j, :]
        attended = torch.matmul(attn_weights, V)

        # ── Step 6: Merge heads and apply output projection ───────────────────
        # (B, h, N, d_k) → (B, N, D)
        attended = self._merge_heads(attended, B, N)

        # Final linear mix: MultiHead(X) = Concat(heads) W_O
        out = self.W_o(attended)  # (B, N, D)

        if return_attention:
            return out, attn_weights
        return out
