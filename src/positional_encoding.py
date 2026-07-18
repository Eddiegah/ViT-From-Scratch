"""
positional_encoding.py
======================
Adds learnable positional embeddings to patch embeddings.

Why positional embeddings are necessary
----------------------------------------
Self-attention computes relationships between all pairs of tokens, but the
attention operation itself is *permutation-invariant* — if you shuffled all
patches, the attention weights would shift but the model would still produce
the same output (just for a different arrangement).

Without positional information, the model cannot know that patch 0 is
top-left and patch 8 is bottom-right. By adding a unique learnable vector to
each position, we inject spatial location into the token representation.

The ViT paper (Dosovitskiy et al., 2020) tries several positional encoding
strategies and finds that 1D learnable embeddings (one vector per position,
treating the patch sequence as a flat list) work just as well as 2D variants
on typical benchmarks — so that is what we implement here.

Positional embeddings are learned jointly with the rest of the model via
backpropagation, not fixed sinusoids (as in the original "Attention Is All
You Need" paper).
"""

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Adds a learnable 1-D positional embedding to every token in the sequence,
    including the [CLS] token at position 0.

    Args:
        num_patches : int — number of image patches (NOT counting [CLS])
        embed_dim   : int — embedding dimension D
    """

    def __init__(self, num_patches: int, embed_dim: int):
        super().__init__()

        # Total sequence length = num_patches + 1 ([CLS] token occupies position 0)
        seq_len = num_patches + 1

        # pos_embedding: (1, seq_len, embed_dim)
        # The leading "1" dimension broadcasts across the batch.
        # Each of the seq_len positions gets its own D-dimensional learnable vector.
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

        # Initialise with small values so positional signals start subtle
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor of shape (B, seq_len, embed_dim)
                — patch embeddings with [CLS] prepended

        Returns:
            Tensor of shape (B, seq_len, embed_dim)
            — same tensor with positional embeddings added element-wise
        """
        # Simple element-wise addition; pos_embedding broadcasts over batch dim
        return x + self.pos_embedding
