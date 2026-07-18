"""
patch_embedding.py
==================
Converts a batch of images into a sequence of patch embeddings, then prepends
a learnable [CLS] token — the first step in every ViT forward pass.

Mathematical background
-----------------------
Given an image x ∈ R^(H × W × C), we divide it into N non-overlapping patches
of size (P × P), where N = (H/P) * (W/P).

Each patch p_i ∈ R^(P*P*C) is flattened and linearly projected:
    z_i = p_i @ E + b_E    where E ∈ R^(P*P*C × D) is the patch projection matrix

This gives a sequence z ∈ R^(N × D) of patch embeddings.

Efficient implementation via strided Conv2d
-------------------------------------------
A Conv2d with:
  - kernel_size = patch_size
  - stride      = patch_size
  - out_channels = embed_dim

is mathematically identical to "split into patches → flatten → linear project":

  For each patch at position (i, j), the conv kernel slides over exactly that
  patch (no overlap because stride == kernel_size), computes a dot product
  between the flattened kernel weights and the flattened patch pixels, and
  outputs one scalar per output channel.

  With out_channels = embed_dim output channels, that's exactly the same as
  multiplying the flattened patch (length P*P*C) by a matrix of shape
  (P*P*C × embed_dim) and adding a bias — which is the linear projection E above.

  Using Conv2d is therefore NOT a shortcut that avoids understanding — it is the
  standard, numerically identical, hardware-efficient way to implement patch
  embedding in all major ViT codebases (including the original JAX implementation).
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Splits an image into non-overlapping patches and projects each patch into
    a D-dimensional embedding vector.

    Then prepends a learnable [CLS] token to the resulting sequence.

    Args:
        image_size  : int  — height/width of the (square) input image (e.g. 32)
        patch_size  : int  — height/width of each (square) patch (e.g. 4)
        in_channels : int  — number of image channels (3 for RGB)
        embed_dim   : int  — dimension D of each patch embedding vector
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 128,
    ):
        super().__init__()

        assert image_size % patch_size == 0, (
            f"Image size {image_size} must be divisible by patch size {patch_size}"
        )

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Number of patches along each spatial dimension
        self.num_patches_per_side = image_size // patch_size
        # Total number of patches in the sequence (excluding [CLS])
        self.num_patches = self.num_patches_per_side ** 2

        # ---- Patch projection (strided Conv2d) --------------------------------
        # kernel_size = patch_size, stride = patch_size → each kernel application
        # covers exactly one non-overlapping patch.
        # out_channels = embed_dim → each patch maps to a D-dim vector.
        # This is equivalent to: flatten_patch → linear(P*P*C, embed_dim).
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # ---- Learnable [CLS] token -------------------------------------------
        # Shape: (1, 1, embed_dim) — the "1, 1" dims let us broadcast across batch.
        # The [CLS] token is not derived from any patch; it starts as a learned
        # parameter and gathers global image information through self-attention.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Weight initialisation: small normal for projection, zeros for cls_token
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor of shape (B, C, H, W)

        Returns:
            Tensor of shape (B, num_patches + 1, embed_dim)
            — the +1 is the prepended [CLS] token
        """
        B, C, H, W = x.shape

        # Step 1: Project patches → (B, embed_dim, H/P, W/P)
        x = self.projection(x)

        # Step 2: Flatten spatial grid → sequence
        # (B, embed_dim, n_h, n_w) → (B, embed_dim, N) → (B, N, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        # x is now (B, num_patches, embed_dim)

        # Step 3: Expand [CLS] token to batch size and prepend
        # cls_token: (1, 1, D) → (B, 1, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # Concatenate along the sequence dimension: (B, N+1, D)
        x = torch.cat([cls_tokens, x], dim=1)

        return x
