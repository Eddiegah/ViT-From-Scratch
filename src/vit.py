"""
vit.py
======
Full Vision Transformer (ViT) model, assembling all components.

Architecture overview
---------------------
Given an image x ∈ R^(B × C × H × W):

  1. PatchEmbedding
     - Split into N = (H/P)^2 non-overlapping patches of size P×P
     - Linearly project each patch to a D-dimensional vector
     - Prepend a learnable [CLS] token → sequence length = N + 1

  2. PositionalEncoding
     - Add a learnable positional embedding to each token
     - (sequence is now position-aware)

  3. Transformer Encoder (L blocks stacked)
     Each block:  x = x + MHSA(LN(x))
                  x = x + MLP(LN(x))

  4. Classification head
     - Take the [CLS] token's final representation (index 0 of the sequence)
     - Apply a final LayerNorm (standard ViT practice)
     - Pass through a Linear(D → num_classes) projection

Why the [CLS] token for classification?
----------------------------------------
The [CLS] token has no fixed spatial meaning — it starts as a learned
embedding and must attend to all patches to gather global image information.
By the final layer its representation (hopefully) encodes a holistic summary
of the image, which is why we use it as the classification feature.

An alternative is global average pooling over all patch tokens — in practice
both work similarly, but [CLS] is the approach in the original ViT paper.

Default CIFAR-10 configuration (small, CPU-feasible)
------------------------------------------------------
  image_size  = 32
  patch_size  = 4     → N = (32/4)^2 = 64 patches
  embed_dim   = 128
  num_heads   = 4     → head_dim = 32
  num_layers  = 6
  mlp_dim     = 256
  num_classes = 10

Parameter count ≈ 1.8 M — small enough to train on CPU in a few hours.
"""

import torch
import torch.nn as nn

from src.patch_embedding import PatchEmbedding
from src.positional_encoding import PositionalEncoding
from src.transformer_block import TransformerBlock


class ViT(nn.Module):
    """
    Vision Transformer.

    Args:
        image_size   : int   — height/width of the (square) input image
        patch_size   : int   — height/width of each (square) patch
        in_channels  : int   — number of image channels (3 for RGB)
        num_classes  : int   — number of output classes
        embed_dim    : int   — embedding dimension D
        num_heads    : int   — number of attention heads per block
        num_layers   : int   — number of stacked Transformer blocks
        mlp_dim      : int   — MLP hidden dimension inside each block
        attn_dropout : float — dropout on attention weights
        mlp_dropout  : float — dropout inside MLP
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 6,
        mlp_dim: int = 256,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()

        # ── Component 1: Patch embedding + [CLS] token ────────────────────────
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches  # N

        # ── Component 2: Positional encoding ─────────────────────────────────
        self.pos_enc = PositionalEncoding(
            num_patches=num_patches,
            embed_dim=embed_dim,
        )

        # ── Component 3: Dropout after embedding (light regularisation) ───────
        self.embed_dropout = nn.Dropout(mlp_dropout)

        # ── Component 4: Stack of Transformer Encoder blocks ─────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
            )
            for _ in range(num_layers)
        ])

        # ── Component 5: Final LayerNorm ──────────────────────────────────────
        # Applied to the full sequence after all blocks.
        # Standard ViT practice — stabilises the [CLS] representation before
        # the classification head.
        self.norm = nn.LayerNorm(embed_dim)

        # ── Component 6: Classification head ──────────────────────────────────
        # We use *only* the [CLS] token (index 0) — it aggregated global context
        # through all attention layers.
        self.head = nn.Linear(embed_dim, num_classes)

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        """
        Initialise weights following common ViT practice:
        - Linear layers: truncated normal for weights, zero for bias
        - LayerNorm: weight=1, bias=0 (identity at init)
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Args:
            x               : Tensor (B, C, H, W) — input images
            return_attention : bool — if True, return attention weights from
                               every block (for visualisation)

        Returns:
            logits        : Tensor (B, num_classes)
            attention_maps: list of Tensors (B, h, N+1, N+1), one per block
                            (only returned if return_attention=True)
        """
        # ── Step 1: Patch embed → (B, N+1, D) ────────────────────────────────
        x = self.patch_embed(x)

        # ── Step 2: Add positional encodings ──────────────────────────────────
        x = self.pos_enc(x)
        x = self.embed_dropout(x)

        # ── Step 3: Pass through each Transformer block ───────────────────────
        attention_maps = []
        for block in self.blocks:
            if return_attention:
                x, attn = block(x, return_attention=True)
                attention_maps.append(attn)
            else:
                x = block(x)

        # ── Step 4: Final LayerNorm ───────────────────────────────────────────
        x = self.norm(x)

        # ── Step 5: Extract [CLS] token (position 0) and classify ─────────────
        cls_output = x[:, 0, :]        # (B, D)
        logits = self.head(cls_output)  # (B, num_classes)

        if return_attention:
            return logits, attention_maps
        return logits

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_small_vit(num_classes: int = 10) -> ViT:
    """
    Convenience factory: the default small CIFAR-10 configuration.

    Patch size 4 on 32×32 images → 64 patches → sequence length 65 (with CLS).
    ~1.8 M parameters. Trainable on CPU in a few hours.
    """
    return ViT(
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=128,
        num_heads=4,
        num_layers=6,
        mlp_dim=256,
        attn_dropout=0.1,
        mlp_dropout=0.1,
    )
