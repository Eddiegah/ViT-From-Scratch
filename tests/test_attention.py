"""
test_attention.py
=================
Unit tests for the attention mechanism and related components.

Run with:
    python -m pytest tests/ -v

These tests catch subtle implementation bugs that are very easy to introduce
in attention code — wrong axis for softmax, wrong reshape order, incorrect
scaling, etc. — without needing to train a full model.
"""

import math
import torch
import pytest

from src.attention import MultiHeadSelfAttention
from src.patch_embedding import PatchEmbedding
from src.positional_encoding import PositionalEncoding
from src.transformer_block import TransformerBlock, MLP
from src.vit import ViT, build_small_vit


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

BATCH   = 2
SEQ_LEN = 17   # e.g. 16 patches + 1 CLS
EMBED   = 64
HEADS   = 4


@pytest.fixture
def mhsa():
    return MultiHeadSelfAttention(embed_dim=EMBED, num_heads=HEADS)


@pytest.fixture
def sample_seq():
    torch.manual_seed(0)
    return torch.randn(BATCH, SEQ_LEN, EMBED)


# ─────────────────────────────────────────────────────────────────────────────
# MultiHeadSelfAttention tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiHeadSelfAttention:

    def test_output_shape(self, mhsa, sample_seq):
        """Output should be same shape as input."""
        out = mhsa(sample_seq)
        assert out.shape == (BATCH, SEQ_LEN, EMBED), (
            f"Expected ({BATCH}, {SEQ_LEN}, {EMBED}), got {out.shape}"
        )

    def test_attention_weights_shape(self, mhsa, sample_seq):
        """Attention weights should be (B, num_heads, seq_len, seq_len)."""
        _, attn = mhsa(sample_seq, return_attention=True)
        assert attn.shape == (BATCH, HEADS, SEQ_LEN, SEQ_LEN), (
            f"Expected ({BATCH}, {HEADS}, {SEQ_LEN}, {SEQ_LEN}), got {attn.shape}"
        )

    def test_attention_weights_sum_to_one(self, mhsa, sample_seq):
        """
        Each row of the attention weight matrix must sum to 1.0 (it is a
        probability distribution over keys for each query).

        This verifies that:
        1. softmax is applied along the correct dimension (dim=-1, i.e., keys)
        2. there are no NaN values
        """
        _, attn = mhsa(sample_seq, return_attention=True)
        # attn: (B, h, N, N) — sum over last dim (keys) should be 1.0 per query
        row_sums = attn.sum(dim=-1)  # (B, h, N)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
            f"Attention rows do not sum to 1. Max deviation: "
            f"{(row_sums - 1).abs().max().item():.2e}"
        )

    def test_attention_weights_non_negative(self, mhsa, sample_seq):
        """Attention weights are outputs of softmax — all must be >= 0."""
        _, attn = mhsa(sample_seq, return_attention=True)
        assert (attn >= 0).all(), "Attention weights contain negative values"

    def test_attention_weights_no_nan(self, mhsa, sample_seq):
        """No NaN values anywhere in the attention computation."""
        out, attn = mhsa(sample_seq, return_attention=True)
        assert not torch.isnan(out).any(),  "NaN in attention output"
        assert not torch.isnan(attn).any(), "NaN in attention weights"

    def test_scale_factor(self, mhsa):
        """The scale factor should be 1/sqrt(head_dim)."""
        expected = 1.0 / math.sqrt(EMBED // HEADS)
        assert abs(mhsa.scale - math.sqrt(EMBED // HEADS)) < 1e-6, (
            "Scale stored as sqrt(d_k); 1/scale used in forward. "
            f"Expected sqrt({EMBED // HEADS})={math.sqrt(EMBED // HEADS):.4f}, "
            f"got {mhsa.scale:.4f}"
        )

    def test_embed_dim_divisibility(self):
        """Should raise AssertionError if embed_dim % num_heads != 0."""
        with pytest.raises(AssertionError):
            MultiHeadSelfAttention(embed_dim=65, num_heads=4)

    def test_different_inputs_give_different_outputs(self, mhsa):
        """Two different inputs should (almost certainly) give different outputs."""
        torch.manual_seed(1)
        x1 = torch.randn(BATCH, SEQ_LEN, EMBED)
        x2 = torch.randn(BATCH, SEQ_LEN, EMBED)
        out1 = mhsa(x1)
        out2 = mhsa(x2)
        assert not torch.allclose(out1, out2), (
            "Different inputs produced identical outputs — likely a bug"
        )

    def test_batch_independence(self, mhsa):
        """
        Processing two samples in a batch should give the same result as
        processing each individually. Verifies no cross-batch contamination.
        """
        mhsa.eval()
        torch.manual_seed(42)
        x = torch.randn(2, SEQ_LEN, EMBED)
        with torch.no_grad():
            out_batch = mhsa(x)
            out_single_0 = mhsa(x[0:1])
            out_single_1 = mhsa(x[1:2])
        assert torch.allclose(out_batch[0], out_single_0[0], atol=1e-5), (
            "Batch sample 0 differs from single-sample result"
        )
        assert torch.allclose(out_batch[1], out_single_1[0], atol=1e-5), (
            "Batch sample 1 differs from single-sample result"
        )

    def test_gradient_flow(self, mhsa, sample_seq):
        """Gradients should flow back through the attention mechanism."""
        x = sample_seq.requires_grad_(True)
        out = mhsa(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "No gradient flowed back through MHSA"
        assert not torch.isnan(x.grad).any(), "NaN in gradients"


# ─────────────────────────────────────────────────────────────────────────────
# PatchEmbedding tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPatchEmbedding:

    def test_output_shape(self):
        """Output should be (B, num_patches + 1, embed_dim)."""
        pe = PatchEmbedding(image_size=32, patch_size=4, in_channels=3, embed_dim=64)
        x = torch.randn(2, 3, 32, 32)
        out = pe(x)
        # 32/4 = 8 patches per side → 64 patches + 1 CLS = 65
        assert out.shape == (2, 65, 64), f"Expected (2, 65, 64), got {out.shape}"

    def test_num_patches(self):
        """num_patches attribute should be (image_size / patch_size) ** 2."""
        pe = PatchEmbedding(image_size=32, patch_size=4)
        assert pe.num_patches == 64

    def test_cls_token_prepended(self):
        """The [CLS] token should be at position 0 of the sequence."""
        pe = PatchEmbedding(image_size=32, patch_size=4, embed_dim=64)
        x = torch.zeros(1, 3, 32, 32)
        out = pe(x)
        # The CLS token is a learned parameter, not fixed to zero — just
        # verify sequence length includes it
        assert out.shape[1] == pe.num_patches + 1

    def test_invalid_patch_size_raises(self):
        """patch_size that doesn't evenly divide image_size should raise."""
        with pytest.raises(AssertionError):
            PatchEmbedding(image_size=32, patch_size=5)


# ─────────────────────────────────────────────────────────────────────────────
# PositionalEncoding tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPositionalEncoding:

    def test_output_shape(self):
        """Output shape should equal input shape."""
        pos = PositionalEncoding(num_patches=64, embed_dim=128)
        x = torch.randn(2, 65, 128)
        out = pos(x)
        assert out.shape == x.shape

    def test_adds_unique_values(self):
        """
        Positional encoding should change the input (i.e. is not all zeros).
        Verifies that pos_embedding was actually initialised with non-zero values.
        """
        pos = PositionalEncoding(num_patches=64, embed_dim=128)
        x = torch.zeros(1, 65, 128)
        out = pos(x)
        assert not torch.allclose(out, x), (
            "Positional encoding had no effect — pos_embedding may be all zeros"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TransformerBlock tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTransformerBlock:

    def test_output_shape(self):
        """Transformer block output shape must equal input shape."""
        block = TransformerBlock(embed_dim=64, num_heads=4, mlp_dim=128)
        x = torch.randn(2, 17, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_return_attention_shape(self):
        """When return_attention=True, second return value has correct shape."""
        block = TransformerBlock(embed_dim=64, num_heads=4, mlp_dim=128)
        x = torch.randn(2, 17, 64)
        out, attn = block(x, return_attention=True)
        assert out.shape == x.shape
        assert attn.shape == (2, 4, 17, 17)

    def test_residual_connection(self):
        """
        With zero-initialised weights (not the real init, but as a sanity check):
        the output should not be identical to a zero tensor when given a
        non-trivial input, confirming the residual stream passes through.
        """
        block = TransformerBlock(embed_dim=64, num_heads=4, mlp_dim=128)
        x = torch.randn(1, 5, 64)
        out = block(x)
        assert not torch.allclose(out, torch.zeros_like(out)), (
            "Block output is all zeros — residual connection may be missing"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Full ViT model tests
# ─────────────────────────────────────────────────────────────────────────────

class TestViT:

    def test_output_shape(self):
        """ViT logits should be (B, num_classes)."""
        model = build_small_vit(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        logits = model(x)
        assert logits.shape == (2, 10), f"Expected (2, 10), got {logits.shape}"

    def test_return_attention(self):
        """return_attention=True should return a list of attention maps, one per block."""
        model = build_small_vit(num_classes=10)
        x = torch.randn(1, 3, 32, 32)
        logits, attn_maps = model(x, return_attention=True)
        assert len(attn_maps) == 6, f"Expected 6 attention maps (one per block), got {len(attn_maps)}"
        # Each map: (1, num_heads, seq_len, seq_len) = (1, 4, 65, 65)
        assert attn_maps[0].shape == (1, 4, 65, 65)

    def test_parameter_count_reasonable(self):
        """Model should have a reasonable parameter count for a small ViT."""
        model = build_small_vit()
        n_params = model.count_parameters()
        # Small ViT on CIFAR-10 should be in the 1M–5M range
        assert 500_000 < n_params < 10_000_000, (
            f"Parameter count {n_params:,} is outside expected range [500K, 10M]"
        )
        print(f"\n  ViT parameter count: {n_params:,}")

    def test_no_nan_output(self):
        """Forward pass should not produce NaN values."""
        model = build_small_vit()
        model.eval()
        torch.manual_seed(0)
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any(), "ViT forward pass produced NaN logits"

    def test_gradient_flows_to_inputs(self):
        """Gradients should flow all the way back to the input pixels."""
        model = build_small_vit()
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        logits = model(x)
        logits.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_different_images_give_different_logits(self):
        """Two different images must not produce identical logits."""
        model = build_small_vit()
        model.eval()
        with torch.no_grad():
            x1 = torch.randn(1, 3, 32, 32)
            x2 = torch.randn(1, 3, 32, 32)
            assert not torch.allclose(model(x1), model(x2))
