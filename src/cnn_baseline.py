"""
cnn_baseline.py
===============
A small CNN baseline for fair comparison against the ViT on CIFAR-10.

Design goals
------------
- Comparable parameter count to the small ViT (~1–2 M parameters)
- Classic conv → BN → ReLU → pool pattern — readable and representative
- Built using standard nn.Conv2d etc. (appropriate since this is the
  *comparison model*, not the subject of the "from scratch" exercise)

Architecture
------------
  Input: (B, 3, 32, 32)

  Block 1: Conv(3→32, k=3, p=1) → BN → ReLU → Conv(32→32, k=3, p=1) → BN → ReLU → MaxPool(2)
  Block 2: Conv(32→64, k=3, p=1) → BN → ReLU → Conv(64→64, k=3, p=1) → BN → ReLU → MaxPool(2)
  Block 3: Conv(64→128, k=3, p=1) → BN → ReLU → Conv(128→128, k=3, p=1) → BN → ReLU → MaxPool(2)

  After 3 max-pool(2) operations on 32×32: spatial size = 4×4
  Flatten → (B, 128*4*4) = (B, 2048)

  Classifier:
    Linear(2048 → 512) → ReLU → Dropout(0.3)
    Linear(512 → 256)  → ReLU → Dropout(0.3)
    Linear(256 → 10)

Total: ~2.0 M parameters — comparable to the small ViT.

Why CNNs typically outperform ViTs at this scale
-------------------------------------------------
Convolutional layers have a strong inductive bias baked in: local spatial
connections (the kernel sees a neighbourhood, not the full image) and
translation equivariance (the same filter is applied everywhere). These are
excellent priors for images.

ViT has *no* such prior — it treats the image as a flat sequence of patches
and must learn spatial relationships entirely from data through attention.
At small data scales (< ~100 K images, limited compute), the CNN's head start
from its inductive bias is hard to overcome.

At ImageNet scale (~1.2 M images) with sufficient compute, ViTs can match or
exceed CNNs — but getting there requires the data to "teach" the ViT what
CNNs already "know" by construction.
"""

import torch
import torch.nn as nn


def _conv_bn_relu(in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1) -> nn.Sequential:
    """Helper: Conv2d + BatchNorm2d + ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CNNBaseline(nn.Module):
    """
    Small CNN baseline for CIFAR-10.

    Args:
        num_classes : int — number of output classes (default 10)
        dropout     : float — dropout rate in the classifier head
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()

        # ── Feature extractor ─────────────────────────────────────────────────
        self.features = nn.Sequential(
            # Block 1 — 32×32 → 16×16
            _conv_bn_relu(3, 32),
            _conv_bn_relu(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2 — 16×16 → 8×8
            _conv_bn_relu(32, 64),
            _conv_bn_relu(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3 — 8×8 → 4×4
            _conv_bn_relu(64, 128),
            _conv_bn_relu(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # After feature extractor: (B, 128, 4, 4)

        # ── Classifier head ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),                         # (B, 128*4*4) = (B, 2048)
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Tensor (B, C, H, W)

        Returns:
            logits : Tensor (B, num_classes)
        """
        x = self.features(x)
        return self.classifier(x)

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
