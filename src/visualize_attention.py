"""
visualize_attention.py
======================
Visualises what the ViT's attention mechanism actually learned to focus on.

For each sample image we:
1. Run a forward pass with return_attention=True to extract attention weight
   matrices from every Transformer block.
2. Focus on the last block's attention weights (most semantically meaningful).
3. Extract the row corresponding to the [CLS] token — this row tells us how
   much the classification token attends to each patch.
4. Reshape that vector from (num_patches,) back to the 2D patch grid and
   upsample it to the original image size.
5. Overlay the attention map as a heatmap on the original image.

Why this is interesting
-----------------------
The [CLS] token has no fixed spatial location — it must gather whatever
information is useful for classification by attending to patches. Visualising
its attention weights shows us *which parts of the image the model decided
were most informative* for the correct class label. This is something a CNN
doesn't give you as directly (there are GradCAM etc., but they require
post-hoc gradient tricks; here the attention weights are intrinsic to the
architecture).

Usage
-----
    python -m src.visualize_attention [--num_samples N] [--layer L] [--device cpu|cuda|auto]
"""

import argparse
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from src.vit import build_small_vit

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def unnormalise(img_tensor):
    """
    Reverse CIFAR-10 normalisation for display.
    img_tensor: (C, H, W) normalised tensor → numpy (H, W, C) in [0, 1]
    """
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std  = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def get_attention_map(attn_weights, patch_grid_size: int):
    """
    Extract and reshape the [CLS] attention map from a single block's weights.

    Args:
        attn_weights  : Tensor (1, num_heads, seq_len, seq_len)
                        — attention from one block, single image
        patch_grid_size : int — number of patches per side (e.g. 8 for 32/4)

    Returns:
        attn_map : np.ndarray (patch_grid_size, patch_grid_size)
                   — mean-over-heads attention of [CLS] → patches, normalised
    """
    # Average over attention heads: (1, num_heads, seq_len, seq_len) → (seq_len, seq_len)
    attn = attn_weights[0].mean(dim=0)  # (seq_len, seq_len)

    # Row 0 is the [CLS] token query; columns 1: are the patch key tokens
    # (column 0 is [CLS] attending to itself — we drop it)
    cls_attn = attn[0, 1:]  # (num_patches,)

    # Normalise to [0, 1] for display
    cls_attn = cls_attn - cls_attn.min()
    if cls_attn.max() > 0:
        cls_attn = cls_attn / cls_attn.max()

    # Reshape to 2D patch grid
    attn_map = cls_attn.reshape(patch_grid_size, patch_grid_size).cpu().numpy()
    return attn_map


def visualize_sample(model, image, label, device, layer_idx: int, patch_size: int = 4,
                     image_size: int = 32, save_path: str = None):
    """
    Produce a 3-panel figure: original image | attention heatmap | overlay.

    Args:
        model      : trained ViT
        image      : Tensor (C, H, W) — normalised image
        label      : int — true class index
        device     : torch.device
        layer_idx  : int — which Transformer block's attention to visualise
        patch_size : int
        image_size : int
        save_path  : str or None — if given, save to this path
    """
    model.eval()
    patch_grid = image_size // patch_size  # e.g. 32 // 4 = 8

    with torch.no_grad():
        x = image.unsqueeze(0).to(device)  # (1, C, H, W)
        logits, attention_maps = model(x, return_attention=True)
        pred = logits.argmax(dim=1).item()

    # attention_maps is a list of (1, h, N+1, N+1) tensors, one per block
    attn_weights = attention_maps[layer_idx]  # (1, h, N+1, N+1)
    attn_map = get_attention_map(attn_weights, patch_grid)  # (grid, grid)

    # Upsample attention map to full image resolution for overlay
    attn_upsampled = np.kron(attn_map, np.ones((patch_size, patch_size)))  # (H, W)

    # Original image for display
    img_display = unnormalise(image)  # (H, W, 3)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    true_label = CIFAR10_CLASSES[label]
    pred_label = CIFAR10_CLASSES[pred]
    color = "green" if pred == label else "red"
    fig.suptitle(
        f"True: {true_label}  |  Predicted: {pred_label}  |  Block {layer_idx+1}",
        fontsize=12, color=color
    )

    # Panel 1: Original image
    axes[0].imshow(img_display)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Panel 2: Attention map (patch grid, not upsampled)
    im = axes[1].imshow(attn_map, cmap="inferno", interpolation="nearest")
    axes[1].set_title(f"[CLS] Attention\n(averaged over {attn_weights.shape[1]} heads)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Overlay
    axes[2].imshow(img_display)
    axes[2].imshow(attn_upsampled, cmap="inferno", alpha=0.5, interpolation="bilinear")
    axes[2].set_title("Attention Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {save_path}")
    else:
        plt.show()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualise ViT attention maps")
    parser.add_argument("--num_samples", type=int, default=8,
                        help="Number of test images to visualise")
    parser.add_argument("--layer", type=int, default=-1,
                        help="Which Transformer block layer to visualise (-1 = last)")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        "cpu" if args.device == "auto" else args.device
    )

    # Load model
    model = build_small_vit().to(device)
    ckpt = "results/checkpoints/vit.pth"
    if not os.path.exists(ckpt):
        print(f"ERROR: Checkpoint not found at {ckpt}")
        print("Run `python -m src.train` first to train the model.")
        return

    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"Loaded ViT checkpoint from {ckpt}")

    num_layers = len(model.blocks)
    layer_idx = args.layer if args.layer >= 0 else num_layers + args.layer
    print(f"Visualising block {layer_idx + 1}/{num_layers}")

    # Test dataset (no augmentation, just normalise)
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    ds = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=tf)

    # Sample evenly across classes for variety
    samples_per_class = max(1, args.num_samples // 10)
    selected = []
    class_counts = {i: 0 for i in range(10)}
    for img, label in ds:
        if class_counts[label] < samples_per_class and len(selected) < args.num_samples:
            selected.append((img, label))
            class_counts[label] += 1
        if len(selected) >= args.num_samples:
            break

    print(f"Visualising {len(selected)} samples...")
    for i, (image, label) in enumerate(selected):
        save_path = f"results/attention_visualizations/sample_{i:03d}_{CIFAR10_CLASSES[label]}.png"
        visualize_sample(
            model=model,
            image=image,
            label=label,
            device=device,
            layer_idx=layer_idx,
            patch_size=4,
            image_size=32,
            save_path=save_path,
        )

    # Also produce a multi-layer comparison for one image
    print("\nGenerating multi-layer attention comparison for one sample...")
    img, label = selected[0]
    model.eval()
    with torch.no_grad():
        x = img.unsqueeze(0).to(device)
        _, attention_maps = model(x, return_attention=True)

    patch_grid = 32 // 4
    fig, axes = plt.subplots(2, num_layers // 2, figsize=(3 * (num_layers // 2), 7))
    axes = axes.flat
    fig.suptitle(
        f"[CLS] Attention Across All {num_layers} Layers — {CIFAR10_CLASSES[label]}",
        fontsize=12
    )
    for l_idx, ax in enumerate(axes):
        attn_map = get_attention_map(attention_maps[l_idx], patch_grid)
        ax.imshow(attn_map, cmap="inferno", interpolation="nearest")
        ax.set_title(f"Block {l_idx + 1}")
        ax.axis("off")
    plt.tight_layout()
    multi_path = f"results/attention_visualizations/all_layers_{CIFAR10_CLASSES[label]}.png"
    os.makedirs("results/attention_visualizations", exist_ok=True)
    plt.savefig(multi_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Multi-layer comparison saved → {multi_path}")
    print("\nDone. All attention visualizations saved to results/attention_visualizations/")


if __name__ == "__main__":
    main()
