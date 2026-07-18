"""
train.py
========
Training script: trains both the ViT and the CNN baseline on CIFAR-10 with
identical training budgets, then saves training curves and model checkpoints.

Usage
-----
    python -m src.train [--epochs N] [--batch_size N] [--lr LR] [--device cpu|cuda]

Defaults are conservative for CPU training:
    --epochs     30
    --batch_size 64
    --lr         3e-4
    --device     auto (uses CUDA if available, else CPU)

Expected runtime (approximate)
-------------------------------
  CPU (modern laptop):    ~3–6 min per epoch → ~2–3 hours for 30 epochs per model
  GPU (e.g. Colab T4):    ~20 sec per epoch  → ~10 min for 30 epochs per model
"""

import argparse
import os
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.vit import build_small_vit
from src.cnn_baseline import CNNBaseline


# ── Data transforms ───────────────────────────────────────────────────────────

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def get_transforms(augment: bool = True):
    """
    Returns train and test transforms for CIFAR-10.
    Augmentation: random horizontal flip + crop (standard practice).
    """
    train_tf = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ] if augment else [
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    return train_tf, test_tf


def get_dataloaders(batch_size: int, num_workers: int = 2):
    """Download CIFAR-10 (if not cached) and return train/test DataLoaders."""
    train_tf, test_tf = get_transforms(augment=True)

    train_ds = torchvision.datasets.CIFAR10(
        root="data", train=True,  download=True, transform=train_tf
    )
    test_ds = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=test_tf
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


# ── Per-epoch training and evaluation ────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, desc="Train"):
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, desc="Eval"):
    """Run one evaluation pass. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# ── Training loop for one model ───────────────────────────────────────────────

def train_model(model, train_loader, test_loader, args, model_name: str, device):
    """
    Full training loop: trains `model` for args.epochs epochs.

    Returns a dict with training history and training time.
    """
    criterion = nn.CrossEntropyLoss()

    # AdamW with cosine annealing LR schedule
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    print(f"\n{'='*60}")
    print(f"  Training {model_name}")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}  |  Batch size: {args.batch_size}  |  LR: {args.lr}")
    print(f"{'='*60}")

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            desc=f"[{model_name}] E{epoch:03d} Train"
        )
        val_loss, val_acc = eval_epoch(
            model, test_loader, criterion, device,
            desc=f"[{model_name}] E{epoch:03d} Val  "
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

    elapsed = time.time() - start_time
    history["training_time_seconds"] = elapsed
    print(f"\n  Done. Training time: {elapsed/60:.1f} min\n")

    # Save checkpoint
    os.makedirs("results/checkpoints", exist_ok=True)
    ckpt_path = f"results/checkpoints/{model_name.lower().replace(' ', '_')}.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Checkpoint saved → {ckpt_path}")

    return history


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_training_curves(vit_history, cnn_history, save_path="results/training_curves.png"):
    """Save a 2×2 grid comparing ViT vs CNN training curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(1, len(vit_history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ViT vs CNN Baseline — CIFAR-10 Training Curves", fontsize=14, fontweight="bold")

    titles = [
        ("Train Loss",  "train_loss", "Loss"),
        ("Val Loss",    "val_loss",   "Loss"),
        ("Train Acc",   "train_acc",  "Accuracy"),
        ("Val Acc",     "val_acc",    "Accuracy"),
    ]

    for ax, (title, key, ylabel) in zip(axes.flat, titles):
        ax.plot(epochs, vit_history[key], label="ViT",      linewidth=2, color="#e07b39")
        ax.plot(epochs, cnn_history[key], label="CNN",      linewidth=2, color="#3a7ebf")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves saved → {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT and CNN on CIFAR-10")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--device",     type=str,   default="auto",
                        help="'auto', 'cpu', or 'cuda'")
    parser.add_argument("--num_workers", type=int,  default=2)
    return parser.parse_args()


def main():
    args = parse_args()

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\nUsing device: {device}")
    if device.type == "cpu":
        print(
            "  NOTE: CPU training will be slow. Expect ~3–6 min/epoch per model.\n"
            "  Consider using the Colab notebook (notebooks/colab_version.ipynb)\n"
            "  for GPU-accelerated training.\n"
        )

    # Data
    train_loader, test_loader = get_dataloaders(args.batch_size, args.num_workers)
    print(f"  CIFAR-10 loaded: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test")

    # Models
    vit = build_small_vit(num_classes=10).to(device)
    cnn = CNNBaseline(num_classes=10).to(device)

    # Train ViT
    vit_history = train_model(vit, train_loader, test_loader, args, "ViT", device)

    # Train CNN
    cnn_history = train_model(cnn, train_loader, test_loader, args, "CNN", device)

    # Plot
    plot_training_curves(vit_history, cnn_history)

    # Save raw histories for evaluate.py
    os.makedirs("results", exist_ok=True)
    with open("results/vit_history.json", "w") as f:
        json.dump(vit_history, f, indent=2)
    with open("results/cnn_history.json", "w") as f:
        json.dump(cnn_history, f, indent=2)

    print("\nTraining complete. Run `python -m src.evaluate` to generate the comparison report.")


if __name__ == "__main__":
    main()
