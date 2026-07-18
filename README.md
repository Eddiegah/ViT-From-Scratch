# ViT From Scratch

A Vision Transformer (ViT) implemented entirely from scratch in PyTorch — patch embedding, positional encoding, and multi-head self-attention all built manually — trained on CIFAR-10 and benchmarked against a hand-built CNN baseline.

The goal is genuine understanding of the architecture, not just a working model. Code clarity and mathematical correctness are prioritised over final accuracy.

---

## Table of Contents

1. [Compute Requirements](#compute-requirements)
2. [Setup](#setup)
3. [Project Structure](#project-structure)
4. [How ViT Differs from CNNs](#how-vit-differs-from-cnns)
5. [Default Configuration](#default-configuration)
6. [Running the Project](#running-the-project)
7. [Results](#results)
8. [Troubleshooting](#troubleshooting)
9. [Future Work](#future-work)

---

## Compute Requirements

**Read this before starting a training run.**

Vision Transformers are data- and compute-hungry compared to CNNs because they have no built-in spatial priors — the model must learn that nearby pixels are related from data alone, via attention. CNNs assume this by construction (the convolution kernel looks at a local neighbourhood). This difference matters a lot at small scale.

| Environment | Approx. time per epoch (per model) | Recommended? |
|---|---|---|
| CPU (modern laptop) | 3–6 minutes | Feasible for a quick run |
| GPU (free Colab T4) | ~20 seconds | **Recommended** |
| GPU (local NVIDIA) | ~15–30 seconds | Ideal |

This project targets **CIFAR-10** (50,000 training images, 32×32 pixels, 10 classes) with a **small ViT configuration** (~1.8 M parameters). This is deliberately not ImageNet scale — the goal is to understand the architecture at manageable compute cost.

**Expected accuracy** (30 epochs, default config):
- CNN baseline: ~82–87%
- ViT from scratch: ~70–80%

The ViT underperforming the CNN at this scale is **expected and well-documented** — not a bug. See [How ViT Differs from CNNs](#how-vit-differs-from-cnns) for the explanation.

---

## Setup

### Prerequisites

- Python 3.9–3.12 (Python 3.11 recommended on Windows)
- Windows 10/11 (primary supported environment), macOS, or Linux

### 1. Check your Python version

```cmd
py -0
```

You should see a `3.9`, `3.10`, `3.11`, or `3.12` entry. If not, download Python 3.11 from [python.org](https://www.python.org/downloads/).

### 2. Create and activate a virtual environment

```cmd
py -3.11 -m venv venv
venv\Scripts\activate
```

Your prompt should now start with `(venv)`.

### 3. Install dependencies

```cmd
pip install -r requirements.txt
```

### 4. Verify the installation

```cmd
python -c "import torch; import torchvision; import matplotlib; print('CUDA available:', torch.cuda.is_available()); print('All imports OK')"
```

You should see:
```
CUDA available: True   # (or False if no NVIDIA GPU — that's fine)
All imports OK
```

If this fails with a **DLL load error**, see [Troubleshooting](#troubleshooting).

---

## Project Structure

```
vit-from-scratch/
├── src/
│   ├── patch_embedding.py      # Image → patch sequence + [CLS] token
│   ├── positional_encoding.py  # Learnable positional embeddings
│   ├── attention.py            # Multi-head self-attention (from raw math)
│   ├── transformer_block.py    # MHSA + MLP + LayerNorm + residuals
│   ├── vit.py                  # Full ViT model
│   ├── cnn_baseline.py         # Small CNN for comparison
│   ├── train.py                # Training script (both models)
│   ├── evaluate.py             # Test metrics + comparison report
│   └── visualize_attention.py  # Attention map visualizations
├── tests/
│   └── test_attention.py       # Unit tests for attention mechanism
├── results/
│   ├── training_curves.png          # Generated after training
│   ├── attention_visualizations/    # Generated after visualize_attention.py
│   └── comparison_report.md         # Generated after evaluate.py
├── notebooks/
│   └── colab_version.ipynb     # GPU-accelerated Colab alternative
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How ViT Differs from CNNs

This is the conceptual core of the project.

### The CNN's inductive bias

A convolutional layer encodes two powerful assumptions about images:

1. **Local spatial structure**: The kernel operates on a small neighbourhood (e.g. 3×3 pixels). Nearby pixels are processed together; distant pixels are not. This reflects a genuine property of natural images — edges, textures, and objects are all locally coherent.

2. **Translation equivariance**: The same filter is applied at every spatial position. A "cat ear detector" trained on the top-left will generalise to the bottom-right without extra learning.

These are very good priors for images. The CNN doesn't need to *discover* that local structure matters — it's baked into the architecture. This is why CNNs can learn effectively from relatively small datasets.

### The ViT's lack of priors

The core operation in a ViT is self-attention:

```
Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
```

This computes a weighted mixture of all values, where the weights come from comparing every query token against every key token. **Every patch attends to every other patch equally at initialisation.** There is no locality assumption.

The only spatial information the ViT receives is through learned positional embeddings added to the patch sequence. The model must learn from data that patch 0 (top-left) is spatially close to patch 1 (top-second-from-left), and that this proximity is meaningful.

With 50,000 CIFAR-10 training images, the ViT simply doesn't get enough examples to fully learn the spatial relationships that a CNN assumes from the start. This is why, at small scale, the CNN wins.

### At large scale, the story flips

The original ViT paper (Dosovitskiy et al., "An Image is Worth 16x16 Words", 2020) found that ViTs trained on ImageNet alone (~1.2 M images) underperformed CNNs. But when pre-trained on JFT-300M (300 million images), ViTs matched or exceeded the best CNNs.

The intuition: given enough data, the ViT learns spatial relationships so thoroughly that the CNN's inductive bias becomes a constraint rather than a head start. The ViT's more general attention mechanism can then find patterns a CNN would miss.

---

## Default Configuration

| Hyperparameter | Value | Notes |
|---|---|---|
| Image size | 32×32 | CIFAR-10 native resolution |
| Patch size | 4×4 | → 64 patches per image |
| Sequence length | 65 | 64 patches + 1 [CLS] token |
| Embedding dim (D) | 128 | |
| Attention heads | 4 | Head dim = 32 |
| Transformer layers | 6 | |
| MLP hidden dim | 256 | 2× embed_dim |
| Attention dropout | 0.1 | |
| MLP dropout | 0.1 | |
| Parameters | ~1.8 M | |

Training config (defaults):

| Setting | Value |
|---|---|
| Epochs | 30 |
| Batch size | 64 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Learning rate | 3e-4 |
| LR schedule | Cosine annealing |
| Augmentation | RandomHorizontalFlip + RandomCrop(32, padding=4) |

---

## Running the Project

All commands assume you're in the project root with the virtual environment activated.

### Run unit tests first (recommended)

```cmd
python -m pytest tests/ -v
```

All tests should pass before training. This catches subtle bugs in the attention implementation.

### Train both models

```cmd
python -m src.train
```

With custom settings:

```cmd
python -m src.train --epochs 30 --batch_size 64 --lr 3e-4
```

CIFAR-10 (~170 MB) downloads automatically to `data/` on first run.

Outputs:
- `results/checkpoints/vit.pth` — trained ViT weights
- `results/checkpoints/cnn.pth` — trained CNN weights
- `results/training_curves.png` — loss/accuracy comparison plots
- `results/vit_history.json`, `results/cnn_history.json` — raw training logs

### Generate attention visualizations

```cmd
python -m src.visualize_attention --num_samples 8
```

Outputs to `results/attention_visualizations/` — one image per sample showing original image, attention heatmap, and overlay.

### Generate comparison report

```cmd
python -m src.evaluate
```

Outputs `results/comparison_report.md` with final test accuracy, parameter counts, training times, and a plain-language explanation of the results.

### Use Colab for GPU training

Open `notebooks/colab_version.ipynb` in Google Colab. Set the runtime to **T4 GPU** (`Runtime → Change runtime type`), then run all cells. Training takes ~10–15 minutes instead of hours.

---

## Results

*This section is populated after running `src/evaluate.py`.*

See `results/comparison_report.md` for the full generated report, including honest interpretation of the results and why they came out the way they did.

---

## Troubleshooting

### DLL load error on Windows (torch import fails)

Install the Microsoft Visual C++ 2015–2022 Redistributable:
- [vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe) (64-bit — required)
- [vc_redist.x86.exe](https://aka.ms/vs/17/release/vc_redist.x86.exe) (32-bit — also install this)

Restart your machine and retry.

### `ModuleNotFoundError: No module named 'src'`

Run commands from the project root directory (the folder containing `src/`), not from inside `src/`. Always use `python -m src.train` (module form), not `python src/train.py`.

### Training is extremely slow

This is expected on CPU. Options:
- Use the Colab notebook (`notebooks/colab_version.ipynb`) for free GPU access.
- Reduce epochs: `--epochs 10` for a quick sanity check.
- Reduce batch size if memory is tight: `--batch_size 32`.

### CIFAR-10 download fails

If the automatic download fails, download manually:
1. Go to https://www.cs.toronto.edu/~kriz/cifar.html
2. Download `CIFAR-10 python version`
3. Extract to `data/cifar-10-batches-py/`

### `num_workers` warning on Windows

Windows sometimes shows warnings about DataLoader workers. If you see errors, add `--num_workers 0` to the train command.

### OneDrive path issues

If your project is inside a OneDrive-synced folder, move it to a non-synced path (e.g. `C:\Projects\vit-from-scratch`) before running. OneDrive file locking can interfere with PyTorch's checkpoint saving and dataset caching.

---

## Future Work

- **Scale up**: Larger patch size (16×16), deeper encoder (12 layers), larger embedding dim (768) — approaching ViT-Base, but requires significantly more compute and data.
- **Data augmentation**: CutMix, MixUp, RandAugment — these disproportionately help ViTs and can close a significant portion of the gap with CNNs at small scale.
- **Transfer learning**: Pre-train on a larger dataset (e.g. ImageNet-1K), then fine-tune on CIFAR-10. This is the standard approach for getting strong ViT performance without massive compute.
- **Sinusoidal positional encoding**: Compare learnable vs. fixed sinusoidal embeddings.
- **DeiT-style distillation**: Train the ViT with a CNN teacher (knowledge distillation), which is one of the techniques that makes ViTs competitive at smaller scales.

---

## References

- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (2020) — [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Vaswani et al., "Attention Is All You Need" (2017) — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Touvron et al., "Training data-efficient image transformers & distillation through attention" (DeiT, 2021) — [arXiv:2012.12877](https://arxiv.org/abs/2012.12877)
