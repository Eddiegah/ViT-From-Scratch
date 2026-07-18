<div align="center">

# 🔬 ViT From Scratch

### Vision Transformer — Built from the Math Up

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-00B4D8?style=for-the-badge&logo=databricks&logoColor=white)](https://www.cs.toronto.edu/~kriz/cifar.html)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-25%20Passing-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey?style=for-the-badge&logo=windows&logoColor=white)](https://github.com/Eddiegah/ViT-From-Scratch)

<br/>

**Multi-head self-attention, patch embedding, and positional encoding — all implemented manually, without `nn.MultiheadAttention` or pre-built transformer blocks.** Trained on CIFAR-10, benchmarked against a hand-built CNN baseline.

*The goal is genuine understanding — not just a working model.*

<br/>

[🚀 Quick Start](#setup) · [🏗 Architecture](#architecture) · [📊 Results](#results) · [⚡ Colab (GPU)](#google-colab-gpu) · [🔍 Attention Viz](#attention-visualization)

</div>

---

## ✨ What's Built Here

This project implements every component of the Vision Transformer from scratch — each file maps directly to a piece of the original paper's math:

| File | What it implements | Key insight |
|---|---|---|
| `src/patch_embedding.py` | Image → patch sequence + [CLS] token | Strided Conv2d ≡ flatten + linear project |
| `src/positional_encoding.py` | Learnable 1D positional embeddings | Attention is permutation-invariant without this |
| `src/attention.py` | Scaled dot-product + multi-head attention | `Q·Kᵀ / √d_k` → softmax → weight `V` |
| `src/transformer_block.py` | Pre-LN encoder: MHSA + MLP + residuals | Why pre-norm stabilises training |
| `src/vit.py` | Full ViT model end-to-end | [CLS] token aggregates global context |
| `src/cnn_baseline.py` | CNN baseline (~2M params) | Inductive bias vs. learned spatial structure |
| `src/train.py` | AdamW + cosine LR + tqdm training loop | Identical budget for fair comparison |
| `src/evaluate.py` | Metrics + auto-generated comparison report | Honest interpretation of results |
| `src/visualize_attention.py` | [CLS] attention maps, all layers | *See what the model actually learned to look at* |

---

## 🏗 Architecture

### Vision Transformer (ViT)

```
Input Image (3×32×32)
        │
        ▼
┌──────────────────────┐
│   Patch Embedding    │  32×32 → 64 patches of 4×4
│   + [CLS] Token      │  + 1 learnable class token
└──────────────────────┘
        │
        ▼
┌──────────────────────┐
│  Positional Encoding │  Inject spatial location into tokens
└──────────────────────┘
        │
        ▼  ×6 blocks
┌──────────────────────┐
│  LayerNorm           │
│  ↓                   │
│  Multi-Head          │  Q = XW_Q,  K = XW_K,  V = XW_V
│  Self-Attention      │  Attn = softmax(QKᵀ/√d_k) · V
│  ↓                   │
│  Residual (+x)       │
│                      │
│  LayerNorm           │
│  ↓                   │
│  MLP (GELU)          │  Linear → GELU → Dropout → Linear
│  ↓                   │
│  Residual (+x)       │
└──────────────────────┘
        │
        ▼
  [CLS] Token Output
        │
        ▼
┌──────────────────────┐
│  LayerNorm → Linear  │  → 10 class logits
└──────────────────────┘
```

**Default CIFAR-10 configuration** (~811K parameters):

| Hyperparameter | Value |
|---|---|
| Image size | 32×32 |
| Patch size | 4×4 → 64 patches |
| Sequence length | 65 (64 patches + [CLS]) |
| Embedding dim (D) | 128 |
| Attention heads | 4 (head dim = 32) |
| Transformer layers | 6 |
| MLP hidden dim | 256 |

### The Attention Formula (from code)

```python
# src/attention.py — every line maps to the math

Q = self.W_q(x)                              # Q = X W_Q
K = self.W_k(x)                              # K = X W_K
V = self.W_v(x)                              # V = X W_V

scores = torch.matmul(Q, K.transpose(-2,-1)) / self.scale   # QKᵀ / √d_k
attn   = F.softmax(scores, dim=-1)           # softmax over keys
out    = torch.matmul(attn, V)               # weighted sum of values
out    = self.W_o(merge_heads(out))          # output projection
```

---

## 🧠 Why ViT Needs More Data Than CNN

This is the conceptual core of the project.

### The CNN's built-in advantage

```
Conv kernel: looks at a 3×3 neighbourhood only
             → local spatial structure assumed by architecture
             → translation equivariant by construction
```

A CNN **assumes** nearby pixels are related. It doesn't have to learn this. That's a powerful prior for natural images — and it's why CNNs learn so effectively from small datasets.

### The ViT's blank slate

```python
# Every patch can attend to every other patch equally at initialisation
scores = Q @ K.T / sqrt(d_k)    # all pairs, all positions, equal footing
attn   = softmax(scores)        # model must learn which pairs matter
```

Self-attention is **permutation-invariant** — shuffle all patches and the math still works. The ViT must discover from data that the top-left patch is spatially adjacent to the ones next to it. On 50,000 CIFAR-10 images, this is hard. With 300 million images (Google's JFT-300M), the story flips.

| Scale | Who wins | Why |
|---|---|---|
| CIFAR-10 (50K images) | CNN | Inductive bias > learned structure |
| ImageNet (1.2M images) | Roughly tied | ViT catches up |
| JFT-300M (300M images) | ViT | Learned structure > baked-in bias |

---

## 📊 Results

> Results are generated after a full training run via `python -m src.evaluate`.
> See [`results/comparison_report.md`](results/comparison_report.md) for the full auto-generated report.

| Metric | ViT (from scratch) | CNN Baseline |
|---|---|---|
| Test Accuracy | *run training* | *run training* |
| Parameters | ~811K | ~2.0M |
| Training Time | *see report* | *see report* |

**Training curves** (generated by `src/train.py`):

> `results/training_curves.png` — ViT vs CNN loss and accuracy over 30 epochs.

**Attention visualizations** (generated by `src/visualize_attention.py`):

> `results/attention_visualizations/` — what patches the [CLS] token attends to, per layer.

---

## 🔍 Attention Visualization

One of the most interesting outputs of this project is being able to *see* what the model learned to focus on. After training, `src/visualize_attention.py` extracts the [CLS] token's attention weights and overlays them on the input image.

```
Original Image | Attention Heatmap | Overlay
     🐦        |    🔥🔥..🔥.     |   combined
```

The [CLS] token has no fixed spatial location — it attends to the patches that are most informative for classification. By the final layer, it should be attending to the discriminative regions of the image.

Run it after training:

```cmd
python -m src.visualize_attention --num_samples 8
```

---

## ⚙️ Setup

### Prerequisites

- Python 3.9–3.12 (Python 3.11 recommended)
- Windows 10/11, macOS, or Linux

### 1. Check your Python

```cmd
py -0
```

You need a `3.9`, `3.10`, `3.11`, or `3.12` entry. Get it at [python.org](https://www.python.org/downloads/).

### 2. Create a virtual environment

```cmd
py -3.11 -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```cmd
pip install -r requirements.txt
```

### 4. Verify

```cmd
python -c "import torch; import torchvision; print('CUDA:', torch.cuda.is_available()); print('All OK')"
```

---

## 🚀 Running

All commands from the project root with venv activated.

### Run unit tests (do this first)

```cmd
python -m pytest tests/ -v
```

All 25 tests should pass before training. They verify attention weights sum to 1, shapes are correct, gradients flow, and more.

### Train both models

```cmd
python -m src.train
```

CIFAR-10 (~170 MB) downloads automatically on first run.

**Optional flags:**

```cmd
python -m src.train --epochs 30 --batch_size 64 --lr 3e-4
```

**Outputs:**
- `results/checkpoints/vit.pth` — trained ViT
- `results/checkpoints/cnn.pth` — trained CNN
- `results/training_curves.png`
- `results/vit_history.json`, `results/cnn_history.json`

### Generate the comparison report

```cmd
python -m src.evaluate
```

Writes `results/comparison_report.md` with accuracy, parameter counts, training times, and a plain-language explanation.

### Visualize attention

```cmd
python -m src.visualize_attention --num_samples 8
```

Saves attention overlay images to `results/attention_visualizations/`.

---

## ⚡ Google Colab (GPU)

CPU training takes **3–5 hours**. The Colab notebook does it in **~15 minutes** on a free T4 GPU.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/colab_version.ipynb)

1. Open `notebooks/colab_version.ipynb` in Google Colab
2. Set runtime: `Runtime → Change runtime type → T4 GPU`
3. Run all cells — uploads project, trains, downloads results

---

## 💻 Compute Requirements

| Environment | Time per epoch (per model) | Recommended? |
|---|---|---|
| CPU (modern laptop) | 5–10 min | Feasible, slow |
| GPU (free Colab T4) | ~20 sec | ✅ Recommended |
| GPU (local NVIDIA) | ~15–30 sec | ✅ Best |

This project targets CIFAR-10 scale only. Scaling to ImageNet would require significantly more compute and is a documented future step.

---

## 🧪 Unit Tests

25 tests in `tests/test_attention.py` covering:

```
✅ Attention output shapes
✅ Attention weights sum to 1 (softmax correctness)
✅ No negative attention weights
✅ No NaN values in forward pass
✅ Scale factor = 1/√d_k
✅ Gradient flow through full model
✅ Batch independence (no cross-contamination)
✅ Patch embedding output shapes
✅ CLS token is prepended
✅ Invalid patch sizes raise AssertionError
✅ Positional encoding adds non-zero values
✅ Transformer block residual connection
✅ Full ViT output shape (B, num_classes)
✅ Attention maps returned per block
✅ Parameter count in expected range
✅ Different images → different outputs
```

---

## 🔧 Troubleshooting

<details>
<summary><b>DLL load error on Windows (torch import fails)</b></summary>

Install Microsoft Visual C++ 2015–2022 Redistributable:
- [vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe) — 64-bit (required)
- [vc_redist.x86.exe](https://aka.ms/vs/17/release/vc_redist.x86.exe) — 32-bit (also install)

Restart and retry.
</details>

<details>
<summary><b>ModuleNotFoundError: No module named 'src'</b></summary>

Run from the project root using module form:
```cmd
python -m src.train        ✅
python src/train.py        ❌
```
</details>

<details>
<summary><b>Training is very slow</b></summary>

Use the Colab notebook for free GPU training (~15 min vs ~5 hrs). Or reduce epochs for a quick sanity check:
```cmd
python -m src.train --epochs 5
```
</details>

<details>
<summary><b>num_workers warning on Windows</b></summary>

Add `--num_workers 0` to the training command:
```cmd
python -m src.train --num_workers 0
```
</details>

<details>
<summary><b>OneDrive path issues</b></summary>

Move the project to a non-OneDrive path (e.g. `C:\Projects\ViT-From-Scratch`). OneDrive file locking can interfere with checkpoint saving.
</details>

---

## 🗺 Project Structure

```
vit-from-scratch/
├── src/
│   ├── patch_embedding.py      # Image → patch sequence + [CLS] token
│   ├── positional_encoding.py  # Learnable 1D positional embeddings
│   ├── attention.py            # Multi-head self-attention (raw math)
│   ├── transformer_block.py    # Pre-LN encoder block
│   ├── vit.py                  # Full ViT model
│   ├── cnn_baseline.py         # CNN comparison baseline
│   ├── train.py                # Training loop (both models)
│   ├── evaluate.py             # Metrics + comparison report
│   └── visualize_attention.py  # Attention map visualizations
├── tests/
│   └── test_attention.py       # 25 unit tests
├── results/
│   ├── training_curves.png          # Generated after training
│   ├── attention_visualizations/    # Generated after visualization
│   └── comparison_report.md         # Generated after evaluation
├── notebooks/
│   └── colab_version.ipynb     # GPU Colab alternative
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔮 Future Work

- **Better augmentation** (CutMix, MixUp, RandAugment) — disproportionately helps ViTs at small scale
- **DeiT-style distillation** — train ViT with CNN teacher, closes the gap on small datasets
- **Sinusoidal vs learnable positional encoding** — ablation study
- **ViT-Base config** (12 layers, 768 dim) — requires ImageNet-scale compute
- **Transfer learning** — pre-train on ImageNet-1K, fine-tune on CIFAR-10

---

## 📚 References

- Dosovitskiy et al., **"An Image is Worth 16x16 Words"** (2020) — [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Vaswani et al., **"Attention Is All You Need"** (2017) — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Touvron et al., **"Training data-efficient image transformers"** (DeiT, 2021) — [arXiv:2012.12877](https://arxiv.org/abs/2012.12877)

---

<div align="center">

Built with 🧠 and PyTorch — no shortcuts, all math.

</div>
