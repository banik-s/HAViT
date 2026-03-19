# HAViT — Historical Attention Vision Transformer

A research framework for training and evaluating Vision Transformer (ViT) variants with novel attention mechanisms on image classification benchmarks.

---

## Overview

Vision Transformers have excelled in computer vision, but their attention mechanisms operate **independently across layers**, limiting information flow and feature learning.

This work proposes a **cross-layer attention propagation** method that preserves and integrates historical attention matrices across encoder layers — enabling progressive refinement of attention patterns throughout the transformer hierarchy with **minimal architectural changes**.

Key results from our paper:

| Dataset | Baseline ViT | HAViT | Δ |
|---|---|---|---|
| CIFAR-100 | 75.74% | 77.07% | **+1.33%** |
| Tiny ImageNet | 57.82% | 59.07% | **+1.25%** |

Cross-architecture validation yields similar gains (e.g., CaiT **+1.01%**). For the full methodology, ablations, and analysis → *see our paper*.

---

## Repository Structure

```
HAViT/
├── main.py                  # Main training & evaluation entry point
├── models/
│   ├── create_model.py      # Model factory — maps --model name to class
│   ├── vitlucidrains.py     # Baseline ViT (lucidrains-style)
│   ├── vitlucidrains_mod_ver1.py  # HAViT-v1: inter-layer attention blending
│   ├── vitlucidrains_mod_ver2.py  # ... (further variants)
│   ├── ...
│   ├── pit*.py              # Pooling-in-Transformer variants
│   ├── cait*.py             # Class-Attention in Image Transformers variants
│   └── vit_original.py      # Standard ViT baseline
├── utils/
│   ├── dataloader.py        # Dataset loading & normalization statistics
│   ├── autoaug.py           # Auto-augmentation policies (CIFAR, SVHN, ImageNet)
│   ├── transforms.py        # Custom transforms
│   ├── random_erasing.py    # Random Erasing augmentation
│   ├── mix.py               # CutMix & MixUp implementations
│   ├── sampler.py           # Repeated Augmentation (RA) sampler
│   ├── scheduler.py         # Learning rate scheduler utilities
│   ├── cosine_annealing_with_warmup.py
│   ├── losses.py            # Label Smoothing Cross-Entropy loss
│   ├── logger_dict.py       # CSV/console training logger
│   ├── print_progress.py    # Progress bar
│   └── training_functions.py # Top-1 accuracy computation
└── Optimizers/
    ├── AdaBelief.py         # AdaBelief optimizer
    ├── Radam.py             # Rectified Adam
    ├── diffGrad.py          # diffGrad optimizer
    ├── AdamNorm.py          # Adam with gradient norm
    └── ...                  # Inject / Norm variants of each optimizer
```

---

## Supported Datasets

| Flag Value | Dataset               | Classes | Image Size |
|------------|-----------------------|---------|------------|
| `CIFAR10`  | CIFAR-10              | 10      | 32×32      |
| `CIFAR100` | CIFAR-100             | 100     | 32×32      |
| `T-IMNET`  | Tiny ImageNet-200     | 200     | 64×64      |

CIFAR-10, CIFAR-100, and SVHN are downloaded automatically via `torchvision`.  
**Tiny ImageNet must be downloaded and placed manually** (see setup below).

---

## Supported Models (`--model`)

| Model Name | Description |
|---|---|
| `vitlucidrains` | Baseline lucidrains-style ViT |
| `vitlucidrains_mod_ver1` – `ver13` | HAViT model variants with attention modifications |
| `vit_original` | Standard ViT-Small |
| `pit` / `pit_mod_ver1` | Pooling-in-Transformer |
| `cait` / `cait_mod_ver1` | Class-Attention in Image Transformers |

---

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.10 (with CUDA recommended)

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

### CIFAR-10 / CIFAR-100
These are downloaded **automatically** into `./datasets/` on first run.

### Tiny ImageNet-200 (`T-IMNET`)
Download manually and place it under your dataset directory:

```bash
# From the project root
mkdir -p ./datasets/tiny-imagenet-200
cd ./datasets/tiny-imagenet-200

# Download
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
mv tiny-imagenet-200/* .
rmdir tiny-imagenet-200
```

The expected directory layout after extraction:

```
datasets/
└── tiny-imagenet-200/
    ├── train/
    │   ├── n01443537/
    │   │   └── images/
    │   └── ...
    └── val/
        ├── n01443537/
        │   └── images/
        └── ...
```

> **Important:** The `val/` folder in the raw Tiny ImageNet download uses a flat layout with a `val_annotations.txt` file. You need to restructure it into class subfolders. A helper script is provided below.

<details>
<summary><b>Script to restructure the val/ directory</b></summary>

```python
# run from datasets/tiny-imagenet-200/
import os, shutil

val_dir = 'val'
ann_file = os.path.join(val_dir, 'val_annotations.txt')

with open(ann_file) as f:
    for line in f:
        parts = line.strip().split('\t')
        img, cls = parts[0], parts[1]
        dest = os.path.join(val_dir, cls, 'images')
        os.makedirs(dest, exist_ok=True)
        shutil.move(os.path.join(val_dir, 'images', img), os.path.join(dest, img))
```
</details>

---

## How to Run

### Basic Usage

```bash
python3 main.py \
  --model <model_name> \
  --data_path <path_to_datasets> \
  --dataset <dataset_name> \
  --exp_name <experiment_name>
```

### Example — Reproduce Paper Results (HAViT-v1 on Tiny ImageNet)

```bash
python3 main.py \
  --model vitlucidrains_mod_ver1 \
  --data_path ./datasets/ \
  --dataset T-IMNET \
  --exp_name vitlucidrains_mod_ver1tiny
```

### Example — Train on CIFAR-100

```bash
python3 main.py \
  --model vitlucidrains_mod_ver1 \
  --data_path ./datasets/ \
  --dataset CIFAR100 \
  --exp_name havit_v1_cifar100 \
  --epochs 100 \
  --batch_size 128 \
  --lr 0.003
```


## Outputs

All outputs are saved under:

```
CheckpointsResults/<model_name>-<tag>-<dataset>-LR[<lr>]-Seed<seed>-<exp_name>/
├── checkpoint.pth   # Latest epoch checkpoint
├── best.pth         # Best validation accuracy checkpoint
├── history.csv      # Per-epoch training log
└── results.txt      # Epoch-by-epoch loss/accuracy/FLOPs text log

tensorboard/<experiment_name>/   # TensorBoard event files
```

### Monitoring with TensorBoard

```bash
tensorboard --logdir ./tensorboard
```
Then open `http://localhost:6006` in your browser.

---

## Resuming Training

If training was interrupted, resume from the last saved checkpoint:

```bash
python3 main.py \
  --model vitlucidrains_mod_ver1 \
  --data_path ./datasets/ \
  --dataset T-IMNET \
  --exp_name vitlucidrains_mod_ver1tiny \
  --resume
```

---

## Model Architecture — HAViT (`vitlucidrains_mod_ver1`)

The core idea is straightforward: at each encoder layer, the raw (pre-softmax) attention scores are **blended with the previous layer's attention history** before applying softmax:

```
Ã_{l+1} = α · A_{l+1}  +  (1 − α) · H_l
Attention_{l+1} = softmax(Ã_{l+1}) · V_{l+1}
```

where `H_l` is the historical attention matrix carried forward from layer `l`, and `α` is a fixed hyperparameter. The first layer's history `H_0` can be initialized randomly or with zeros — a design choice we analyze in depth in the paper.

This requires **no extra parameters** — only attention matrix storage and a blending operation.

For the full derivation, initialization analysis, optimal `α` search, and cross-architecture experiments → *read our paper*.

**Default config:**  dim=192, depth=9, heads=12, mlp_dim=512

---

## If you find any of our work helpful, please cite our paper in your publications. Thank you.