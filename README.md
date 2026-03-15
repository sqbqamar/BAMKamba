# BAMKamba

**Boundary-Aware Multi-Scale State-Space Networks with Adaptive Feature Calibration for Medical Image Segmentation**

Official PyTorch implementation.

## Architecture

BAMKamba is a U-shaped encoder-decoder segmentation framework that addresses limitations of SSM-based architectures through three complementary modules:

| Module | Location | Purpose |
|--------|----------|---------|
| **MSDC** | Bottleneck | Multi-scale context via parallel dilated convolutions (r=1,3,5,7) |
| **AFC** | Skip connections | Channel-spatial gating for adaptive encoder-decoder fusion |
| **BAS** | 3rd decoder stage | Sobel-based boundary supervision during training (zero inference cost) |

**Encoder:** Conv Block ×3 → LACE Block ×2 (with VSSM + local conv + channel attention)
**Decoder:** LACE Block ×2 → DConv Block ×3

Parameters: **6.18M** | GFlops: **2.34**

## Project Structure

```
BAMKamba/
├── models/
│   ├── __init__.py          # Package exports
│   ├── bamkamba.py          # Full BAMKamba architecture (Section 2.1)
│   ├── ss2d.py              # 2D Selective Scan with 4-direction scanning (Section 2.2.1)
│   ├── vssm.py              # Vision State-Space Module — dual-branch (Section 2.2.1)
│   ├── lace_block.py        # LACE block: VSSM + local conv + channel attention (Section 2.2, 2.5)
│   ├── msdc.py              # Multi-Scale Dilated Context module (Section 2.2)
│   ├── afc.py               # Adaptive Feature Calibration gates (Section 2.3)
│   ├── bas.py               # Boundary-Aware Supervision module (Section 2.4)
│   └── conv_blocks.py       # Encoder/decoder conv blocks, patch embedding
├── utils/
│   ├── __init__.py
│   └── utils.py             # Metrics (Dice, IoU, Acc, Spec) + dataset loader
├── train.py                 # Training script (Section 3.2 configuration)
└── README.md
```

## Results

### ISIC 2018 (Skin Lesion Segmentation)

| Model | Params (M) | GFlops | Dice (%) | IoU (%) |
|-------|-----------|--------|----------|---------|
| U-Net | 14.75 | 25.19 | 84.30 | 75.21 |
| ACC-UNet | 16.77 | 45.33 | 87.71 | 79.12 |
| H-vmunet | 6.43 | 1.48 | 86.61 | 76.38 |
| **BAMKamba** | **6.18** | **2.34** | **89.12** | **80.74** |

### BUSI (Breast Ultrasound Segmentation)

| Model | Params (M) | GFlops | Dice (%) | IoU (%) |
|-------|-----------|--------|----------|---------|
| U-Net | 14.75 | 25.19 | 76.80 | 68.30 |
| U-KAN | 9.38 | 6.89 | 80.01 | 70.12 |
| H-vmunet | 6.43 | 1.48 | 79.92 | 66.55 |
| **BAMKamba** | **6.18** | **2.34** | **83.45** | **72.38** |

### CVC-ClinicDB (Polyp Segmentation)

| Model | Params (M) | GFlops | Dice (%) | IoU (%) |
|-------|-----------|--------|----------|---------|
| U-Net | 14.75 | 25.19 | 82.41 | 73.62 |
| ACC-UNet | 16.77 | 45.33 | 86.12 | 77.88 |
| H-vmunet | 6.43 | 1.48 | 85.90 | 77.50 |
| **BAMKamba** | **6.18** | **2.34** | **88.27** | **80.14** |

## Requirements

```
torch >= 2.0
torchvision
numpy
Pillow
```

## Dataset Preparation

Organize each dataset as:

```
dataset_root/
├── images/
│   ├── img001.png
│   ├── img002.png
│   └── ...
└── masks/
    ├── img001.png
    ├── img002.png
    └── ...
```

**Datasets used:**
- [ISIC 2018](https://challenge.isic-archive.com/data/#2018) — 3,694 dermoscopic images (2,594 train / 100 val / 1,000 test)
- [BUSI](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) — 647 breast ultrasound images (80:20 split)
- [CVC-ClinicDB](https://www.dropbox.com/s/p5qe9eotetjnbmq/CVC-ClinicDB.rar) — 612 colonoscopy frames (80:20 split)

## Training

```bash
# ISIC 2018
python train.py --dataset_root /path/to/ISIC2018 --dataset isic2018

# BUSI
python train.py --dataset_root /path/to/BUSI --dataset busi

# CVC-ClinicDB
python train.py --dataset_root /path/to/CVC-ClinicDB --dataset cvc
```

All hyperparameters match Section 3.2: Adam optimizer, lr=1e-4, weight_decay=1e-4, 400 epochs, batch size 16, input size 256×256, λ=0.3.

## Quick Test

```python
import torch
from models import BAMKamba

model = BAMKamba(in_channels=3, num_classes=1)
x = torch.randn(1, 3, 256, 256)
seg_out, _ = model(x)
print(seg_out.shape)  # torch.Size([1, 1, 256, 256])

# Count parameters
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {n / 1e6:.2f}M")
```

## Citation

If you use this code, please cite our work.
