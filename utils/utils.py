# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18:22:20 2026

@author: drsaq
"""

"""
Utilities: Metrics and Dataset
===============================
Evaluation metrics from Section 3.2: Dice, IoU, Accuracy, Specificity.
Dataset loader for ISIC 2018, BUSI, and CVC-ClinicDB.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


# ════════════════════════════════════════════════════════════════
# EVALUATION METRICS (Section 3.2)
# ════════════════════════════════════════════════════════════════

def compute_metrics(pred, target, threshold=0.5):
    """
    Compute segmentation metrics: Dice, IoU, Accuracy, Specificity.

    Args:
        pred: (B, 1, H, W) predicted probabilities (after sigmoid)
        target: (B, 1, H, W) binary ground-truth mask
        threshold: binarization threshold

    Returns:
        dict with 'dice', 'iou', 'accuracy', 'specificity'
    """
    pred_bin = (pred > threshold).float()
    target_bin = target.float()

    # Flatten
    p = pred_bin.view(-1)
    t = target_bin.view(-1)

    tp = (p * t).sum()
    tn = ((1 - p) * (1 - t)).sum()
    fp = (p * (1 - t)).sum()
    fn = ((1 - p) * t).sum()

    # Dice coefficient
    dice = (2 * tp + 1e-8) / (2 * tp + fp + fn + 1e-8)

    # IoU (Jaccard index)
    iou = (tp + 1e-8) / (tp + fp + fn + 1e-8)

    # Accuracy
    accuracy = (tp + tn + 1e-8) / (tp + tn + fp + fn + 1e-8)

    # Specificity = TN / (TN + FP)
    specificity = (tn + 1e-8) / (tn + fp + 1e-8)

    return {
        'dice': dice.item() * 100,
        'iou': iou.item() * 100,
        'accuracy': accuracy.item() * 100,
        'specificity': specificity.item() * 100,
    }


# ════════════════════════════════════════════════════════════════
# DATASET LOADER (Section 3.1)
# ════════════════════════════════════════════════════════════════

class SegmentationDataset(Dataset):
    """
    Generic segmentation dataset loader.

    Expects directory structure:
        root/
            images/
                img001.png
                img002.png
                ...
            masks/
                img001.png
                img002.png
                ...

    All images are resized to 256×256 as stated in Section 3.2.

    Args:
        root (str): Path to dataset root directory.
        split (str): 'train' or 'test'.
        image_size (int): Target size. Default: 256.
        augment (bool): Apply training augmentations. Default: False.
    """

    def __init__(self, root, split='train', image_size=256, augment=False):
        self.root = root
        self.image_size = image_size
        self.augment = augment and (split == 'train')

        image_dir = os.path.join(root, 'images')
        mask_dir = os.path.join(root, 'masks')

        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        self.mask_paths = sorted([
            os.path.join(mask_dir, f)
            for f in os.listdir(mask_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])

        assert len(self.image_paths) == len(self.mask_paths), \
            f"Mismatch: {len(self.image_paths)} images vs {len(self.mask_paths)} masks"

        # Standard normalization
        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        # Training augmentations (Section 3.2)
        if self.augment:
            self.spatial_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, scale=(0.85, 1.15)),
            ])
        else:
            self.spatial_aug = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        # Apply spatial augmentations (same transform for image and mask)
        if self.spatial_aug is not None:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            image = self.spatial_aug(image)
            torch.manual_seed(seed)
            mask = self.spatial_aug(mask)

        image = self.img_transform(image)
        mask = self.mask_transform(mask)

        # Binarize mask
        mask = (mask > 0.5).float()

        return image, mask
