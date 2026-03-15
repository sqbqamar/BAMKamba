# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18:22:20 2026

@author: drsaq



Training configuration:
    - Input size: 256 × 256
    - Epochs: 400
    - Optimizer: Adam (lr=1e-4, momentum=0.9, weight_decay=1e-4)
    - Loss: Dice + BCE (segmentation) + λ·(BCE + Dice) (boundary)
    - λ = 0.3
    - Batch size: 16
    - GPU: NVIDIA A100
    - Augmentation: horizontal/vertical flip, rotation ±15°, scale 0.85-1.15

Usage:
    python train.py --dataset_root /path/to/ISIC2018 --dataset isic2018
    python train.py --dataset_root /path/to/BUSI --dataset busi
    python train.py --dataset_root /path/to/CVC-ClinicDB --dataset cvc
"""

import os
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models import BAMKamba
from utils import SegmentationDataset, compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='BAMKamba Training')
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Path to dataset root (images/ and masks/ subdirs)')
    parser.add_argument('--dataset', type=str, default='isic2018',
                        choices=['isic2018', 'busi', 'cvc'],
                        help='Dataset name for split configuration')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (Section 3.2)')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lambda_bnd', type=float, default=0.3,
                        help='Boundary loss weight λ (Section 2.4)')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_seg = 0.0
    total_bnd = 0.0
    n_batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        seg_out, loss_dict = model(images, gt_mask=masks)
        loss = loss_dict['total_loss']
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_seg += loss_dict['seg_loss'].item()
        total_bnd += loss_dict['bnd_loss'].item()
        n_batches += 1

    return {
        'total_loss': total_loss / n_batches,
        'seg_loss': total_seg / n_batches,
        'bnd_loss': total_bnd / n_batches,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate on validation/test set."""
    model.eval()
    all_metrics = {'dice': 0, 'iou': 0, 'accuracy': 0, 'specificity': 0}
    n_batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        seg_out, _ = model(images)
        seg_prob = torch.sigmoid(seg_out)

        metrics = compute_metrics(seg_prob, masks)
        for k in all_metrics:
            all_metrics[k] += metrics[k]
        n_batches += 1

    for k in all_metrics:
        all_metrics[k] /= n_batches

    return all_metrics


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Dataset setup (Section 3.1) ──
    full_dataset = SegmentationDataset(
        root=args.dataset_root,
        split='train',
        image_size=args.image_size,
        augment=True,
    )

    # Split ratios per dataset (Section 3.1)
    n_total = len(full_dataset)
    if args.dataset == 'isic2018':
        # Standard ISIC split: 2594 train, 100 val, 1000 test
        # Assume the dataset root contains only training images
        n_val = 100
        n_train = n_total - n_val
    else:
        # BUSI and CVC-ClinicDB: 80:20 random split
        n_val = int(0.2 * n_total)
        n_train = n_total - n_val

    train_set, val_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # Disable augmentation for validation
    val_dataset = SegmentationDataset(
        root=args.dataset_root,
        split='test',
        image_size=args.image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model ──
    model = BAMKamba(
        in_channels=3,
        num_classes=1,
        channels=[16, 32, 128, 160, 256],
        lambda_bnd=args.lambda_bnd,
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.2f}M")

    # ── Optimizer (Section 3.2) ──
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    # ── Training loop ──
    best_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_losses = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == args.epochs:
            val_metrics = evaluate(model, val_loader, device)
            elapsed = time.time() - t0

            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"Loss: {train_losses['total_loss']:.4f} "
                f"(seg: {train_losses['seg_loss']:.4f}, "
                f"bnd: {train_losses['bnd_loss']:.4f}) | "
                f"Dice: {val_metrics['dice']:.2f}% | "
                f"IoU: {val_metrics['iou']:.2f}% | "
                f"Acc: {val_metrics['accuracy']:.2f}% | "
                f"Spec: {val_metrics['specificity']:.2f}% | "
                f"Time: {elapsed:.1f}s"
            )

            # Save best model
            if val_metrics['dice'] > best_dice:
                best_dice = val_metrics['dice']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_dice': best_dice,
                    'metrics': val_metrics,
                }, os.path.join(args.save_dir, 'best_model.pth'))
                print(f"  → Saved best model (Dice: {best_dice:.2f}%)")
        else:
            elapsed = time.time() - t0
            if epoch % 50 == 0:
                print(
                    f"Epoch {epoch:03d}/{args.epochs} | "
                    f"Loss: {train_losses['total_loss']:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

    print(f"\nTraining complete. Best Dice: {best_dice:.2f}%")
    print(f"Best model saved to: {os.path.join(args.save_dir, 'best_model.pth')}")


if __name__ == '__main__':
    main()
