# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18:22:20 2026

@author: drsaq
"""
"""
Boundary-Aware Supervision (BAS) Module
=========================================
Implements the auxiliary boundary supervision from Section 2.4.

During training:
  1. Edge maps E are generated from ground-truth masks M using Sobel operators:
       E = sqrt( Sx(M)^2 + Sy(M)^2 )                       (Eq. 6)
  2. E is binarized at threshold 0.5.
  3. A lightweight boundary prediction head (3x3 conv + upsample + sigmoid)
     attached to the 3rd decoder stage predicts E'.
  4. Boundary loss = BCE(E', E) + (1 - Dice(E', E))         (Eq. 7)
  5. Total loss = L_seg + λ · L_bnd                          (Eq. 8)
     where λ = 0.3.

At inference, the boundary head is discarded — zero added cost.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryExtractor(nn.Module):
    """
    Extracts edge maps from binary ground-truth masks using Sobel operators.

    The Sobel kernels are fixed (non-learnable) and deterministic.
    The output edge map is binarized at threshold 0.5.
    """

    def __init__(self):
        super().__init__()
        # Horizontal Sobel kernel Sx
        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)

        # Vertical Sobel kernel Sy
        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)

        # Register as buffers (not parameters — no gradient)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, mask):
        """
        Args:
            mask: (B, 1, H, W) binary ground-truth segmentation mask
        Returns:
            edge: (B, 1, H, W) binarized edge map
        """
        # Apply Sobel operators (Eq. 6)
        gx = F.conv2d(mask, self.sobel_x, padding=1)
        gy = F.conv2d(mask, self.sobel_y, padding=1)
        edge = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

        # Normalize to [0, 1] and binarize at 0.5
        edge = edge / (edge.max() + 1e-8)
        edge = (edge > 0.5).float()
        return edge


class BoundaryHead(nn.Module):
    """
    Lightweight boundary prediction head.

    Attached to the 3rd decoder stage during training.
    Consists of: 3x3 conv → bilinear upsample → sigmoid.

    Args:
        in_channels (int): Number of input channels from decoder stage.
        target_size (int): Target spatial size for upsampling to match GT.
    """

    def __init__(self, in_channels, target_size=256):
        super().__init__()
        self.target_size = target_size
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature map from 3rd decoder stage
        Returns:
            pred: (B, 1, target_size, target_size) predicted boundary map
        """
        pred = self.conv(x)  # (B, 1, H, W)
        pred = F.interpolate(
            pred, size=(self.target_size, self.target_size),
            mode='bilinear', align_corners=False
        )
        pred = torch.sigmoid(pred)
        return pred


def dice_loss(pred, target, smooth=1.0):
    """
    Dice loss for binary segmentation.

    Args:
        pred: (B, 1, H, W) predicted probabilities
        target: (B, 1, H, W) ground-truth binary map
    Returns:
        loss: scalar Dice loss
    """
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1.0 - (2.0 * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )


def boundary_loss(pred_boundary, gt_boundary):
    """
    Combined BCE + Dice boundary loss (Eq. 7).

    Args:
        pred_boundary: (B, 1, H, W) predicted boundary map
        gt_boundary: (B, 1, H, W) ground-truth edge map
    Returns:
        loss: scalar boundary loss
    """
    bce = F.binary_cross_entropy(pred_boundary, gt_boundary, reduction='mean')
    dl = dice_loss(pred_boundary, gt_boundary)
    return bce + dl
