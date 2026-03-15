# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18:22:20 2026

@author: drsaq
"""



"""
Adaptive Feature Calibration (AFC) Gates
==========================================
Implements the skip-connection gating from Section 2.3 of the manuscript.

For encoder feature map x_{e,i} at stage i:

    g_c = σ(W2 · ReLU(W1 · GAP(x_{e,i})))           — channel gate (Eq. 3)
    g_s = σ(Conv3x3(AvgPool(x_{e,i}) ‖ MaxPool(x_{e,i})))  — spatial gate (Eq. 4)
    x'_{e,i} = g_c · g_s · x_{e,i} + α · x_{e,i}   — calibrated output (Eq. 5)

where:
    - GAP = global average pooling
    - W1, W2 are linear projections with bottleneck ratio 4
    - ‖ denotes channel concatenation
    - α is a learnable scaling parameter initialized to 0.1
"""

import torch
import torch.nn as nn


class AFCGate(nn.Module):
    """
    Adaptive Feature Calibration Gate.

    Applies channel-spatial gating to skip connection features
    to selectively amplify informative features and suppress
    noisy or redundant activations.

    Args:
        channels (int): Number of input channels at this encoder stage.
        reduction (int): Bottleneck ratio for channel gate. Default: 4.
        alpha_init (float): Initial value for residual scaling. Default: 0.1.
    """

    def __init__(self, channels, reduction=4, alpha_init=0.1):
        super().__init__()

        # ── Channel Gate (Eq. 3) ──
        mid = max(channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.channel_gate = nn.Sequential(
            nn.Linear(channels, mid, bias=False),    # W1
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),    # W2
            nn.Sigmoid(),
        )

        # ── Spatial Gate (Eq. 4) ──
        # Input: concatenation of avg-pooled and max-pooled features (2 channels)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )

        # ── Learnable residual scaling α (Eq. 5) ──
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) encoder feature map x_{e,i}
        Returns:
            out: (B, C, H, W) calibrated feature map x'_{e,i}
        """
        B, C, H, W = x.shape

        # ── Channel gate g_c ──
        gc = self.gap(x).view(B, C)             # (B, C)
        gc = self.channel_gate(gc).view(B, C, 1, 1)  # (B, C, 1, 1)

        # ── Spatial gate g_s ──
        avg_pool = x.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        max_pool = x.max(dim=1, keepdim=True)[0]  # (B, 1, H, W)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2, H, W)
        gs = self.spatial_gate(spatial_input)     # (B, 1, H, W)

        # ── Calibrated output (Eq. 5) ──
        out = gc * gs * x + self.alpha * x
        return out
