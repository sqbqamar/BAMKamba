# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18:22:20 2026

@author: drsaq
"""

"""
Local Aware Channel Enhancement (LACE) Block
=============================================
Implements the LACE block from Sections 2.2 and 2.5 of the manuscript.

For an input feature map F^D (B x C x H x W):

Step 1 — VSSM + learnable skip:
    V^l = VSSM(LN(F^D)) + s · F^D          (Eq. 1)

Step 2 — Local conv + Channel Attention + learnable skip:
    F^D' = CA(Conv(LN(V^l))) + s' · V^l     (Eq. 2)

The local convolution uses a bottleneck design: channels are reduced
by factor β, processed with a 3x3 conv, then expanded back.

Channel Attention (SE-style) suppresses redundant channels that
arise from the large hidden-state dimensions of the SSM.

Learnable scale factors s, s' ∈ R^C modulate the residual connections.
"""

import torch
import torch.nn as nn
from .vssm import VSSM


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.

    Reference: Hu et al., "Squeeze-and-Excitation Networks" (CVPR 2018).
    Used in LACE to select the most informative channels after SSM processing.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W) channel-reweighted features
        """
        B, C, _, _ = x.shape
        w = self.pool(x).view(B, C)        # (B, C)
        w = self.fc(w).view(B, C, 1, 1)    # (B, C, 1, 1)
        return x * w


class LocalConvBlock(nn.Module):
    """
    Local convolution with bottleneck design.

    Restores neighborhood information lost during the flattening
    of 2D feature maps into 1D sequences for the SSM.

    Channels are reduced by factor β, processed with 3x3 conv,
    then expanded back to the original dimension.
    """

    def __init__(self, channels, bottleneck_factor=4):
        super().__init__()
        mid = max(channels // bottleneck_factor, 16)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return self.conv(x)


class LACEBlock(nn.Module):
    """
    Local Aware Channel Enhancement Block.

    Combines VSSM-based global modeling with local convolution
    and channel attention for medical image segmentation.

    Args:
        dim (int): Number of input/output channels.
        d_state (int): SSM state dimension. Default: 16.
        bottleneck_factor (int): Channel reduction factor β. Default: 4.
    """

    def __init__(self, dim, d_state=16, bottleneck_factor=4):
        super().__init__()
        self.dim = dim

        # Step 1: LayerNorm -> VSSM
        self.norm1 = nn.LayerNorm(dim)
        self.vssm = VSSM(dim=dim, d_state=d_state)
        # Learnable scale factor s for first residual (Eq. 1)
        self.scale1 = nn.Parameter(torch.ones(dim) * 0.1)

        # Step 2: LayerNorm -> Local Conv -> Channel Attention
        self.norm2 = nn.LayerNorm(dim)
        self.local_conv = LocalConvBlock(dim, bottleneck_factor)
        self.channel_attn = ChannelAttention(dim)
        # Learnable scale factor s' for second residual (Eq. 2)
        self.scale2 = nn.Parameter(torch.ones(dim) * 0.1)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input feature map
        Returns:
            out: (B, C, H, W) enhanced feature map
        """
        B, C, H, W = x.shape

        # Convert to (B, H, W, C) for LayerNorm and VSSM
        x_hwc = x.permute(0, 2, 3, 1)  # (B, H, W, C)

        # ── Step 1: VSSM + scaled residual (Eq. 1) ──
        vl = self.vssm(self.norm1(x_hwc))  # (B, H, W, C)
        scale1 = self.scale1.view(1, 1, 1, C)
        vl = vl + scale1 * x_hwc           # V^l

        # ── Step 2: Local Conv + Channel Attention + scaled residual (Eq. 2) ──
        vl_normed = self.norm2(vl)
        # Convert to (B, C, H, W) for conv operations
        vl_bchw = vl_normed.permute(0, 3, 1, 2)
        out = self.local_conv(vl_bchw)     # (B, C, H, W)
        out = self.channel_attn(out)        # (B, C, H, W)

        # Scaled residual in (B, C, H, W) format
        vl_bchw_orig = vl.permute(0, 3, 1, 2)
        scale2 = self.scale2.view(1, C, 1, 1)
        out = out + scale2 * vl_bchw_orig   # F^D' (Eq. 2)

        return out
