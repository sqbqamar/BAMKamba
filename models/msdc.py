# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18:22:20 2026

@author: drsaq
"""
"""
Multi-Scale Dilated Context (MSDC) Module
==========================================
Implements the bottleneck module from Section 2.2 of the manuscript.

Given the bottleneck feature map F_b (B x C5 x H/32 x W/32):

    F_k = DilConv(F_b, r_k)  for k = 1,2,3,4       (Eq. 1)
    F_msdc = Conv_1x1(Concat(F1,F2,F3,F4)) + F_b    (Eq. 2)

where DilConv is a 3x3 depthwise separable convolution with
dilation rate r_k ∈ {1, 3, 5, 7}.

Depthwise separable convolutions keep computational cost low.
The residual connection preserves direct information flow.
"""

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution with dilation.

    Splits a standard convolution into:
      1. Depthwise conv: applies a single filter per channel
      2. Pointwise conv: 1x1 conv to mix channels
    """

    def __init__(self, channels, dilation=1):
        super().__init__()
        padding = dilation  # maintains spatial size with 3x3 kernel
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3,
            padding=padding, dilation=dilation,
            groups=channels, bias=False
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class MSDC(nn.Module):
    """
    Multi-Scale Dilated Context Module.

    Applies parallel dilated convolutions at rates {1, 3, 5, 7},
    concatenates the outputs, projects back to original channels,
    and adds a residual connection.

    Args:
        channels (int): Number of input/output channels (C5 = 256).
        dilation_rates (list): Dilation rates. Default: [1, 3, 5, 7].
    """

    def __init__(self, channels, dilation_rates=None):
        super().__init__()
        if dilation_rates is None:
            dilation_rates = [1, 3, 5, 7]

        self.branches = nn.ModuleList([
            DepthwiseSeparableConv(channels, dilation=r)
            for r in dilation_rates
        ])

        # 1x1 pointwise convolution to project concatenated features
        # back to original channel dimension
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * len(dilation_rates), channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) bottleneck feature map F_b
        Returns:
            out: (B, C, H, W) multi-scale enriched features F_msdc
        """
        # Parallel dilated convolutions (Eq. 1)
        branch_outputs = [branch(x) for branch in self.branches]

        # Channel-wise concatenation + 1x1 projection (Eq. 2)
        concat = torch.cat(branch_outputs, dim=1)
        fused = self.fuse(concat)

        # Residual connection from original bottleneck features
        out = fused + x
        return out
