# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18:22:20 2026

@author: drsaq
"""
"""
Convolutional Blocks for Encoder and Decoder
=============================================
Implements the standard convolutional blocks from Section 2.1.

Encoder ConvBlock:
    Two 3x3 convolutions, each followed by BatchNorm + ReLU,
    then MaxPooling for 2x spatial downsampling.

Decoder ConvBlock (DConvBlock):
    Bilinear upsample 2x, concatenate with skip features,
    then two 3x3 convolutions with BatchNorm + ReLU.

Patch Embedding:
    2x2 convolution with stride 2 that decreases spatial dimensions
    while expanding channel depth. Used between Conv3 and LACE1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Encoder convolutional block.

    Applies two 3x3 convolutions (each with BN + ReLU),
    then MaxPool for 2x downsampling.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Returns:
            features: (B, out_ch, H, W) features before pooling (for skip)
            pooled: (B, out_ch, H/2, W/2) downsampled features
        """
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class DConvBlock(nn.Module):
    """
    Decoder convolutional block.

    Upsamples 2x via bilinear interpolation, concatenates with
    skip-connected features, then applies two 3x3 convolutions.

    Args:
        in_channels (int): Channels from the previous decoder stage.
        skip_channels (int): Channels from the corresponding skip connection.
        out_channels (int): Output channels.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        """
        Args:
            x: (B, in_ch, H, W) decoder features (will be upsampled)
            skip: (B, skip_ch, 2H, 2W) skip-connected encoder features
        Returns:
            out: (B, out_ch, 2H, 2W) decoded features
        """
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                          align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class PatchEmbedding(nn.Module):
    """
    Patch Embedding layer.

    Implemented as a 2x2 convolution with stride 2, which decreases
    spatial dimensions by half while expanding channel depth.

    Used to transition from convolutional stages to LACE stages.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=2,
                              stride=2, bias=False)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        """
        Args:
            x: (B, in_ch, H, W)
        Returns:
            out: (B, out_ch, H/2, W/2)
        """
        x = self.proj(x)  # (B, out_ch, H/2, W/2)
        B, C, H, W = x.shape
        # Apply LayerNorm in (B, H, W, C) format
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class LACEDecoderBlock(nn.Module):
    """
    Decoder block that uses LACE instead of standard convolutions.

    Upsamples via linear layer + rearrangement, concatenates skip features,
    then processes through a LACE block.

    Args:
        in_channels (int): Channels from previous decoder stage.
        skip_channels (int): Channels from skip connection.
        out_channels (int): Output channels.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        from .lace_block import LACEBlock

        # Linear upsample: project channels, then pixel-shuffle-like rearrange
        self.up_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # Fuse concatenated features
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.lace = LACEBlock(dim=out_channels)

    def forward(self, x, skip):
        """
        Args:
            x: (B, in_ch, H, W) decoder features
            skip: (B, skip_ch, 2H, 2W) skip features
        Returns:
            out: (B, out_ch, 2H, 2W)
        """
        x = self.up_proj(x)
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                          align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        x = self.lace(x)
        return x
