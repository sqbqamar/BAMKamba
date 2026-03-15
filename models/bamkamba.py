# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18:22:20 2026

@author: drsaq
"""

"""
BAMKamba: Full Architecture
============================
Implements the complete U-shaped BAMKamba network from Section 2.1.

Architecture overview:
    ENCODER:
        Stage 1: ConvBlock  (3 → 16 ch)    → x_{e,1}  (H/2  × W/2)
        Stage 2: ConvBlock  (16 → 32 ch)   → x_{e,2}  (H/4  × W/4)
        Stage 3: ConvBlock  (32 → 128 ch)  → x_{e,3}  (H/8  × W/8)
        Stage 4: PatchEmbed + LACE (128 → 160 ch) → x_{e,4}  (H/16 × W/16)
        Stage 5: PatchEmbed + LACE (160 → 256 ch) → x_{e,5}  (H/32 × W/32)

    BOTTLENECK:
        MSDC module on x_{e,5} (dilation rates = 1,3,5,7)

    SKIP CONNECTIONS (with AFC gates):
        x_{e,1} → AFC1 → s1
        x_{e,2} → AFC2 → s2
        x_{e,3} → AFC3 → s3
        x_{e,4} → AFC4 → s4

    DECODER:
        Stage 5: LACEDecoder (256 → 160 ch, skip=s4)
        Stage 4: LACEDecoder (160 → 128 ch, skip=s3)
        Stage 3: DConvBlock  (128 → 32 ch,  skip=s2)
        Stage 2: DConvBlock  (32 → 16 ch,   skip=s1)
        Stage 1: Final 1×1 conv → segmentation output

    BAS MODULE (train only):
        Boundary head attached to 3rd decoder stage (128 ch)

Parameters: ~6.18M, GFlops: ~2.34
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_blocks import ConvBlock, DConvBlock, PatchEmbedding, LACEDecoderBlock
from .lace_block import LACEBlock
from .msdc import MSDC
from .afc import AFCGate
from .bas import BoundaryHead, BoundaryExtractor, boundary_loss, dice_loss


class BAMKamba(nn.Module):
    """
    BAMKamba: Boundary-Aware Multi-Scale State-Space Network
    for Medical Image Segmentation.

    Args:
        in_channels (int): Input image channels. Default: 3.
        num_classes (int): Number of output classes. Default: 1.
        channels (list): Channel dims at each stage. Default: [16,32,128,160,256].
        lambda_bnd (float): Boundary loss weight λ. Default: 0.3.
    """

    def __init__(self, in_channels=3, num_classes=1, channels=None,
                 lambda_bnd=0.3):
        super().__init__()
        if channels is None:
            channels = [16, 32, 128, 160, 256]
        c1, c2, c3, c4, c5 = channels
        self.lambda_bnd = lambda_bnd

        # ════════════ ENCODER ════════════

        # Stage 1-3: Convolutional blocks
        self.enc1 = ConvBlock(in_channels, c1)   # → (c1, H/2,  W/2)
        self.enc2 = ConvBlock(c1, c2)            # → (c2, H/4,  W/4)
        self.enc3 = ConvBlock(c2, c3)            # → (c3, H/8,  W/8)

        # Stage 4: Patch Embedding + LACE block
        self.patch_embed4 = PatchEmbedding(c3, c4)  # → (c4, H/16, W/16)
        self.enc4 = LACEBlock(dim=c4)

        # Stage 5: Patch Embedding + LACE block
        self.patch_embed5 = PatchEmbedding(c4, c5)  # → (c5, H/32, W/32)
        self.enc5 = LACEBlock(dim=c5)

        # ════════════ BOTTLENECK ════════════

        self.msdc = MSDC(channels=c5, dilation_rates=[1, 3, 5, 7])

        # ════════════ SKIP CONNECTIONS (AFC gates) ════════════

        self.afc1 = AFCGate(c1)
        self.afc2 = AFCGate(c2)
        self.afc3 = AFCGate(c3)
        self.afc4 = AFCGate(c4)

        # ════════════ DECODER ════════════

        # Stage 5 → 4: LACE decoder (upsample + skip from stage 4)
        self.dec5 = LACEDecoderBlock(c5, c4, c4)    # → (c4, H/16, W/16)

        # Stage 4 → 3: LACE decoder (upsample + skip from stage 3)
        self.dec4 = LACEDecoderBlock(c4, c3, c3)    # → (c3, H/8,  W/8)

        # Stage 3 → 2: DConv decoder (upsample + skip from stage 2)
        self.dec3 = DConvBlock(c3, c2, c2)           # → (c2, H/4,  W/4)

        # Stage 2 → 1: DConv decoder (upsample + skip from stage 1)
        self.dec2 = DConvBlock(c2, c1, c1)           # → (c1, H/2,  W/2)

        # Final upsampling + 1x1 convolution for segmentation output
        self.final_conv = nn.Conv2d(c1, num_classes, kernel_size=1)

        # ════════════ BAS MODULE (train only) ════════════

        # Boundary head on 3rd decoder stage = DConv Block 3 (c2 channels, H/4)
        # Section 2.4: "attached to the third decoder stage"
        self.boundary_head = BoundaryHead(in_channels=c2, target_size=256)
        self.boundary_extractor = BoundaryExtractor()

    def forward(self, x, gt_mask=None):
        """
        Args:
            x: (B, C_in, H, W) input image, e.g. (B, 3, 256, 256)
            gt_mask: (B, 1, H, W) ground-truth mask (only during training)

        Returns:
            seg_out: (B, num_classes, H, W) segmentation prediction
            loss_dict: dict with 'seg_loss', 'bnd_loss', 'total_loss'
                       (only if gt_mask is provided)
        """
        input_size = x.shape[2:]  # (H, W)

        # ── Encoder ──
        xe1, p1 = self.enc1(x)       # xe1: (B,c1,H/2,W/2),  p1: (B,c1,H/4,W/4)
        xe2, p2 = self.enc2(p1)      # xe2: (B,c2,H/4,W/4),  p2: (B,c2,H/8,W/8)
        xe3, p3 = self.enc3(p2)      # xe3: (B,c3,H/8,W/8),  p3: (B,c3,H/16,W/16)

        pe4 = self.patch_embed4(p3)  # (B, c4, H/16, W/16)
        xe4 = self.enc4(pe4)         # (B, c4, H/16, W/16)

        pe5 = self.patch_embed5(xe4) # (B, c5, H/32, W/32)
        xe5 = self.enc5(pe5)         # (B, c5, H/32, W/32)

        # ── Bottleneck ──
        bottleneck = self.msdc(xe5)  # (B, c5, H/32, W/32)

        # ── Skip connections through AFC gates ──
        s1 = self.afc1(xe1)  # (B, c1, H/2,  W/2)
        s2 = self.afc2(xe2)  # (B, c2, H/4,  W/4)
        s3 = self.afc3(xe3)  # (B, c3, H/8,  W/8)
        s4 = self.afc4(xe4)  # (B, c4, H/16, W/16)

        # ── Decoder ──
        d5 = self.dec5(bottleneck, s4)  # 1st dec stage: (B, c4, H/16, W/16)
        d4 = self.dec4(d5, s3)          # 2nd dec stage: (B, c3, H/8,  W/8)
        d3 = self.dec3(d4, s2)          # 3rd dec stage: (B, c2, H/4,  W/4) ← BAS branches here
        d2 = self.dec2(d3, s1)          # 4th dec stage: (B, c1, H/2,  W/2)

        # ── Final output ──
        seg_out = F.interpolate(d2, size=input_size, mode='bilinear',
                                align_corners=False)
        seg_out = self.final_conv(seg_out)  # (B, num_classes, H, W)

        # ── Loss computation (training only) ──
        if gt_mask is not None and self.training:
            # Segmentation loss: Dice + BCE (Eq. 8, L_seg)
            seg_prob = torch.sigmoid(seg_out)
            seg_loss = (
                F.binary_cross_entropy_with_logits(seg_out, gt_mask)
                + dice_loss(seg_prob, gt_mask)
            )

            # Boundary loss (Eq. 7, 8)
            # BAS branches from 3rd decoder stage (DConv Block 3 = d3)
            gt_edge = self.boundary_extractor(gt_mask)   # (B, 1, H, W)
            pred_edge = self.boundary_head(d3)           # (B, 1, H, W)
            bnd_loss = boundary_loss(pred_edge, gt_edge)

            # Total loss (Eq. 8)
            total_loss = seg_loss + self.lambda_bnd * bnd_loss

            loss_dict = {
                'seg_loss': seg_loss,
                'bnd_loss': bnd_loss,
                'total_loss': total_loss,
            }
            return seg_out, loss_dict

        return seg_out, {}
