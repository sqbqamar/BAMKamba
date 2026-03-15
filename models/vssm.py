# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18:22:20 2026

@author: drsaq
"""

"""
Vision State-Space Module (VSSM)
================================
Implements the dual-branch VSSM from Section 2.2.1 of the manuscript.

Two parallel branches process the input X (H x W x C):
  Branch 1: Linear -> DWConv -> SiLU -> 2D-SSM -> LayerNorm
  Branch 2: Linear -> SiLU

The branches are fused via Hadamard (element-wise) product,
then projected back to C channels:
    X1 = LN(2D-SSM(SiLU(DWConv(Linear(X)))))
    X2 = SiLU(Linear(X))
    Xout = Linear(X1 ⊙ X2)

The channel expansion factor is λ (default=2).
"""

import torch
import torch.nn as nn
from .ss2d import SS2D


class VSSM(nn.Module):
    """
    Vision State-Space Module.

    Args:
        dim (int): Input/output channel dimension C.
        d_state (int): State dimension for the SSM. Default: 16.
        expand (int): Channel expansion factor λ. Default: 2.
    """

    def __init__(self, dim, d_state=16, expand=2):
        super().__init__()
        self.dim = dim
        self.expand = expand
        inner_dim = dim * expand

        # Branch 1: Linear -> DWConv -> SiLU -> 2D-SSM -> LN
        self.linear1 = nn.Linear(dim, inner_dim)
        self.dwconv = nn.Conv2d(
            inner_dim, inner_dim, kernel_size=3, padding=1,
            groups=inner_dim, bias=True
        )
        self.act1 = nn.SiLU()
        self.ss2d = SS2D(d_model=inner_dim, d_state=d_state)
        self.norm = nn.LayerNorm(inner_dim)

        # Branch 2: Linear -> SiLU
        self.linear2 = nn.Linear(dim, inner_dim)
        self.act2 = nn.SiLU()

        # Output projection: inner_dim -> dim
        self.out_proj = nn.Linear(inner_dim, dim)

    def forward(self, x):
        """
        Args:
            x: (B, H, W, C) input feature map
        Returns:
            out: (B, H, W, C) output feature map
        """
        B, H, W, C = x.shape

        # ── Branch 1 ──
        x1 = self.linear1(x)               # (B, H, W, inner_dim)
        # DWConv needs (B, C, H, W) format
        x1_conv = x1.permute(0, 3, 1, 2)   # (B, inner_dim, H, W)
        x1_conv = self.dwconv(x1_conv)
        x1 = x1_conv.permute(0, 2, 3, 1)   # (B, H, W, inner_dim)
        x1 = self.act1(x1)
        x1 = self.ss2d(x1)                 # (B, H, W, inner_dim)
        x1 = self.norm(x1)

        # ── Branch 2 ──
        x2 = self.act2(self.linear2(x))    # (B, H, W, inner_dim)

        # ── Hadamard product + output projection ──
        out = self.out_proj(x1 * x2)       # (B, H, W, C)
        return out
