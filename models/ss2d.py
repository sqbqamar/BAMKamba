# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18:22:20 2026

@author: drsaq
"""
"""
2D Selective Scan Module (2D-SSM)
=================================
Implements the four-directional selective scan for 2D feature maps.

The standard Mamba processes inputs causally in one direction.
For non-causal 2D images, the 2D-SSM flattens the feature map into
four 1D sequences by scanning in four directions:
  - top-left to bottom-right
  - bottom-right to top-left
  - top-right to bottom-left
  - bottom-left to top-right

Each sequence is processed using the discrete state-space equation.
The outputs are summed and reshaped back to the original 2D layout.

Reference: Section 2.2.1, Section 2.5 of the manuscript.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelectiveSSM(nn.Module):
    """
    Selective State-Space Model for a single 1D sequence.

    Implements the discrete state-space recurrence:
        h_t = A_bar * h_{t-1} + B_bar * u_t
        y_t = C_t * h_t + D * u_t

    where A_bar, B_bar are input-dependent (selective) discretizations.
    """

    def __init__(self, d_model, d_state=16, dt_rank=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, d_model // 16)

        # Continuous-time state matrix A (initialized with HiPPO)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).repeat(d_model, 1)  # (d_model, d_state)
        self.A_log = nn.Parameter(torch.log(A))

        # D skip connection parameter
        self.D = nn.Parameter(torch.ones(d_model))

        # Input-dependent projections for dt, B, C
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)
        self.x_to_dtBC = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)

        # Initialize dt bias for stable discretization
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x):
        """
        Args:
            x: (B, L, D) input sequence
        Returns:
            y: (B, L, D) output sequence
        """
        B, L, D = x.shape

        # Compute input-dependent dt, B_input, C_input
        dtBC = self.x_to_dtBC(x)  # (B, L, dt_rank + 2*d_state)
        dt_input = dtBC[:, :, :self.dt_rank]
        B_input = dtBC[:, :, self.dt_rank:self.dt_rank + self.d_state]
        C_input = dtBC[:, :, self.dt_rank + self.d_state:]

        # Project dt_rank to d_model and apply softplus
        dt = F.softplus(self.dt_proj(dt_input))  # (B, L, D)

        # Discretize: A_bar = exp(dt * A), B_bar = dt * B
        A = -torch.exp(self.A_log)  # (D, N)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, D, N)
        dB = dt.unsqueeze(-1) * B_input.unsqueeze(2)  # (B, L, D, N)

        # Sequential scan (recurrence)
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)  # (B, D, N)
            y_t = (h * C_input[:, t].unsqueeze(1)).sum(-1)  # (B, D)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, L, D)

        # Skip connection
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y


class SS2D(nn.Module):
    """
    2D Selective Scan Module.

    Scans the 2D feature map in four directions, processes each
    with a selective SSM, then sums and reshapes back to 2D.

    Reference: Section 2.2.1 of the manuscript.
    """

    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Four independent SSMs for four scan directions
        self.ssm_forward = SelectiveSSM(d_model, d_state)
        self.ssm_backward = SelectiveSSM(d_model, d_state)
        self.ssm_forward_t = SelectiveSSM(d_model, d_state)
        self.ssm_backward_t = SelectiveSSM(d_model, d_state)

    def _scan_directions(self, x):
        """
        Flatten 2D feature map into four 1D sequences.

        Args:
            x: (B, H, W, C) feature map
        Returns:
            list of four (B, H*W, C) sequences
        """
        B, H, W, C = x.shape

        # Direction 1: top-left to bottom-right (row-major)
        seq_tlbr = x.reshape(B, H * W, C)

        # Direction 2: bottom-right to top-left (reversed row-major)
        seq_brtl = seq_tlbr.flip(1)

        # Direction 3: top-right to bottom-left (flip columns, row-major)
        seq_trbl = x.flip(2).reshape(B, H * W, C)

        # Direction 4: bottom-left to top-right (flip columns, reversed)
        seq_bltr = seq_trbl.flip(1)

        return [seq_tlbr, seq_brtl, seq_trbl, seq_bltr]

    def _merge_directions(self, outputs, H, W):
        """
        Reverse the four scan directions and sum the results.

        Args:
            outputs: list of four (B, H*W, C) sequences
            H, W: original spatial dimensions
        Returns:
            merged: (B, H, W, C) feature map
        """
        B = outputs[0].shape[0]
        C = outputs[0].shape[2]

        # Reverse direction 1: already in correct order
        out1 = outputs[0]

        # Reverse direction 2: flip back
        out2 = outputs[1].flip(1)

        # Reverse direction 3: reshape and flip columns back
        out3 = outputs[2].reshape(B, H, W, C).flip(2).reshape(B, H * W, C)

        # Reverse direction 4: flip then reshape and flip columns
        out4 = outputs[3].flip(1).reshape(B, H, W, C).flip(2).reshape(B, H * W, C)

        # Sum all directions and reshape to 2D
        merged = (out1 + out2 + out3 + out4).reshape(B, H, W, C)
        return merged

    def forward(self, x):
        """
        Args:
            x: (B, H, W, C) input feature map
        Returns:
            out: (B, H, W, C) output feature map
        """
        B, H, W, C = x.shape

        # Flatten to four directional sequences
        sequences = self._scan_directions(x)

        # Process each direction with its own SSM
        ssm_list = [self.ssm_forward, self.ssm_backward,
                     self.ssm_forward_t, self.ssm_backward_t]
        outputs = [ssm(seq) for ssm, seq in zip(ssm_list, sequences)]

        # Merge four directions back to 2D
        out = self._merge_directions(outputs, H, W)
        return out
