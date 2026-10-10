# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Test-only torch oracles extracted from ``kernels/gemm/rdna4_fused_mlp_nmajor.py``."""

import torch
import torch.nn.functional as F


def reference_swiglu_mlp(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Eager reference: two GEMMs + SiLU×mul + down (f32 math)."""
    orig = x.shape
    x2d = x.reshape(-1, orig[-1]).float()
    gate = x2d @ w_gate.float().T
    up = x2d @ w_up.float().T
    mid = F.silu(gate) * up
    y = mid @ w_down.float().T
    return y.reshape(*orig[:-1], w_down.shape[0])
