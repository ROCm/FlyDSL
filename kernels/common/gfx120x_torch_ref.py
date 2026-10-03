# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Pure-PyTorch reference paths for ``FLYDSL_DISPATCH_MODE=force_hip``.

These are portable baselines with no third-party packages, so size-gate
bypasses work on a stock ROCm/PyTorch install.
"""

import torch

__all__ = ["int8_linear_torch", "scaled_mm_fp8_torch"]


def int8_linear_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """W8A16-style reference: activations × (8-bit weights * per-channel scale)."""
    if out_dtype is None:
        out_dtype = x.dtype
    x2 = x if x.ndim == 2 else x.reshape(-1, x.shape[-1])
    w = weight.to(dtype=torch.float32)
    scale = weight_scale.to(device=w.device, dtype=torch.float32).reshape(-1)
    if scale.numel() == 1:
        w = w * scale.item()
    elif scale.numel() == w.shape[0]:
        w = w * scale.unsqueeze(1)
    elif scale.numel() == w.shape[1]:
        w = w * scale.unsqueeze(0)
    else:
        raise ValueError(f"weight_scale numel={scale.numel()} incompatible with weight shape {tuple(w.shape)}")
    y = x2.to(dtype=torch.float32) @ w.transpose(0, 1)
    if bias is not None:
        y = y + bias.to(device=y.device, dtype=torch.float32).reshape(1, -1)
    if x.ndim != 2:
        y = y.reshape(*x.shape[:-1], y.shape[-1])
    return y.to(dtype=out_dtype)


def scaled_mm_fp8_torch(
    a: torch.Tensor,
    b_kn: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """FP8 scaled GEMM reference: (a * scale_a) @ (b * scale_b) with b as [K, N]."""
    sa = float(scale_a.to(device=a.device, dtype=torch.float32).reshape(1).item())
    sb = float(scale_b.to(device=a.device, dtype=torch.float32).reshape(1).item())
    af = a.to(dtype=torch.float32) * sa
    bf = b_kn.to(dtype=torch.float32) * sb
    return (af @ bf).to(dtype=out_dtype)
