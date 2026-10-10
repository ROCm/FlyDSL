# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Test-only torch oracles extracted from ``kernels/quant/rdna4_int8_convrot.py``."""

import math

import torch


def reference_quantize_int8_convrot_weight(weight: torch.Tensor, group_size: int = 256) -> tuple[object, object]:
    """Torch reference matching eager ConvRot + rowwise INT8."""
    import torch

    def _build_hadamard(size: int, device: object, dtype: object) -> torch.Tensor:
        if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
            raise ValueError(f"Regular Hadamard size must be a power of 4, got {size}")
        h4 = torch.tensor(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            dtype=dtype,
            device=device,
        )
        h = h4
        cur = 4
        while cur < size:
            h = torch.kron(h, h4)
            cur *= 4
        return h / (size**0.5)

    orig_shape = tuple(weight.shape)
    k = int(orig_shape[-1])
    w2d = weight.reshape(-1, k).to(torch.float32)
    # Match product host: soft-pad K to group_size (never reject odd K).
    pad = (group_size - (k % group_size)) % group_size
    if pad:
        w2d = torch.nn.functional.pad(w2d, (0, pad))
    k_pad = int(w2d.shape[-1])
    h = _build_hadamard(group_size, w2d.device, torch.float32)
    n_groups = k_pad // group_size
    wg = w2d.reshape(-1, n_groups, group_size)
    # H symmetric → W @ H.T == W @ H
    rotated = torch.matmul(wg, h.T).reshape(-1, k_pad)
    abs_max = rotated.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max / 127.0).clamp(min=1e-30)
    # Match HIP: round-half-to-even after exact divide (ref uses IEEE /).
    q = torch.round(rotated / scale).clamp(-128, 127).to(torch.int8)
    # Crop values to logical K; scales stay row-wise (one per row).
    q = q[..., :k].contiguous()
    return q.reshape(orig_shape), scale.reshape(*orig_shape[:-1], 1).to(torch.float32)


def reference_dequantize_int8_convrot_weight(
    q: torch.Tensor, scale: torch.Tensor | float | None, group_size: int = 256, out_dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Torch oracle for the dequant kernel. Not a runtime path."""
    import torch

    if out_dtype is None:
        out_dtype = torch.bfloat16

    def _build_hadamard(size: int, device: object, dtype: object) -> torch.Tensor:
        h4 = torch.tensor(
            [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
            dtype=dtype,
            device=device,
        )
        h = h4
        cur = 4
        while cur < size:
            h = torch.kron(h, h4)
            cur *= 4
        return h / (size**0.5)

    orig = tuple(q.shape)
    k = int(orig[-1])
    q2 = q.reshape(-1, k).float()
    pad = (group_size - (k % group_size)) % group_size
    if pad:
        q2 = torch.nn.functional.pad(q2, (0, pad))
    k_pad = int(q2.shape[-1])
    if isinstance(scale, (int, float)):
        sc = torch.full((q2.shape[0], 1), float(scale), device=q.device, dtype=torch.float32)
    else:
        sc = scale.to(device=q.device, dtype=torch.float32).reshape(-1, 1)
    deq = q2 * sc
    h = _build_hadamard(group_size, q.device, torch.float32)
    n_groups = k_pad // group_size
    dg = deq.reshape(-1, n_groups, group_size)
    out = torch.matmul(dg, h.T).reshape(-1, k_pad)[..., :k].reshape(orig)
    return out.to(out_dtype)
