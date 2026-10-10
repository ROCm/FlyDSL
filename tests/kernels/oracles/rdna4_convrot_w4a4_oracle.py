# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Test-only torch oracles extracted from ``kernels/quant/rdna4_convrot_w4a4.py``."""

import math

import torch

_INT4_GROUP_SIZE = 64
_INT4_MAX = 7
_SCALE_FLOOR = 1e-10


def _build_hadamard(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    import torch

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


def _pack_int4_row_major(values: object) -> torch.Tensor:
    """Wire format: low nibble = even column, high = odd."""
    import torch

    if values.shape[-1] % 2 != 0:
        raise ValueError(f"last dim must be even, got {values.shape[-1]}")
    lo = values[..., 0::2].to(torch.int32) & 0x0F
    hi = values[..., 1::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.int8)


def reference_quantize_convrot_w4a4_weight(
    weight: torch.Tensor,
    convrot_groupsize: int = 256,
    quant_group_size: int = _INT4_GROUP_SIZE,
) -> tuple[object, object]:
    """Torch reference matching eager reference ConvRot W4A4."""
    import torch

    if quant_group_size != _INT4_GROUP_SIZE:
        raise ValueError(f"quant_group_size must be {_INT4_GROUP_SIZE}")
    n, k = weight.shape
    w2d = weight.reshape(-1, k).to(torch.float32)
    h = _build_hadamard(convrot_groupsize, w2d.device, torch.float32)
    n_groups = k // convrot_groupsize
    wg = w2d.reshape(-1, n_groups, convrot_groupsize)
    rotated = torch.matmul(wg, h.T).reshape(-1, k)
    abs_max = rotated.abs().amax(dim=-1, keepdim=True).clamp(min=_SCALE_FLOOR)
    scale = abs_max / float(_INT4_MAX)
    q = torch.round(rotated / scale).clamp(-_INT4_MAX, _INT4_MAX).to(torch.int8)
    return _pack_int4_row_major(q).reshape(n, k // 2), scale.reshape(n).to(torch.float32)
