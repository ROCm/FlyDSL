# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Shared INT4 pack/unpack + groupwise dequant helpers (packed nibble wire).

Nunchaku-style contract (same as ConvRot W4A4, SVDQuant W4A4, AWQ uint4):

* Pack: low nibble = even column, high = odd
  ``(v0 & 0xF) | ((v1 & 0xF) << 4)`` → int8 ``(..., K//2)``.
* Signed storage range: ``[-8, 7]`` (full nibble).
* Signed **quantizer emission** (SVDQuant / ConvRot absmax): ``[-7, 7]``
  (skip ``-8`` so dequant is symmetric about 0 with ``scale = max/7``).
* Unsigned storage / emission: ``[0, 15]`` (AWQ indices; post-GELU u4 acts).

Does **not** duplicate ConvRot / Asym / AWQ layout math — only the shared
codec surface those modules keep private today. Prefer importing from here
for SVDQuant / AWQ pack helpers.

"""

import torch

_DEFAULT_GROUP = 64


def pack_int4_row_major(values: torch.Tensor) -> torch.Tensor:
    """Pack ``(..., K)`` int4 values into ``(..., K//2)`` int8 (low = even).

    Accepts signed ``[-8, 7]`` or unsigned ``[0, 15]``; masks via ``& 0x0F``.
    """
    import torch

    if values.shape[-1] % 2 != 0:
        raise ValueError(f"last dim must be even, got {values.shape[-1]}")
    lo = values[..., 0::2].to(torch.int32) & 0x0F
    hi = values[..., 1::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.int8)


def unpack_int4_row_major(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_int4_row_major` with **signed** nibble ``[-8, 7]``.

    CUDA/gfx120x uses device ``expand_signed_i4``; CPU keeps a torch codec for
    offline oracles in ``tests/kernels/oracles.py``.
    """
    import torch

    if packed.is_cuda:
        from kernels.quant.rdna4_convrot_w4a4 import expand_signed_i4

        out = expand_signed_i4(packed)
        return out.reshape(*packed.shape[:-1], packed.shape[-1] * 2)

    x32 = packed.to(torch.int32)
    lo = x32 & 0x0F
    hi = (x32 >> 4) & 0x0F
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    stacked = torch.stack([lo, hi], dim=-1)
    return stacked.reshape(*packed.shape[:-1], -1).to(torch.int8)


def unpack_uint4_row_major(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_int4_row_major` with **unsigned** nibble ``[0, 15]``."""
    import torch

    x32 = packed.to(torch.int32)
    lo = x32 & 0x0F
    hi = (x32 >> 4) & 0x0F
    stacked = torch.stack([lo, hi], dim=-1)
    return stacked.reshape(*packed.shape[:-1], -1).to(torch.int8)


def dequant_int4_groupwise_signed(
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    group_size: int = _DEFAULT_GROUP,
) -> torch.Tensor:
    """Host dequant: signed packed INT4 × per-group scales → compute dtype.

    Layout (reference SVDQuant / ConvRot-style groupwise)::

        qweight[N, K//2] int8   (signed nibble pack)
        wscales[K//G, N]        bf16/fp16/fp32

        W[n, k] = q_signed[n, k] * wscales[k // G, n]
    """

    if qweight.dim() != 2:
        raise ValueError(f"qweight must be 2D, got {tuple(qweight.shape)}")
    if qweight.dtype != torch.int8:
        raise ValueError(f"qweight must be int8, got {qweight.dtype}")
    if wscales.device != qweight.device:
        raise ValueError(f"wscales device mismatch: {wscales.device} vs {qweight.device}")
    if wscales.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"wscales dtype unsupported: {wscales.dtype}")
    n, k_half = qweight.shape
    k = k_half * 2
    groups = (k + group_size - 1) // group_size
    if tuple(wscales.shape) != (groups, n):
        raise ValueError(f"wscales must be {(groups, n)}, got {tuple(wscales.shape)}")
    compute_dtype = wscales.dtype
    w_int = unpack_int4_row_major(qweight).to(compute_dtype)
    if k % group_size == 0:
        w_g = w_int.view(n, groups, group_size)
        scales_ng = wscales.t().unsqueeze(-1)  # (n, groups, 1)
        return (w_g * scales_ng).view(n, k)
    g_idx = torch.arange(k, device=qweight.device) // group_size
    return w_int * wscales.t()[:, g_idx]


def dequant_uint4_groupwise_awq(
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    wzeros: torch.Tensor | None,
    group_size: int = _DEFAULT_GROUP,
) -> torch.Tensor:
    """Host AWQ-style dequant: ``(u4 - 8) * scale + zero`` (reference AWQ wire).

    Thin shared helper — preferred for new call sites; AWQ device path stays in
    ``rdna4_awq_w4a16``.
    """

    if qweight.dim() != 2:
        raise ValueError(f"qweight must be 2D, got {tuple(qweight.shape)}")
    if qweight.dtype != torch.int8:
        raise ValueError(f"qweight must be int8, got {qweight.dtype}")
    if wscales.device != qweight.device or wzeros.device != qweight.device:
        raise ValueError("wscales/wzeros must be on the qweight device")
    if wscales.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"wscales dtype unsupported: {wscales.dtype}")
    n, k_half = qweight.shape
    k = k_half * 2
    groups = (k + group_size - 1) // group_size
    if tuple(wscales.shape) != (groups, n):
        raise ValueError(f"wscales must be {(groups, n)}, got {tuple(wscales.shape)}")
    if tuple(wzeros.shape) != (groups, n):
        raise ValueError(f"wzeros must be {(groups, n)}, got {tuple(wzeros.shape)}")
    compute_dtype = wscales.dtype
    w_uint = unpack_uint4_row_major(qweight).to(compute_dtype)
    if k % group_size == 0:
        w_g = w_uint.view(n, groups, group_size)
        scales_ng = wscales.t().unsqueeze(-1)
        zeros_ng = wzeros.t().unsqueeze(-1)
        return ((w_g - 8.0) * scales_ng + zeros_ng).view(n, k)
    g_idx = torch.arange(k, device=qweight.device) // group_size
    scales_nk = wscales.t()[:, g_idx]
    zeros_nk = wzeros.t()[:, g_idx]
    return (w_uint - 8.0) * scales_nk + zeros_nk
