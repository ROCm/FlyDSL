# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Preparation and inspection of the mono indexer's row-major MXFP4 cache.

Each 68-byte row contains 128 E2M1 values (low nibble first), followed by
four E8M0 scales. This is a logical row format, not ATOM's shuffled cache ABI.
Conversion is setup work and must be performed before Graph capture.
"""

import torch

INDEX_FP4_BYTES = 68


def pack_index_fp4(values: torch.Tensor) -> torch.Tensor:
    """Round BF16 keys to E2M1 RNE, using a power-of-two scale per 32 values."""
    if values.ndim not in (2, 3) or values.shape[-1] != 128 or values.dtype is not torch.bfloat16:
        raise ValueError("index keys must be BF16 [rows,128] or [pages,page_size,128]")
    flat = values.reshape(-1, 128)
    result = torch.empty((flat.shape[0], INDEX_FP4_BYTES), dtype=torch.uint8, device=values.device)
    # Bound temporary memory even when preparing million-token caches.
    for begin in range(0, flat.shape[0], 16384):
        x = flat[begin : begin + 16384].float().reshape(-1, 4, 32)
        if not bool(torch.isfinite(x).all()):
            raise ValueError("index keys must be finite")
        amax = x.abs().amax(-1).clamp_min(1e-10)
        exponent = torch.ceil(torch.log2(amax / 6)).clamp(-126, 127)
        q = x / torch.exp2(exponent)[..., None]
        magnitude = q.abs()
        code = torch.zeros_like(q, dtype=torch.uint8)
        # Midpoints with ties-to-even E2M1 mantissas. No device conversion
        # intrinsic is used here, so this also serves as a packing oracle.
        for lower, midpoint in enumerate((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)):
            up = magnitude > midpoint if lower % 2 == 0 else magnitude >= midpoint
            code += up.to(torch.uint8)
        code |= torch.signbit(q).to(torch.uint8) << 3
        code = code.reshape(-1, 128)
        chunk = result[begin : begin + x.shape[0]]
        chunk[:, :64] = code[:, ::2] | (code[:, 1::2] << 4)
        chunk[:, 64:] = (exponent + 127).to(torch.uint8)
    return result.reshape(*values.shape[:-1], INDEX_FP4_BYTES)


def unpack_index_fp4(cache: torch.Tensor) -> torch.Tensor:
    """Decode the logical cache for references and diagnostics, outside capture."""
    if cache.ndim not in (2, 3) or cache.shape[-1] != INDEX_FP4_BYTES or cache.dtype is not torch.uint8:
        raise ValueError("FP4 index cache must be uint8 [...,68]")
    flat = cache.reshape(-1, INDEX_FP4_BYTES)
    result = torch.empty((flat.shape[0], 128), dtype=torch.bfloat16, device=cache.device)
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=cache.device)
    for begin in range(0, flat.shape[0], 16384):
        chunk = flat[begin : begin + 16384]
        code = torch.stack((chunk[:, :64] & 15, chunk[:, :64] >> 4), dim=-1).reshape(-1, 4, 32)
        value = lut[(code & 7).long()] * torch.where((code & 8) != 0, -1.0, 1.0)
        value *= torch.exp2(chunk[:, 64:].float() - 127)[..., None]
        result[begin : begin + chunk.shape[0]] = value.reshape(-1, 128).bfloat16()
    return result.reshape(*cache.shape[:-1], 128)
