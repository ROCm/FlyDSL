# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Independent OCP MXFP4/E8M0 decoder for the SmoothQuant acceptance test."""

from __future__ import annotations

import torch


MX_GROUP_SIZE = 32
_E2M1_LUT = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def decode_mxfp4_e8m0(payload: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Decode row-major packed E2M1 with one E8M0 byte per 32 values."""
    if payload.dtype == torch.float4_e2m1fn_x2:
        raw = payload.view(torch.uint8)
    elif payload.dtype == torch.uint8:
        raw = payload
    else:
        raise TypeError(f"unexpected MXFP4 payload dtype {payload.dtype}")
    if raw.ndim != 2 or scales.dtype != torch.uint8 or scales.ndim != 2:
        raise TypeError("MXFP4 payload/scales must be rank-2 byte tensors")
    rows, packed_cols = raw.shape
    cols = packed_cols * 2
    expected = (rows, cols // MX_GROUP_SIZE)
    if cols % MX_GROUP_SIZE or tuple(scales.shape) != expected:
        raise ValueError(
            f"invalid MXFP4 shapes payload={tuple(raw.shape)} scales={tuple(scales.shape)}"
        )
    nibble = torch.empty((rows, cols), dtype=torch.long, device=raw.device)
    nibble[:, 0::2] = (raw & 0xF).long()
    nibble[:, 1::2] = (raw >> 4).long()
    values = torch.tensor(_E2M1_LUT, dtype=torch.float32, device=raw.device)[nibble]
    scale_f32 = (scales.to(torch.int32) << 23).contiguous().view(torch.float32)
    return values * scale_f32.repeat_interleave(MX_GROUP_SIZE, dim=1)
