# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Independent decoder for MXFP4-transported inputs used by the M13 oracle."""

from __future__ import annotations

import torch

MX_GROUP_SIZE = 32

# OCP E2M1 values indexed by the raw four-bit payload. Packed MXFP4 stores
# the even logical element in the low nibble and the odd element in the high.
_E2M1_LUT = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _payload_bytes(payload: torch.Tensor) -> torch.Tensor:
    if payload.dtype == torch.uint8:
        raw = payload
    elif payload.dtype == torch.float4_e2m1fn_x2:
        raw = payload.view(torch.uint8)
    else:
        raise TypeError(
            "MXFP4 payload must be uint8 or torch.float4_e2m1fn_x2, "
            f"got {payload.dtype}"
        )
    if raw.ndim != 2:
        raise ValueError(f"MXFP4 payload must be rank 2, got shape={tuple(raw.shape)}")
    return raw.contiguous()


def decode_mxfp4_e8m0(payload: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Decode row-major packed E2M1 with one row-major E8M0 byte per 32 values."""
    raw = _payload_bytes(payload)
    if scales.dtype != torch.uint8 or scales.ndim != 2:
        raise TypeError("MXFP4 scales must be a rank-2 uint8 tensor")
    rows, packed_cols = raw.shape
    cols = packed_cols * 2
    if cols % MX_GROUP_SIZE:
        raise ValueError(f"decoded width {cols} must be divisible by {MX_GROUP_SIZE}")
    expected_scale_shape = (rows, cols // MX_GROUP_SIZE)
    if tuple(scales.shape) != expected_scale_shape:
        raise ValueError(
            f"MXFP4 scales must have shape {expected_scale_shape}, got {tuple(scales.shape)}"
        )
    if scales.device != raw.device:
        raise ValueError("MXFP4 payload and scales must be on the same device")

    nibble = torch.empty((rows, cols), dtype=torch.long, device=raw.device)
    nibble[:, 0::2] = (raw & 0xF).long()
    nibble[:, 1::2] = (raw >> 4).long()
    lut = torch.tensor(_E2M1_LUT, dtype=torch.float32, device=raw.device)
    values = lut[nibble]

    # E8M0 is the exponent field of an IEEE float with a zero mantissa.
    scale_bits = scales.to(torch.int32) << 23
    scale_f32 = scale_bits.contiguous().view(torch.float32)
    return values * scale_f32.repeat_interleave(MX_GROUP_SIZE, dim=1)
