#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Shared test utilities for the grouped FP8 blockscale GEMM tests.

Used by both `test_grouped_gemm_blockscale_contiguous.py` and
`test_grouped_gemm_blockscale_masked.py`. Contains FP8 dtype selection,
E8M0 quantization helpers, and the byte-identical B-tensor quantization
path. The contiguous-vs-masked-specific A-tensor quantization
(`quantize_to_fp8` 2D vs `quantize_a_masked_to_fp8` 3D) lives in the
respective test files.
"""

import torch

from flydsl.runtime.device import get_rocm_arch


ARCH = get_rocm_arch()
# Use appropriate FP8 dtype for the architecture
DTYPE_FP8 = torch.float8_e4m3fn if "gfx95" in ARCH else torch.float8_e4m3fnuz
# gfx950 uses hardware E8M0 block scaling — quantization must use E8M0-truncated scales
USE_UE8M0 = "gfx95" in ARCH


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def fp32_to_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Round FP32 scale UP to E8M0 precision (ceiling on exponent).

    Rounding up is required so that x / scale_e8m0 <= fp8_max — truncation
    would shrink the scale, causing FP8 saturation and a systematic bias on
    every block.
    """
    bits = scale.abs().float().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float32)


def fp32_e8m0_to_byte(scale_e8m0_f32: torch.Tensor) -> torch.Tensor:
    """Extract the E8M0 byte from a float that was previously rounded by
    fp32_to_e8m0. Returns uint8. Use this when handing scales to the kernel
    so dequant uses bit-exact the same scale the kernel will apply (HW E8M0
    path uses byte 0 of the i32 scale operand)."""
    bits = scale_e8m0_f32.view(torch.int32)
    return ((bits >> 23) & 0xFF).to(torch.uint8)


def quantize_b_to_fp8(
    b: torch.Tensor, scale_block_n: int = 128, scale_block_k: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize B tensor to FP8 with per-block scaling.

    Args:
        b: Input tensor [num_groups, N, K]
        scale_block_n: N-dimension block size
        scale_block_k: K-dimension block size

    Returns:
        (b_fp8, scale_b): FP8 tensor and scale factors [num_groups, scale_n, scale_k]
    """
    num_groups, N, K = b.shape
    nblk_n = N // scale_block_n
    nblk_k = K // scale_block_k

    # Reshape to [num_groups, nblk_n, scale_block_n, nblk_k, scale_block_k]
    b_blocks = b.view(num_groups, nblk_n, scale_block_n, nblk_k, scale_block_k)

    # Compute per-block max
    b_amax = b_blocks.abs().amax(dim=(2, 4)).clamp(min=1e-12)

    fp8_max = torch.finfo(DTYPE_FP8).max
    scale = b_amax / fp8_max

    # Round to E8M0 when hardware scaling is used (gfx950); keep as FP32 for the
    # quantization divide, then pre-pack to uint8 below for kernel consumption.
    if USE_UE8M0:
        scale = fp32_to_e8m0(scale)

    # Quantize
    b_scaled = b_blocks / scale.view(num_groups, nblk_n, 1, nblk_k, 1)
    b_fp8 = b_scaled.to(DTYPE_FP8).view(num_groups, N, K)

    if USE_UE8M0:
        scale = fp32_e8m0_to_byte(scale)

    return b_fp8, scale


def _as_i8(t: torch.Tensor) -> torch.Tensor:
    """View FP8 tensor as int8 for kernel interface."""
    return t.view(torch.int8)
