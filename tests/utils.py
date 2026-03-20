# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Shared utilities for FlyDSL kernel testing."""

import torch
import functools
from typing import Optional


# Simple dtypes namespace used by pertoken_quant
class dtypes:
    fp32 = torch.float32
    fp16 = torch.float16
    bf16 = torch.bfloat16
    i8 = torch.int8
    i32 = torch.int32


@functools.lru_cache()
def get_dtype_max(dtype):
    """Get max value for a given dtype."""
    try:
        return torch.finfo(dtype).max
    except Exception:
        return torch.iinfo(dtype).max


def pertoken_quant(
    x,
    scale=None,
    x_scale=None,  # smooth_scale
    scale_dtype=dtypes.fp32,
    quant_dtype=dtypes.i8,
    dtypeMax=None,
):
    x = x.to(dtypes.fp32)
    if x_scale is None:
        hidden_states = x
    else:
        # smooth quant
        hidden_states = x * x_scale

    if dtypeMax is None:
        dtypeMax = get_dtype_max(quant_dtype)

    # Be robust to rare non-finite values (can appear from FP8 pipelines at extreme shapes):
    # - Avoid producing inf scales (which would later lead to 0*inf -> NaN in dequant).
    # - Avoid propagating NaN/Inf into the quantized tensor.
    hidden_states = torch.nan_to_num(
        hidden_states,
        nan=0.0,
        posinf=float(dtypeMax),
        neginf=-float(dtypeMax),
    )

    per_token_scale = scale
    if scale is None:
        # [m, 1]
        # Avoid materializing a full-size abs() temporary (can be huge for MoE weights).
        # max(abs(x)) = max(max(x), -min(x))
        per_token_max = torch.amax(hidden_states, dim=-1, keepdim=True)
        per_token_min = torch.amin(hidden_states, dim=-1, keepdim=True)
        per_token_amax = torch.maximum(per_token_max, -per_token_min)
        per_token_scale = per_token_amax / dtypeMax
        per_token_scale[per_token_scale == 0] = 1

    per_token_scale = torch.nan_to_num(per_token_scale, nan=1.0, posinf=1.0, neginf=1.0)

    # quant hidden_states
    y = (hidden_states / per_token_scale).to(dtype=quant_dtype)
    y_scale = per_token_scale.to(scale_dtype)
    return y, y_scale


def shuffle_weight(x: torch.Tensor, layout=(16, 16), use_int4=False) -> torch.Tensor:
    # Hardcode BLOCK_K and BLOCK_N
    x_type = x.dtype
    if hasattr(torch, "float4_e2m1fn_x2") and x_type == torch.float4_e2m1fn_x2:
        x = x.view(torch.uint8)

    IN, IK = layout
    BK = IK * 2
    K = 16 // x.element_size() if not use_int4 else 32
    BN = IN
    assert x.shape[-2] % BN == 0, f"{x.shape[-2]} % {BN} == {x.shape[-2] % BN }"
    assert x.shape[-1] % BK == 0, f"{x.shape[-1]} % {BK} == {x.shape[-1] % BK }"

    x_ = x
    x_ = x_.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K)
    x_ = x_.permute(0, 1, 3, 4, 2, 5)
    x_ = x_.contiguous()
    x_ = x_.view(*x.shape)
    x_ = x_.view(x_type)
    x_.is_shuffled = True
    return x_


def shuffle_scale_for_int4(scale: torch.Tensor, group_size: int = 32, layout=(16, 16)) -> torch.Tensor:
    """Prepare scale tensor for W4A16 groupwise scale kernel.

    NOTE: Despite the name, this function does NOT shuffle the scale tensor.
    The kernel uses the [E, num_groups, N] layout (Opt 0: cache-friendly) where
    adjacent threads read adjacent N elements (stride-1 access).

    Scale indexing uses: scale_idx = expert_offset*(G-1) + n_global + group_idx*N_pe

    Args:
        scale: Scale tensor of shape [E, num_groups, N] where num_groups = K_dim // group_size
        group_size: Group size for quantization (must be 32 for FlyDSL)
        layout: Tile layout (unused, kept for API compatibility)

    Returns:
        Scale tensor in [E, num_groups, N] layout, ready for kernel consumption.
    """
    if group_size != 32:
        raise ValueError(
            f"shuffle_scale_for_int4 only supports group_size=32, got {group_size}. "
            f"This is due to int4 preshuffle layout constraints."
        )

    return scale.contiguous()