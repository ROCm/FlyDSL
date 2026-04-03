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


def shuffle_weight_flat_int4(
    w_i8: torch.Tensor,
    tile_k: int,
    layout=(16, 16),
) -> torch.Tensor:
    """Pre-shuffle + pack int8 weights into CK-tile flat layout for W4A16.

    Produces a packed int4 tensor where each thread's data across K tiles is
    contiguous in memory, enabling ``buffer_load_dwordx4`` (16B) loads per
    N column per K tile instead of 8 separate 4B loads.

    Layout: ``[N0, KLane=4, NLane=16, K0 * KPack]``
    where K0 = K_packed / (4 * KPack), KPack = tile_k // 8 bytes per thread.

    Args:
        w_i8: Weight tensor of shape ``[..., N, K]`` with int8 values in [-8, 7].
        tile_k: K tile size used by the kernel (must be multiple of 128).
        layout: Tile layout parameters (default (16, 16)).

    Returns:
        Packed int4 tensor (as int8 dtype, half the K dimension) with flat layout.
    """
    NLane = layout[0]  # 16
    KLane = 4

    # Step 1: Apply existing preshuffle (reorders within 16N x 32K blocks)
    w_shuf = shuffle_weight(w_i8, layout=layout, use_int4=True)

    # Step 2: Pack int4 (same packing as _pack_shuffled_int8_to_packed_int4_no_perm)
    flat = w_shuf.contiguous().view(-1).to(torch.int16)
    assert flat.numel() % 8 == 0
    u = (flat & 0xF).to(torch.uint8).view(-1, 8)
    packed = torch.empty((u.shape[0], 4), device=u.device, dtype=torch.uint8)
    packed[:, 0] = u[:, 0] | (u[:, 4] << 4)
    packed[:, 1] = u[:, 1] | (u[:, 5] << 4)
    packed[:, 2] = u[:, 2] | (u[:, 6] << 4)
    packed[:, 3] = u[:, 3] | (u[:, 7] << 4)
    w_packed = packed.view(-1).to(torch.int8)

    # Original shape: [..., N, K] -> packed: [..., N, K//2]
    orig_shape = list(w_i8.shape)
    N = orig_shape[-2]
    K = orig_shape[-1]
    K_packed = K // 2
    batch_dims = orig_shape[:-2]
    batch_size = 1
    for d in batch_dims:
        batch_size *= d

    w_packed = w_packed.view(batch_size, N, K_packed)

    # Step 3: Reshape to [batch, N0, NLane, K0, KLane, KPack]
    # Current preshuffle layout within packed data:
    #   Within each (N0, K0) block of size (NLane=16, KLane*KPack=4*KPack):
    #     data is ordered as [KLane, NLane, KPack] (from shuffle_weight permute)
    #   But across K0 blocks, K0 is the outer dimension.
    #
    # The preshuffle reorders the data within 16N x 32K blocks. After packing,
    # each 16N x 32K block becomes 16N x 16B (32 int4 = 16 packed bytes).
    # Within each block: [KLane=4, NLane=16, KPack_block=1 (for 32K block)]
    #
    # For tile_k=128 with K=128: 4 blocks of 32K each.
    # For flat layout we need all 4 blocks' data to be contiguous per thread.
    #
    # Preshuffle block size in packed bytes:
    block_k_packed = (layout[1] * 2) // 2  # BK=32 int4 values -> 16 packed bytes
    K0_blocks = K_packed // block_k_packed  # number of K blocks

    # Reshape: [batch, N//16, 16, K_packed//block_k_packed, block_k_packed]
    # But within each block, the preshuffle has already arranged data as
    # [KLane, NLane, KPack_intra], so the packed block of 16 bytes has:
    #   [KLane=4, NLane=16, KPack_intra=1] ... wait, 4*16*1=64 but block is 16 bytes.
    #
    # Actually, after shuffle_weight + pack for int4:
    # Each 16N x 32K block (32 int4 values per N row) becomes:
    #   Shuffled: [BK//K, NLane, K] = [1, 16, 32] for int4
    #   After pack: [1, 16, 16] bytes (32 int4 -> 16 packed bytes)
    # So within a block: NLane is the outer dim, 16 bytes inner (contiguous per N lane).
    #
    # The full packed tensor shape: [batch, N0, K0, NLane, block_k_packed]
    # For flat layout we want: [batch, N0, KLane, NLane, K0, KPack_per_tile]
    # where KLane and KPack split the block_k_packed dimension.
    #
    # Within each block's 16 packed bytes per N-lane:
    #   The MFMA K decomposition: KLane=4 threads, each gets 4 packed bytes.
    #   byte[b] for b=0..3 -> thread at KLane = b (covers 8 int4 = 2 K16 MFMAs)
    # So: block_k_packed = KLane * bytes_per_klane = 4 * 4 = 16. ✓
    bytes_per_klane_per_block = block_k_packed // KLane  # 4 bytes

    w_5d = w_packed.view(batch_size, N // NLane, K0_blocks, NLane, KLane, bytes_per_klane_per_block)
    # Current order: [batch, N0, K0, NLane, KLane, KPack_intra]
    # Target order:  [batch, N0, KLane, NLane, K0, KPack_intra]
    w_flat = w_5d.permute(0, 1, 4, 3, 2, 5).contiguous()
    w_flat = w_flat.view(batch_size, N, K_packed)

    # Restore batch dims
    w_flat = w_flat.view(*batch_dims, N, K_packed)
    return w_flat