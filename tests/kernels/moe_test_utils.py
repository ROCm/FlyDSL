#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Shared test helpers for MoE kernel harnesses."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

# Optional: use aiter's exact routing/sorting implementation (matches
# `aiter/op_tests/test_moe_2stage.py`). Some environments ship aiter python but
# miss required JIT .so dependencies; we fall back gracefully.
try:
    import aiter  # noqa: F401
    from aiter.fused_moe import moe_sorting as aiter_moe_sorting

    HAS_AITER = True
except Exception:
    HAS_AITER = False
    aiter_moe_sorting = None


RoutingBuffers = Tuple[
    torch.Tensor,  # sorted_token_ids
    torch.Tensor,  # sorted_weights
    torch.Tensor,  # sorted_expert_ids
    torch.Tensor,  # num_valid_ids (shape [1], i32)
    int,           # sorted_size
    int,           # blocks
]


def moe_sorting_torch_native(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    block_size: int,
    expert_mask: Optional[torch.Tensor] = None,
    num_local_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch reference for aiter's moe_sorting."""

    assert topk_ids.is_cuda and topk_weights.is_cuda
    device = topk_ids.device
    tokens, topk = topk_ids.shape
    topk = topk_ids.shape[1]

    max_num_tokens_padded = int(
        topk_ids.numel() + int(num_experts) * int(block_size) - int(topk)
    )
    max_num_m_blocks = int(
        (max_num_tokens_padded + int(block_size) - 1) // int(block_size)
    )

    init_val = (int(topk) << 24) | int(tokens)
    sorted_ids = torch.full(
        (max_num_tokens_padded,), init_val, dtype=torch.int32, device=device
    )
    sorted_weights = torch.empty(
        (max_num_tokens_padded,), dtype=torch.float32, device=device
    )
    sorted_expert_ids = torch.full(
        (max_num_m_blocks,), -1, dtype=torch.int32, device=device
    )
    num_tokens_post_pad = torch.empty((2,), dtype=torch.int32, device=device)

    if num_local_tokens is not None:
        topk_ids = topk_ids[: num_local_tokens.item()]

    sorted_ids_begin = 0
    sorted_expert_ids_begin = 0
    skip_expert_num = 0
    for expert_id in range(int(num_experts)):
        if expert_mask is not None and int(expert_mask[expert_id].item()) == 0:
            skip_expert_num += 1
            continue
        token_id, topk_id = torch.where(topk_ids == expert_id)
        tokens_num = int(token_id.numel())
        sorted_expert_ids_num = int(
            (tokens_num + int(block_size) - 1) // int(block_size)
        )
        tokens_num_pad = int(sorted_expert_ids_num * int(block_size))
        sorted_ids[sorted_ids_begin : sorted_ids_begin + tokens_num] = (
            (topk_id.to(torch.int32) << 24) | token_id.to(torch.int32)
        )
        sorted_weights[sorted_ids_begin : sorted_ids_begin + tokens_num] = (
            topk_weights[token_id, topk_id].to(torch.float32)
        )
        sorted_ids_begin = int(sorted_ids_begin + tokens_num_pad)
        sorted_expert_ids[
            sorted_expert_ids_begin : sorted_expert_ids_begin + sorted_expert_ids_num
        ] = int(expert_id - skip_expert_num)
        sorted_expert_ids_begin = int(
            sorted_expert_ids_begin + sorted_expert_ids_num
        )

    num_tokens_post_pad[0] = int(sorted_ids_begin)
    num_tokens_post_pad[1] = int(topk_ids.shape[0])
    return sorted_ids, sorted_weights, sorted_expert_ids, num_tokens_post_pad


def _maybe_aiter_moe_sorting(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    model_dim: int,
    block_m: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids) or None."""

    if not HAS_AITER or aiter_moe_sorting is None:
        return None
    try:
        topk_ids_i32 = topk_ids.to(torch.int32)
        topk_w_f32 = topk_weights.to(torch.float32)
        sorted_ids, sorted_w, sorted_expert_ids, num_valid_ids, _moe_buf = (
            aiter_moe_sorting(
                topk_ids_i32,
                topk_w_f32,
                num_experts,
                model_dim,
                torch.float16,
                block_m,
            )
        )
        if num_valid_ids.numel() > 1:
            num_valid_ids = num_valid_ids[:1].contiguous()
        return sorted_ids, sorted_w, sorted_expert_ids, num_valid_ids
    except Exception:
        return None


def build_routing_buffers(
    *,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    experts: int,
    tile_m: int,
    model_dim: Optional[int] = None,
    moe_sort_mode: Optional[str] = None,
    expert_mask: Optional[torch.Tensor] = None,
    num_local_tokens: Optional[torch.Tensor] = None,
) -> RoutingBuffers:
    """Build routing buffers once, reusable across stage1 + stage2."""

    default_mode = "aiter" if HAS_AITER else "torch"
    sort_mode = str(
        moe_sort_mode or os.environ.get("flydsl_MOE_SORT_MODE", default_mode)
    ).lower().strip()
    if sort_mode not in ("aiter", "torch"):
        raise ValueError(
            f"invalid moe_sort_mode={sort_mode!r} (expected 'aiter' or 'torch')"
        )

    if sort_mode == "torch":
        (
            sorted_token_ids,
            sorted_weights,
            sorted_expert_ids,
            num_tokens_post_pad,
        ) = moe_sorting_torch_native(
            topk_ids=topk_ids.to(torch.int32),
            topk_weights=topk_weights.to(torch.float32),
            num_experts=int(experts),
            block_size=int(tile_m),
            expert_mask=expert_mask,
            num_local_tokens=num_local_tokens,
        )
        num_valid_ids = num_tokens_post_pad[:1].contiguous()
        sorted_size = int(sorted_token_ids.numel())
        blocks = int(sorted_expert_ids.numel())
        return (
            sorted_token_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            sorted_size,
            blocks,
        )

    if not HAS_AITER:
        raise RuntimeError(
            "aiter is not available; cannot build routing buffers (moe_sort_mode='aiter')."
        )
    if model_dim is None:
        raise ValueError("model_dim is required when moe_sort_mode='aiter'")

    res = _maybe_aiter_moe_sorting(
        topk_ids,
        topk_weights,
        num_experts=experts,
        model_dim=model_dim,
        block_m=tile_m,
    )
    if res is None:
        raise RuntimeError("aiter moe_sorting failed/unavailable; cannot build routing buffers.")
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids = res
    sorted_token_ids = sorted_token_ids.contiguous()
    sorted_weights = sorted_weights.contiguous()
    sorted_expert_ids = sorted_expert_ids.contiguous()
    sorted_size = int(sorted_token_ids.numel())
    blocks = int(sorted_expert_ids.numel())
    return (
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        sorted_size,
        blocks,
    )


def get_topk_valid_mask(
    topk_ids: torch.Tensor,
    expert_mask: Optional[torch.Tensor] = None,
    *,
    dtype: torch.dtype = torch.int8,
) -> torch.Tensor:
    """Build valid_mask [tokens, topk] for optional EP-style masking."""

    if expert_mask is None:
        return torch.ones(topk_ids.shape, dtype=dtype, device=topk_ids.device)
    return expert_mask[topk_ids].to(dtype)
