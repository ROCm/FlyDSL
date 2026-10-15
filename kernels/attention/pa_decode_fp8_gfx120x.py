# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x paged-attention decode launch. Same arguments as ``pa_decode_ps_launch``.

The WMMA paged kernel writes the combined output. A 5D cache is the vectorized
layout and a 4D cache is linear. ``sliding_window`` is the causal left window.
"""

from __future__ import annotations

import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.attention.pa_decode_tile_gfx120x import pa_decode_tile
from kernels.common.utils import cdiv

KV_COMPUTE_BLOCK = 256


def get_recommended_splits(
    num_sequences: int,
    num_kv_heads: int,
    split_kv_blocks: int = 1,
    *,
    sliding_window: int = 0,
    context_partition_size: int = KV_COMPUTE_BLOCK,
    query_length: int = 1,
) -> int:
    """Host split hint. The WMMA launch does not partition the cache."""
    if sliding_window > 0:
        return cdiv(sliding_window + query_length - 1, context_partition_size) + 1
    props = torch.cuda.get_device_properties(torch.device("cuda"))
    num_sm = props.multi_processor_count * 2
    denom = max(1, num_sequences * num_kv_heads * split_kv_blocks)
    n = cdiv(num_sm, denom) * split_kv_blocks
    return max(4, min(n, 8))


def pa_decode_ps_launch(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    softmax_scale: float,
    key_scale: torch.Tensor = None,
    value_scale: torch.Tensor = None,
    *,
    sliding_window: int = 0,
    metadata: dict = None,
    block_tables: torch.Tensor = None,
    max_context_partition_num: int = 0,
    exp_sums: torch.Tensor = None,
    max_logits: torch.Tensor = None,
    temporary_output: torch.Tensor = None,
    stream=None,
) -> str:
    """Launch gfx120x decode. ``kv_page_indices`` and ``metadata`` are unused."""
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx120"), f"pa_decode_ps_launch requires gfx120x, got {arch}"
    del kv_page_indices, kv_indptr, metadata
    if max_context_partition_num <= 0:
        raise ValueError("max_context_partition_num must be positive.")
    if block_tables is None:
        raise ValueError("pa_decode_ps_launch requires block_tables")
    if query.dtype != key_cache.dtype or query.dtype != value_cache.dtype:
        raise ValueError(
            "gfx120x pa_decode_ps_launch keeps Q and KV in one dtype, "
            f"got {query.dtype}/{key_cache.dtype}/{value_cache.dtype}"
        )
    pa_decode_tile(
        output,
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        key_scale,
        value_scale,
        softmax_scale=softmax_scale,
        stream=stream,
        num_partitions=max_context_partition_num,
        pmax=max_logits,
        psum=exp_sums,
        pout=temporary_output,
        sliding_window=int(sliding_window),
    )
    if sliding_window > 0:
        return "ps_sw_fused_partitioned"
    return "ps_small_block"
