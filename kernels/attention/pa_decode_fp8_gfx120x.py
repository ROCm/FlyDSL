# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x paged-attention decode launch. Same arguments as ``pa_decode_ps_launch``.

The WMMA paged kernel writes the combined output. A 5D cache is the vectorized
layout and a 4D cache is linear. ``sliding_window`` is the causal left window.
A missing block table is built on the host from the CSR page list. A bf16 or
fp16 query with an fp8 or int8 cache dequantizes the pages named in that table,
then uses the same-dtype launch.
"""

from __future__ import annotations

import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.attention.pa_decode_tile_gfx120x import pa_decode_tile
from kernels.common.utils import cdiv

KV_COMPUTE_BLOCK = 256

_FP16_QUERY = (torch.float16, torch.bfloat16)
_QUANT_CACHE = tuple(
    getattr(torch, name)
    for name in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz", "int8")
    if hasattr(torch, name)
)


def _block_table_from_csr(
    kv_page_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    num_seqs: int,
) -> torch.Tensor:
    """Rectangular ``[num_seqs, max_pages]`` int32 table. Copies page ids only."""
    indptr = kv_indptr.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    indices = kv_page_indices.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    if int(indptr.numel()) != num_seqs + 1:
        raise ValueError(f"kv_indptr length {int(indptr.numel())} != num_seqs + 1 ({num_seqs + 1})")
    if num_seqs == 0:
        return torch.zeros((0, 0), dtype=torch.int32)
    lengths = indptr[1:] - indptr[:-1]
    if int(lengths.min().item()) < 0:
        raise ValueError("kv_indptr must be non-decreasing")
    end = int(indptr[-1].item())
    start0 = int(indptr[0].item())
    if start0 < 0 or end > int(indices.numel()):
        raise ValueError(f"kv_indptr range [{start0}, {end}) is outside kv_page_indices length {int(indices.numel())}")
    used_ids = indices[start0:end]
    if used_ids.numel() and (int(used_ids.min().item()) < 0 or int(used_ids.max().item()) > 0x7FFFFFFF):
        raise ValueError("kv_page_indices must fit in non-negative int32")
    max_pages = int(lengths.max().item())
    table = torch.zeros((num_seqs, max_pages), dtype=torch.int32)
    if max_pages == 0 or used_ids.numel() == 0:
        return table
    positions = torch.arange(start0, end, dtype=torch.int64)
    row_starts = torch.repeat_interleave(indptr[:-1], lengths)
    cols = positions - row_starts
    seq_ids = torch.repeat_interleave(torch.arange(num_seqs, dtype=torch.int64), lengths)
    table[seq_ids, cols] = used_ids.to(torch.int32)
    return table


def _cache_kind(cache: torch.Tensor, which: str) -> str:
    if cache.dim() == 4:
        return "linear"
    if cache.dim() == 5:
        return "vectorized_k" if which == "k" else "vectorized_v"
    raise ValueError(f"gfx120x cache rank {cache.dim()} is not 4 or 5")


def _heads_and_page(cache: torch.Tensor, kind: str) -> tuple[int, int]:
    if kind == "linear":
        return int(cache.shape[2]), int(cache.shape[1])
    if kind == "vectorized_k":
        return int(cache.shape[1]), int(cache.shape[3])
    return int(cache.shape[1]), int(cache.shape[2]) * int(cache.shape[4])


def _scale_factor(scale, num_blocks: int, num_heads: int, page: int, device: torch.device):
    """Per-tensor scalar, or a factor shaped from ``[blocks, heads, page]``."""
    if scale is None:
        return 1.0
    if not isinstance(scale, torch.Tensor):
        return float(scale)
    factor = scale.detach().to(device=device, dtype=torch.float32)
    if factor.numel() == 1:
        return factor.reshape(())
    per_token = (num_blocks, num_heads, page)
    if factor.dim() != 3 and factor.numel() == num_blocks * num_heads * page:
        factor = factor.reshape(per_token)
    if factor.shape[:1] != (num_blocks,):
        raise ValueError(f"scale shape {tuple(scale.shape)} does not match {num_blocks} cache blocks")
    return factor


def _broadcast_page_scale(factor, values: torch.Tensor, kind: str):
    if not isinstance(factor, torch.Tensor):
        return float(factor)
    if factor.ndim == 0 or factor.numel() == 1:
        return factor.reshape(())
    n = int(values.shape[0])
    if kind == "linear":
        page, heads = int(values.shape[1]), int(values.shape[2])
        if tuple(factor.shape) == (n, heads, page):
            return factor.permute(0, 2, 1).unsqueeze(-1)
        if tuple(factor.shape) == (n, heads):
            return factor.reshape(n, 1, heads, 1)
        if tuple(factor.shape) == (n,):
            return factor.reshape(n, 1, 1, 1)
    elif kind == "vectorized_k":
        heads, page = int(values.shape[1]), int(values.shape[3])
        if tuple(factor.shape) == (n, heads, page):
            return factor.reshape(n, heads, 1, page, 1)
        if tuple(factor.shape) == (n, heads):
            return factor.reshape(n, heads, 1, 1, 1)
        if tuple(factor.shape) == (n,):
            return factor.reshape(n, 1, 1, 1, 1)
    elif kind == "vectorized_v":
        heads, groups, vec = int(values.shape[1]), int(values.shape[2]), int(values.shape[4])
        page = groups * vec
        if tuple(factor.shape) == (n, heads, page):
            return factor.reshape(n, heads, groups, 1, vec)
        if tuple(factor.shape) == (n, heads):
            return factor.reshape(n, heads, 1, 1, 1)
        if tuple(factor.shape) == (n,):
            return factor.reshape(n, 1, 1, 1, 1)
    else:
        raise ValueError(f"unknown cache kind {kind}")
    try:
        torch.broadcast_shapes(tuple(factor.shape), tuple(values.shape))
    except RuntimeError as exc:
        raise ValueError(
            f"{kind} scale shape {tuple(factor.shape)} does not broadcast onto cache page {tuple(values.shape)}"
        ) from exc
    return factor


def _dequant_selected(
    cache: torch.Tensor, pages: torch.Tensor, scale, out_dtype: torch.dtype, kind: str
) -> torch.Tensor:
    """Dequant ``cache[pages]`` into ``out_dtype``. Other blocks are not read."""
    page_index = pages.to(device=cache.device, dtype=torch.long)
    src = cache.index_select(0, page_index)
    heads, page = _heads_and_page(cache, kind)
    factor = _scale_factor(scale, int(cache.shape[0]), heads, page, cache.device)
    if isinstance(factor, torch.Tensor) and factor.ndim > 0 and int(factor.shape[0]) == int(cache.shape[0]):
        factor = factor.index_select(0, page_index)
    view = _broadcast_page_scale(factor, src, kind)
    vals = src.to(torch.float32)
    if isinstance(view, torch.Tensor):
        vals = vals * view.to(dtype=torch.float32)
    else:
        vals = vals * float(view)
    return vals.to(dtype=out_dtype)


def _dequant_referenced_caches(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    key_scale,
    value_scale,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compact caches holding only the block-table pages, plus a remapped table."""
    if int(key_cache.shape[0]) != int(value_cache.shape[0]):
        raise ValueError(f"key/value block count mismatch {int(key_cache.shape[0])} vs {int(value_cache.shape[0])}")
    if key_cache.device != value_cache.device:
        raise ValueError("key_cache and value_cache must be on the same device")
    flat = block_tables.detach().to(dtype=torch.int64).reshape(-1)
    pages = torch.unique(flat)
    remap = torch.searchsorted(pages, flat).to(dtype=torch.int32).reshape(block_tables.shape)
    remap = remap.to(device=key_cache.device)
    if pages.numel() == 0:
        return key_cache[:0].to(dtype=out_dtype), value_cache[:0].to(dtype=out_dtype), remap
    first = int(pages[0].item())
    last = int(pages[-1].item())
    n_blocks = int(key_cache.shape[0])
    if first < 0 or last >= n_blocks:
        raise ValueError(f"block table page id outside [0, {n_blocks}): {first}..{last}")
    k_kind = _cache_kind(key_cache, "k")
    v_kind = _cache_kind(value_cache, "v")
    key_out = _dequant_selected(key_cache, pages, key_scale, out_dtype, k_kind)
    value_out = _dequant_selected(value_cache, pages, value_scale, out_dtype, v_kind)
    return key_out, value_out, remap


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
    key_scale: float | torch.Tensor | None = None,
    value_scale: float | torch.Tensor | None = None,
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
    """Launch gfx120x decode.

    ``metadata`` is accepted and unused; persistent-scheduling metadata stays
    on the host. ``block_tables`` may be omitted when ``kv_indptr`` and
    ``kv_page_indices`` are set: the rectangular int32 table is built from
    that CSR on the host. A bf16 or fp16 query with an fp8 e4m3, fp8 e5m2,
    or int8 cache dequantizes only the pages listed in the block table
    (``key_scale`` and ``value_scale`` default to 1) and then calls the
    same-dtype tile. One partition writes the final output into
    ``temporary_output`` when the partial tensors are present. ``exp_sums``
    and ``max_logits`` stay unused.
    """
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx120"), f"pa_decode_ps_launch requires gfx120x, got {arch}"
    del metadata
    if max_context_partition_num <= 0:
        raise ValueError("max_context_partition_num must be positive.")
    num_seqs = int(context_lengths.shape[0])
    if block_tables is None:
        if kv_page_indices is None or kv_indptr is None:
            raise ValueError("pa_decode_ps_launch requires block_tables, or kv_indptr and kv_page_indices")
        block_tables = _block_table_from_csr(kv_page_indices, kv_indptr, num_seqs).to(device=query.device)
    same_dtype = query.dtype == key_cache.dtype and query.dtype == value_cache.dtype
    if same_dtype:
        launch_key, launch_value = key_cache, value_cache
        launch_k_scale, launch_v_scale = key_scale, value_scale
    elif query.dtype in _FP16_QUERY and key_cache.dtype == value_cache.dtype and key_cache.dtype in _QUANT_CACHE:
        launch_key, launch_value, block_tables = _dequant_referenced_caches(
            key_cache,
            value_cache,
            block_tables,
            key_scale,
            value_scale,
            query.dtype,
        )
        # Scales are folded into the temporary cache. The float tile must not apply them again.
        launch_k_scale, launch_v_scale = None, None
    else:
        raise ValueError(
            "gfx120x pa_decode_ps_launch keeps Q and KV in one dtype, "
            "or a bf16/fp16 query with an fp8 or int8 cache, "
            f"got {query.dtype}/{key_cache.dtype}/{value_cache.dtype}"
        )
    pa_decode_tile(
        output,
        query,
        launch_key,
        launch_value,
        block_tables,
        context_lengths,
        launch_k_scale,
        launch_v_scale,
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
