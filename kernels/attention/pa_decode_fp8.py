# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlyDSL Paged Attention Decode with Persistent Scheduling — FP8.

Persistent scheduling (PS) mode:
- Grid = (num_SM, 1, 4) so each CTA handles one 256-token sub-tile of a 1024-token KV page
- Outer work loop iterates over pre-computed worklist from get_pa_metadata_v1
- Inner KV loop iterates pages from kv_page_indices
- Supports split-reduce for load balancing across CUs

Requires: aiter's get_pa_metadata_v1 (module_pa_metadata.so)
"""

from __future__ import annotations

import torch

from kernels.attention.pa_decode_swa import compile_pa_decode_sw, compile_pa_decode_sw_reduce
from kernels.attention.pa_decode_tile import pa_decode_tile
from kernels.attention.pa_metadata import compile_pa_decode_metadata
from kernels.common.tensor_shim import _run_compiled
from kernels.common.utils import cdiv

# ── Kernel geometry constants ────────────────────────────────────────
KV_COMPUTE_BLOCK = 256  # tile size (matches SP3 kTileKV)
# Persistent-grid oversubscription for the metadata decode path: launch
# CU_count * this many workgroups so the HW keeps multiple workgroups resident
# per CU (memory-latency hiding).  1 = original (1 wg/CU).
_PA_METADATA_GRID_OVERSUB = 3
MFMA_N = 16


# =====================================================================
# Launch API — Persistent Scheduling mode
# =====================================================================


def get_pa_metadata(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    kv_indptr: torch.Tensor,
    num_query_heads: int,
    num_kv_heads: int,
    partition_size: int = KV_COMPUTE_BLOCK,
):
    """Compute PA metadata (worklist, reduce maps) via get_pa_metadata_v1.

    The worklist is now load-balanced at **partition** granularity
    (``partition_size`` tokens, default ``KV_COMPUTE_BLOCK=256``) rather than at
    physical block granularity: ``kv_granularity = partition_size``, so each
    scheduled work unit is one partition and ``work_info.kv_start/kv_end`` are
    cumulative **partition** indices (in ``partition_size``-token units), not
    page indices. The partition↔block relationship for the consumer is:
    ``partition_size > block_size`` → ``partition_size // block_size`` blocks per
    partition; otherwise ``block_size // partition_size`` partitions per block.

    NOTE: the consuming decode kernel must interpret kv_start/kv_end as partition
    indices accordingly.

    Returns a dict with: work_indptr, work_info_flat, reduce_indptr,
    reduce_final_map, reduce_partial_map, num_sm, partial_output,
    partial_lse, stride_po_partial, stride_pl_partial.
    """
    from kernels.attention.pa_metadata import get_pa_metadata_info_v1, get_pa_metadata_v1

    dev = query.device
    batch_size = context_lengths.shape[0]
    query_length = query.shape[0] // batch_size
    head_size = query.shape[-1]

    props = torch.cuda.get_device_properties(dev)
    # Oversubscribe the persistent grid: the decode kernel is memory-latency-bound
    # and only ~3 workgroups/CU fit by VGPR, but the worklist defaults to 1 wg/CU
    # (grid = CU count).  Distributing work across num_cu = CU_count * OVERSUB bins
    # (and launching that many workgroups) lets the HW keep multiple workgroups
    # resident per CU → more waves in flight → better latency hiding.
    base_cu = props.multi_processor_count
    num_sm = base_cu * _PA_METADATA_GRID_OVERSUB
    num_sm = (num_sm // num_kv_heads) * num_kv_heads  # keep divisible by num_kv_heads

    seqlens_qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=dev) * query_length

    # Cumulative-partition prefix sum (in partition_size-token units).  The decode
    # kernel needs partition_base[batch] = partition_indptr[batch] to convert a
    # global cumulative partition index (work_info.kv_start/kv_end) into a local
    # within-sequence partition index.
    _parts_per_batch = (context_lengths.to(torch.int32) + (partition_size - 1)) // partition_size
    partition_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=dev)
    partition_indptr[1:] = torch.cumsum(_parts_per_batch, dim=0).to(torch.int32)

    (
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = get_pa_metadata_info_v1(batch_size, num_kv_heads, num_cu=num_sm)

    work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type, device=dev)
    work_info = torch.empty(work_info_set_size, dtype=work_info_set_type, device=dev)
    reduce_indptr = torch.empty(reduce_indptr_size, dtype=reduce_indptr_type, device=dev)
    reduce_final_map = torch.empty(reduce_final_map_size, dtype=reduce_final_map_type, device=dev)
    reduce_partial_map = torch.empty(reduce_partial_map_size, dtype=reduce_partial_map_type, device=dev)

    get_pa_metadata_v1(
        seqlens_qo_indptr,
        context_lengths,
        work_indptr,
        work_info,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        query_group_size=num_query_heads // num_kv_heads,
        num_kv_heads=num_kv_heads,
        kv_granularity=partition_size,
        query_length=query_length,
        num_cu=num_sm,
        stream=torch.cuda.current_stream(dev),
    )

    # The FlyDSL get_pa_metadata_v1 produces the reduce_* maps natively
    # (faithful to the C++ kernel), so work_info / reduce_* are consumed directly
    # (no post-hoc expansion). work_info.kv_start/kv_end are partition indices and
    # work_info[:,1] (partial_qo_loc) is -1 for direct works or a partition-row
    # offset for split works.
    work_info_flat = work_info.reshape(-1).contiguous()

    # Number of partial slots = reduce_indptr[-1] (= last_reduce_indptr). Each
    # split partial occupies query_length rows in the partial buffer.
    num_partials = int(reduce_indptr[-1].item())
    max_qlen = query_length
    partial_output = torch.empty(
        ((num_partials + 1) * max_qlen, 1, num_query_heads, head_size), dtype=torch.float32, device=dev
    )
    partial_lse = torch.empty(((num_partials + 1) * max_qlen, 1, num_query_heads, 1), dtype=torch.float32, device=dev)

    stride_po_partial = query_length * num_query_heads * head_size
    stride_pl_partial = query_length * num_query_heads
    stride_po_ql = num_query_heads * head_size
    stride_pl_ql = num_query_heads

    return {
        "work_indptr": work_indptr,
        "work_info_flat": work_info_flat,
        "partition_indptr": partition_indptr,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
        "num_sm": num_sm,
        "partial_output": partial_output,
        "partial_lse": partial_lse,
        "stride_po_partial": stride_po_partial,
        "stride_pl_partial": stride_pl_partial,
        "stride_po_ql": stride_po_ql,
        "stride_pl_ql": stride_pl_ql,
        "query_length": query_length,
    }


def _is_current_stream_capturing() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_current_stream_capturing()
    except RuntimeError:
        return False


def _prepare_scale_tensor(
    name: str,
    scale,
    *,
    device: torch.device,
    is_graph_capturing: bool,
) -> torch.Tensor:
    if isinstance(scale, torch.Tensor):
        if is_graph_capturing:
            if scale.device != device:
                raise ValueError(
                    f"CUDA graph capture requires `{name}` to already be on {device}, " f"got {scale.device}."
                )
            if scale.dtype != torch.float32:
                raise ValueError(f"CUDA graph capture requires `{name}` to already be float32, " f"got {scale.dtype}.")
            return scale
        return scale.to(device=device, dtype=torch.float32)

    if is_graph_capturing:
        raise ValueError(
            f"CUDA graph capture requires `{name}` to be passed as a pre-created "
            "float32 tensor on the target device."
        )

    return torch.tensor([float(scale or 1.0)], device=device, dtype=torch.float32)


def _get_query_input_dtype(query: torch.Tensor) -> str:
    if query.dtype == torch.bfloat16:
        return "bf16"
    if query.dtype == torch.float16:
        return "f16"
    raise ValueError(f"Unsupported query dtype for pa_decode_ps_launch: {query.dtype}. Expected bf16 or f16.")


def _get_output_dtype_str(output: torch.Tensor) -> str:
    if output.dtype == torch.bfloat16:
        return "bf16"
    if output.dtype == torch.float16:
        return "f16"
    if output.dtype == torch.float32:
        return "f32"
    raise ValueError(
        f"Unsupported output dtype for pa_decode_ps_launch reduce: {output.dtype}. " "Expected bf16, f16, or f32."
    )


def get_recommended_splits(
    num_sequences: int,
    num_kv_heads: int,
    split_kv_blocks: int = 1,
    *,
    sliding_window: int = 0,
    context_partition_size: int = KV_COMPUTE_BLOCK,
    query_length: int = 1,
) -> int:
    """Recommend ``max_context_partition_num`` for PS partitioned paths.

    For sliding-window PS, this includes the old
    ``get_sw_ps_max_context_partition_num`` token-window calculation. For
    non-sliding PS, this mirrors ``get_recommended_splits`` in
    ``aiter/ops/triton/gluon/pa_decode_gluon.py`` so FlyDSL callers do not need
    to depend on aiter for the host-side split count.
    """
    if sliding_window > 0:
        window_token_count = sliding_window + query_length
        return cdiv(window_token_count - 1, context_partition_size) + 1

    props = torch.cuda.get_device_properties(torch.device("cuda"))
    # Reference uses occupancy = 2 (see `get_occupancy()` in the Gluon module).
    occupancy = 2
    num_sm = props.multi_processor_count * occupancy
    denom = max(1, num_sequences * num_kv_heads * split_kv_blocks)
    n = cdiv(num_sm, denom) * split_kv_blocks
    return max(4, min(n, 8))


# Small block sizes use the standalone tile kernel; the metadata decode path
# below is reserved for 1024-token physical pages.
_PA_DECODE_PS_SMALL_BLOCK_SIZES = (16, 64)


def pa_decode_ps_launch(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    kv_page_indices: torch.Tensor,  # [total_pages] int32
    kv_indptr: torch.Tensor,  # [num_seqs + 1] int32
    softmax_scale: float,
    key_scale: torch.Tensor = None,
    value_scale: torch.Tensor = None,
    *,
    sliding_window: int = 0,
    metadata: dict = None,
    block_tables: torch.Tensor = None,  # [num_seqs, max_blocks_per_seq] i32
    max_context_partition_num: int = 0,
    exp_sums: torch.Tensor = None,
    max_logits: torch.Tensor = None,
    temporary_output: torch.Tensor = None,
    stream=None,
) -> str:
    """Launch PA decode with persistent scheduling.

    Args:
        metadata: Pre-computed metadata dict from get_pa_metadata().
                  If None, calls get_pa_metadata() internally.
    """
    num_query_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]
    trans_v = len(value_cache.shape) == 5
    query_input_dtype = _get_query_input_dtype(query)

    dev = query.device
    is_graph_capturing = _is_current_stream_capturing()

    key_scale = _prepare_scale_tensor(
        "key_scale",
        key_scale,
        device=dev,
        is_graph_capturing=is_graph_capturing,
    )
    value_scale = _prepare_scale_tensor(
        "value_scale",
        value_scale,
        device=dev,
        is_graph_capturing=is_graph_capturing,
    )
    # Detect per-token vs per-tensor quantization from scale tensor
    # dimensionality: a >1-D scale tensor carries one scale per (block, head,
    # token), which enables the per-token K/V path in the metadata kernel.
    per_token_kv = key_scale.ndim > 1

    query_length = query.shape[0] // context_lengths.shape[0]
    query_group_size = num_query_heads // num_kv_heads

    # Strides for key_scale/value_scale
    if per_token_kv:
        stride_ks_block = key_scale.stride(0)
        stride_ks_head = key_scale.stride(1)
    else:
        stride_ks_block = 0
        stride_ks_head = 0

    s = stream or torch.cuda.current_stream()

    if sliding_window > 0:
        # Launch one CTA per 256-token context partition in the sliding window:
        # grid = (batch, kv_heads, max_context_partition_num).
        batch_size = context_lengths.shape[0]
        head_size = query.shape[-1]
        eqgs = query_length * query_group_size
        context_partition_size = KV_COMPUTE_BLOCK
        if max_context_partition_num == 0:
            max_context_partition_num = get_recommended_splits(
                batch_size,
                num_kv_heads,
                sliding_window=sliding_window,
                context_partition_size=context_partition_size,
                query_length=query_length,
            )
        if is_graph_capturing and (exp_sums is None or max_logits is None or temporary_output is None):
            raise ValueError(
                "CUDA graph capture requires preallocated `exp_sums`, `max_logits`, "
                "and `temporary_output` for the sliding-window path."
            )
        if exp_sums is None:
            exp_sums = torch.zeros(
                batch_size, num_kv_heads, max_context_partition_num, eqgs, device=dev, dtype=torch.float32
            )
        if max_logits is None:
            max_logits = torch.full(
                (batch_size, num_kv_heads, max_context_partition_num, eqgs),
                float("-inf"),
                device=dev,
                dtype=torch.float32,
            )
        if temporary_output is None:
            temporary_output = torch.zeros(
                batch_size, num_kv_heads, max_context_partition_num, eqgs, head_size, device=dev, dtype=torch.bfloat16
            )

        # The fused SW kernel is useful only when there is no real cross-partition
        # parallelism to exploit.  For the 1023-token window case, one CTA would
        # serialize six 256-token partitions and regress badly versus the
        # partitioned main kernel plus reduce.
        fuse_sw_partitions = max_context_partition_num <= 1
        sw_mtp_groups = (eqgs + MFMA_N - 1) // MFMA_N
        sw_grid_y = num_kv_heads * sw_mtp_groups
        output_5d = output.reshape(batch_size, query_length, num_kv_heads, query_group_size, head_size)

        compiled_sw = compile_pa_decode_sw(
            sliding_window=sliding_window,
            softmax_scale=softmax_scale,
            trans_v=trans_v,
            query_group_size=query_group_size,
            per_token_kv=per_token_kv,
            query_length=query_length,
            query_input_dtype=query_input_dtype,
            fuse_partitions=fuse_sw_partitions,
            head_dim=int(head_size),
        )

        _run_compiled(
            compiled_sw["launch"],
            exp_sums.data_ptr(),
            max_logits.data_ptr(),
            temporary_output.data_ptr(),
            output_5d.data_ptr(),
            query.data_ptr(),
            key_cache.data_ptr(),
            value_cache.data_ptr(),
            block_tables.data_ptr(),
            context_lengths.data_ptr(),
            key_scale.data_ptr(),
            value_scale.data_ptr(),
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            value_cache.stride(0),
            value_cache.stride(1),
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            temporary_output.stride(0),
            temporary_output.stride(1),
            temporary_output.stride(2),
            temporary_output.stride(3),
            output_5d.stride(0),
            output_5d.stride(1),
            output_5d.stride(2),
            output_5d.stride(3),
            block_tables.stride(0),
            stride_ks_block,
            stride_ks_head,
            batch_size,
            sw_grid_y,
            1 if fuse_sw_partitions else max_context_partition_num,
            s,
        )

        if fuse_sw_partitions:
            return "ps_sw_fused_partitioned"

        compiled_sw_reduce = compile_pa_decode_sw_reduce(
            max_context_partition_num=max_context_partition_num,
            query_seq_len=query_length,
            query_group_size=query_group_size,
            head_size=head_size,
            output_dtype_str=_get_output_dtype_str(output),
        )
        _run_compiled(
            compiled_sw_reduce["launch"],
            output_5d.data_ptr(),
            exp_sums.data_ptr(),
            max_logits.data_ptr(),
            temporary_output.data_ptr(),
            output_5d.stride(0),
            output_5d.stride(1),
            output_5d.stride(2),
            output_5d.stride(3),
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            temporary_output.stride(0),
            temporary_output.stride(1),
            temporary_output.stride(2),
            temporary_output.stride(3),
            batch_size,
            num_kv_heads,
            s,
        )
        return "ps_sw_partitioned"

    # ── small-block (block_size 16/64) → tile kernel ──
    # Key cache shape is [num_blocks, num_kv_heads, head_size // 16, block_size, 16].
    block_size = key_cache.shape[-2]
    if block_size in _PA_DECODE_PS_SMALL_BLOCK_SIZES:
        if block_tables is None:
            raise ValueError(
                f"pa_decode_ps_launch: block_size={block_size} requires `block_tables` "
                "(per-sequence physical block index table)."
            )
        batch_size = context_lengths.shape[0]
        if is_graph_capturing:
            # Buffer sizes must be fixed ahead of capture and stay identical
            # across every replay, so require the caller to have preallocated
            # exp_sums/max_logits/temporary_output, exactly as the other PS
            # paths already require.
            if exp_sums is None or max_logits is None or temporary_output is None:
                raise ValueError(
                    "CUDA graph capture requires preallocated `exp_sums`, `max_logits`, "
                    "and `temporary_output` for the tile-backed small-block PS path."
                )
        # pa_decode_tile requires an exact [num_blocks, num_kv_heads,
        # block_size] per-token scale shape; callers here may pass an extra
        # trailing singleton dim (e.g. from a pertoken-quant helper), which
        # reshape away without changing the strides.
        if per_token_kv:
            num_blocks = key_cache.shape[0]
            key_scale = key_scale.reshape(num_blocks, num_kv_heads, block_size)
            value_scale = value_scale.reshape(num_blocks, num_kv_heads, block_size)
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
            stream=s,
            num_partitions=max_context_partition_num,
            pmax=max_logits,
            psum=exp_sums,
            pout=temporary_output,
        )
        return "ps_small_block"

    if metadata is None:
        if is_graph_capturing:
            raise ValueError(
                "CUDA graph capture requires precomputed `metadata`; "
                "call `get_pa_metadata()` before capture and pass it via `metadata=`."
            )
        metadata = get_pa_metadata(query, key_cache, context_lengths, kv_indptr, num_query_heads, num_kv_heads)

    work_indptr = metadata["work_indptr"]
    work_info_flat = metadata["work_info_flat"]
    partition_indptr = metadata["partition_indptr"]
    partial_output = metadata["partial_output"]
    partial_lse = metadata["partial_lse"]
    stride_po_partial = metadata["stride_po_partial"]
    stride_pl_partial = metadata["stride_pl_partial"]
    num_sm = metadata["num_sm"]

    metadata_block_size = key_cache.shape[-2]
    compiled = compile_pa_decode_metadata(
        softmax_scale=softmax_scale,
        trans_v=trans_v,
        query_group_size=query_group_size,
        per_token_kv=per_token_kv,
        query_length=query_length,
        query_input_dtype=query_input_dtype,
        head_dim=int(query.shape[-1]),
        block_size=int(metadata_block_size),
        output_dtype_str=_get_output_dtype_str(output),
    )

    stride_po_ql = metadata.get("stride_po_ql", num_query_heads * query.shape[-1])
    stride_pl_ql = metadata.get("stride_pl_ql", num_query_heads)

    _run_compiled(
        compiled["launch"],
        output.data_ptr(),
        partial_output.data_ptr(),
        partial_lse.data_ptr(),
        query.data_ptr(),
        key_cache.data_ptr(),
        value_cache.data_ptr(),
        context_lengths.data_ptr(),
        key_scale.data_ptr(),
        value_scale.data_ptr(),
        work_indptr.data_ptr(),
        work_info_flat.data_ptr(),
        kv_page_indices.data_ptr(),
        kv_indptr.data_ptr(),
        partition_indptr.data_ptr(),
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        value_cache.stride(0),
        value_cache.stride(1),
        output.stride(0),
        output.stride(1),
        stride_po_partial,
        stride_pl_partial,
        stride_ks_block,
        stride_ks_head,
        stride_po_ql,
        stride_pl_ql,
        num_sm,
        s,
    )

    from kernels.attention.pa_metadata import pa_metadata_reduce

    # Deterministic FlyDSL reduce replaces the racy aiter pa_reduce_v1/mla_reduce_v1
    # (root cause of the flaky test_pa NaN). Same partial layout / reduce maps.
    pa_metadata_reduce(
        partial_output=partial_output[query_length:],
        partial_lse=partial_lse[query_length:],
        reduce_indptr=metadata["reduce_indptr"],
        reduce_final_map=metadata["reduce_final_map"],
        reduce_partial_map=metadata["reduce_partial_map"],
        max_seqlen_q=query_length,
        final_output=output,
        num_query_heads=num_query_heads,
        head_dim=int(query.shape[-1]),
        stream=s,
    )

    return "ps_split_reduce"
