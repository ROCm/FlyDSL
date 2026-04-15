# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""PS-only paged-attention regression harness."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytest
import torch
import triton

import aiter
from aiter import dtypes
from aiter import per_tensor_quant, pertoken_quant
from aiter.ops.triton.gluon.pa_decode_gluon import get_recommended_splits
from aiter.test_common import checkAllclose

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from triton.experimental import gluon  # noqa: F401
    from triton.experimental.gluon import language as gl  # noqa: F401
    HAS_GLUON = False
except ImportError:
    HAS_GLUON = False
    print("Warning: Triton Gluon is unavailable; Gluon reference checks will fail.")

try:
    from kernels.pa_decode_fp8 import (
        get_sw_ps_max_context_partition_num,
        get_pa_metadata as flydsl_get_pa_metadata,
        pa_decode_ps_launch as flydsl_ps_launch,
    )
    HAS_FLYDSL_PS = True
except ImportError as exc:
    HAS_FLYDSL_PS = False
    print(f"Warning: FlyDSL PA decode PS not available: {exc}")

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)

TRITON_VERSION = triton.__version__
TEST_NAME = "ps_accuracy"
UNIFORM_RANGE = (-1, 1)
USE_CUDA_GRAPH_TEST = False

STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
}

CASE_SET_NAME_OPTIONS = [
    "normal_accuracy",
    "sliding_window_accuracy",
    "global_window_accuracy",
]

COMPUTE_TYPE_OPTIONS = ["fp8"]
KV_VARLEN_OPTIONS = [False, True]
TRANS_V_OPTIONS = [True]
CONTEXT_PARTITION_SIZE_OPTIONS = [256]
QUANT_MODE_OPTIONS = ["per_token", "per_tensor"]
HEAD_DIMENSION_OPTIONS = [128]
BLOCK_SIZE_OPTIONS = [1024]
HEAD_CONFIGURATIONS = [(8, 1), (16, 1)]
QUERY_LENGTH_OPTIONS = [1, 2, 3, 4]
CONTEXT_LENGTH_OPTIONS = [1027]
BATCH_SIZE_OPTIONS = [3, 81]
SLIDING_WINDOW_OPTIONS = [0]
GLOBAL_WINDOW_OPTIONS = [0]


def setup_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def compare_arrays(
    arr1: np.ndarray,
    arr2: np.ndarray,
    k: int = 5,
    thresholds: List[float] = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1],
) -> Dict[str, object]:
    if arr1.shape != arr2.shape:
        raise ValueError("Input arrays must have the same shape")
    arr1 = arr1.astype(np.float32)
    arr2 = arr2.astype(np.float32)
    diff = np.abs(arr1 - arr2)
    total_elements = arr1.size
    result: Dict[str, object] = {
        "top_k_diff": [],
        "threshold_stats": [],
        "max_diff": float(diff.max()),
        "max_diff_thr": float((diff / (1.0 + np.abs(arr2))).max()),
    }
    flat_diff = diff.flatten()
    top_k_indices = np.argpartition(flat_diff, -k)[-k:]
    top_k_indices = top_k_indices[np.argsort(-flat_diff[top_k_indices])]
    orig_indices = np.unravel_index(top_k_indices, diff.shape)
    for i in range(k):
        idx = tuple(dim[i] for dim in orig_indices)
        result["top_k_diff"].append(
            {
                "value": float(diff[idx]),
                "position": idx,
                "arr1_value": float(arr1[idx]),
                "arr2_value": float(arr2[idx]),
            }
        )
    for i in range(len(thresholds) - 1):
        lower = thresholds[i]
        upper = thresholds[i + 1]
        mask = (diff >= lower) & (diff < upper)
        count = int(np.sum(mask))
        result["threshold_stats"].append(
            {
                "range": f"[{lower:.1e}, {upper:.1e})",
                "count": count,
                "percentage": 100.0 * count / total_elements,
            }
        )
    mask = diff >= thresholds[-1]
    count = int(np.sum(mask))
    result["threshold_stats"].append(
        {
            "range": f">={thresholds[-1]:.1e}",
            "count": count,
            "percentage": 100.0 * count / total_elements,
        }
    )
    print(f"diff.abs.max={result['max_diff']}")
    print(f"max_diff_thr={result['max_diff_thr']}")
    return result


def get_kv_cache_torch_dtype(
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
) -> torch.dtype:
    if isinstance(cache_dtype, str):
        if cache_dtype == "auto":
            if isinstance(model_dtype, str):
                return STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            if isinstance(model_dtype, torch.dtype):
                return model_dtype
            raise ValueError(f"Invalid model dtype: {model_dtype}")
        if cache_dtype in ["half", "bfloat16", "float"]:
            return STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        if cache_dtype == "fp8":
            return torch.uint8
        raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    if isinstance(cache_dtype, torch.dtype):
        return cache_dtype
    raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")


def create_kv_cache(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: int = 0,
    device: Optional[str] = "cuda",
    itemsize: int = 1,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(f"Does not support fp8 key cache with head_size={head_size}")
    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)
    elements_per_vector = 16 // itemsize
    key_cache_shape = (
        num_blocks,
        num_heads,
        head_size // elements_per_vector,
        block_size,
        elements_per_vector,
    )
    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    key_caches: List[torch.Tensor] = []
    value_caches: List[torch.Tensor] = []
    setup_seed(seed)
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape, dtype=torch_dtype, device=device)
        value_cache = torch.empty(size=value_cache_shape, dtype=torch_dtype, device=device)
        key_cache.uniform_(*UNIFORM_RANGE)
        value_cache.uniform_(*UNIFORM_RANGE)
        key_caches.append(key_cache)
        value_caches.append(value_cache)
    return key_caches, value_caches



def reference_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float,
    output_dtype: torch.dtype,
    is_causal: bool = True,
    sliding_window=0,
    global_window=0,
) -> torch.Tensor:
    """Reference implementation of masked attention."""
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)
    num_query_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    s_q = query.shape[0]
    s_k = key.shape[0]
    key = key.repeat_interleave(num_query_heads // num_kv_heads, dim=1)
    value = value.repeat_interleave(num_query_heads // num_kv_heads, dim=1)

    attention_weights = torch.einsum("qhd,khd->hqk", query, key) * softmax_scale

    if is_causal:
        query_len = query.shape[0]
        key_len = key.shape[0]
        attention_bias = torch.zeros(
            query_len, key_len, dtype=torch.float32, device=query.device
        )
        causal_mask = torch.ones(
            query_len, key_len, dtype=torch.bool, device=query.device
        ).tril(diagonal=key_len - query_len)
        # attention_bias.masked_fill_(causal_mask.logical_not(), float(-3.4e38))
        attention_bias.masked_fill_(causal_mask.logical_not(), float(-3.4e38))
        attention_weights += attention_bias

    # Handle position calculation for both context and generation phases
    if s_q == s_k:
        # Context phase: standard position calculation
        query_positions = torch.arange(s_q, device=query.device)
        key_positions = torch.arange(s_k, device=query.device)
    else:
        # Generation phase: query is at position s_k (after the cache)
        query_positions = torch.arange(
            s_k - s_q, s_k, device=query.device
        )  # [s_k] for s_q=1
        key_positions = torch.arange(s_k, device=query.device)  # [0,1,2,...,s_k-1]

    # Create position difference matrix: query_pos - key_pos
    pos_diff = query_positions.unsqueeze(1) - key_positions.unsqueeze(0)  # [s_q, s_k]

    # Fallback: initialize the mask to all True, then progressively tighten with AND
    window_mask = torch.ones_like(attention_weights, dtype=torch.bool)
    if sliding_window > 0:
        # Sliding window mask: allow attention only if 0 <= pos_diff < sliding_window_size
        # sliding window size does not cover the diagonals
        sliding_window_mask = pos_diff >= sliding_window + 1
        window_mask &= sliding_window_mask

    if global_window > 0:
        # Global window mask: allow attention only if key_positions <= global_window
        global_window_mask = key_positions.unsqueeze(0) >= global_window
        window_mask &= global_window_mask

    if sliding_window > 0 or global_window > 0:
        attention_weights.masked_fill_(window_mask, float("-inf"))
    # torch.save(attention_weights, "/data00/fengjunda.aml/debug/attention_weights.pt")

    attention_weights = torch.softmax(attention_weights, dim=-1)
    output = torch.einsum("hqk,khd->qhd", attention_weights, value)
    return output.to(output_dtype)


def torch_mha_extend(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    query_output_indptr: torch.Tensor,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    sliding_window=0,
    global_window=0,
) -> torch.Tensor:
    """PyTorch reference implementation of paged attention."""
    num_blocks, num_heads, head_size, block_size = value_cache.shape
    softmax_scale = 1.0 / (head_size**0.5)

    output_dtype = query.dtype
    kv_dtype = key_cache.dtype

    queries_split = torch.tensor_split(query, query_output_indptr.tolist()[1:])
    key_cache_flat = (
        key_cache.permute(0, 3, 1, 2, 4).contiguous().view(-1, num_heads, head_size)
    )
    value_cache_flat = (
        value_cache.permute(0, 3, 1, 2).contiguous().view(-1, num_heads, head_size)
    )

    batch_size = query_output_indptr.shape[0] - 1
    outputs = []

    for batch_idx in range(batch_size):
        current_query = queries_split[batch_idx]
        current_block_table = block_tables[batch_idx]
        current_context_length = context_lengths[batch_idx].item()

        token_indices = (
            current_block_table.repeat_interleave(block_size)[:current_context_length]
            * block_size
            + torch.arange(current_context_length, device=current_block_table.device)
            % block_size
        )

        gathered_keys = (
            key_cache_flat.view(torch.int8)[token_indices]
            .view(kv_dtype)
            .to(torch.float)
        )
        if key_scale is not None:
            gathered_keys *= key_scale[:, token_indices].t().unsqueeze(-1)

        gathered_values = (
            value_cache_flat.view(torch.int8)[token_indices]
            .view(kv_dtype)
            .to(torch.float)
        )
        if value_scale is not None:
            gathered_values *= value_scale[:, token_indices].t().unsqueeze(-1)

        attention_output = reference_masked_attention(
            current_query,
            gathered_keys,
            gathered_values,
            softmax_scale,
            output_dtype,
            is_causal=True,
            sliding_window=sliding_window,
            global_window=global_window,
        )
        outputs.append(attention_output)

    return torch.cat(outputs)


def quantize_kv_cache_symmetric(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    quant_dtype: torch.dtype,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    num_blocks, num_heads, head_dim, block_size = value_cache.shape
    total_tokens = num_blocks * block_size
    key_cache_reshaped = (
        key_cache.permute(0, 1, 3, 2, 4)
        .reshape(num_blocks, num_heads, block_size, -1)
        .contiguous()
    )
    value_cache_reshaped = (
        value_cache.permute(0, 1, 3, 2)
        .reshape(num_blocks, num_heads, block_size, -1)
        .contiguous()
    )
    quantized_keys, key_scales_original = pertoken_quant(
        key_cache_reshaped, quant_dtype=quant_dtype
    )
    quantized_values, value_scales_original = pertoken_quant(
        value_cache_reshaped, quant_dtype=quant_dtype
    )
    elements_per_vector = 16 // quant_dtype.itemsize
    quantized_keys = (
        quantized_keys.view(
            num_blocks,
            num_heads,
            block_size,
            head_dim // elements_per_vector,
            elements_per_vector,
        )
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    quantized_values = (
        quantized_values.view(num_blocks, num_heads, block_size, head_dim)
        .permute(0, 1, 3, 2)
        .contiguous()
    )
    key_scales_flat = (
        key_scales_original.permute(1, 0, 2, 3).contiguous().view(num_heads, total_tokens)
    )
    value_scales_flat = (
        value_scales_original.permute(1, 0, 2, 3).contiguous().view(num_heads, total_tokens)
    )
    return (
        quantized_keys,
        key_scales_flat,
        quantized_values,
        value_scales_flat,
        key_scales_original,
        value_scales_original,
    )


def quantize_kv_cache_per_tensor(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    quant_dtype: torch.dtype,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    num_blocks, num_heads, head_dim, block_size = value_cache.shape
    elements_per_vector = 16 // quant_dtype.itemsize
    key_cache_reshaped = (
        key_cache.permute(0, 1, 3, 2, 4)
        .reshape(num_blocks, num_heads, block_size, -1)
        .contiguous()
    )
    key_cache_reshaped = (
        key_cache_reshaped.view(
            num_blocks,
            num_heads,
            block_size,
            head_dim // elements_per_vector,
            elements_per_vector,
        )
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    quantized_keys, key_scales_original = per_tensor_quant(
        key_cache_reshaped, quant_dtype=quant_dtype
    )
    quantized_values, value_scales_original = per_tensor_quant(
        value_cache, quant_dtype=quant_dtype
    )
    key_scales_flat = key_scales_original.expand(num_heads, num_blocks * block_size)
    value_scales_flat = value_scales_original.expand(num_heads, num_blocks * block_size)
    return (
        quantized_keys,
        key_scales_flat,
        quantized_values,
        value_scales_flat,
        key_scales_original,
        value_scales_original,
    )


def shuffle_value_cache_layout(value_cache: torch.Tensor) -> torch.Tensor:
    elements_per_vector = 16 // value_cache.element_size()
    num_blocks, num_kv_heads, head_size, block_size = value_cache.shape
    value_cache_reshaped = value_cache.view(
        num_blocks,
        num_kv_heads,
        head_size,
        block_size // elements_per_vector,
        elements_per_vector,
    )
    return value_cache_reshaped.permute(0, 1, 3, 2, 4).contiguous()


def measure_us(
    fn,
    *,
    warmup: int = 1,
    iters: int = 3,
    use_cuda_graph: Optional[bool] = None,
) -> float:
    if use_cuda_graph is None:
        use_cuda_graph = USE_CUDA_GRAPH_TEST
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = None
    if use_cuda_graph:
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        try:
            with torch.cuda.stream(capture_stream):
                fn()
            torch.cuda.current_stream().wait_stream(capture_stream)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(capture_stream):
                with torch.cuda.graph(graph, stream=capture_stream):
                    fn()
            torch.cuda.current_stream().wait_stream(capture_stream)
            torch.cuda.synchronize()
        except RuntimeError as exc:
            graph = None
            print(
                f"Warning: measure_us cuda graph capture failed, falling back to eager execution: {exc}"
            )
            torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        if graph is not None:
            graph.replay()
        else:
            fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def get_gluon_partition_count(
    num_seqs: int,
    num_kv_heads: int,
    block_size: int,
    context_partition_size: int,
    sliding_window: int,
    global_window: int,
) -> int:
    if sliding_window > 0:
        return get_sw_ps_max_context_partition_num(
            sliding_window,
            global_window,
            context_partition_size,
        )
    split_kv_blocks = triton.cdiv(block_size, context_partition_size)
    return get_recommended_splits(num_seqs, num_kv_heads, split_kv_blocks)


def run_gluon_ps(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    block_tables: torch.Tensor,
    softmax_scale: float,
    query_length: int,
    max_context_partition_num: int,
    context_partition_size: int,
    compute_type: torch.dtype,
    query_scale: Optional[torch.Tensor],
    key_scale: Optional[torch.Tensor],
    value_scale: Optional[torch.Tensor],
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    temporary_output: torch.Tensor,
    *,
    sliding_window: int,
    global_window: int,
) -> None:
    torch.ops.aiter.pa_decode_gluon(
        output,
        query,
        key_cache,
        value_cache,
        context_lengths,
        block_tables,
        softmax_scale,
        query_length,
        max_context_partition_num,
        context_partition_size,
        compute_type,
        query_scale,
        key_scale,
        value_scale,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=temporary_output,
        alibi_slopes=None,
        sinks=None,
        sliding_window=sliding_window,
        global_window=global_window,
        ps=True,
    )


def build_ps_page_data(
    block_tables_list: List[List[int]],
    context_lengths: torch.Tensor,
    block_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = context_lengths.shape[0]
    actual_blocks = (context_lengths + block_size - 1) // block_size
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(actual_blocks, dim=0)
    kv_page_indices_list: List[int] = []
    for batch_idx, num_blocks in enumerate(actual_blocks.tolist()):
        kv_page_indices_list.extend(block_tables_list[batch_idx][:num_blocks])
    kv_page_indices = torch.tensor(kv_page_indices_list, dtype=torch.int32, device=device)
    return kv_page_indices, kv_indptr


def run_flydsl_ps(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    softmax_scale: float,
    key_scale: Union[float, torch.Tensor],
    value_scale: Union[float, torch.Tensor],
    metadata: Dict[str, torch.Tensor],
    *,
    sliding_window: int,
    global_window: int,
    block_tables: torch.Tensor,
    max_context_partition_num: int,
    exp_sums: Optional[torch.Tensor] = None,
    max_logits: Optional[torch.Tensor] = None,
    temporary_output: Optional[torch.Tensor] = None,
) -> None:
    flydsl_ps_launch(
        output,
        query,
        key_cache,
        value_cache,
        context_lengths,
        kv_page_indices,
        kv_indptr,
        softmax_scale,
        key_scale=key_scale,
        value_scale=value_scale,
        sliding_window=sliding_window,
        global_window=global_window,
        metadata=metadata,
        block_tables=block_tables,
        max_context_partition_num=max_context_partition_num,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=temporary_output,
    )


def get_tolerance(*, kv_varlen: bool, sliding_window: int) -> float:
    diff_tolerance = 5e-3
    if kv_varlen:
        diff_tolerance = 5e-2
    if sliding_window > 0:
        diff_tolerance = max(diff_tolerance, 5e-2)
        if kv_varlen:
            diff_tolerance = 6e-2
    return diff_tolerance



def get_ps_vs_gluon_tolerance(ps_tolerance: float, gluon_tolerance: float) -> float:
    """Cross-check tolerance should not be stricter than the Gluon reference itself."""
    return max(ps_tolerance, gluon_tolerance)


def dtype_to_name(dtype: torch.dtype) -> str:
    for name, candidate in dtypes.d_dtypes.items():
        if candidate == dtype:
            return name
    return str(dtype)


def summarize_comparison(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> Tuple[int, Dict[str, object]]:
    err = checkAllclose(expected, actual, atol=atol, rtol=rtol, msg=f"[{name}]")
    err = 1 if err > 0 else 0
    diff_result = compare_arrays(
        actual.to(torch.float32).detach().cpu().numpy(),
        expected.to(torch.float32).detach().cpu().numpy(),
    )
    print(f"{name} {'PASSED' if err == 0 else 'FAILED'}")
    return err, diff_result


def run_pa_decode_ps_test(
    context_length: int,
    batch_size: int,
    num_heads: Tuple[int, int],
    head_size: int,
    block_size: int,
    compute_type: torch.dtype,
    query_length: int,
    quant_mode: str,
    context_partition_size: int,
    trans_v: bool,
    kv_varlen: bool,
    sliding_window: int,
    global_window: int,
) -> Dict[str, Union[float, int, str, bool, Tuple[int, int]]]:
    if not HAS_FLYDSL_PS:
        raise RuntimeError("FlyDSL `pa_decode_ps_launch` is not available.")
    if compute_type != aiter.dtypes.fp8:
        raise ValueError("This PS-only harness only keeps fp8 cases.")
    if block_size != 1024:
        raise ValueError("This PS-only harness only keeps block_size=1024 cases.")
    if head_size != 128:
        raise ValueError("This PS-only harness only keeps head_size=128 cases.")
    results: Dict[str, Union[float, int, str, bool, Tuple[int, int]]] = {
        "compute_type": dtype_to_name(compute_type),
        "quant_mode": quant_mode,
        "trans_v": trans_v,
        "kv_varlen": kv_varlen,
        "context_partition_size": context_partition_size,
        "block_size": block_size,
        "num_heads": num_heads,
        "context_length": context_length,
        "batch_size": batch_size,
        "query_length": query_length,
        "head_size": head_size,
        "sliding_window": sliding_window,
        "global_window": global_window,
        "quant_q": False,
        "quant_kv": True,
    }
    seed = 123
    setup_seed(seed)
    device = torch.device("cuda:0")
    torch.set_default_device(device)
    num_query_heads, num_kv_heads = num_heads
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("Query heads must be divisible by KV heads")
    data_type = torch.bfloat16 if compute_type == aiter.dtypes.fp8 else compute_type
    softmax_scale = 1.0 / (head_size**0.5)
    total_queries = batch_size * query_length
    query_output_indptr = torch.arange(
        0,
        (batch_size + 1) * query_length,
        query_length,
        dtype=torch.int32,
        device=device,
    )
    qkv_tensor = torch.randn(
        total_queries,
        num_query_heads + 2 * num_kv_heads,
        head_size,
        dtype=data_type,
        device=device,
    )
    query, key, value = torch.split(
        qkv_tensor, [num_query_heads, num_kv_heads, num_kv_heads], dim=1
    )
    query.uniform_(*UNIFORM_RANGE)
    if kv_varlen:
        kv_len_list = [random.randint(query_length, context_length) for _ in range(batch_size)]
    else:
        kv_len_list = [context_length] * batch_size
    context_lengths = torch.tensor(kv_len_list, dtype=torch.int32, device=device)
    max_context_length = max(16384, context_length)
    max_blocks_per_sequence = triton.cdiv(max_context_length, block_size)
    total_blocks = max_blocks_per_sequence * batch_size
    blocks_per_sequence = triton.cdiv(context_length, block_size)
    block_tables_list: List[List[int]] = []
    for _ in range(batch_size):
        block_tables_list.append(
            [random.randint(0, total_blocks - 1) for _ in range(blocks_per_sequence)]
        )
    block_tables = torch.tensor(block_tables_list, dtype=torch.int32, device=device)
    key_caches, value_caches = create_kv_cache(
        total_blocks,
        block_size,
        1,
        num_kv_heads,
        head_size,
        "auto",
        data_type,
        seed,
        str(device),
        1,
    )
    key_cache = key_caches[0]
    value_cache = value_caches[0]

    query_scale_factors = None
    quantized_query = query
    if quant_mode == "per_token":
        (
            quantized_keys,
            key_scale_factors_flat,
            quantized_values,
            value_scale_factors_flat,
            key_scale_original,
            value_scale_original,
        ) = quantize_kv_cache_symmetric(
            key_cache,
            value_cache,
            quant_dtype=aiter.dtypes.fp8,
        )
    else:
        (
            quantized_keys,
            key_scale_factors_flat,
            quantized_values,
            value_scale_factors_flat,
            key_scale_original,
            value_scale_original,
        ) = quantize_kv_cache_per_tensor(
            key_cache,
            value_cache,
            quant_dtype=aiter.dtypes.fp8,
        )
    reference_output = torch_mha_extend(
        query,
        quantized_keys,
        quantized_values,
        block_tables,
        context_lengths,
        query_output_indptr,
        key_scale_factors_flat,
        value_scale_factors_flat,
        sliding_window=sliding_window,
        global_window=global_window,
    ).to(data_type)
    quantized_values = shuffle_value_cache_layout(quantized_values) if trans_v else quantized_values
    if HAS_GLUON:
        max_context_partition_num = get_gluon_partition_count(
            batch_size,
            num_kv_heads,
            block_size,
            context_partition_size,
            sliding_window,
            global_window,
        )
        equivalent_query_group_size = query_length * (num_query_heads // num_kv_heads)
        intermediate_shape = (
            batch_size,
            num_kv_heads,
            max_context_partition_num,
            equivalent_query_group_size,
        )
        exp_sums = torch.empty(intermediate_shape, dtype=torch.float32, device=device)
        max_logits = torch.empty(intermediate_shape, dtype=torch.float32, device=device)
        temporary_output = torch.empty(
            *intermediate_shape,
            head_size,
            dtype=reference_output.dtype,
            device=device,
        )
        gluon_output = torch.empty_like(reference_output)

        def gluon_call() -> None:
            run_gluon_ps(
                gluon_output,
                quantized_query,
                quantized_keys,
                quantized_values,
                context_lengths,
                block_tables,
                softmax_scale,
                query_length,
                max_context_partition_num,
                context_partition_size,
                compute_type,
                query_scale_factors,
                key_scale_original,
                value_scale_original,
                exp_sums,
                max_logits,
                temporary_output,
                sliding_window=sliding_window,
                global_window=global_window,
            )

        gluon_time = measure_us(gluon_call)
        gluon_tol = get_tolerance(kv_varlen=kv_varlen, sliding_window=sliding_window)
        print("\nGluon vs Torch:")
        err_gluon, gluon_diff = summarize_comparison(
            "Gluon vs Torch",
            gluon_output,
            reference_output,
            atol=gluon_tol,
            rtol=gluon_tol,
        )

    kv_page_indices, kv_indptr = build_ps_page_data(
        block_tables_list,
        context_lengths,
        block_size,
        device,
    )
    # Match Gluon's query path: launch with bf16 query and let the PS launcher
    # cast to fp8 internally with a unit query scale.
    flydsl_ps_query = query
    ps_metadata = flydsl_get_pa_metadata(
        flydsl_ps_query,
        quantized_keys,
        context_lengths,
        kv_indptr,
        num_query_heads,
        num_kv_heads,
    )
    ps_key_scale: torch.Tensor = key_scale_original
    ps_value_scale: torch.Tensor = value_scale_original
    flydsl_ps_output = torch.empty_like(reference_output)
    
    max_context_partition_num = get_sw_ps_max_context_partition_num(
        sliding_window,
        global_window,
        context_partition_size,
    )
    flydsl_exp_sums = None
    flydsl_max_logits = None
    flydsl_temporary_output = None
    if sliding_window > 0:
        intermediate_shape = (
            batch_size,
            num_kv_heads,
            max_context_partition_num,
            query_length * (num_query_heads // num_kv_heads),
        )
        flydsl_exp_sums = torch.empty(
            intermediate_shape, dtype=torch.float32, device=device
        )
        flydsl_max_logits = torch.empty(
            intermediate_shape, dtype=torch.float32, device=device
        )
        flydsl_temporary_output = torch.empty(
            *intermediate_shape,
            head_size,
            dtype=reference_output.dtype,
            device=device,
        )

    def flydsl_ps_call() -> None:
        run_flydsl_ps(
            flydsl_ps_output,
            flydsl_ps_query,
            quantized_keys,
            quantized_values,
            context_lengths,
            kv_page_indices,
            kv_indptr,
            softmax_scale,
            ps_key_scale,
            ps_value_scale,
            ps_metadata,
            sliding_window=sliding_window,
            global_window=global_window,
            block_tables=block_tables,
            max_context_partition_num=max_context_partition_num,
            exp_sums=flydsl_exp_sums,
            max_logits=flydsl_max_logits,
            temporary_output=flydsl_temporary_output,
        )

    flydsl_ps_time = measure_us(flydsl_ps_call)
    ps_tol = get_tolerance(kv_varlen=kv_varlen, sliding_window=sliding_window)
    print("\nFlyDSL PS vs Torch:")
    err_flydsl_ps, flydsl_ps_diff = summarize_comparison(
        "FlyDSL PS vs Torch",
        flydsl_ps_output,
        reference_output,
        atol=ps_tol,
        rtol=ps_tol,
    )

    if HAS_GLUON:
        results["us_gluon"] = gluon_time
        results["err_gluon"] = err_gluon
        results["gluon_max_diff"] = float(gluon_diff["max_diff"])
        results["gluon_max_diff_thr"] = float(gluon_diff["max_diff_thr"])

    results["us_flydsl_ps"] = flydsl_ps_time
    results["err_flydsl_ps"] = err_flydsl_ps
    results["flydsl_ps_max_diff"] = float(flydsl_ps_diff["max_diff"])
    results["flydsl_ps_max_diff_thr"] = float(flydsl_ps_diff["max_diff_thr"])

    return results


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="PS-only paged attention decode regression test",
    )
    parser.add_argument("--compute_type", type=str, default=None, help="Compute type")
    parser.add_argument(
        "-n",
        "--num_heads",
        type=dtypes.str2tuple,
        default=None,
        help="Number of heads as q_heads,kv_heads",
    )
    parser.add_argument(
        "-q",
        "--query_length",
        type=int,
        choices=QUERY_LENGTH_OPTIONS,
        default=None,
        help="Query length",
    )
    parser.add_argument("-c", "--context_length", type=int, default=None, help="Context length")
    parser.add_argument("-b", "--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("-d", "--head_dim", type=int, default=None, help="Head dimension")
    parser.add_argument("--block_size", type=int, default=None, help="Block size")
    parser.add_argument(
        "--quant_mode",
        type=str,
        choices=["per_token", "per_tensor", "both"],
        default=None,
        help="KV quantization mode",
    )
    parser.add_argument(
        "--trans_v",
        type=lambda x: str(x).lower() == "true",
        default=None,
        help="Use transposed V layout for Gluon",
    )
    parser.add_argument(
        "--kv_varlen",
        type=lambda x: str(x).lower() == "true",
        default=None,
        help="Use variable KV lengths",
    )
    parser.add_argument(
        "--context_partition_size",
        type=int,
        default=None,
        help="Context partition size for Gluon reduce",
    )
    parser.add_argument(
        "--sliding_window",
        type=int,
        default=None,
        help="Sliding window size; 0 disables sliding window",
    )
    parser.add_argument(
        "--global_window",
        type=int,
        default=None,
        help="Global prefix size; tokens with index < global_window stay visible",
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=1.0,
        help="Randomly sample test cases from the selected case set",
    )
    parser.add_argument(
        "--use_cuda_graph",
        action="store_true",
        help="Enable CUDA graph timing mode for the selected test run",
    )
    return parser


def process_arguments(args: argparse.Namespace) -> tuple:
    compute_types = [dtypes.d_dtypes[key] for key in COMPUTE_TYPE_OPTIONS]
    block_sizes = BLOCK_SIZE_OPTIONS
    head_configs = HEAD_CONFIGURATIONS
    context_lengths = CONTEXT_LENGTH_OPTIONS
    batch_sizes = BATCH_SIZE_OPTIONS
    head_sizes = HEAD_DIMENSION_OPTIONS
    query_lengths = QUERY_LENGTH_OPTIONS
    quant_modes = QUANT_MODE_OPTIONS
    trans_v = TRANS_V_OPTIONS
    kv_varlen = KV_VARLEN_OPTIONS
    context_partition_sizes = CONTEXT_PARTITION_SIZE_OPTIONS
    sliding_window_options = SLIDING_WINDOW_OPTIONS
    global_window_options = GLOBAL_WINDOW_OPTIONS
    if args.compute_type is not None:
        compute_types = [dtypes.d_dtypes[args.compute_type]]
    if args.num_heads is not None:
        head_configs = [args.num_heads]
    if args.query_length is not None:
        query_lengths = [args.query_length]
    if args.context_length is not None:
        context_lengths = [args.context_length]
    if args.batch_size is not None:
        batch_sizes = [args.batch_size]
    if args.head_dim is not None:
        head_sizes = [args.head_dim]
    if args.block_size is not None:
        block_sizes = [args.block_size]
    if args.quant_mode is not None:
        quant_modes = ["per_token", "per_tensor"] if args.quant_mode == "both" else [args.quant_mode]
    if args.trans_v is not None:
        trans_v = [args.trans_v]
    if args.kv_varlen is not None:
        kv_varlen = [args.kv_varlen]
    if args.context_partition_size is not None:
        context_partition_sizes = [args.context_partition_size]
    if args.sliding_window is not None:
        sliding_window_options = [args.sliding_window]
    if args.global_window is not None:
        global_window_options = [args.global_window]
    return (
        block_sizes,
        head_configs,
        context_lengths,
        batch_sizes,
        head_sizes,
        query_lengths,
        quant_modes,
        trans_v,
        kv_varlen,
        compute_types,
        context_partition_sizes,
        args.sample_rate,
        sliding_window_options,
        global_window_options,
    )


def _run_single_test(args: Tuple[Dict[str, object], int, int]) -> Dict[str, object]:
    test_config, current, total = args
    print(
        f"\n[{current}/{total}] Testing: "
        f"compute_type={test_config['compute_type']}, "
        f"quant_mode={test_config['quant_mode']}, "
        f"trans_v={test_config['trans_v']}, "
        f"kv_varlen={test_config['kv_varlen']}, "
        f"context_partition_size={test_config['context_partition_size']}, "
        f"block_size={test_config['block_size']}, "
        f"num_heads={test_config['num_heads']}, "
        f"context_length={test_config['context_length']}, "
        f"batch_size={test_config['batch_size']}, "
        f"query_length={test_config['query_length']}, "
        f"head_size={test_config['head_size']}, "
        f"sliding_window={test_config['sliding_window']}, "
        f"global_window={test_config['global_window']}"
    )
    return run_pa_decode_ps_test(**test_config)


def run_multi_pa_decode_ps_test(
    block_sizes: List[int],
    head_configs: List[Tuple[int, int]],
    context_lengths: List[int],
    batch_sizes: List[int],
    head_sizes: List[int],
    query_lengths: List[int],
    quant_modes: List[str],
    trans_v: List[bool],
    kv_varlen: List[bool],
    compute_types: List[torch.dtype],
    context_partition_sizes: List[int],
    *,
    sample_rate: float = 1.0,
    sliding_window_options: List[int],
    global_window_options: List[int],
) -> pd.DataFrame:
    test_configs: List[Dict[str, object]] = []
    for compute_type in compute_types:
        for trans_v_mode in trans_v:
            for kv_varlen_mode in kv_varlen:
                for context_partition_size in context_partition_sizes:
                    for quant_mode in quant_modes:
                        for block_size in block_sizes:
                            for head_size in head_sizes:
                                for query_length in query_lengths:
                                    for batch_size in batch_sizes:
                                        for context_length in context_lengths:
                                            for head_config in head_configs:
                                                for sliding_window in sliding_window_options:
                                                    for global_window in global_window_options:
                                                        if global_window > 0 and sliding_window <= 0:
                                                            continue
                                                        test_configs.append(
                                                            {
                                                                "compute_type": compute_type,
                                                                "quant_mode": quant_mode,
                                                                "trans_v": trans_v_mode,
                                                                "kv_varlen": kv_varlen_mode,
                                                                "context_partition_size": context_partition_size,
                                                                "block_size": block_size,
                                                                "num_heads": head_config,
                                                                "context_length": context_length,
                                                                "batch_size": batch_size,
                                                                "query_length": query_length,
                                                                "head_size": head_size,
                                                                "sliding_window": sliding_window,
                                                                "global_window": global_window,
                                                            }
                                                        )
    total = len(test_configs)
    print(f"\nTotal test cases: {total}")
    if sample_rate < 1.0:
        sampler = random.Random(1234)
        test_configs = [cfg for cfg in test_configs if sampler.random() < sample_rate]
        print(
            f"Using random sampling: running {len(test_configs)} out of {total} cases "
            f"(sample_rate={sample_rate:.2%})"
        )
    else:
        print(f"Running all {total} cases")
    if not test_configs:
        raise RuntimeError("No test cases selected")
    results = []
    for idx, test_config in enumerate(test_configs):
        results.append(_run_single_test((test_config, idx + 1, len(test_configs))))
    return pd.DataFrame(results)


def parse_arg_and_run_test(sample_rate0: float = None, *, output_tag: str = TEST_NAME) -> None:
    print(f"Triton version: {triton.__version__}")
    parser = create_argument_parser()
    running_via_pytest = "pytest" in sys.argv[0] or sys.argv[0].endswith("py.test")
    args = parser.parse_args([] if running_via_pytest else None)
    global USE_CUDA_GRAPH_TEST
    USE_CUDA_GRAPH_TEST = args.use_cuda_graph
    (
        block_sizes,
        head_configs,
        context_lengths,
        batch_sizes,
        head_sizes,
        query_lengths,
        quant_modes,
        trans_v,
        kv_varlen,
        compute_types,
        context_partition_sizes,
        sample_rate1,
        sliding_window_options,
        global_window_options,
    ) = process_arguments(args)
    sample_rate = sample_rate1 if sample_rate0 is None else sample_rate0
    results_df = run_multi_pa_decode_ps_test(
        block_sizes,
        head_configs,
        context_lengths,
        batch_sizes,
        head_sizes,
        query_lengths,
        quant_modes,
        trans_v,
        kv_varlen,
        compute_types,
        context_partition_sizes,
        sample_rate=sample_rate,
        sliding_window_options=sliding_window_options,
        global_window_options=global_window_options,
    )
    output_file = (
        f"run_pa_decode_ps_test.{output_tag}.block_size_{block_sizes[0]}.triton.{TRITON_VERSION}.csv"
    )
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    print(f"\nSummary:\n{results_df}")
    flydsl_errors = int(results_df["err_flydsl_ps"].sum())

    if flydsl_errors:
        raise AssertionError(
            f"{flydsl_errors} FlyDSL PS case(s) exceeded the Torch-reference tolerance"
        )

    print("\nAll PS-only tests passed!")


def normal_accuracy_test() -> None:
    global BLOCK_SIZE_OPTIONS
    global QUERY_LENGTH_OPTIONS
    global BATCH_SIZE_OPTIONS
    global HEAD_CONFIGURATIONS
    global CONTEXT_LENGTH_OPTIONS
    global COMPUTE_TYPE_OPTIONS
    global QUANT_MODE_OPTIONS
    global HEAD_DIMENSION_OPTIONS
    global TRANS_V_OPTIONS
    global KV_VARLEN_OPTIONS
    global CONTEXT_PARTITION_SIZE_OPTIONS
    global SLIDING_WINDOW_OPTIONS
    global GLOBAL_WINDOW_OPTIONS
    COMPUTE_TYPE_OPTIONS = ["fp8"]
    CONTEXT_PARTITION_SIZE_OPTIONS = [256]
    HEAD_DIMENSION_OPTIONS = [128]
    HEAD_CONFIGURATIONS = [(8, 1), (16, 1)]
    QUERY_LENGTH_OPTIONS = [1, 2, 3, 4]
    QUANT_MODE_OPTIONS = ["per_token", "per_tensor"]
    CONTEXT_LENGTH_OPTIONS = [1027]
    BATCH_SIZE_OPTIONS = [3, 81]
    TRANS_V_OPTIONS = [True]
    KV_VARLEN_OPTIONS = [False, True]
    BLOCK_SIZE_OPTIONS = [1024]
    SLIDING_WINDOW_OPTIONS = [0]
    GLOBAL_WINDOW_OPTIONS = [0]
    parse_arg_and_run_test(output_tag="ps_normal_accuracy")


def sliding_window_accuracy_test() -> None:
    global BLOCK_SIZE_OPTIONS
    global QUERY_LENGTH_OPTIONS
    global BATCH_SIZE_OPTIONS
    global HEAD_CONFIGURATIONS
    global CONTEXT_LENGTH_OPTIONS
    global COMPUTE_TYPE_OPTIONS
    global QUANT_MODE_OPTIONS
    global HEAD_DIMENSION_OPTIONS
    global TRANS_V_OPTIONS
    global KV_VARLEN_OPTIONS
    global CONTEXT_PARTITION_SIZE_OPTIONS
    global SLIDING_WINDOW_OPTIONS
    global GLOBAL_WINDOW_OPTIONS
    COMPUTE_TYPE_OPTIONS = ["fp8"]
    CONTEXT_PARTITION_SIZE_OPTIONS = [256]
    HEAD_DIMENSION_OPTIONS = [128]
    HEAD_CONFIGURATIONS = [(8, 1), (16, 1)]
    QUERY_LENGTH_OPTIONS = [1, 2, 3, 4]
    QUANT_MODE_OPTIONS = ["per_token"]
    CONTEXT_LENGTH_OPTIONS = [8192]
    BATCH_SIZE_OPTIONS = [128]
    TRANS_V_OPTIONS = [True]
    KV_VARLEN_OPTIONS = [True]
    BLOCK_SIZE_OPTIONS = [1024]
    SLIDING_WINDOW_OPTIONS = [0, 128, 1023]
    GLOBAL_WINDOW_OPTIONS = [0]
    parse_arg_and_run_test(output_tag="ps_sliding_window_accuracy")


def global_window_accuracy_test() -> None:
    global BLOCK_SIZE_OPTIONS
    global QUERY_LENGTH_OPTIONS
    global BATCH_SIZE_OPTIONS
    global HEAD_CONFIGURATIONS
    global CONTEXT_LENGTH_OPTIONS
    global COMPUTE_TYPE_OPTIONS
    global QUANT_MODE_OPTIONS
    global HEAD_DIMENSION_OPTIONS
    global TRANS_V_OPTIONS
    global KV_VARLEN_OPTIONS
    global CONTEXT_PARTITION_SIZE_OPTIONS
    global SLIDING_WINDOW_OPTIONS
    global GLOBAL_WINDOW_OPTIONS
    COMPUTE_TYPE_OPTIONS = ["fp8"]
    CONTEXT_PARTITION_SIZE_OPTIONS = [256]
    HEAD_DIMENSION_OPTIONS = [128]
    HEAD_CONFIGURATIONS = [(8, 2), (12, 2)]
    QUERY_LENGTH_OPTIONS = [4]
    QUANT_MODE_OPTIONS = ["per_token"]
    CONTEXT_LENGTH_OPTIONS = [32768]
    BATCH_SIZE_OPTIONS = [64]
    TRANS_V_OPTIONS = [True]
    KV_VARLEN_OPTIONS = [True]
    BLOCK_SIZE_OPTIONS = [1024]
    SLIDING_WINDOW_OPTIONS = [1023]
    GLOBAL_WINDOW_OPTIONS = [3]

    parse_arg_and_run_test(output_tag="ps_global_window_accuracy")


@pytest.mark.parametrize("case_set_name", CASE_SET_NAME_OPTIONS)
def test_multi_case_set(case_set_name: str) -> None:
    if case_set_name == "normal_accuracy":
        normal_accuracy_test()
    elif case_set_name == "sliding_window_accuracy":
        sliding_window_accuracy_test()
    elif case_set_name == "global_window_accuracy":
        global_window_accuracy_test()
    else:
        raise ValueError(f"Unsupported case set: {case_set_name}")


if __name__ == "__main__":
    # sliding_window_accuracy_test()
    global_window_accuracy_test()
