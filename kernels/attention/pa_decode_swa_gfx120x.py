# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x sliding-window decode. Same builder arguments as ``compile_pa_decode_sw``.

The flash kernel applies the window. This module does not launch the CDNA
pointer ABI. ``pa_decode_ps_launch`` is the tensor entry.
"""

from __future__ import annotations

from flydsl.runtime.device import get_rocm_arch


def compile_pa_decode_sw(
    sliding_window: int,
    max_context_partition_num: int,
    softmax_scale=None,
    trans_v=False,
    query_group_size=16,
    per_token_kv=False,
    query_length: int = 1,
    query_input_dtype: str = "bf16",
    head_dim: int = 128,
):
    """Record a windowed decode. The tensor launch is ``pa_decode_ps_launch``."""
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx120"), f"compile_pa_decode_sw requires gfx120x, got {arch}"
    if sliding_window <= 0:
        raise ValueError("compile_pa_decode_sw requires sliding_window > 0")
    if query_input_dtype not in ("bf16", "f16", "fp8_e4m3fn", "fp8_e5m2", "int8"):
        raise ValueError(f"compile_pa_decode_sw dtype {query_input_dtype!r} is not a gfx120x attention dtype")
    return {
        "sliding_window": int(sliding_window),
        "max_context_partition_num": int(max_context_partition_num),
        "softmax_scale": softmax_scale,
        "trans_v": bool(trans_v),
        "query_group_size": int(query_group_size),
        "per_token_kv": bool(per_token_kv),
        "query_length": int(query_length),
        "query_input_dtype": query_input_dtype,
        "head_dim": int(head_dim),
    }


def compile_pa_decode_sw_reduce(
    max_context_partition_num: int,
    query_seq_len: int,
    query_group_size: int,
    head_size: int,
    output_dtype_str: str,
):
    """The WMMA launch already reduced the window. The record is the host check."""
    arch = str(get_rocm_arch())
    assert arch.startswith("gfx120"), f"compile_pa_decode_sw_reduce requires gfx120x, got {arch}"
    if output_dtype_str not in ("bf16", "f16", "fp8_e4m3fn", "fp8_e5m2", "int8"):
        raise ValueError(f"compile_pa_decode_sw_reduce dtype {output_dtype_str!r} is not a gfx120x dtype")
    return {
        "max_context_partition_num": int(max_context_partition_num),
        "query_seq_len": int(query_seq_len),
        "query_group_size": int(query_group_size),
        "head_size": int(head_size),
        "output_dtype_str": output_dtype_str,
    }
