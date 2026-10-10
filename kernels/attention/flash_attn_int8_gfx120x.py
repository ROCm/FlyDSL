# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x int8 FlashAttention. The device kernel is the shared quant builder."""

from collections.abc import Callable


def build_flash_attn_func_int8_module_primary(
    num_heads: int,
    head_dim: int,
    causal: bool = True,
    dtype_str: str = "int8",
    sm_scale: float | None = None,
    waves_per_eu: int = 2,
    flat_work_group_size: int | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    unsafe_fp_math: bool = True,
    fast_fp_math: bool = True,
    daz: bool = True,
    path_tag: str = "auto",
    has_attn_bias: bool = False,
    has_per_head_bias: bool = False,
    return_lse: bool = False,
    has_sink: bool = False,
    num_kv_heads: int | None = None,
    sign_a: bool = True,
    sign_b: bool = True,
    logical_head_dim: int | None = None,
    varlen: bool = False,
    paged: bool = False,
    page_size: int = 16,
    kv_cache_layout: str = "linear",
    split_k: bool = False,
    sliding_window: tuple[int, int] | None = None,
    bias_bottom_right: bool = False,
    has_alibi: bool = False,
    alibi_per_head: bool = False,
) -> Callable[..., None]:
    """Build iu8 FlashAttention through the shared fp8/int8 quant kernel."""
    from kernels.attention.flash_attn_fp8_gfx120x import build_flash_attn_func_fp8_module_primary

    if dtype_str not in ("int8", "i8"):
        raise ValueError(f"int8 gfx120x FA expects dtype_str int8, got {dtype_str!r}")
    return build_flash_attn_func_fp8_module_primary(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str="int8",
        sm_scale=sm_scale,
        waves_per_eu=waves_per_eu,
        flat_work_group_size=flat_work_group_size,
        block_m=block_m,
        block_n=block_n,
        unsafe_fp_math=unsafe_fp_math,
        fast_fp_math=fast_fp_math,
        daz=daz,
        path_tag=path_tag,
        has_attn_bias=has_attn_bias,
        has_per_head_bias=has_per_head_bias,
        return_lse=return_lse,
        has_sink=has_sink,
        num_kv_heads=num_kv_heads,
        sign_a=sign_a,
        sign_b=sign_b,
        logical_head_dim=logical_head_dim,
        varlen=varlen,
        paged=paged,
        page_size=page_size,
        kv_cache_layout=kv_cache_layout,
        split_k=split_k,
        sliding_window=sliding_window,
        bias_bottom_right=bias_bottom_right,
        has_alibi=has_alibi,
        alibi_per_head=alibi_per_head,
    )


build_flash_attn_func_int8_module = build_flash_attn_func_int8_module_primary
build_flash_attn_func_int8_module_gfx120x = build_flash_attn_func_int8_module_primary
