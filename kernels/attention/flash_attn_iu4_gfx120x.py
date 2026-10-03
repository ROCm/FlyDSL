# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""iu4 FlashAttention builder.

Tries the native iu4 module. If that import fails, or ``prefer_native`` is
false, returns the int8 module. This function does not unpack nibbles and
does not look at bias, ALiBi, sink, or LSE. The host does that.
"""

from collections.abc import Callable

KERNEL_NAME = "flash_attn_func_iu4_gfx120x_kernel"
_NATIVE_BUILD_ERROR = None


def build_flash_attn_func_iu4_module_primary(
    num_heads: int,
    head_dim: int,
    causal: bool = True,
    dtype_str: str = "iu4",
    sm_scale: float | None = None,
    waves_per_eu: int = 2,
    flat_work_group_size: int | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    unsafe_fp_math: bool = True,
    fast_fp_math: bool = True,
    daz: bool = True,
    path_tag: str = "auto",
    prefer_native: bool = True,
    num_kv_heads: int | None = None,
) -> Callable[..., None]:
    """Build the iu4 module, or the int8 module if native build fails."""
    global _NATIVE_BUILD_ERROR
    if prefer_native:
        try:
            from kernels.attention._flash_attn_iu4_native_gfx120x import build_flash_attn_func_iu4_native_module

            return build_flash_attn_func_iu4_native_module(
                num_heads=num_heads,
                head_dim=head_dim,
                causal=causal,
                dtype_str=dtype_str,
                sm_scale=sm_scale,
                waves_per_eu=waves_per_eu,
                flat_work_group_size=flat_work_group_size,
                block_m=block_m,
                block_n=block_n,
                unsafe_fp_math=unsafe_fp_math,
                fast_fp_math=fast_fp_math,
                daz=daz,
                path_tag=path_tag,
                num_kv_heads=num_kv_heads,
            )
        except Exception as exc:
            _NATIVE_BUILD_ERROR = f"{type(exc).__name__}: {exc}"
    from kernels.attention.flash_attn_int8_gfx120x import build_flash_attn_func_int8_module

    return build_flash_attn_func_int8_module(
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
        num_kv_heads=num_kv_heads,
    )


build_flash_attn_func_iu4_module = build_flash_attn_func_iu4_module_primary
build_flash_attn_func_iu4_module_gfx120x = build_flash_attn_func_iu4_module_primary


def native_iu4_build_error() -> str | None:
    """Return the reason the native iu4 body cannot be built, or None."""
    return _NATIVE_BUILD_ERROR
