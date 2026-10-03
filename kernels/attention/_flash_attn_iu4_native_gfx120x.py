from collections.abc import Callable


# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
def build_flash_attn_func_iu4_native_module(**kwargs) -> Callable[..., None]:
    """Build the native packed-iu4 FlashAttention module. Dense BSHD only."""
    from kernels.attention._flash_attn_iu4_native_body_gfx120x import build_flash_attn_func_iu4_native_body

    return build_flash_attn_func_iu4_native_body(**kwargs)
