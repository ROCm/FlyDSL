# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""CPU checks that the gfx120x default hosts exist and the missing name does not.

Does not launch a kernel. Device coverage stays in the test_rdna4_* and
test_flash_attn_gfx120x files.
"""

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]


def test_default_hosts_are_callable() -> None:
    from kernels.gemm import rdna4_scaled_mm_fp8 as plain
    from kernels.gemm.rdna4_int8_linear_dispatch import int8_linear_auto, int8_linear_dispatched
    from kernels.gemm.rdna4_scaled_mm_fp8 import build_scaled_mm_fp8_module
    from kernels.gemm.rdna4_scaled_mm_fp8_auto import scaled_mm_fp8_auto

    assert callable(int8_linear_auto)
    assert callable(int8_linear_dispatched)
    assert callable(scaled_mm_fp8_auto)
    assert callable(build_scaled_mm_fp8_module)
    assert not hasattr(plain, "scaled_mm_fp8")


def test_hosts_the_audits_found_uncalled_are_still_exported() -> None:
    """Import check only. Device tests cover the paths that launch."""
    from kernels.attention.flash_attn_gfx120x_host import (
        flydsl_flash_attn_paged_func,
        flydsl_flash_attn_varlen_func,
        flydsl_flash_attn_varlen_paged_func,
    )
    from kernels.gemm.rdna4_w8a16_linear import w8a16_linear
    from kernels.norm.rms_rope_gfx120x import (
        build_rms_rope_qk_fused_module,
        build_rms_rope_split_module,
        build_rms_rope_split_qk_fused_module,
    )
    from kernels.norm.rope_gfx120x import build_rope_split_half_qk_fused_module
    from kernels.quant.rdna4_int4_codec import dequant_uint4_groupwise_awq
    from kernels.quant.rdna4_int8_convrot import quantize_and_rotate_rowwise

    for fn in (
        flydsl_flash_attn_varlen_func,
        flydsl_flash_attn_paged_func,
        flydsl_flash_attn_varlen_paged_func,
        w8a16_linear,
        build_rope_split_half_qk_fused_module,
        build_rms_rope_split_module,
        build_rms_rope_qk_fused_module,
        build_rms_rope_split_qk_fused_module,
        dequant_uint4_groupwise_awq,
        quantize_and_rotate_rowwise,
    ):
        assert callable(fn)
