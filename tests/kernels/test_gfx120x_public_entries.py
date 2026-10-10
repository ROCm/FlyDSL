# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""CPU checks that the gfx120x fixed product hosts exist.

Does not launch a kernel. Device coverage stays in the test_rdna4_* and
test_flash_attn_gfx120x files.
"""

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]


def test_default_hosts_are_callable() -> None:
    from kernels.gemm import rdna4_scaled_mm_fp8 as plain
    from kernels.gemm.rdna4_int8_linear import create_wmma_int8_linear_module, int8_linear
    from kernels.gemm.rdna4_int8_linear_fused import int8_linear_fused
    from kernels.gemm.rdna4_scaled_mm_fp8 import (
        TileConfig,
        build_scaled_mm_fp8_module,
        pick_tile_config,
        scaled_mm_fp8,
    )
    from kernels.gemm.rdna4_scaled_mm_fp8_fused import scaled_mm_fp8_fused
    from kernels.gemm.rdna4_w8a16_linear import w8a16_gemm, w8a16_linear

    assert callable(int8_linear)
    assert callable(create_wmma_int8_linear_module)
    assert callable(int8_linear_fused)
    assert callable(scaled_mm_fp8)
    assert callable(build_scaled_mm_fp8_module)
    assert callable(pick_tile_config)
    assert callable(scaled_mm_fp8_fused)
    assert callable(w8a16_linear)
    assert callable(w8a16_gemm)
    assert TileConfig is not None
    assert hasattr(plain, "scaled_mm_fp8")
    assert not hasattr(plain, "scaled_mm_fp8_auto")
    assert not hasattr(plain, "int8_linear_auto")


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


def test_capability_catalog_is_arch_gated_and_callable() -> None:
    """A host can list gfx120x calls. Other arches, including gfx1250, get none."""
    from kernels.common.gfx120x_capabilities import available_for_arch, catalog, resolve

    ops = catalog()
    names = [op.name for op in ops]
    assert len(names) == len(set(names))
    assert "flash_attn" in names
    assert "scaled_mm_fp8" in names
    assert "mxfp4_block_gemm" in names
    assert "iu4_gemm" in names
    assert available_for_arch("gfx950") == ()
    assert available_for_arch("gfx942") == ()
    assert available_for_arch("gfx1250") == ()
    assert available_for_arch("gfx11") == ()
    assert available_for_arch(None) == ()
    assert available_for_arch("gfx1201") == ops
    assert available_for_arch("gfx1200:xnack-") == ops
    for op in ops:
        fn = resolve(op.name)
        if not callable(fn):
            raise AssertionError(f"{op.name} did not resolve to a callable")
