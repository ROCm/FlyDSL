"""Architecture-aware test filtering for GPU kernel tests.

Automatically skips tests that are incompatible with the current GPU:
  - MFMA-based kernels (preshuffle GEMM, MoE, blockscale, PA) require CDNA (gfx9xx)
  - WMMA-based kernels require RDNA4 (gfx12xx) — handled by per-test _requires_rdna4()
  - Generic kernels (softmax, layernorm, vec_add, quant, etc.) run on all architectures
"""

import pytest

# Test files that require CDNA (gfx9xx) GPUs.
# Reasons: MFMA instructions, hardcoded wave64, or imports from CDNA-only kernels.
_CDNA_ONLY_TEST_FILES = frozenset({
    "test_preshuffle_gemm.py",         # MFMA preshuffle pipeline
    "test_blockscale_preshuffle_gemm.py",  # MFMA blockscale pipeline
    "test_moe_gemm.py",                # MFMA MoE 2-stage
    "test_moe_blockscale.py",          # MFMA MoE blockscale
    "test_moe_reduce.py",              # imports moe_gemm_2stage (MFMA)
    "test_pa.py",                      # MFMA paged attention
    "test_quant.py",                   # hardcodes WARP_SIZE=64
})


def _get_gpu_arch():
    try:
        from flydsl.runtime.device import get_rocm_arch
        return str(get_rocm_arch())
    except Exception:
        return ""


def pytest_collection_modifyitems(config, items):
    arch = _get_gpu_arch()
    if not arch:
        return

    is_cdna = "gfx9" in arch
    is_rdna4 = "gfx12" in arch

    if is_cdna:
        return

    skip_marker = pytest.mark.skip(
        reason=f"Test requires CDNA (gfx9xx) GPU, current arch: {arch}"
    )
    for item in items:
        filename = item.fspath.basename
        if filename in _CDNA_ONLY_TEST_FILES:
            item.add_marker(skip_marker)
