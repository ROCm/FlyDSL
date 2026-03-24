"""Architecture-aware test filtering for GPU kernel tests.

Automatically skips tests that are incompatible with the current GPU:
  - MFMA-based kernels (preshuffle GEMM, MoE, blockscale, PA) require CDNA (gfx9xx)
  - WMMA-based kernels require RDNA4 (gfx12xx) — handled by per-test _requires_rdna4()
  - Generic kernels (softmax, layernorm, vec_add, quant, etc.) run on all architectures

Configuration lives in tests/arch_compat.py (single source of truth).
"""

import pytest
from tests.arch_compat import CDNA_ONLY_TESTS


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

    if is_cdna:
        return

    skip_marker = pytest.mark.skip(
        reason=f"Test requires CDNA (gfx9xx) GPU, current arch: {arch}"
    )
    for item in items:
        filename = item.fspath.basename
        if filename in CDNA_ONLY_TESTS:
            item.add_marker(skip_marker)
