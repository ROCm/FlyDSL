# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Linear paged value head that is not a multiple of 32.

Query head 64, value head 40. The launch must accept it and return width 40.
"""

import sys
from pathlib import Path

import pytest

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.attention.flash_attn_gfx120x_host import flydsl_flash_attn_paged_func  # noqa: E402

pytestmark = [
    pytest.mark.l2_device,
    pytest.mark.rocm_lower,
    pytest.mark.skipif(
        not str(get_rocm_arch() or "").startswith("gfx120"),
        reason="requires gfx120x",
    ),
]


def test_linear_paged_value_head_not_multiple_of_32() -> None:
    """Uniform linear cache: Dv=40 is tiled, the caller still sees 40."""
    torch.manual_seed(0)
    batch, sq, page, heads, dq, dv = 1, 4, 16, 1, 64, 40
    q = torch.randn(batch, sq, heads, dq, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(1, page, heads, dq, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn(1, page, heads, dv, device="cuda", dtype=torch.bfloat16)
    block_table = torch.zeros(batch, 1, device="cuda", dtype=torch.int32)
    seqlen_k = torch.tensor([page], device="cuda", dtype=torch.int32)
    k_ptr, v_ptr = k_cache.data_ptr(), v_cache.data_ptr()

    out = flydsl_flash_attn_paged_func(
        q,
        k_cache,
        v_cache,
        block_table,
        seqlen_k,
        page_size=page,
        causal=False,
        kv_cache_layout="linear",
    )

    assert k_cache.data_ptr() == k_ptr
    assert v_cache.data_ptr() == v_ptr
    assert tuple(v_cache.shape) == (1, page, heads, dv)
    assert tuple(out.shape) == (batch, sq, heads, dv)
