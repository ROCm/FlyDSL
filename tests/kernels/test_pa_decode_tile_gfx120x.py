# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""gfx120x paged decode against the dense flash kernel."""

import sys
from pathlib import Path

import pytest

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

from kernels.attention.flash_attn_gfx120x_host import flydsl_flash_attn_func  # noqa: E402
from kernels.attention.pa_decode_tile_gfx120x import pa_decode_tile  # noqa: E402


def _pack_vectorized(k_linear, v_linear):
    """linear ``[blocks, page, H, D]`` to the decode K and V layouts."""
    nb, page, h, d = k_linear.shape
    vec = 16 // k_linear.element_size()
    dv = int(v_linear.shape[-1])
    k5 = k_linear.permute(0, 2, 1, 3).reshape(nb, h, page, d // vec, vec).permute(0, 1, 3, 2, 4).contiguous()
    v5 = v_linear.permute(0, 2, 1, 3).reshape(nb, h, page // vec, vec, dv).permute(0, 1, 2, 4, 3).contiguous()
    return k5, v5


def test_bf16_decode_matches_dense() -> None:
    torch.manual_seed(0)
    page, h, d, sk = 16, 2, 64, 32
    q = torch.randn(1, 1, h, d, device="cuda", dtype=torch.bfloat16)
    k_lin = torch.randn(2, page, h, d, device="cuda", dtype=torch.bfloat16)
    v_lin = torch.randn(2, page, h, d, device="cuda", dtype=torch.bfloat16)
    k5, v5 = _pack_vectorized(k_lin, v_lin)
    bt = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
    sk_t = torch.tensor([sk], device="cuda", dtype=torch.int32)
    out = torch.empty(1, h, d, device="cuda", dtype=torch.bfloat16)
    pa_decode_tile(out, q.reshape(1, h, d), k5, v5, bt, sk_t, None, None, softmax_scale=d**-0.5)
    dense_k = k_lin.reshape(1, sk, h, d)
    dense_v = v_lin.reshape(1, sk, h, d)
    ref = flydsl_flash_attn_func(q, dense_k, dense_v, causal=False, sm_scale=d**-0.5)
    err = (out.float() - ref.reshape(1, h, d).float()).abs().max().item()
    assert err < 1e-2, err
