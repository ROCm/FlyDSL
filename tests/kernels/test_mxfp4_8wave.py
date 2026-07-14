#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""8-wave a8w4 GEMM (fp8 E4M3 A x preshuffled MXFP4 B) correctness harness.

Kernel implementation: ``kernels/gemm/mxfp4_8wave.py`` (gfx950 only).

C[M,N] = A[M,K] @ B[N,K]^T with A = per-1x32 e8m0-scaled fp8 codes (row-major),
B = ``shuffle_weight_w4`` preshuffled MXFP4, both scales ``shuffle_scale_w4``
preshuffled, bf16 output. Mirrors ``test_fp4_gemm_4wave`` / the a8w4 path of
``bench_a8w4.py``.
"""

import math
import os
import sys

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.gemm.mxfp4_8wave import compile_mxfp4_8w  # noqa: E402
from tests.kernels.utils import gemm_common_utils as fp4_utils  # noqa: E402

OUT_DTYPE = torch.bfloat16
ARCH = str(get_rocm_arch())

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


def _ptr(t):
    return flyc.from_c_void_p(fx.Uint8, t.contiguous().data_ptr())


def _per_1x32_f8_quant(x):
    block = 32
    dtype_max = 2.0 ** int(math.log2(448.0))
    shp = x.shape
    xb = x.reshape(-1, block)
    max_abs = torch.amax(torch.abs(xb.float()), 1)
    scale_e8m0 = fp4_utils.f32_to_e8m0(max_abs / dtype_max)
    scale_f32 = fp4_utils.e8m0_to_f32(scale_e8m0)
    y = xb.float() / scale_f32.view(-1, 1)
    codes = y.to(torch.float8_e4m3fn).view(*shp).contiguous()
    scale = scale_e8m0.view(shp[0], -1).view(torch.uint8)
    return codes, scale


def _make_inputs(M, N, K, device="cuda"):
    M32 = (M + 31) // 32 * 32
    N32 = (N + 31) // 32 * 32
    a_f = torch.randn(M, K, device=device, dtype=torch.float32)
    b_f = torch.randn(N, K, device=device, dtype=torch.float32)
    a_fp = torch.zeros(M32, K, device=device, dtype=torch.float32)
    b_fp = torch.zeros(N32, K, device=device, dtype=torch.float32)
    a_fp[:M] = a_f
    b_fp[:N] = b_f

    a_codes, scale_a_orig = _per_1x32_f8_quant(a_fp)
    a_codes = a_codes[:M]
    scale_a = fp4_utils.shuffle_scale_w4(scale_a_orig, 1, False)

    b_q, scale_b, _ = fp4_utils.per_1x32_f4_quant(b_fp)
    b_q = b_q[:N]
    b_shuf = fp4_utils.shuffle_weight_w4(b_q, 16, False, False)
    scale_b_shuf = fp4_utils.shuffle_scale_w4(scale_b, 1, False)

    a_deq = fp4_utils.fp8_e4m3_to_f32(a_codes.view(torch.uint8)) * fp4_utils.e8m0_to_f32(
        scale_a_orig[:M].repeat_interleave(32, dim=1)
    )
    b_deq = fp4_utils.mxfp4_to_f32(b_q) * fp4_utils.e8m0_to_f32(scale_b[:N].repeat_interleave(32, dim=1))
    c_ref = torch.mm(a_deq, b_deq.T).to(torch.float32)

    c_out = torch.zeros((M, N), dtype=OUT_DTYPE, device=device)
    return a_codes, b_shuf, scale_a, scale_b_shuf, c_out, c_ref


def _cos_sim(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return (torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)).item()


@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n",
    [
        pytest.param(2048, 4096, 4096, 256, 256, id="2048x4096x4096"),
        pytest.param(1024, 2048, 2048, 256, 256, id="1024x2048x2048"),
        pytest.param(8192, 8192, 8192, 256, 256, marks=pytest.mark.large_shape, id="8192x8192x8192"),
        pytest.param(4096, 4096, 8192, 256, 256, marks=pytest.mark.large_shape, id="4096x4096x8192"),
    ],
)
def test_mxfp4_8wave_a8w4(M, N, K, tile_m, tile_n):
    if ARCH != "gfx950":
        pytest.skip(f"mxfp4 8-wave a8w4 GEMM requires gfx950, got {ARCH}")

    a, b, sa, sb, c, c_ref = _make_inputs(M, N, K)
    fn = compile_mxfp4_8w(K=K, N=N, BLOCK_M=tile_m, BLOCK_N=tile_n, BLOCK_K=256)
    fn(_ptr(c), _ptr(a), _ptr(b), _ptr(sa), _ptr(sb), M, N, torch.cuda.current_stream())
    torch.cuda.synchronize()

    cos = _cos_sim(c.float(), c_ref)
    assert cos >= 0.99, f"cos={cos} below threshold for {M}x{N}x{K}"
