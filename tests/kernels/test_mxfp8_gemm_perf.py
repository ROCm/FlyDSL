#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MXFP8 (a8w8) dense GEMM perf-regression guard (gfx950).

Exercises the merged block-scale path (`compile_mxfp8_gemm_8w` in
`kernels/gemm/fp8_gemm_8wave.py`, via the `gemm_mxfp8_flydsl_kernel` host entry) on
compute-bound shapes and asserts both correctness (cos) and a conservative TFLOPS
floor. The floor is set well below the observed ~2600-2850 TF (fx.gemm scaled-atom
path, HipKittens FP8_8wave / PR #390 pipeline) so gfx950 clock noise (~+-14%) does
not make it flaky, while a structural regression (e.g. streaming B from gmem instead
of LDS, which drops 8192^3 to ~2400 TF) still trips it. Uses best-of-N (peak clock).

Marked `benchmark` -> opt-in (`pytest -m benchmark`), not in the default fast suite.
"""

import math
import os
import sys

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower, pytest.mark.benchmark]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.gemm.mxfp8_gemm_kernel import gemm_mxfp8_flydsl_kernel  # noqa: E402
from tests.kernels.benchmark_common import bench_kernel_us  # noqa: E402
from tests.kernels.utils import gemm_common_utils as U  # noqa: E402

ARCH = str(get_rocm_arch())

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


def _per_1x32_f8_quant(x):
    block = 32
    dtype_max = 2.0 ** int(math.log2(448.0))
    shp = x.shape
    xb = x.reshape(-1, block)
    scale_e8m0 = U.f32_to_e8m0(torch.amax(torch.abs(xb.float()), 1) / dtype_max)
    scale_f32 = U.e8m0_to_f32(scale_e8m0)
    codes = (xb.float() / scale_f32.view(-1, 1)).to(torch.float8_e4m3fn).view(*shp).contiguous()
    return codes, scale_e8m0.view(shp[0], -1).view(torch.uint8)


def _cos(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return (torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)).item()


@pytest.mark.parametrize(
    "M, N, K, floor_tf",
    [
        # Floors ~15-20% under the fx.gemm-atom baseline (4096x4096x8192 ~2620,
        # 8192^3 ~2820); catch a real regression, tolerate clock noise.
        pytest.param(4096, 4096, 8192, 2200.0, id="4096x4096x8192"),
        pytest.param(8192, 8192, 8192, 2350.0, marks=pytest.mark.large_shape, id="8192x8192x8192"),
    ],
)
def test_mxfp8_gemm_perf_floor(M, N, K, floor_tf):
    if ARCH != "gfx950":
        pytest.skip(f"mxfp8 8-wave a8w8 GEMM requires gfx950, got {ARCH}")

    a = torch.randn(M, K, device="cuda")
    b = torch.randn(N, K, device="cuda")
    a_codes, a_scale = _per_1x32_f8_quant(a)
    b_codes, b_scale = _per_1x32_f8_quant(b)
    a_deq = U.fp8_e4m3_to_f32(a_codes.view(torch.uint8)) * U.e8m0_to_f32(a_scale.repeat_interleave(32, dim=1))
    b_deq = U.fp8_e4m3_to_f32(b_codes.view(torch.uint8)) * U.e8m0_to_f32(b_scale.repeat_interleave(32, dim=1))
    c_ref = torch.mm(a_deq, b_deq.T).to(torch.float32)

    out = gemm_mxfp8_flydsl_kernel(a_codes, a_scale, b_codes, b_scale, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    cos = _cos(out.float(), c_ref)
    assert cos >= 0.99, f"cos={cos} below threshold for {M}x{N}x{K}"

    def run():
        gemm_mxfp8_flydsl_kernel(a_codes, a_scale, b_codes, b_scale, out_dtype=torch.bfloat16)

    # best-of-N (min us = peak clock) so a single low-clock window does not fail the floor.
    best_us = min(bench_kernel_us(run, warmup=15, iters=40, flush_l2=False) for _ in range(5))
    tflops = (2 * M * N * K) / (best_us / 1e6) / 1e12
    print(f"\n[mxfp8 {M}x{N}x{K}] {tflops:.1f} TFLOPS ({best_us:.1f} us), cos={cos:.6f}, floor={floor_tf} TF")
    assert tflops >= floor_tf, f"mxfp8 {M}x{N}x{K}: {tflops:.1f} TF < floor {floor_tf} TF (perf regression)"
