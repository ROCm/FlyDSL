#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MXFP8 (a8w8) dense GEMM correctness harness (gfx950)."""

import os
import sys

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.gemm.mxfp8_gemm_interface import gemm_mxfp8  # noqa: E402
from tests.kernels.utils import gemm_common_utils  # noqa: E402
from tests.test_common import run_perftest, verify_output  # noqa: E402

ARCH = str(get_rocm_arch())

DEFAULT_BENCH_ITERS = 20
DEFAULT_BENCH_WARMUP = 3

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


def run_torch(a_codes, a_scale, b_codes, b_scale, dtype=torch.float32):
    a_deq = gemm_common_utils.fp8_e4m3_to_f32(a_codes.view(torch.uint8)) * gemm_common_utils.e8m0_to_f32(
        a_scale.repeat_interleave(32, dim=1)
    )
    b_deq = gemm_common_utils.fp8_e4m3_to_f32(b_codes.view(torch.uint8)) * gemm_common_utils.e8m0_to_f32(
        b_scale.repeat_interleave(32, dim=1)
    )
    return torch.mm(a_deq, b_deq.T).to(dtype)


@pytest.mark.parametrize(
    "M, N, K, out_dtype",
    [
        pytest.param(256, 256, 256, torch.bfloat16, id="256x256x256-bf16"),
        pytest.param(1024, 1024, 512, torch.bfloat16, id="1024x1024x512-bf16"),
        pytest.param(100, 256, 256, torch.bfloat16, id="ragged_m-100x256x256"),
        pytest.param(256, 300, 256, torch.bfloat16, id="ragged_n-256x300x256"),
        pytest.param(512, 512, 256, torch.float16, id="512x512x256-fp16"),
        pytest.param(4096, 4096, 8192, torch.bfloat16, marks=pytest.mark.large_shape, id="4096x4096x8192-bf16"),
        pytest.param(8192, 8192, 8192, torch.bfloat16, marks=pytest.mark.large_shape, id="8192x8192x8192-bf16"),
    ],
)
@pytest.mark.l2_device
@pytest.mark.rocm_lower
def test_mxfp8_gemm_a8w8(
    M,
    N,
    K,
    out_dtype,
    *,
    bench_iters: int = DEFAULT_BENCH_ITERS,
    bench_warmup: int = DEFAULT_BENCH_WARMUP,
):
    """A8W8: MXFP8 (E4M3) A x MXFP8 (E4M3) B, fused scale preshuffle -- gfx950 only."""
    if ARCH != "gfx950":
        pytest.skip(f"mxfp8 8-wave a8w8 GEMM requires gfx950, got {ARCH}")

    _out_name = str(out_dtype).removeprefix("torch.")
    print("=" * 80)
    print(f"MXFP8 A8W8 8-wave GEMM Test (M={M}, N={N}, K={K}, out={_out_name})")
    print("=" * 80)

    device = torch.device("cuda")
    torch.manual_seed(0)

    a_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
    b_fp32 = torch.randn(N, K, device=device, dtype=torch.float32)

    # Raw E8M0 block scales: the kernel preshuffles them itself, fused into the launch.
    a_codes, a_scale = gemm_common_utils.per_1x32_f8_quant(a_fp32)
    b_codes, b_scale = gemm_common_utils.per_1x32_f8_quant(b_fp32)
    c_ref = run_torch(a_codes, a_scale, b_codes, b_scale)

    def launch_kernel(a, sa, b, sb):
        return gemm_mxfp8(a, sa, b, sb, out_dtype=out_dtype)

    bench_iters = max(2, int(bench_iters))
    c_out, us = run_perftest(
        launch_kernel,
        a_codes,
        a_scale,
        b_codes,
        b_scale,
        num_iters=bench_iters,
        num_warmup=int(bench_warmup),
    )
    torch.cuda.synchronize()

    assert c_out.shape == (M, N)
    assert c_out.dtype == out_dtype
    assert verify_output(c_out.to(torch.float32), c_ref, rtol=0.1, atol=0.1)

    bytes_moved = M * K + N * K + M * N * 2 + (M + N) * (K // 32)
    tflops = (2 * M * N * K) / (us / 1e6) / 1e12
    tbps = bytes_moved / 1e12 / (us / 1e6)
    print(f"[flyc] MXFP8 A8W8 Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s")


def test_mxfp8_gemm_rejects_tn():
    """NT only: trans_b=False must fail loudly rather than compute garbage."""
    if ARCH != "gfx950":
        pytest.skip(f"mxfp8 8-wave a8w8 GEMM requires gfx950, got {ARCH}")

    a_codes, a_scale = gemm_common_utils.per_1x32_f8_quant(torch.randn(256, 256, device="cuda"))
    b_codes, b_scale = gemm_common_utils.per_1x32_f8_quant(torch.randn(256, 256, device="cuda"))
    with pytest.raises(NotImplementedError, match="NT only"):
        gemm_mxfp8(a_codes, a_scale, b_codes, b_scale, trans_b=False)
