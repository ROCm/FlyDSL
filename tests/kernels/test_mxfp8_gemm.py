#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MXFP8 (a8w8) dense GEMM correctness harness (gfx950).

Kernel: ``compile_mxfp8_gemm_8w`` in ``kernels/gemm/fp8_gemm_8wave.py``, driven
through the ``gemm_mxfp8_flydsl_kernel`` host entry (which fuses the raw-E8M0 ->
broadcast-int32 scale preshuffle into the GEMM launch).

C[M,N] = A[M,K] @ B[N,K]^T with both operands per-1x32 E8M0 block-scaled fp8
(E4M3), bf16/fp16 output. Only correctness is asserted; achieved TFLOPS are
printed for information (``pytest -s``) but never gate the suite.

Shapes deliberately include the ragged cases the kernel guards for:
  * M not a multiple of 64 -- the A-scale workspace is sized ``cdiv(M, 64)`` and
    the preshuffle masks the partial group's OOB rows to 0.
  * N not a multiple of 256 -- the combined-B scale groups are sized
    ``cdiv(N, 256) * 4`` and the epilogue clamps OOB columns.
E5M2 operands (cbsz/blgp=1) are not covered: the shared quant helper only emits
E4M3 codes.
"""

import os
import sys

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

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


def _cos(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return (torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)).item()


def _dequant(codes, scale):
    return U.fp8_e4m3_to_f32(codes.view(torch.uint8)) * U.e8m0_to_f32(scale.repeat_interleave(32, dim=1))


def _report_perf(M, N, K, out_dtype, cos, run):
    """Print achieved TFLOPS alongside the cos (visible under ``pytest -s``).

    Informational only -- nothing is asserted, so a slow or busy machine never
    fails the suite. Only the ``large_shape`` cases are compute-bound; the small
    and ragged ones are launch-overhead dominated (all land around 20 us
    regardless of size) and their TFLOPS say nothing about kernel throughput."""
    us = bench_kernel_us(run, warmup=5, iters=20, flush_l2=False)
    tflops = (2 * M * N * K) / (us / 1e6) / 1e12
    print(
        f"\n[mxfp8 {M}x{N}x{K} {str(out_dtype).removeprefix('torch.')}] {tflops:7.1f} TFLOPS ({us:7.1f} us), cos={cos:.6f}"
    )


@pytest.mark.parametrize(
    "M, N, K, out_dtype",
    [
        pytest.param(256, 256, 256, torch.bfloat16, id="256x256x256-bf16"),
        pytest.param(1024, 1024, 512, torch.bfloat16, id="1024x1024x512-bf16"),
        pytest.param(100, 256, 256, torch.bfloat16, id="ragged_m-100x256x256"),
        pytest.param(256, 300, 256, torch.bfloat16, id="ragged_n-256x300x256"),
        pytest.param(512, 512, 256, torch.float16, id="512x512x256-fp16"),
        # Compute-bound shapes: the only ones whose printed TFLOPS mean anything.
        # large_shape -> skipped by the default suite, run with RUN_TESTS_FULL=1
        # (or `pytest -m large_shape -s` to just read the numbers).
        pytest.param(4096, 4096, 8192, torch.bfloat16, marks=pytest.mark.large_shape, id="4096x4096x8192-bf16"),
        pytest.param(8192, 8192, 8192, torch.bfloat16, marks=pytest.mark.large_shape, id="8192x8192x8192-bf16"),
    ],
)
def test_mxfp8_gemm_a8w8(M, N, K, out_dtype):
    if ARCH != "gfx950":
        pytest.skip(f"mxfp8 8-wave a8w8 GEMM requires gfx950, got {ARCH}")

    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(N, K, device="cuda")
    a_codes, a_scale = U.per_1x32_f8_quant(a)
    b_codes, b_scale = U.per_1x32_f8_quant(b)
    c_ref = torch.mm(_dequant(a_codes, a_scale), _dequant(b_codes, b_scale).T).to(torch.float32)

    out = gemm_mxfp8_flydsl_kernel(a_codes, a_scale, b_codes, b_scale, out_dtype=out_dtype)
    torch.cuda.synchronize()

    assert out.shape == (M, N)
    assert out.dtype == out_dtype
    cos = _cos(out.float(), c_ref)
    assert cos >= 0.99, f"cos={cos} below threshold for {M}x{N}x{K} ({out_dtype})"

    _report_perf(
        M,
        N,
        K,
        out_dtype,
        cos,
        lambda: gemm_mxfp8_flydsl_kernel(a_codes, a_scale, b_codes, b_scale, out_dtype=out_dtype),
    )


def test_mxfp8_gemm_rejects_tn():
    """NT only: trans_b=False must fail loudly rather than compute garbage."""
    if ARCH != "gfx950":
        pytest.skip(f"mxfp8 8-wave a8w8 GEMM requires gfx950, got {ARCH}")

    a_codes, a_scale = U.per_1x32_f8_quant(torch.randn(256, 256, device="cuda"))
    b_codes, b_scale = U.per_1x32_f8_quant(torch.randn(256, 256, device="cuda"))
    with pytest.raises(NotImplementedError, match="NT only"):
        gemm_mxfp8_flydsl_kernel(a_codes, a_scale, b_codes, b_scale, trans_b=False)
