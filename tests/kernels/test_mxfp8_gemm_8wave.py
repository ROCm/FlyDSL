#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MXFP8 8-wave GEMM correctness + perf harness.

Kernel implementation: ``kernels/gemm/mxfp8_gemm_8wave.py`` (gfx950 only).

``C[M,N] = A[M,K] @ B[N,K]^T`` with per-1x32 E8M0 block scales on both operands,
bf16 output. A and B are row-major fp8 e4m3; both scales are ``shuffle_scale_w4``
permuted (the kernel reads that layout directly, see its module docstring).
"""

import os
import sys

import pytest
import torch

import flydsl.compiler as flyc

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch  # noqa: E402
from kernels.gemm.fp8_gemm_utils import preshuffle_b  # noqa: E402
from kernels.gemm.mxfp8_gemm_8wave import compile_mxfp8_gemm_8w  # noqa: E402
from tests.kernels.utils import gemm_common_utils  # noqa: E402
from tests.test_common import run_perftest, verify_output  # noqa: E402

OUT_DTYPE = torch.bfloat16
ARCH = str(get_rocm_arch())

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


def _ref_mxfp8(a_q, b_q, a_scale, b_scale):
    """Dequantize fp8 + per-1x32 E8M0 scale, then mm in fp32."""
    a_f32 = a_q.to(torch.float32) * gemm_common_utils.e8m0_to_f32(a_scale).repeat_interleave(32, dim=1)
    b_f32 = b_q.to(torch.float32) * gemm_common_utils.e8m0_to_f32(b_scale).repeat_interleave(32, dim=1)
    return torch.mm(a_f32, b_f32.T)


def _run_mxfp8_gemm(
    M,
    N,
    K,
    *,
    b_preshuffled=False,
    xcd_swizzle=0,
    num_warmups=10,
    num_iters=100,
    report=False,
):
    if ARCH != "gfx950":
        pytest.skip(f"MXFP8 8-wave GEMM requires gfx950, got {ARCH}")
    assert M % 256 == 0 and N % 256 == 0, "kernel requires M/N aligned to 256"

    device = torch.device("cuda")
    torch.manual_seed(0)
    a_f32 = torch.randn(M, K, device=device, dtype=torch.float32)
    b_f32 = torch.randn(N, K, device=device, dtype=torch.float32)

    a_q, a_scale = gemm_common_utils.per_1x32_f8_quant(a_f32)
    b_q, b_scale = gemm_common_utils.per_1x32_f8_quant(b_f32)
    c_ref = _ref_mxfp8(a_q, b_q, a_scale, b_scale)
    c_out = torch.zeros((M, N), dtype=OUT_DTYPE, device=device)

    # preshuffle_b only reorders K inside a 16x64 brick, so b_scale is unchanged.
    b_in = preshuffle_b(b_q.view(torch.int8)).view(torch.float8_e4m3fn) if b_preshuffled else b_q
    a_scale_in = gemm_common_utils.shuffle_scale_w4(a_scale, 1, False)
    b_scale_in = gemm_common_utils.shuffle_scale_w4(b_scale, 1, False)
    launch_fn = compile_mxfp8_gemm_8w(K=K, b_preshuffled=b_preshuffled, xcd_swizzle=xcd_swizzle)

    def _args(c, a, b, sa, sb):
        return (
            a.contiguous().view(torch.int8).view(-1),
            b.contiguous().view(torch.int8).view(-1),
            c.contiguous().view(-1),
            sa.contiguous().view(-1),
            sb.contiguous().view(-1),
            M,
            N,
            torch.cuda.current_stream(),
        )

    compiled = flyc.compile(launch_fn, *_args(c_out, a_q, b_in, a_scale_in, b_scale_in))

    def _launch(c, a, b, sa, sb):
        compiled(*_args(c, a, b, sa, sb))

    _, us = run_perftest(_launch, c_out, a_q, b_in, a_scale_in, b_scale_in, num_iters=num_iters, num_warmup=num_warmups)
    torch.cuda.synchronize()

    if report:
        tflops = 2.0 * M * N * K / (us * 1e-6) / 1e12
        tag = f"B={'pre' if b_preshuffled else 'raw'}"
        print(f"\n[mxfp8_gemm_8wave] M={M} N={N} K={K} {tag}: {us:.1f} us, {tflops:.1f} TFLOPS")

    assert verify_output(c_out.to(torch.float32), c_ref, rtol=0.05, atol=0.05)


@pytest.mark.parametrize("b_preshuffled", [False, True])
@pytest.mark.parametrize(
    "M, N, K, xcd_swizzle",
    [
        (256, 256, 512, 0),
        (256, 256, 768, 0),
        (1024, 1024, 2048, 0),
        # xcd_swizzle only reaches the remap itself at 8192^3: xcd_remap_pid falls
        # back to row-major below 4*32*num_xcds workgroups, so the 1024 case covers
        # the fallback select and the 8192 one covers the remap.
        (1024, 1024, 2048, 4),
        (2048, 2048, 4096, 0),
        pytest.param(8192, 8192, 8192, 0, marks=pytest.mark.large_shape),
        pytest.param(8192, 8192, 8192, 4, marks=pytest.mark.large_shape),
    ],
)
def test_mxfp8_gemm_8wave(M, N, K, xcd_swizzle, b_preshuffled):
    _run_mxfp8_gemm(M, N, K, b_preshuffled=b_preshuffled, xcd_swizzle=xcd_swizzle, report=True)
