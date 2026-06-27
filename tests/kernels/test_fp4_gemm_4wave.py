#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FP4 (MXFP4) 4-wave GEMM correctness + perf harness.

Kernel implementation: ``kernels/fp4_gemm_4wave.py`` (gfx950 only).

C[M,N] = A[M,K] @ B[N,K]^T with per-1x32 E8M0 block scales on both A and B,
bf16 output. A is row-major fp4 (uint8, 2 fp4/byte); B is ``shuffle_weight_w4``
preshuffled; both scales are ``shuffle_scale_w4`` preshuffled.
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
from kernels.fp4_gemm_4wave import compile_fp4_gemm_4w  # noqa: E402
from tests.kernels.utils import fp4_utils  # noqa: E402
from tests.test_common import run_perftest, verify_output  # noqa: E402

OUT_DTYPE = torch.bfloat16
ARCH = str(get_rocm_arch())

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


def _run_torch_w4(x_q, w_q, x_scale, w_scale, dtype=torch.float32):
    """Reference: dequantize fp4 + per-32 e8m0 scale, then mm."""
    x_f32 = fp4_utils.mxfp4_to_f32(x_q)
    w_f32 = fp4_utils.mxfp4_to_f32(w_q)
    x_s = fp4_utils.e8m0_to_f32(x_scale[: x_q.shape[0]].repeat_interleave(32, dim=1))
    w_s = fp4_utils.e8m0_to_f32(w_scale[: w_q.shape[0]].repeat_interleave(32, dim=1))
    return torch.mm(x_f32 * x_s, (w_f32 * w_s).T).to(dtype)


def _as_u8(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype in (torch.uint8, torch.int8) else t.view(torch.uint8)


def _bench_fp4_gemm(M, N, K, tile_m=256, tile_n=256, num_warmups=10, num_iters=100):
    if ARCH != "gfx950":
        pytest.skip(f"FP4 4-wave GEMM requires gfx950, got {ARCH}")

    device = torch.device("cuda")
    M_a = (M + 31) // 32 * 32
    N_a = (N + 31) // 32 * 32

    a_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
    b_fp32 = torch.randn(N, K, device=device, dtype=torch.float32)
    a_pad = torch.zeros(M_a, K, device=device, dtype=torch.float32)
    b_pad = torch.zeros(N_a, K, device=device, dtype=torch.float32)
    a_pad[:M] = a_fp32
    b_pad[:N] = b_fp32

    a_q, scale_a_orig, _ = fp4_utils.per_1x32_f4_quant(a_pad)
    a_q = a_q[:M]
    b_q, scale_b_orig, _ = fp4_utils.per_1x32_f4_quant(b_pad)
    b_q = b_q[:N]

    c_ref = _run_torch_w4(a_q, b_q, scale_a_orig, scale_b_orig)

    # Kernel inputs: A row-major fp4; B preshuffled (16,16); both scales preshuffled.
    b_shuffled = fp4_utils.shuffle_weight_w4(b_q, 16, False, False)
    scale_a = fp4_utils.shuffle_scale_w4(scale_a_orig, 1, False)
    scale_b = fp4_utils.shuffle_scale_w4(scale_b_orig, 1, False)

    c_out = torch.zeros((M, N), dtype=OUT_DTYPE, device=device)

    launch_fn = compile_fp4_gemm_4w(
        K=K, BLOCK_M=tile_m, BLOCK_N=tile_n, mn_aligned=(M % tile_m == 0 and N % tile_n == 0)
    )
    print(f"\n[fp4_gemm_4wave] M={M} N={N} K={K} BLOCK_M={tile_m} BLOCK_N={tile_n}")

    def _args(c, a, b, sa, sb):
        # kernel signature: (A, B_T, C, A_scale, B_scale, c_m, c_n, stream)
        return (
            _as_u8(a).contiguous().view(-1),
            _as_u8(b).contiguous().view(-1),
            c.contiguous().view(-1),
            _as_u8(sa).contiguous().view(-1),
            _as_u8(sb).contiguous().view(-1),
            M,
            N,
            torch.cuda.current_stream(),
        )

    compiled = flyc.compile(launch_fn, *_args(c_out, a_q, b_shuffled, scale_a, scale_b))

    def _launch(c, a, b, sa, sb):
        compiled(*_args(c, a, b, sa, sb))

    num_iters = max(2, int(num_iters))
    _, us = run_perftest(
        _launch,
        c_out,
        a_q,
        b_shuffled,
        scale_a,
        scale_b,
        num_iters=num_iters,
        num_warmup=num_warmups,
    )
    torch.cuda.synchronize()

    assert verify_output(c_out.to(torch.float32), c_ref, rtol=0.1, atol=0.1)

    flops = 2 * M * N * K
    size_a = (M * K) // 2
    size_b = (N * K) // 2
    bytes_moved = size_a + size_b + M * N * 2 + (M + N) * (K // 32)
    tflops = flops / (us / 1e6) / 1e12
    tbps = bytes_moved / 1e12 / (us / 1e6)
    print(f"[flyc] Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s")
    return tflops


@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n",
    [
        pytest.param(8192, 8192, 8192, 256, 256, marks=pytest.mark.large_shape, id="8192x8192x8192"),
        pytest.param(16384, 16384, 16384, 256, 256, marks=pytest.mark.large_shape, id="16384x16384x16384"),
    ],
)
def test_fp4_gemm_4wave(M, N, K, tile_m, tile_n):
    _bench_fp4_gemm(M=M, N=N, K=K, tile_m=tile_m, tile_n=tile_n)


if __name__ == "__main__":
    _bench_fp4_gemm(8192, 8192, 8192)
    _bench_fp4_gemm(16384, 16384, 16384)
