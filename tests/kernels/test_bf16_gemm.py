#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""BF16 GEMM correctness + perf harness.

Computes ``C = A @ B_T.T`` with bf16 A/B inputs, f32 output, and a k-major
(row-major, K contiguous) layout for both operands. Kernel implementation
lives in ``kernels/bf16_gemm.py``.
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

from kernels.bf16_gemm import compile_bf16_gemm  # noqa: E402
from tests.test_common import run_perftest, verify_output  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


DEFAULT_BENCH_ITERS = 10
DEFAULT_BENCH_WARMUP = 2


def _run_torch(a, b_t, dtype=torch.float32):
    """Reference GEMM: C = A @ B_T.T accumulated in f32."""
    c = torch.mm(a.to(torch.float32), b_t.to(torch.float32).T)
    return c.to(dtype)


def _bench_bf16_gemm(
    M: int,
    N: int,
    K: int,
    *,
    num_warmups: int = DEFAULT_BENCH_WARMUP,
    num_iters: int = DEFAULT_BENCH_ITERS,
):
    """Run + verify a single (M, N, K) configuration. Returns TFLOPS."""

    device = torch.device("cuda")

    # k-major (row-major, K contiguous) bf16 operands.
    a = torch.empty(M, K, device=device, dtype=torch.bfloat16).uniform_(-1, 1)
    b_t = torch.empty(N, K, device=device, dtype=torch.bfloat16).uniform_(-1, 1)
    c_out = torch.zeros((M, N), device=device, dtype=torch.float32)

    a = a.contiguous()
    b_t = b_t.contiguous()

    c_ref = _run_torch(a, b_t)

    launch_fn = compile_bf16_gemm(K=K)
    print(f"\n[bf16_gemm] M={M} N={N} K={K}")

    def _args(c, a_, b_):
        return (
            a_.contiguous().view(-1),
            b_.contiguous().view(-1),
            c.contiguous().view(-1),
            M,
            N,
            torch.cuda.current_stream(),
        )

    compiled = flyc.compile(launch_fn, *_args(c_out, a, b_t))

    def _launch(c, a_, b_):
        compiled(*_args(c, a_, b_))

    num_iters = max(2, int(num_iters))
    _, us = run_perftest(
        _launch,
        c_out,
        a,
        b_t,
        num_iters=num_iters,
        num_warmup=num_warmups,
    )
    torch.cuda.synchronize()

    assert verify_output(c_out, c_ref, rtol=0.1, atol=0.1)

    flops = 2 * M * N * K
    bytes_moved = (M * K * 2) + (N * K * 2) + (M * N * 4)
    tflops = flops / (us / 1e6) / 1e12
    tbps = bytes_moved / 1e12 / (us / 1e6)
    print(f"[flyc] Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s")

    return tflops


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BF16 GEMM benchmark")
    parser.add_argument("-M", type=int, default=4096)
    parser.add_argument("-N", type=int, default=4096)
    parser.add_argument("-K", type=int, default=4096)
    parser.add_argument("--num_iters", type=int, default=DEFAULT_BENCH_ITERS)
    parser.add_argument("--num_warmups", type=int, default=DEFAULT_BENCH_WARMUP)
    args = parser.parse_args()

    torch.set_default_device("cuda")

    try:
        _bench_bf16_gemm(
            M=args.M,
            N=args.N,
            K=args.K,
            num_warmups=args.num_warmups,
            num_iters=args.num_iters,
        )
    except pytest.skip.Exception as e:
        print(f"Skipped: {e}")
