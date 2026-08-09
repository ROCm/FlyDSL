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
from kernels.gemm.fp4_gemm_4wave import compile_fp4_gemm_4w  # noqa: E402
from tests.kernels.utils import gemm_common_utils  # noqa: E402
from tests.test_common import run_perftest, verify_output  # noqa: E402

OUT_DTYPE = torch.bfloat16
ARCH = str(get_rocm_arch())

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


def _run_torch_w4(x_q, w_q, x_scale, w_scale, dtype=torch.float32):
    """Reference: dequantize fp4 + per-32 e8m0 scale, then mm."""
    x_f32 = gemm_common_utils.mxfp4_to_f32(x_q)
    w_f32 = gemm_common_utils.mxfp4_to_f32(w_q)
    x_s = gemm_common_utils.e8m0_to_f32(x_scale[: x_q.shape[0]].repeat_interleave(32, dim=1))
    w_s = gemm_common_utils.e8m0_to_f32(w_scale[: w_q.shape[0]].repeat_interleave(32, dim=1))
    return torch.mm(x_f32 * x_s, (w_f32 * w_s).T).to(dtype)


def _as_u8(t: torch.Tensor) -> torch.Tensor:
    return t if t.dtype in (torch.uint8, torch.int8) else t.view(torch.uint8)


def _bench_fp4_gemm(M, N, K, num_warmups=10, num_iters=100):
    if ARCH != "gfx950":
        pytest.skip(f"FP4 4-wave GEMM requires gfx950, got {ARCH}")
    # The kernel is hardcoded to a 256x256 block and an all-in-bounds epilogue.
    assert M % 256 == 0 and N % 256 == 0, "kernel requires M/N aligned to 256"

    device = torch.device("cuda")
    M_a = (M + 31) // 32 * 32
    N_a = (N + 31) // 32 * 32

    a_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
    b_fp32 = torch.randn(N, K, device=device, dtype=torch.float32)
    a_pad = torch.zeros(M_a, K, device=device, dtype=torch.float32)
    b_pad = torch.zeros(N_a, K, device=device, dtype=torch.float32)
    a_pad[:M] = a_fp32
    b_pad[:N] = b_fp32

    a_q, scale_a_orig, _ = gemm_common_utils.per_1x32_f4_quant(a_pad)
    a_q = a_q[:M]
    b_q, scale_b_orig, _ = gemm_common_utils.per_1x32_f4_quant(b_pad)
    b_q = b_q[:N]

    c_ref = _run_torch_w4(a_q, b_q, scale_a_orig, scale_b_orig)

    # Kernel inputs: A row-major fp4; B preshuffled (16,16); both scales preshuffled.
    b_shuffled = gemm_common_utils.shuffle_weight_w4(b_q, 16, False, False)
    scale_a = gemm_common_utils.shuffle_scale_w4(scale_a_orig, 1, False)
    scale_b = gemm_common_utils.shuffle_scale_w4(scale_b_orig, 1, False)

    c_out = torch.zeros((M, N), dtype=OUT_DTYPE, device=device)

    launch_fn = compile_fp4_gemm_4w(K=K, MN=(M, N))
    print(f"\n[fp4_gemm_4wave] M={M} N={N} K={K}")

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
    "M, N, K",
    [
        pytest.param(8192, 8192, 8192, marks=pytest.mark.large_shape, id="8192x8192x8192"),
        pytest.param(16384, 16384, 16384, marks=pytest.mark.large_shape, id="16384x16384x16384"),
    ],
)
def test_fp4_gemm_4wave(M, N, K):
    _bench_fp4_gemm(M=M, N=N, K=K)


# ---------------------------------------------------------------------------
# Steady-state benchmark vs aiter
# ---------------------------------------------------------------------------
# The correctness harness above uses run_perftest, which is right for a smoke
# check but reads a few percent low and drifts run to run. The one below is
# built for A/B-ing a single optimization instead: it costs ~1500 launches per
# shape (~30 s for the four shapes) and needs aiter installed, so it is
# deliberately NOT a pytest test -- run it via
#
#     python3 tests/kernels/test_fp4_gemm_4wave.py --vs-aiter

BENCH_PAIRS = int(os.environ.get("FP4_BENCH_PAIRS", "3"))
BENCH_ITERS = int(os.environ.get("FP4_BENCH_ITERS", "500"))
BENCH_WARMUP = int(os.environ.get("FP4_BENCH_WARMUP", "500"))
BENCH_SETS = int(os.environ.get("FP4_BENCH_SETS", "5"))


def _steady_state_us(step, base):
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    for n in range(BENCH_WARMUP):
        step(base + n)
    st.record()
    for n in range(BENCH_ITERS):
        step(base + n)
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / BENCH_ITERS * 1e3


def _make_bench_steps(aiter, M, N, K):
    from aiter.ops.shuffle import shuffle_weight

    device = torch.device("cuda")
    quant = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    stream = torch.cuda.current_stream()
    fly_args, ait_args = [], []
    c = torch.zeros(M * N, dtype=OUT_DTYPE, device=device)

    for s in range(BENCH_SETS):
        torch.manual_seed(s)
        a = torch.randn(M, K, device=device)
        b = torch.randn(N, K, device=device)

        a_q, scale_a, _ = gemm_common_utils.per_1x32_f4_quant(a)
        b_q, scale_b, _ = gemm_common_utils.per_1x32_f4_quant(b)
        flat = [
            _as_u8(t).contiguous().view(-1)
            for t in (
                a_q,
                gemm_common_utils.shuffle_weight_w4(b_q, 16, False, False),
                gemm_common_utils.shuffle_scale_w4(scale_a, 1, False),
                gemm_common_utils.shuffle_scale_w4(scale_b, 1, False),
            )
        ]
        fly_args.append((flat[0], flat[1], c, flat[2], flat[3], M, N, stream))

        xq, xs = quant(a.to(OUT_DTYPE), shuffle=True)
        wq, ws = quant(b.to(OUT_DTYPE), shuffle=True)
        ait_args.append((xq, shuffle_weight(wq, layout=(16, 16)), xs, ws))
        del a, b, a_q, b_q, wq

    compiled = flyc.compile(compile_fp4_gemm_4w(K=K, MN=(M, N)), *fly_args[0])

    def fly_step(i):
        compiled(*fly_args[i % BENCH_SETS])

    def aiter_step(i):
        x, w, xs, ws = ait_args[i % BENCH_SETS]
        aiter.gemm_a4w4(x, w, xs, ws, bpreshuffle=True)

    return fly_step, aiter_step


def bench_vs_aiter(M, N, K):
    import aiter

    assert ARCH == "gfx950", f"FP4 4-wave GEMM requires gfx950, got {ARCH}"

    fly_step, aiter_step = _make_bench_steps(aiter, M, N, K)
    fly_step(0)  # first-call costs: module load, aiter's kernel-config lookup
    aiter_step(0)
    torch.cuda.synchronize()

    flops = 2 * M * N * K
    print(
        f"\n[fp4_gemm_4wave] {M}x{N}x{K}  ({BENCH_SETS} input sets, warmup {BENCH_WARMUP}, {BENCH_PAIRS}x{BENCH_ITERS} iters)"
    )
    fly_best = ait_best = float("inf")
    for p in range(BENCH_PAIRS):
        f = _steady_state_us(fly_step, p * BENCH_ITERS)
        a = _steady_state_us(aiter_step, p * BENCH_ITERS)
        fly_best, ait_best = min(fly_best, f), min(ait_best, a)
        print(
            f"  pair {p}:  fly {f:8.1f} us {flops / (f / 1e6) / 1e12:6.0f} TFLOPS"
            f"   |  aiter {a:8.1f} us {flops / (a / 1e6) / 1e12:6.0f} TFLOPS"
        )
    print(
        f"  BEST:    fly {flops / (fly_best / 1e6) / 1e12:6.0f} TFLOPS"
        f"   |  aiter {flops / (ait_best / 1e6) / 1e12:6.0f} TFLOPS"
        f"   |  fly is {(ait_best / fly_best - 1) * 100:+.2f}%"
    )


if __name__ == "__main__":
    for shape in ((8192, 8192, 8192), (8192, 8192, 16384), (16384, 16384, 8192), (16384, 16384, 16384)):
        bench_vs_aiter(*shape)
