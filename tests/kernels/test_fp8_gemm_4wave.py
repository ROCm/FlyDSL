#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FP8 4-wave GEMM correctness + perf harness.

Kernel implementation: ``kernels/fp8_gemm_4wave.py``.
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
from kernels.fp8_gemm_4wave import compile_fp8_gemm, preshuffle_a, preshuffle_b  # noqa: E402
from tests.test_common import run_perftest, verify_output  # noqa: E402
from tests.utils import pertoken_quant  # noqa: E402

FP8_DTYPE = torch.float8_e4m3fn
OUT_DTYPE = torch.bfloat16
ARCH = str(get_rocm_arch())

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


def _run_torch(a, b, scale_a, scale_b, dtype=torch.float32):
    a_f32 = a.to(torch.float32) * scale_a.view(-1, 1)
    b_f32 = b.to(torch.float32) * scale_b.view(-1, 1)
    return torch.mm(a_f32, b_f32.T).to(dtype)


def _as_i8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int8) if "float8" in str(t.dtype) else t


def _bench_torch_scaled_mm(M, N, K, a_q, b_q, scale_a, scale_b, c_ref, num_warmups, num_iters):
    """Run torch._scaled_mm with the same inputs / perf harness as flydsl."""
    b_t = b_q.t()
    sa_v = scale_a.view(M, 1).to(torch.float32).contiguous()
    sb_v = scale_b.view(1, N).to(torch.float32).contiguous()
    out = torch.empty((M, N), dtype=OUT_DTYPE, device="cuda")

    def _launch(_out):
        torch._scaled_mm(a_q, b_t, scale_a=sa_v, scale_b=sb_v, out_dtype=OUT_DTYPE, out=_out)

    _, us = run_perftest(_launch, out, num_iters=num_iters, num_warmup=num_warmups)
    torch.cuda.synchronize()
    ok = verify_output(out.to(torch.float32), c_ref, rtol=0.1, atol=0.1)
    return us, ok


def _bench_fp8_gemm_4wave(
    M: int,
    N: int,
    K: int,
    *,
    tile_m: int,
    tile_n: int,
    disable_xcd_remap: bool = False,
    num_warmups: int = 2,
    num_iters: int = 10,
    vs_torch: bool = False,
    a_preshuffled: bool = False,
    b_preshuffled: bool = False,
):
    """Run + verify a single (M, N, K, tile) configuration. Returns TFLOPS."""
    if "gfx95" not in ARCH:
        pytest.skip("FP8 4-wave GEMM requires CDNA4 (gfx95*)")

    device = torch.device("cuda")
    a_fp32 = torch.rand(M, K, device=device, dtype=torch.float32)
    b_fp32_t = torch.rand(N, K, device=device, dtype=torch.float32)
    c_out_raw = torch.zeros((M, N), dtype=OUT_DTYPE, device=device)

    a_q, scale_a = pertoken_quant(a_fp32, quant_dtype=FP8_DTYPE)
    b_q, scale_b = pertoken_quant(b_fp32_t, quant_dtype=FP8_DTYPE)

    a_q = a_q.contiguous()
    b_q = b_q.contiguous()
    scale_a = scale_a.squeeze().contiguous()
    scale_b = scale_b.squeeze().contiguous()

    c_ref = _run_torch(a_q, b_q, scale_a, scale_b)

    a_kernel = preshuffle_a(a_q) if a_preshuffled else a_q
    b_kernel = preshuffle_b(b_q) if b_preshuffled else b_q

    launch_fn = compile_fp8_gemm(
        M=M,
        N=N,
        K=K,
        BLOCK_M=tile_m,
        BLOCK_N=tile_n,
        use_xcd_remap=not disable_xcd_remap,
        a_preshuffled=a_preshuffled,
        b_preshuffled=b_preshuffled,
    )
    print(
        f"\n[fp8_gemm_4wave] M={M} N={N} K={K} "
        f"BLOCK_M={tile_m} BLOCK_N={tile_n} xcd_remap={not disable_xcd_remap} "
        f"preshuffle_a={a_preshuffled} preshuffle_b={b_preshuffled}"
    )

    def _args(c, a, b, sa, sb):
        return (
            _as_i8(a).contiguous().view(-1),
            _as_i8(b).contiguous().view(-1),
            c.contiguous().view(-1),
            sa.contiguous().view(-1),
            sb.contiguous().view(-1),
            torch.cuda.current_stream(),
        )

    compiled = flyc.compile(launch_fn, *_args(c_out_raw, a_kernel, b_kernel, scale_a, scale_b))

    def _launch(c, a, b, sa, sb):
        compiled(*_args(c, a, b, sa, sb))

    num_iters = max(2, int(num_iters))
    _, us = run_perftest(
        _launch,
        c_out_raw,
        a_kernel,
        b_kernel,
        scale_a,
        scale_b,
        num_iters=num_iters,
        num_warmup=num_warmups,
    )
    torch.cuda.synchronize()

    c_out_f32 = c_out_raw.to(torch.float32)
    assert verify_output(c_out_f32, c_ref, rtol=0.1, atol=0.1)

    flops = 2 * M * N * K
    bytes_moved = M * K + N * K + M * N * 2 + (M + N) * 4
    tflops = flops / (us / 1e6) / 1e12
    tbps = bytes_moved / 1e12 / (us / 1e6)
    print(f"[flyc] Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s")

    if vs_torch:
        try:
            us_t, ok_t = _bench_torch_scaled_mm(
                M,
                N,
                K,
                a_q,
                b_q,
                scale_a,
                scale_b,
                c_ref,
                num_warmups,
                num_iters,
            )
            tflops_t = flops / (us_t / 1e6) / 1e12
            print(
                f"[torch._scaled_mm] Throughput: {us_t:.1f} us, "
                f"{tflops_t:.2f} TFLOPS, flydsl/torch={us_t / us * 100:.1f}%, ok={ok_t}"
            )
        except Exception as e:
            print(f"[torch._scaled_mm] failed: {e}")

    return tflops


@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n",
    [
        pytest.param(512, 2112, 7168, 64, 64, id="512x2112x7168"),
        pytest.param(5120, 5120, 8320, 256, 256, id="5120x5120x8320"),
        pytest.param(8192, 8192, 8192, 256, 256, marks=pytest.mark.large_shape, id="8192x8192x8192"),
        pytest.param(9728, 8192, 8320, 256, 256, marks=pytest.mark.large_shape, id="9728x8192x8320"),
    ],
)
def test_fp8_gemm_4wave(M, N, K, tile_m, tile_n):
    _bench_fp8_gemm_4wave(M=M, N=N, K=K, tile_m=tile_m, tile_n=tile_n)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FP8 4-wave GEMM benchmark")
    parser.add_argument("-M", type=int, default=8192)
    parser.add_argument("-N", type=int, default=8192)
    parser.add_argument("-K", type=int, default=8192)
    parser.add_argument("--tile_m", type=int, default=256)
    parser.add_argument("--tile_n", type=int, default=256)
    parser.add_argument("--disable_xcd_remap", action="store_true", default=False)
    parser.add_argument(
        "--num_iters",
        type=int,
        default=100,
        help="Number of timed iterations. Use 100+ for stable steady-state perf",
    )
    parser.add_argument(
        "--num_warmups",
        type=int,
        default=10,
        help="Warmup iterations (>=10 needed to reach steady state on large shapes)",
    )
    parser.add_argument(
        "--vs_torch",
        action="store_true",
        default=False,
        help="Also run torch._scaled_mm with the same input + harness for perf comparison.",
    )
    parser.add_argument(
        "--preshuffle_a",
        action="store_true",
        default=False,
        help="Preshuffle A into K-major slabs (each K atom = M*128 byte contig).",
    )
    parser.add_argument(
        "--preshuffle_b",
        action="store_true",
        default=False,
        help="Preshuffle B into K-major slabs (each K atom = N*128 byte contig).",
    )
    args = parser.parse_args()

    torch.set_default_device("cuda")

    try:
        _bench_fp8_gemm_4wave(
            M=args.M,
            N=args.N,
            K=args.K,
            tile_m=args.tile_m,
            tile_n=args.tile_n,
            disable_xcd_remap=args.disable_xcd_remap,
            num_warmups=args.num_warmups,
            num_iters=args.num_iters,
            vs_torch=args.vs_torch,
            a_preshuffled=args.preshuffle_a,
            b_preshuffled=args.preshuffle_b,
        )
    except pytest.skip.Exception as e:
        print(f"Skipped: {e}")
