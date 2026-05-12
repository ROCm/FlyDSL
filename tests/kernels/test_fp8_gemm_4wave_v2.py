#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FP8 4-Wave GEMM correctness + perf harness.

Kernel implementation: ``kernels/fp8_gemm_4wave_v2.py``.
Direct port of ROCm/FlyDSL PR #505 (test_fp8_gemm_4wave.py); shapes and
block sizes match the PR's benchmark configuration exactly.
"""

import logging
import os
import sys

import pytest
import torch

import flydsl.compiler as flyc

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYFLYDSL_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _PYFLYDSL_SRC not in sys.path:
    sys.path.insert(0, _PYFLYDSL_SRC)

from flydsl.runtime.device import get_rocm_arch
from kernels.fp8_gemm_4wave_v2 import compile_fp8_gemm_4wave
from tests.test_common import run_perftest, verify_output
from tests.utils import pertoken_quant

logging.basicConfig(level=logging.INFO)

FP8_DTYPE = torch.float8_e4m3fn
OUT_DTYPE = torch.bfloat16
ARCH = str(get_rocm_arch())

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


def _run_torch(a, b, scale_a, scale_b, dtype=torch.float32):
    a_f32 = a.to(torch.float32) * scale_a.view(-1, 1)
    b_f32 = b.to(torch.float32) * scale_b.view(-1, 1)
    return torch.mm(a_f32, b_f32.T).to(dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Reusable driver
# ─────────────────────────────────────────────────────────────────────────────


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

    launch_fn = compile_fp8_gemm_4wave(
        M=M,
        N=N,
        K=K,
        BLOCK_M=tile_m,
        BLOCK_N=tile_n,
        use_xcd_remap=not disable_xcd_remap,
    )
    print(
        f"\n[fp8_gemm_4wave_v2] M={M} N={N} K={K} "
        f"BLOCK_M={tile_m} BLOCK_N={tile_n} xcd_remap={not disable_xcd_remap}"
    )

    def _as_i8(t: torch.Tensor) -> torch.Tensor:
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    def _args(c, a, b, sa, sb):
        return (
            _as_i8(a).contiguous().view(-1),
            _as_i8(b).contiguous().view(-1),
            c.contiguous().view(-1),
            sa.contiguous().view(-1),
            sb.contiguous().view(-1),
            torch.cuda.current_stream(),
        )

    compiled = flyc.compile(launch_fn, *_args(c_out_raw, a_q, b_q, scale_a, scale_b))

    def _launch(c, a, b, sa, sb):
        compiled(*_args(c, a, b, sa, sb))

    num_iters = max(2, int(num_iters))
    _, us = run_perftest(
        _launch,
        c_out_raw,
        a_q,
        b_q,
        scale_a,
        scale_b,
        num_iters=num_iters,
        num_warmup=num_warmups,
    )
    torch.cuda.synchronize()

    c_out_f32 = c_out_raw.to(torch.float32)
    assert verify_output(c_out_f32, c_ref, rtol=0.1, atol=0.1)

    bytes_moved = M * K + N * K + M * N * 2 + (M + N) * 4
    flops = 2 * M * N * K
    tflops = flops / (us / 1e6) / 1e12
    tbps = bytes_moved / 1e12 / (us / 1e6)
    print(f"[flyc] Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s")

    return tflops


# ─────────────────────────────────────────────────────────────────────────────
# Pytest parametrisation (PR #505's reported shapes)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "M, N, K, tile_m, tile_n",
    [
        (512, 2112, 7168, 64, 64),
        (5120, 5120, 8320, 256, 256),
        (8192, 8192, 8192, 256, 256),
        (9728, 8192, 8320, 256, 256),
    ],
)
def test_fp8_gemm_4wave_v2(M, N, K, tile_m, tile_n):
    _bench_fp8_gemm_4wave(M=M, N=N, K=K, tile_m=tile_m, tile_n=tile_n)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FP8 4-Wave GEMM (v2) benchmark")
    parser.add_argument("-M", type=int, default=8192)
    parser.add_argument("-N", type=int, default=8192)
    parser.add_argument("-K", type=int, default=8192)
    parser.add_argument("--tile_m", type=int, default=256)
    parser.add_argument("--tile_n", type=int, default=256)
    parser.add_argument("--disable_xcd_remap", action="store_true", default=False)
    parser.add_argument("--num_iters", type=int, default=10)
    parser.add_argument("--num_warmups", type=int, default=2)
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
        )
    except pytest.skip.Exception as e:
        print(f"Skipped: {e}")
