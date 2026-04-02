#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import os
import sys
import logging
import flydsl.compiler as flyc

import torch
import torch.nn.functional as F
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYFLYDSL_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _PYFLYDSL_SRC not in sys.path:
    sys.path.insert(0, _PYFLYDSL_SRC)

from kernels.hgemm_splitk import hgemm_splitk_, hgemm_shuffle_b
from tests.test_common import run_perftest, verify_output
from flydsl.runtime.device import get_rocm_arch

logging.basicConfig(level=logging.INFO)
ARCH = str(get_rocm_arch())

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

try:
    import aiter
    HAS_AITER = True
except Exception:
    HAS_AITER = False

DEFAULT_BENCH_ITERS = 50
DEFAULT_BENCH_WARMUP = 3


def run_torch_acc(a, b, dtype=torch.float32):
    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)
    c = torch.mm(a_f32, b_f32.T)
    return c.to(dtype)


def run_torch_bench(a, b):
    c = torch.mm(a, b.T)
    return c


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "m, n, k, TILE_M, TILE_N, TILE_K, SPLIT_K",
    [
        (32, 384, 7168, 16, 128, 128, 8),
        (4, 384, 7168, 16, 128, 128, 8),
        (65, 1024, 8192, 64, 64, 128, 8),
        (4096, 4096, 4096, 128, 256, 64, 1),
    ]
)
@pytest.mark.parametrize("test_graph", [
    pytest.param(False, id="eager"),
    pytest.param(True, id="graph"),
])
def test_mfma_flyc_preshuffle_splitk_hgemm(
    dtype,
    m, n, k,
    TILE_M, TILE_N, TILE_K, SPLIT_K,
    *,
    test_graph,
    bench_iters: int = DEFAULT_BENCH_ITERS,
    bench_warmup: int = DEFAULT_BENCH_WARMUP,
):
    global ARCH
    if not (ARCH in ["gfx950", "gfx942"]):
        pytest.skip(f"Skip hgemm test: ARCH={ARCH}")
    
    print("=" * 80)
    print(
        f"[flyc] MFMA {dtype.upper()} SplitK-HGEMM Test"
    )
    print("=" * 80)

    bench_iters = max(2, int(bench_iters))
    bench_warmup = int(bench_warmup)

    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    device = torch.device("cuda")
    a_fp32 = torch.rand(m, k, device=device, dtype=torch.float32)
    b_fp32_t = torch.rand(n, k, device=device, dtype=torch.float32)
    a_fp32.uniform_(-1, 1)
    b_fp32_t.uniform_(-1, 1)
    a_q = a_fp32.to(torch_dtype)
    b_q = b_fp32_t.to(torch_dtype)

    _, ref_us = run_perftest(
        run_torch_bench,
        a_q,
        b_q,
        num_iters=bench_iters,
        num_warmup=bench_warmup,
        testGraph=test_graph,
    )
    torch.cuda.synchronize()
    c_ref = run_torch_acc(a_q, b_q, dtype=torch.float32)
    c_out = torch.rand(m, n, device=device, dtype=torch_dtype)

    kwargs = {
        'TILE_M': TILE_M,
        'TILE_N': TILE_N,
        'TILE_K': TILE_K,
        'SPLIT_K': SPLIT_K,
    }

    b_q = hgemm_shuffle_b(b_q)
    print(f"✓ B shuffled")
    hgemm_splitk_(c_out, a_q, b_q, False, kwargs, torch.cuda.current_stream())
    print(f"✓ Kernel prepared: {kwargs}")

    def launch_kernel(c, a, b, shuffle_b, kwargs):
        hgemm_splitk_(c, a, b, shuffle_b, kwargs, torch.cuda.current_stream())

    _, us = run_perftest(
        launch_kernel,
        c_out,
        a_q,
        b_q,
        False, # shuffle_b
        kwargs,
        num_iters=bench_iters,
        num_warmup=bench_warmup,
        testGraph=test_graph,
    )
    torch.cuda.synchronize()
    assert verify_output(c_out.float(), c_ref, rtol=0.1, atol=0.1)

    bytes_moved = (m * k * 2) + (n * k * 2) + (m * n * 2)
    flops = 2 * m * n * k
    tflops = flops / (us / 1e6) / 1e12
    tbps = bytes_moved / 1e12 / (us / 1e6)
    speedup = ref_us / us
    print(f"[flyc] Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s, Torch(us): {ref_us:.1f}, Speedup: {speedup:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preshuffle SplitK HGEMM benchmark")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("-m", type=int, default=4096)
    parser.add_argument("-n", type=int, default=4096)
    parser.add_argument("-k", type=int, default=4096)
    parser.add_argument("--TILE_M", type=int, default=128)
    parser.add_argument("--TILE_N", type=int, default=256)
    parser.add_argument("--TILE_K", type=int, default=64)
    parser.add_argument("--SPLIT_K", type=int, default=1)
    parser.add_argument("--num_warmup", type=int, default=DEFAULT_BENCH_WARMUP)
    parser.add_argument("--num_iters", type=int, default=DEFAULT_BENCH_ITERS)
    parser.add_argument("--test_graph", "-tg", action="store_true", default=False)
    args = parser.parse_args()
    torch.set_default_device("cuda")
    try:
        test_mfma_flyc_preshuffle_splitk_hgemm(
            args.dtype,
            m=args.m, n=args.n, k=args.k,
            TILE_M=args.TILE_M, TILE_N=args.TILE_N, TILE_K=args.TILE_K, SPLIT_K=args.SPLIT_K,
            test_graph=bool(args.test_graph),
            bench_iters=args.num_iters,
            bench_warmup=args.num_warmup,
        )
    except pytest.skip.Exception as e:
        print(f"Skipped: {e}")
