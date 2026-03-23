#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import os
import sys
import logging

import torch
import torch.nn.functional as F
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYFLYDSL_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _PYFLYDSL_SRC not in sys.path:
    sys.path.insert(0, _PYFLYDSL_SRC)

from kernels.preshuffle_splitk_hgemm import compile_hgemm_kernel, HGEMMOut, hgemm_spk_shuffle_b
from tests.test_common import run_perftest, verify_output

logging.basicConfig(level=logging.INFO)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

try:
    import aiter
    HAS_AITER = True
except Exception:
    HAS_AITER = False

DEFAULT_BENCH_ITERS = 20
DEFAULT_BENCH_WARMUP = 3
DEFAULT_RUN_AITER_BENCH = True


def run_torch(a, b, dtype=torch.float32):
    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)
    c = torch.mm(a_f32, b_f32.T)
    return c.to(dtype)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "m, n, k, BLOCK_K, WARP_M_STEPS, WARP_N_STEPS, PACK_N, SPLIT_K",
    [
        (32, 7168, 2048, 128, 2, 2, 2, 4),
        (4096, 4096, 4096, 64, 8, 4, 1, 1),
    ]
)
@pytest.mark.parametrize("test_graph", [
    pytest.param(False, id="eager"),
    pytest.param(True, id="graph"),
])
def test_mfma_flyc_preshuffle_splitk_hgemm(
    dtype,
    m, n, k,
    BLOCK_K, WARP_M_STEPS, WARP_N_STEPS, PACK_N, SPLIT_K,
    *,
    test_graph,
    bench_iters: int = DEFAULT_BENCH_ITERS,
    bench_warmup: int = DEFAULT_BENCH_WARMUP,
    run_aiter_bench: bool = DEFAULT_RUN_AITER_BENCH,
):
    print("=" * 80)
    print(
        f"[flyc] MFMA {dtype.upper()} SplitK-HGEMM Test"
    )
    print("=" * 80)

    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16

    device = torch.device("cuda")
    a_fp32 = torch.rand(m, k, device=device, dtype=torch.float32)
    b_fp32_t = torch.rand(n, k, device=device, dtype=torch.float32)
    a_q = a_fp32.to(torch_dtype)
    b_q = b_fp32_t.to(torch_dtype)

    c_ref = run_torch(a_q, b_q, dtype=torch.float32)
    c_out = HGEMMOut(m, n, torch_dtype, 'cuda')

    kwargs = {
        'BLOCK_K': BLOCK_K,
        'WARP_M_STEPS': WARP_M_STEPS,
        'WARP_N_STEPS': WARP_N_STEPS,
        'PACK_N': PACK_N,
        'SPLIT_K': SPLIT_K,
    }
    if torch_dtype == torch.half:
        launch_kernel = compile_hgemm_kernel('f16', m, n, k, **kwargs)
    elif torch_dtype == torch.bfloat16:
        launch_kernel = compile_hgemm_kernel('bf16', m, n, k, **kwargs)
    else:
        raise NotImplementedError()
    print(f"✓ Kernel prepared")

    b_input = hgemm_spk_shuffle_b(b_q, pack_n=kwargs['PACK_N'])
    print(f"✓ B shuffled")

    bench_iters = max(2, int(bench_iters))
    bench_warmup = int(bench_warmup)
    _, us = run_perftest(
        launch_kernel,
        c_out.get(),
        a_q,
        b_input,
        c_out.get_next(),
        num_iters=bench_iters,
        num_warmup=bench_warmup,
        testGraph=test_graph,
    )
    torch.cuda.synchronize()
    
    c_out.switch()
    launch_kernel(c_out.get(), a_q, b_input, c_out.get_next())
    c_out_ = c_out.get().to(torch.float32)
    assert verify_output(c_out_, c_ref, rtol=0.1, atol=0.1)

    bytes_moved = (m * k * 2) + (n * k * 2) + (m * n * 2)
    flops = 2 * m * n * k
    tflops = flops / (us / 1e6) / 1e12
    tbps = bytes_moved / 1e12 / (us / 1e6)
    print(f"[flyc] Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preshuffle SplitK HGEMM benchmark")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("-m", type=int, default=16)
    parser.add_argument("-n", type=int, default=10240)
    parser.add_argument("-k", type=int, default=8192)
    parser.add_argument("--BLOCK_K", type=int, default=64)
    parser.add_argument("--WARP_M_STEPS", type=int, default=2)
    parser.add_argument("--WARP_N_STEPS", type=int, default=2)
    parser.add_argument("--PACK_N", type=int, default=1)
    parser.add_argument("--SPLIT_K", type=int, default=1)
    parser.add_argument("--num_warmup", type=int, default=DEFAULT_BENCH_WARMUP)
    parser.add_argument("--run_aiter_bench", action="store_true", default=DEFAULT_RUN_AITER_BENCH)
    parser.add_argument("--no_aiter_bench", action="store_false", dest="run_aiter_bench")
    parser.add_argument("--test_graph", "-tg", action="store_true", default=False)
    args = parser.parse_args()
    torch.set_default_device("cuda")
    try:
        test_mfma_a8_flyc_preshuffle(
            args.dtype,
            m=args.m, n=args.n, k=args.k,
            BLOCK_K=args.BLOCK_K, WARP_M_STEPS=args.WARP_M_STEPS, WARP_N_STEPS=args.WARP_N_STEPS,
            PACK_N=args.PACK_N, SPLIT_K=args.SPLIT_K,
            test_graph=bool(args.test_graph),
            bench_iters=args.num_iters,
            bench_warmup=args.num_warmup,
            run_aiter_bench=bool(args.run_aiter_bench),
        )
    except pytest.skip.Exception as e:
        print(f"Skipped: {e}")
