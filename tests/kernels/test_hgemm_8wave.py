#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import argparse

import pytest
import torch

import flydsl.compiler as flyc
from flydsl.runtime.device import get_rocm_arch
from kernels.gemm.hgemm_8wave import compile_hgemm_8w, hgemm_8w_
from tests.test_common import verify_output

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

IS_GFX950 = str(get_rocm_arch()) == "gfx950"

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


@pytest.mark.skipif(not IS_GFX950, reason="8-wave half GEMM requires gfx950")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_hgemm_8wave(dtype):
    m, n, k = 384, 256, 128
    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((n, k), device="cuda", dtype=dtype)
    c = torch.empty((m, n), device="cuda", dtype=dtype)

    hgemm_8w_(c, a, b)
    torch.cuda.synchronize()

    reference = torch.mm(a.float(), b.float().T)
    assert verify_output(c.float(), reference, rtol=0.1, atol=0.1)


@pytest.mark.skipif(not IS_GFX950, reason="8-wave half GEMM requires gfx950")
@pytest.mark.parametrize(
    "num_xcds, group_size_m",
    [(1, 1), (8, 1), (1, 4), (8, 4)],
    ids=["linear", "xcd", "grouped_m", "xcd_grouped_m"],
)
def test_hgemm_8wave_workgroup_mapping(num_xcds, group_size_m):
    # Fifteen workgroups exercise both uneven XCD ranges and a short M group.
    m, n, k = 1280, 768, 128
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    hgemm_8w_(c, a, b, num_xcds=num_xcds, group_size_m=group_size_m)
    torch.cuda.synchronize()

    reference = torch.mm(a.float(), b.float().T)
    assert verify_output(c.float(), reference, rtol=0.1, atol=0.1)


def benchmark_hgemm_8wave(
    *,
    dtype: str,
    m: int,
    n: int,
    k: int,
    warmup: int,
    iters: int,
    rotating: int,
    num_xcds: int,
    group_size_m: int,
):
    """Run a prepared compiled launch; use rocprofv3 for per-dispatch time."""
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    tensor_sets = []
    for _ in range(rotating):
        a = torch.randn((m, k), device="cuda", dtype=torch_dtype)
        b = torch.randn((n, k), device="cuda", dtype=torch_dtype)
        c = torch.empty((m, n), device="cuda", dtype=torch_dtype)
        tensor_sets.append((c, a, b))

    launch = compile_hgemm_8w(
        dtype=dtype,
        n=n,
        k=k,
        num_xcds=num_xcds,
        group_size_m=group_size_m,
    )
    stream = torch.cuda.current_stream()
    c0, a0, b0 = tensor_sets[0]
    compiled = flyc.compile(launch, c0, a0, b0, m, stream)

    for i in range(warmup):
        compiled(*tensor_sets[i % rotating], m, stream)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(iters):
        compiled(*tensor_sets[i % rotating], m, stream)
    end.record()
    end.synchronize()

    mean_us = start.elapsed_time(end) * 1000 / iters
    tflops = 2 * m * n * k / mean_us / 1e6
    print(
        f"{dtype} {m}x{n}x{k}: aggregate event cadence {mean_us:.3f} us, "
        f"{tflops:.2f} sustained TFLOP/s; use rocprofv3 for per-dispatch kernel time"
    )

    c, a, b = tensor_sets[(iters - 1) % rotating]
    reference = torch.mm(a.float(), b.float().T)
    assert verify_output(c.float(), reference, rtol=0.1, atol=0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepared 8-wave BF16/FP16 GEMM benchmark")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("-M", type=int, default=4096)
    parser.add_argument("-N", type=int, default=4096)
    parser.add_argument("-K", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--rotating", type=int, default=3)
    parser.add_argument("--num_xcds", type=int, default=8)
    parser.add_argument("--group_size_m", type=int, default=4)
    args = parser.parse_args()

    benchmark_hgemm_8wave(
        dtype=args.dtype,
        m=args.M,
        n=args.N,
        k=args.K,
        warmup=args.warmup,
        iters=args.iters,
        rotating=args.rotating,
        num_xcds=args.num_xcds,
        group_size_m=args.group_size_m,
    )
