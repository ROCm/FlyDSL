#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
MoE Reduction Kernel Test

Reduces [tokens, topk, model_dim] along the topk dimension.
Designed for MoE stage-2 shapes where topk is small
and model_dim is large and aligned (e.g. 5120, 7168).

MoeReduce(x) = sum(x, dim=1)
"""

import argparse
import logging
import os
import sys
from typing import Tuple

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYTHON_CANDIDATES = [
    os.path.join(_REPO_ROOT, "build", "python_packages"),
    _REPO_ROOT,
]
for _p in reversed(_PYTHON_CANDIDATES):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from kernels.moe_gemm_2stage import compile_moe_reduction
from tests.test_common import verify_output, run_perftest

logging.basicConfig(level=logging.INFO)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


def run_reduce_test(
    tokens: int,
    topk: int,
    model_dim: int,
    dtype_str: str = "f16",
    use_mask: bool = False,
    num_iters: int = 20,
    num_warmup: int = 5,
    compare_torch: bool = True,
):
    """Run reduce kernel test: correctness + performance."""
    dtype_map = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}
    dtype = dtype_map[dtype_str]
    device = torch.device("cuda")

    print(
        f"=== MoE Reduce Kernel: tokens={tokens}, topk={topk}, model_dim={model_dim}, "
        f"use_mask={use_mask}, dtype={dtype_str} ==="
    )

    reduce_exe = compile_moe_reduction(
        topk=topk, model_dim=model_dim, dtype_str=dtype_str, use_mask=use_mask
    )

    X = torch.randn(tokens, topk, model_dim, device=device, dtype=dtype)
    Y = torch.empty(tokens, model_dim, device=device, dtype=dtype)

    if use_mask:
        valid_mask = torch.randint(0, 2, (tokens, topk), device=device, dtype=torch.uint8)
    else:
        valid_mask = torch.empty((0, topk), device=device, dtype=torch.uint8)

    stream = torch.cuda.current_stream()

    def launch(y, x, mask):
        reduce_exe(x, y, mask, tokens, stream)

    _, us = run_perftest(
        launch, Y, X, valid_mask,
        num_iters=num_iters, num_warmup=num_warmup,
    )
    torch.cuda.synchronize()

    # Correctness
    if use_mask:
        X_ref = X * valid_mask.to(torch.bool).unsqueeze(-1)
    else:
        X_ref = X
    Y_ref = torch.sum(X_ref, dim=1)
    assert verify_output(Y.float(), Y_ref.float(), rtol=1e-2, atol=1e-2, msg="[reduce kernel]")

    # Bandwidth
    elem_bytes = X.element_size()
    bytes_moved = (tokens * topk * model_dim + tokens * model_dim) * elem_bytes
    bw_gb_s = bytes_moved / 1e9 / (us / 1e6)
    print(f"[FlyDSL reduce] {us:.1f} us, Bandwidth: {bw_gb_s:.2f} GB/s")

    if compare_torch:
        def launch_torch(y, x_ref):
            torch.sum(x_ref, dim=1, out=y)

        Y_torch = torch.empty_like(Y)
        _, us_torch = run_perftest(
            launch_torch, Y_torch, X_ref,
            num_iters=num_iters, num_warmup=num_warmup,
        )
        torch.cuda.synchronize()
        bw_torch = bytes_moved / 1e9 / (us_torch / 1e6)
        speedup = us_torch / us if us > 0 else 0
        print(f"[torch.sum]     {us_torch:.1f} us, Bandwidth: {bw_torch:.2f} GB/s")
        print(f"[speedup]       {speedup:.2f}x")


@pytest.mark.parametrize(
    "tokens, topk, model_dim, use_mask",
    [
        pytest.param(32769, 8, 7168, False, id="DS-TP8-prefill-L", marks=pytest.mark.large_shape),
        pytest.param(1, 8, 7168, False, id="DS-TP8-decode-S"),
        pytest.param(5, 8, 7168, False, id="DS-TP8-decode-M"),
        pytest.param(65, 8, 7168, False, id="DS-TP8-decode-L"),
        pytest.param(16384, 6, 5120, False, id="EP-K6-prefill", marks=pytest.mark.large_shape),
        pytest.param(1, 6, 5120, False, id="EP-K6-decode-S"),
        pytest.param(5, 6, 5120, False, id="EP-K6-decode-M"),
        pytest.param(65, 6, 5120, False, id="EP-K6-decode-L"),
        pytest.param(129, 8, 7168, True, id="DS-TP8-masked"),
        pytest.param(129, 6, 5120, True, id="EP-K6-masked"),
    ],
)
def test_moe_reduce_kernel(tokens: int, topk: int, model_dim: int, use_mask: bool):
    """Test reduce kernel correctness and performance vs torch.sum."""
    run_reduce_test(tokens=tokens, topk=topk, model_dim=model_dim, use_mask=use_mask)


if __name__ == "__main__":
    torch.set_default_device("cuda")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "MoE Reduction Kernel — correctness & performance test.\n"
            "\n"
            "Reduces [tokens, topk, model_dim] along the topk dimension.\n"
            "Designed for MoE stage-2 shapes where topk is small \n"
            "and model_dim is large and aligned (e.g. 5120, 7168)."
        ),
    )
    parser.add_argument("--tokens", "-t", type=int, default=16384)
    parser.add_argument("--topk", "-k", type=int, default=8)
    parser.add_argument("--model_dim", "-d", type=int, default=7168)
    parser.add_argument("--dtype", type=str, default="f16", choices=["f16", "bf16", "f32"])
    parser.add_argument("--use_mask", action="store_true", default=False)
    parser.add_argument("--num_iters", type=int, default=20)
    parser.add_argument("--num_warmup", type=int, default=5)
    parser.add_argument("--no_compare_torch", action="store_true", default=False)

    args = parser.parse_args()
    run_reduce_test(
        tokens=args.tokens,
        topk=args.topk,
        model_dim=args.model_dim,
        dtype_str=args.dtype,
        use_mask=args.use_mask,
        num_iters=args.num_iters,
        num_warmup=args.num_warmup,
        compare_torch=not args.no_compare_torch,
    )
