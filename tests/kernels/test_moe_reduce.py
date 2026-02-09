#!/usr/bin/env python3
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
from typing import Dict

import pytest
import torch

# -----------------------------------------------------------------------------
# Ensure we use the repo-local `flydsl` when running this file directly.
#
# Some environments have another `flydsl` (e.g. from a sibling checkout) earlier
# on `sys.path`, which can miss newer ROCDL wrappers (notably atomic fadd / MFMA).
# -----------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


from kernels.moe_reduce import compile_moe_reduction
from tests.test_common import verify_output

logging.basicConfig(level=logging.INFO)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


def profile_reduce_kernel(
    tokens: int,
    topk: int,
    model_dim: int,
    dtype: torch.dtype = torch.float16,
    num_iters: int = 20,
    num_warmup: int = 5,
    compare_torch: bool = True,
) -> Dict:
    """Profile reduce kernel bandwidth and latency.

    Args:
        tokens: Number of tokens.
        topk: Top-k value (reduced dimension).
        model_dim: Model dimension (contiguous inner dim).
        dtype: Data type (``torch.float16`` | ``torch.bfloat16`` | ``torch.float32``).
        num_iters: Number of benchmark iterations.
        num_warmup: Number of warmup iterations.
        compare_torch: If True, also benchmark ``torch.sum`` for comparison.

    Returns:
        Dict with profiling results.
    """
    import torch.profiler as tpf

    dtype_str = {torch.float16: "f16", torch.bfloat16: "bf16", torch.float32: "f32"}[dtype]
    reduce_exe = compile_moe_reduction(topk=topk, model_dim=model_dim, dtype_str=dtype_str)

    # Create test tensors
    X = torch.randn(tokens, topk, model_dim, device="cuda", dtype=dtype)
    Y = torch.empty(tokens, model_dim, device="cuda", dtype=dtype)

    # Calculate theoretical bandwidth
    elem_bytes = X.element_size()
    read_bytes = tokens * topk * model_dim * elem_bytes
    write_bytes = tokens * model_dim * elem_bytes
    total_bytes = read_bytes + write_bytes

    def _get_kernel_time_us(prof):
        """Extract CUDA kernel time from profiler (microseconds)."""
        total = 0.0
        for evt in prof.events():
            if str(getattr(evt, "device_type", "")).endswith("CUDA"):
                total += getattr(evt, "self_device_time_total", 0)
        return total

    results: Dict = {"shape": (tokens, topk, model_dim), "dtype": dtype_str}
    stream_ptr = torch.cuda.current_stream().cuda_stream

    # Benchmark FlyDSL reduce
    for _ in range(num_warmup):
        reduce_exe(X, Y, tokens, stream_ptr)
    torch.cuda.synchronize()

    with tpf.profile(activities=[tpf.ProfilerActivity.CUDA]) as prof:
        for _ in range(num_iters):
            reduce_exe(X, Y, tokens, stream_ptr)
        torch.cuda.synchronize()

    flydsl_us = _get_kernel_time_us(prof) / num_iters
    flydsl_bw = (total_bytes / 2**30) / (flydsl_us / 1e6) if flydsl_us > 0 else 0  # GB/s
    results["flydsl"] = {"latency_us": flydsl_us, "bandwidth_gb_s": flydsl_bw}

    # Benchmark torch.sum if requested
    if compare_torch:
        for _ in range(num_warmup):
            torch.sum(X, dim=1, out=Y)
        torch.cuda.synchronize()

        with tpf.profile(activities=[tpf.ProfilerActivity.CUDA]) as prof:
            for _ in range(num_iters):
                torch.sum(X, dim=1, out=Y)
            torch.cuda.synchronize()

        torch_us = _get_kernel_time_us(prof) / num_iters
        torch_bw = (total_bytes / 2**30) / (torch_us / 1e6) if torch_us > 0 else 0
        results["torch"] = {"latency_us": torch_us, "bandwidth_gb_s": torch_bw}
        results["speedup"] = torch_us / flydsl_us if flydsl_us > 0 else 0

    return results


def print_reduce_profile(results: Dict) -> None:
    """Pretty-print reduce profiling results."""
    tokens, topk, model_dim = results["shape"]
    print(f"\n[Reduce Kernel Profile] shape=({tokens}, {topk}, {model_dim}), dtype={results['dtype']}")
    print(f"FlyDSL:  ")
    print(f"Kernel avg time: {results['flydsl']['latency_us']:.4f} us")
    print(f"Bandwidth: {results['flydsl']['bandwidth_gb_s']:.2f} GB/s")
    if "torch" in results:
        print(f"torch:  ")
        print(f"Kernel avg time: {results['torch']['latency_us']:.4f} us")
        print(f"Bandwidth: {results['torch']['bandwidth_gb_s']:.2f} GB/s")
        print(f"speedup: {results['speedup']:.2f}x")


@pytest.mark.parametrize(
    "tokens, topk, model_dim",
    [
        pytest.param(16384, 8, 7168, id="DS-TP8-prefill-S"),
        pytest.param(32768, 8, 7168, id="DS-TP8-prefill-L"),
        pytest.param(1, 8, 7168, id="DS-TP8-decode-bs1"),
        pytest.param(8, 8, 7168, id="DS-TP8-decode-bs8"),
        pytest.param(32768, 6, 5120, id="EP-K6-prefill"),
        pytest.param(1, 6, 5120, id="EP-K6-decode-bs1"),
        pytest.param(8, 6, 5120, id="EP-K6-decode-bs8"),
    ],
)
def test_moe_reduce_kernel(tokens: int, topk: int, model_dim: int):
    """Test reduce kernel correctness and performance vs torch.sum."""
    dtype = torch.float16
    dtype_str = "f16"

    reduce_exe = compile_moe_reduction(topk=topk, model_dim=model_dim, dtype_str=dtype_str)

    # Create test data
    X = torch.randn(tokens, topk, model_dim, device="cuda", dtype=dtype)
    Y_flydsl = torch.empty(tokens, model_dim, device="cuda", dtype=dtype)
    Y_ref = torch.empty(tokens, model_dim, device="cuda", dtype=dtype)

    # Run kernels
    stream_ptr = torch.cuda.current_stream().cuda_stream
    reduce_exe(X, Y_flydsl, tokens, stream_ptr)
    torch.sum(X, dim=1, out=Y_ref)
    torch.cuda.synchronize()

    # Correctness check
    assert verify_output(Y_flydsl.float(), Y_ref.float(), rtol=1e-2, atol=1e-2, msg="[moe reduce kernel]")

    # Performance profiling
    results = profile_reduce_kernel(
        tokens=tokens,
        topk=topk,
        model_dim=model_dim,
        num_iters=20,
        num_warmup=5,
        compare_torch=True,
    )
    print_reduce_profile(results)


def _run_from_args(args: argparse.Namespace) -> None:
    """Run reduce kernel test/benchmark from parsed CLI args."""
    tokens = args.tokens
    topk = args.topk
    model_dim = args.model_dim
    dtype_map = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}
    dtype = dtype_map[args.dtype]
    dtype_str = args.dtype

    print(f"=== MoE Reduce Kernel: tokens={tokens}, topk={topk}, model_dim={model_dim}, dtype={dtype_str} ===")

    reduce_exe = compile_moe_reduction(topk=topk, model_dim=model_dim, dtype_str=dtype_str)

    # Correctness
    X = torch.randn(tokens, topk, model_dim, device="cuda", dtype=dtype)
    Y_flydsl = torch.empty(tokens, model_dim, device="cuda", dtype=dtype)
    Y_ref = torch.empty(tokens, model_dim, device="cuda", dtype=dtype)

    stream_ptr = torch.cuda.current_stream().cuda_stream
    reduce_exe(X, Y_flydsl, tokens, stream_ptr)
    torch.sum(X, dim=1, out=Y_ref)
    torch.cuda.synchronize()

    ok = verify_output(Y_flydsl.float(), Y_ref.float(), rtol=1e-2, atol=1e-2, msg="[reduce kernel]")
    if not ok:
        print("FAIL: correctness check failed!")
        raise SystemExit(1)
    print("PASS: correctness check passed.")

    # Performance
    results = profile_reduce_kernel(
        tokens=tokens,
        topk=topk,
        model_dim=model_dim,
        dtype=dtype,
        num_iters=args.num_iters,
        num_warmup=args.num_warmup,
        compare_torch=args.compare_torch,
    )
    print_reduce_profile(results)


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
    parser.add_argument("--tokens", "-t", type=int, default=16384, help="Number of tokens (default: 16384)")
    parser.add_argument("--topk", "-k", type=int, default=8, help="Top-k value (default: 8)")
    parser.add_argument("--model_dim", "-d", type=int, default=7168, help="Model dimension (default: 7168)")
    parser.add_argument(
        "--dtype",
        type=str,
        default="f16",
        choices=["f16", "bf16", "f32"],
        help="Element dtype (default: f16)",
    )
    parser.add_argument("--num_iters", type=int, default=20, help="Benchmark iterations (default: 20)")
    parser.add_argument("--num_warmup", type=int, default=5, help="Benchmark warmup iterations (default: 5)")
    parser.add_argument(
        "--compare_torch",
        action="store_true",
        default=False,
        help="Whether to compare with torch.sum (default: False).",
    )

    args = parser.parse_args()
    _run_from_args(args)
