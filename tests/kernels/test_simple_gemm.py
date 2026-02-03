#!/usr/bin/env python3
"""
Simple test script for the simple GEMM kernel.

Usage:
    python tests/kernels/test_simple_gemm.py [--size S|M|L|XL|NA1|NA2|all] [--dtype bf16|fp16|all]

Examples:
    python tests/kernels/test_simple_gemm.py                    # Run Small with bf16
    python tests/kernels/test_simple_gemm.py --size M           # Run Medium with bf16
    python tests/kernels/test_simple_gemm.py --dtype all        # Run Small with all dtypes
    python tests/kernels/test_simple_gemm.py --size all         # Run all sizes with bf16
    python tests/kernels/test_simple_gemm.py --size NA1         # Non-aligned test 1
"""

import argparse
import logging
import os
import sys

# Ensure repo-local flydsl is used
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from kernels.simple_gemm import compile_simple_gemm, run_simple_gemm
from tests.test_common import run_perftest, verify_output

# Configure logging to show INFO level messages (required for kernel name display)
logging.basicConfig(level=logging.INFO)

# Test configurations
# Aligned tests: M, N, K are multiples of tile sizes
TEST_CONFIGS = {
    "S": {
        "M": 16,
        "N": 64,
        "K": 128,
        "tile_m": 16,
        "tile_n": 64,
        "tile_k": 128,
        "description": "Small smoke test (single tile)",
    },
    "M": {
        "M": 64,
        "N": 128,
        "K": 256,
        "tile_m": 16,
        "tile_n": 64,
        "tile_k": 128,
        "description": "Medium test (multi-tile)",
    },
    "L": {
        "M": 256,
        "N": 512,
        "K": 512,
        "tile_m": 16,
        "tile_n": 64,
        "tile_k": 128,
        "description": "Large test",
    },
    "XL": {
        "M": 1280,
        "N": 2048,
        "K": 128,
        "tile_m": 16,
        "tile_n": 64,
        "tile_k": 128,
        "description": "Extra large test",
    },
    # Non-aligned tests: M, N, K are NOT multiples of 16
    "NA1": {
        "M": 33,   # Not aligned to 16
        "N": 87,   # Not aligned to 64
        "K": 145,  # Not aligned to 128
        "tile_m": 16,
        "tile_n": 64,
        "tile_k": 128,
        "description": "Non-aligned test 1 (M=33, N=87, K=145)",
    },
    "NA2": {
        "M": 57,   # Not aligned to 16
        "N": 123,  # Not aligned to 64
        "K": 259,  # Not aligned to 128
        "tile_m": 16,
        "tile_n": 64,
        "tile_k": 128,
        "description": "Non-aligned test 2 (M=57, N=123, K=259)",
    },
    "NA3": {
        "M": 100,  # Not aligned to 16
        "N": 200,  # Not aligned to 64
        "K": 300,  # Not aligned to 128
        "tile_m": 16,
        "tile_n": 64,
        "tile_k": 128,
        "description": "Non-aligned test 3 (M=100, N=200, K=300)",
    },
    "NA4": {
        "M": 171,  # Not aligned to 16
        "N": 333,  # Not aligned to 64
        "K": 517,  # Not aligned to 128
        "tile_m": 16,
        "tile_n": 64,
        "tile_k": 128,
        "description": "Non-aligned test 4 (M=171, N=333, K=517)",
    },
}

DTYPES = ["bf16", "fp16"]


def get_torch_dtype(in_dtype: str):
    """Convert string dtype to torch dtype."""
    if in_dtype == "bf16":
        return torch.bfloat16
    elif in_dtype == "fp16":
        return torch.float16
    else:
        raise ValueError(f"Unknown dtype: {in_dtype}")


def _align_up(val: int, align: int) -> int:
    """Round up val to the next multiple of align."""
    return ((val + align - 1) // align) * align


def run_test(
    size: str,
    in_dtype: str,
    num_iters: int = 100,
    num_warmup: int = 5,
    skip_ref: bool = False,
    rtol: float = 1e-2,
    atol: float = 1e-2,
):
    """Run a single GEMM test."""
    config = TEST_CONFIGS[size]
    M = config["M"]
    N = config["N"]
    K = config["K"]
    tile_m = config["tile_m"]
    tile_n = config["tile_n"]
    tile_k = config["tile_k"]

    # Pad all dimensions to tile sizes
    M_pad = _align_up(M, tile_m)
    N_pad = _align_up(N, tile_n)
    K_pad = _align_up(K, tile_k)

    print("=" * 70)
    print(f"Running Simple GEMM Test: size={size} ({config['description']}), dtype={in_dtype}")
    print(f"  M={M}, N={N}, K={K}")
    print(f"  M_pad={M_pad}, N_pad={N_pad}, K_pad={K_pad}")
    print(f"  tile_m={tile_m}, tile_n={tile_n}, tile_k={tile_k}")
    print("=" * 70)

    torch_dtype = get_torch_dtype(in_dtype)
    device = "cuda"

    try:
        # Create random inputs (original size)
        torch.manual_seed(42)
        A_orig = torch.randn(M, K, dtype=torch_dtype, device=device)
        B_orig = torch.randn(N, K, dtype=torch_dtype, device=device)

        # Run reference computation (using float32 for accuracy) with original dimensions
        if not skip_ref:
            A_f32 = A_orig.to(torch.float32)
            B_f32 = B_orig.to(torch.float32)
            C_ref = torch.mm(A_f32, B_f32.T).to(torch_dtype)

        # Create padded tensors
        A_pad = torch.zeros(M_pad, K_pad, dtype=torch_dtype, device=device)
        B_pad = torch.zeros(N_pad, K_pad, dtype=torch_dtype, device=device)
        C_pad = torch.zeros(M_pad, N_pad, dtype=torch_dtype, device=device)
        
        # Copy original data
        A_pad[:M, :K] = A_orig
        B_pad[:N, :K] = B_orig

        # Compile kernel
        print("Compiling kernel...")
        exe = compile_simple_gemm(
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            in_dtype=in_dtype,
        )
        print("Kernel compiled successfully.")

        # Flatten tensors for kernel interface
        A_flat = A_pad.view(-1)
        B_flat = B_pad.view(-1)
        C_flat = C_pad.view(-1)
        C_flat.zero_()

        # Define launch function for run_perftest
        def launch():
            exe(C_flat, A_flat, B_flat, M_pad, N_pad, K_pad, M, N)

        # Warmup and benchmark using run_perftest
        print(f"Running {num_warmup} warmup + {num_iters} benchmark iterations...")
        _, us = run_perftest(
            launch,
            num_iters=num_iters,
            num_warmup=num_warmup,
        )
        torch.cuda.synchronize()

        # Calculate TFLOPS
        flops = 2 * M * N * K  # 2 ops per element (multiply + add)
        tflops = flops / (us / 1e6) / 1e12

        print(f"  Time per iteration: {us:.3f} us ({us/1000:.3f} ms)")
        print(f"  Throughput: {tflops:.2f} TFLOPS")

        # Verify correctness
        if not skip_ref:
            # Run one more time for correctness check
            C_flat.zero_()
            exe(C_flat, A_flat, B_flat, M_pad, N_pad, K_pad, M, N)
            torch.cuda.synchronize()
            # Extract only the M×N portion from the padded output
            C_result = C_pad[:M, :N]

            # Check correctness using verify_output
            passed = verify_output(
                C_result.to(torch.float32),
                C_ref.to(torch.float32),
                rtol=rtol,
                atol=atol,
                msg=f"size={size}, dtype={in_dtype}"
            )

            if not passed:
                # Print more details for debugging
                max_diff = (C_result - C_ref).abs().max().item()
                mean_diff = (C_result - C_ref).abs().mean().item()
                print(f"  Max diff: {max_diff:.6f}")
                print(f"  Mean diff: {mean_diff:.6f}")
                print("\n  Sample values (first 4x4):")
                print(f"  Result:\n{C_result[:4, :4]}")
                print(f"  Reference:\n{C_ref[:4, :4]}")
                return False

        print(f"[PASS] size={size}, dtype={in_dtype}\n")
        return True

    except Exception as e:
        print(f"[FAIL] size={size}, dtype={in_dtype}")
        print(f"  Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Simple GEMM kernel test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--size", "-s",
        type=str,
        choices=list(TEST_CONFIGS.keys()) + ["all", "aligned", "nonaligned"],
        default="S",
        help="Test size: S/M/L/XL (aligned), NA1/NA2/NA3/NA4 (non-aligned), all, aligned, or nonaligned",
    )
    parser.add_argument(
        "--dtype", "-d",
        type=str,
        choices=["bf16", "fp16", "all"],
        default="bf16",
        help="Input data type (default: bf16)",
    )
    parser.add_argument(
        "--num_iters", "-n",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--num_warmup", "-w",
        type=int,
        default=5,
        help="Number of warmup iterations (default: 5)",
    )
    parser.add_argument(
        "--skip_ref",
        action="store_true",
        help="Skip reference correctness check (benchmark only)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for correctness check (default: 1e-2)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for correctness check (default: 1e-2)",
    )

    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm not available. Cannot run GPU tests.")
        sys.exit(1)

    torch.set_default_device("cuda")

    # Determine sizes and dtypes to run
    aligned_sizes = ["S", "M", "L", "XL"]
    nonaligned_sizes = ["NA1", "NA2", "NA3", "NA4"]
    
    if args.size == "all":
        sizes = list(TEST_CONFIGS.keys())
    elif args.size == "aligned":
        sizes = aligned_sizes
    elif args.size == "nonaligned":
        sizes = nonaligned_sizes
    else:
        sizes = [args.size]
    
    dtypes = DTYPES if args.dtype == "all" else [args.dtype]

    print(f"\nRunning Simple GEMM tests: sizes={sizes}, dtypes={dtypes}")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    results = []
    for size in sizes:
        for dtype in dtypes:
            passed = run_test(
                size=size,
                in_dtype=dtype,
                num_iters=args.num_iters,
                num_warmup=args.num_warmup,
                skip_ref=args.skip_ref,
                rtol=args.rtol,
                atol=args.atol,
            )
            results.append((size, dtype, passed))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, _, p in results if p)
    total = len(results)
    for size, dtype, p in results:
        status = "PASS" if p else "FAIL"
        print(f"  [{status}] size={size}, dtype={dtype}")
    print(f"\nTotal: {passed}/{total} passed")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
