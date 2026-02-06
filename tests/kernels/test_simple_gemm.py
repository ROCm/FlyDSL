#!/usr/bin/env python3
"""
Simple test script for the simple GEMM kernel.

Usage:
    python tests/kernels/test_simple_gemm.py [--size S|M|L|XL|NA1|NA2|all] [--dtype bf16|fp16|all] [--waves_per_eu N]

Examples:
    python tests/kernels/test_simple_gemm.py                    # Run Small with bf16
    python tests/kernels/test_simple_gemm.py --size M           # Run Medium with bf16
    python tests/kernels/test_simple_gemm.py --dtype all        # Run Small with all dtypes
    python tests/kernels/test_simple_gemm.py --size all         # Run all sizes with bf16
    python tests/kernels/test_simple_gemm.py --size NA1         # Non-aligned test 1
    python tests/kernels/test_simple_gemm.py --waves_per_eu 2   # Set waves per EU hint to 2
"""

import argparse
import hashlib
import logging
import os
import random
import sys

# Ensure repo-local flydsl is used
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
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

# Tensor initialization range (uniform distribution)
UNIFORM_RANGE = (-1, 1)
DEFAULT_SEED = 123


def setup_seed(seed: int) -> None:
    """Set random seed for reproducibility across all RNG sources."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


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


def compute_md5(tensor: torch.Tensor) -> str:
    """Compute MD5 hash of a tensor's raw bytes."""
    return hashlib.md5(
        tensor.contiguous().view(torch.uint8).detach().cpu().numpy().tobytes()
    ).hexdigest()


def compare_arrays(
    arr1: np.ndarray,
    arr2: np.ndarray,
    k: int = 5,
    thresholds: list = None,
) -> dict:
    """Compare two numpy arrays and compute various difference metrics.

    Args:
        arr1: First input array (result), will be cast to float32.
        arr2: Second input array (reference), will be cast to float32.
        k: Number of top differences to report.
        thresholds: Difference magnitude buckets for histogram.

    Returns:
        Dictionary with top_k_diff, threshold_stats, nan_info, max_diff, max_diff_thr.
    """
    if thresholds is None:
        thresholds = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]

    if arr1.shape != arr2.shape:
        raise ValueError(
            f"Shape mismatch: arr1 {arr1.shape} vs arr2 {arr2.shape}"
        )

    arr1 = arr1.astype(np.float32)
    arr2 = arr2.astype(np.float32)

    result = {"top_k_diff": [], "threshold_stats": [], "nan_info": {}}

    # Check for NaN values
    nan_mask1 = np.isnan(arr1)
    nan_mask2 = np.isnan(arr2)
    if np.any(nan_mask1):
        result["nan_info"]["arr1_nan_count"] = int(np.sum(nan_mask1))
        print(f"  Warning: result contains {result['nan_info']['arr1_nan_count']} NaN values")
    if np.any(nan_mask2):
        result["nan_info"]["arr2_nan_count"] = int(np.sum(nan_mask2))
        print(f"  Warning: reference contains {result['nan_info']['arr2_nan_count']} NaN values")

    # Compute absolute differences
    diff = np.abs(arr1 - arr2)
    total_elements = arr1.size

    max_diff_thr = (diff / (1.0 + np.abs(arr2))).max()
    result["max_diff"] = float(diff.max())
    result["max_diff_thr"] = float(max_diff_thr)

    print(f"  diff.abs.max = {diff.max():.6f}")
    print(f"  diff.abs.mean = {diff.mean():.6f}")
    print(f"  max_diff_thr (rel) = {max_diff_thr:.6e}")

    # Find top k differences
    flat_diff = diff.flatten()
    actual_k = min(k, len(flat_diff))
    top_k_indices = np.argpartition(flat_diff, -actual_k)[-actual_k:]
    top_k_indices = top_k_indices[np.argsort(-flat_diff[top_k_indices])]

    orig_indices = np.unravel_index(top_k_indices, diff.shape)
    print(f"  Top-{actual_k} differences:")
    for i in range(actual_k):
        idx = tuple(dim[i] for dim in orig_indices)
        entry = {
            "value": float(diff[idx]),
            "position": idx,
            "arr1_value": float(arr1[idx]),
            "arr2_value": float(arr2[idx]),
        }
        result["top_k_diff"].append(entry)
        print(f"    [{idx}] result={arr1[idx]:.6f}, ref={arr2[idx]:.6f}, diff={diff[idx]:.6f}")

    # Compute threshold statistics
    print(f"  Threshold distribution ({total_elements} elements):")
    for i in range(len(thresholds) - 1):
        lower, upper = thresholds[i], thresholds[i + 1]
        count = int(np.sum((diff >= lower) & (diff < upper)))
        pct = 100.0 * count / total_elements
        result["threshold_stats"].append(
            {"range": f"[{lower:.0e}, {upper:.0e})", "count": count, "percentage": pct}
        )
        print(f"    [{lower:.0e}, {upper:.0e}): {count:>8d} ({pct:6.2f}%)")

    count = int(np.sum(diff >= thresholds[-1]))
    pct = 100.0 * count / total_elements
    result["threshold_stats"].append(
        {"range": f">={thresholds[-1]:.0e}", "count": count, "percentage": pct}
    )
    print(f"    >={thresholds[-1]:.0e}       : {count:>8d} ({pct:6.2f}%)")

    return result


def run_test(
    size: str,
    in_dtype: str,
    num_iters: int = 100,
    num_warmup: int = 5,
    skip_ref: bool = False,
    rtol: float = 1e-2,
    atol: float = 1e-2,
    waves_per_eu: int = None,
    seed: int = DEFAULT_SEED,
):
    """Run a single GEMM test."""
    config = TEST_CONFIGS[size]
    M = config["M"]
    N = config["N"]
    K = config["K"]
    tile_m = config["tile_m"]
    tile_n = config["tile_n"]
    tile_k = config["tile_k"]

    # K must be padded to tile_k for MFMA vector loads
    K_pad = _align_up(K, tile_k)

    print("=" * 70)
    print(f"Running Simple GEMM Test: size={size} ({config['description']}), dtype={in_dtype}")
    print(f"  M={M}, N={N}, K={K} (K_pad={K_pad})")
    print(f"  tile_m={tile_m}, tile_n={tile_n}, tile_k={tile_k}")
    print("=" * 70)

    torch_dtype = get_torch_dtype(in_dtype)
    device = "cuda"

    try:
        # Create random inputs (uniform distribution in UNIFORM_RANGE)
        setup_seed(seed)
        A_orig = torch.empty(M, K, dtype=torch_dtype, device=device).uniform_(*UNIFORM_RANGE)
        B_orig = torch.empty(N, K, dtype=torch_dtype, device=device).uniform_(*UNIFORM_RANGE)

        # Run reference computation (using float32 for accuracy) with original K
        if not skip_ref:
            A_f32 = A_orig.to(torch.float32)
            B_f32 = B_orig.to(torch.float32)
            C_ref = torch.mm(A_f32, B_f32.T).to(torch_dtype)

        # Pad K for kernel (M and N are handled by kernel mask-based boundary checks)
        if K_pad != K:
            A = torch.zeros(M, K_pad, dtype=torch_dtype, device=device)
            B = torch.zeros(N, K_pad, dtype=torch_dtype, device=device)
            A[:, :K] = A_orig
            B[:, :K] = B_orig
        else:
            A = A_orig
            B = B_orig

        # Create output tensor (original size, no padding needed for M and N)
        C = torch.zeros(M, N, dtype=torch_dtype, device=device)

        # Compile kernel
        print("Compiling kernel...")
        if waves_per_eu is not None:
            print(f"  waves_per_eu={waves_per_eu}")
        exe = compile_simple_gemm(
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            in_dtype=in_dtype,
            waves_per_eu=waves_per_eu,
        )
        print("Kernel compiled successfully.")

        # Flatten tensors for kernel interface
        A_flat = A.view(-1)
        B_flat = B.view(-1)
        C_flat = C.view(-1)
        C_flat.zero_()

        # Define launch function for run_perftest
        def launch():
            exe(C_flat, A_flat, B_flat, M, N, K_pad)

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
            exe(C_flat, A_flat, B_flat, M, N, K_pad)
            torch.cuda.synchronize()
            C_result = C

            # Compute and print MD5 hashes
            result_md5 = compute_md5(C_result)
            ref_md5 = compute_md5(C_ref)
            print(f"  result_md5 = {result_md5}")
            print(f"  ref_md5    = {ref_md5}")
            if result_md5 == ref_md5:
                print("  MD5 match: EXACT (bit-identical)")
            else:
                print("  MD5 match: DIFFER (not bit-identical)")

            # Detailed comparison using compare_arrays
            print("  --- compare_arrays ---")
            compare_arrays(
                C_result.to(torch.float32).detach().cpu().numpy(),
                C_ref.to(torch.float32).detach().cpu().numpy(),
            )

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
    parser.add_argument(
        "--waves_per_eu",
        type=int,
        default=None,
        help="AMDGPU waves-per-eu hint for occupancy optimization (e.g., 1, 2, 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
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
    print(f"seed: {args.seed}")
    if args.waves_per_eu is not None:
        print(f"waves_per_eu: {args.waves_per_eu}")
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
                waves_per_eu=args.waves_per_eu,
                seed=args.seed,
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
