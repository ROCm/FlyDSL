#!/usr/bin/env python3
"""
Simple test script for run_moe_stage1 (no pytest required).

Usage:
    python tests/kernels/test_moe_stage1_simple.py [--size S|M|L] [--dtype fp8|fp16|int8|int4|all]

Examples:
    python tests/kernels/test_moe_stage1_simple.py                    # Run Small with fp8
    python tests/kernels/test_moe_stage1_simple.py --size M           # Run Medium with fp8
    python tests/kernels/test_moe_stage1_simple.py --dtype all        # Run Small with all dtypes
    python tests/kernels/test_moe_stage1_simple.py --size L --dtype fp8  # Run Large with fp8
"""

import argparse
import os
import sys

# Ensure repo-local flydsl is used
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

# Import run_moe_stage1 from the test file
from tests.kernels.test_moe_gemm import run_moe_stage1

# Test configurations (from pytest.param)
TEST_CONFIGS = {
    "S": {
        "tokens": 64,
        "model_dim": 256,
        "inter_dim": 128,
        "experts": 4,
        "topk": 2,
        "tile_m": 32,
        "tile_n": 64,
        "tile_k": 128,
        "doweight_stage1": False,
        "description": "Small smoke test",
    },
    "M": {
        "tokens": 128,
        "model_dim": 1024,
        "inter_dim": 256,
        "experts": 8,
        "topk": 2,
        "tile_m": 64,
        "tile_n": 128,
        "tile_k": 128,
        "doweight_stage1": False,
        "description": "Medium realistic test",
    },
    "L": {
        "tokens": 256,
        "model_dim": 4096,
        "inter_dim": 2048,
        "experts": 17,
        "topk": 9,
        "tile_m": 64,
        "tile_n": 128,
        "tile_k": 128,
        "doweight_stage1": False,
        "description": "Large aiter-style test",
    },
}

DTYPES = ["fp8", "fp16", "int8", "int4"]


def run_test(size: str, in_dtype: str, num_iters: int = 5, num_warmup: int = 2, skip_ref: bool = False):
    """Run a single stage1 test."""
    config = TEST_CONFIGS[size]
    
    print("=" * 70)
    print(f"Running MoE Stage1 Test: size={size} ({config['description']}), dtype={in_dtype}")
    print(f"  tokens={config['tokens']}, model_dim={config['model_dim']}, inter_dim={config['inter_dim']}")
    print(f"  experts={config['experts']}, topk={config['topk']}")
    print(f"  tile_m={config['tile_m']}, tile_n={config['tile_n']}, tile_k={config['tile_k']}")
    print("=" * 70)
    
    try:
        run_moe_stage1(
            tokens=config["tokens"],
            model_dim=config["model_dim"],
            inter_dim=config["inter_dim"],
            experts=config["experts"],
            topk=config["topk"],
            tile_m=config["tile_m"],
            tile_n=config["tile_n"],
            tile_k=config["tile_k"],
            doweight_stage1=config["doweight_stage1"],
            in_dtype=in_dtype,
            seed=0,
            num_iters=num_iters,
            num_warmup=num_warmup,
            compare_aiter_ck=False,  # Skip aiter comparison by default
            moe_sort_mode="torch",   # Use torch sorting for portability
            skip_ref=skip_ref,
        )
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
        description="Simple MoE Stage1 test (no pytest)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--size", "-s",
        type=str,
        choices=["S", "M", "L", "all"],
        default="S",
        help="Test size: S (small), M (medium), L (large), or all",
    )
    parser.add_argument(
        "--dtype", "-d",
        type=str,
        choices=["fp8", "fp16", "int8", "int4", "all"],
        default="fp8",
        help="Input data type (default: fp8)",
    )
    parser.add_argument(
        "--num_iters", "-n",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 5)",
    )
    parser.add_argument(
        "--num_warmup", "-w",
        type=int,
        default=2,
        help="Number of warmup iterations (default: 2)",
    )
    parser.add_argument(
        "--skip_ref",
        action="store_true",
        help="Skip reference correctness check (benchmark only)",
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA/ROCm not available. Cannot run GPU tests.")
        sys.exit(1)
    
    torch.set_default_device("cuda")
    
    # Determine sizes and dtypes to run
    sizes = list(TEST_CONFIGS.keys()) if args.size == "all" else [args.size]
    dtypes = DTYPES if args.dtype == "all" else [args.dtype]
    
    print(f"\nRunning MoE Stage1 tests: sizes={sizes}, dtypes={dtypes}")
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
