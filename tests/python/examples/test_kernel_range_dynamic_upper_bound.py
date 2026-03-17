#!/usr/bin/env python3
"""Regression test for range kernel with dynamic upper bound."""

import argparse
import os
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def range_kernel(loop_count: fx.Int32):
    fx.printf("kernel loop_count={}", loop_count)
    for i in range(loop_count):
        fx.printf("helper i={}", i)


@flyc.jit
def run_case(loop_count: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    range_kernel(loop_count).launch(grid=(1, 1, 1), block=[1, 1, 1], stream=stream.value)


def main():
    parser = argparse.ArgumentParser(
        description="Run range kernel with dynamic upper bound",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--loop_count",
        type=int,
        default=4,
        help="Dynamic upper bound used by range(loop_count)",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("FlyDSL Range Kernel Dynamic Upper Bound Example")
    print("=" * 72)
    print(f"  loop_count: {args.loop_count}")
    print("=" * 72)

    try:
        run_case(args.loop_count)
    except Exception as e:
        print(f"[FAIL] run_case failed: {type(e).__name__}: {e}")
        sys.exit(1)

    print("[OK] run_case succeeded.")


@pytest.mark.parametrize("loop_count", [1, 4, 8])
def test_range_kernel_dynamic_upper_bound(loop_count):
    # Explicitly fail with context if compile/verify/runtime raises.
    try:
        run_case(loop_count)
    except Exception as e:
        pytest.fail(
            "range(loop_count) kernel should compile and run without type mismatch. "
            f"loop_count={loop_count}, error={type(e).__name__}: {e}"
        )


if __name__ == "__main__":
    main()

