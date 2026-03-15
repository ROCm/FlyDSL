#!/usr/bin/env python3
"""Regression test for helper function range() inside @flyc.kernel.

This test verifies the scenario:
1) A plain Python helper function is called by a @flyc.kernel function.
2) The helper takes a dynamic value and uses `for i in range(dynamic_value)`.
3) Compilation/runtime should succeed after AST rewrite improvements.
"""

import argparse
import sys

import flydsl.compiler as flyc
import flydsl.expr as fx


def helper_with_range(loop_count: fx.Int32):
    for i in range(loop_count):
        fx.printf("helper i={}", i)


@flyc.kernel
def call_helper_kernel(loop_count: fx.Int32):
    fx.printf("kernel loop_count={}", loop_count)
    helper_with_range(loop_count)


@flyc.jit
def run_case(loop_count: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    call_helper_kernel(loop_count).launch(grid=(1, 1, 1), block=[1, 1, 1], stream=stream.value)


def run_one(loop_count: int) -> dict:
    """Run one configuration and return structured result."""
    result = {"loop_count": loop_count, "ok": False, "error": ""}
    try:
        run_case(loop_count)
        result["ok"] = True
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run helper range kernel regression example",
    )
    parser.add_argument(
        "--loop_count",
        type=int,
        default=4,
        help="Dynamic loop upper-bound used by helper range()",
    )
    args = parser.parse_args()

    r = run_one(args.loop_count)
    print(f"[CASE] loop_count={args.loop_count}")
    if r["ok"]:
        print("[OK] helper range case passed")
        return 0

    print(f"[FAIL] helper range case failed: {r['error']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# pytest interface
# ---------------------------------------------------------------------------
import pytest  # noqa: E402


@pytest.mark.parametrize("loop_count", [0, 1, 4, 7])
def test_helper_range_in_kernel_helper(loop_count):
    r = run_one(loop_count)
    assert r["ok"], f"helper range case failed for loop_count={loop_count}: {r['error']}"
