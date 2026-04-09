#!/usr/bin/env python3
"""Regression test: ``if precomputed_var:`` inside @flyc.kernel.

Verifies that storing a dynamic comparison result in a variable and
then branching on it (``if flag:``) correctly produces a GPU-level
``scf.IfOp``.

The test has two kernels:
  * **inline_if_kernel** — ``if tid < threshold:``  (Compare node)
  * **precomputed_if_kernel** — ``flag = tid < threshold; if flag:``
    (Name node)

Both kernels conditionally write 1 to an output buffer.  The test checks
that they produce the **same** result.
"""

import argparse
import sys

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, gpu
from flydsl.expr.typing import T

NUM_THREADS = 256


# -- Kernel 1: inline comparison (works) ------------------------------------

@flyc.kernel
def inline_if_kernel(
    out: fx.Tensor,
    threshold: fx.Int32,
):
    rsrc = buffer_ops.create_buffer_resource(out)
    tid = arith.index_cast(T.i32, gpu.thread_id("x"))

    # Inline Compare — AST rewriter generates scf.IfOp correctly.
    if tid < threshold:
        buffer_ops.buffer_store(arith.constant(1), rsrc, tid)


# -- Kernel 2: pre-computed condition (bug) ---------------------------------

@flyc.kernel
def precomputed_if_kernel(
    out: fx.Tensor,
    threshold: fx.Int32,
):
    rsrc = buffer_ops.create_buffer_resource(out)
    tid = arith.index_cast(T.i32, gpu.thread_id("x"))

    # Store result in a variable first — tests that ast.Name is recognised
    # as potentially dynamic by _could_be_dynamic.
    flag = tid < threshold
    if flag:
        buffer_ops.buffer_store(arith.constant(1), rsrc, tid)


# -- Launchers --------------------------------------------------------------

@flyc.jit
def launch_inline(
    out: fx.Tensor,
    threshold: fx.Int32,
    stream: fx.Stream = fx.Stream(None),
):
    inline_if_kernel(out, threshold).launch(
        grid=(1, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream,
    )


@flyc.jit
def launch_precomputed(
    out: fx.Tensor,
    threshold: fx.Int32,
    stream: fx.Stream = fx.Stream(None),
):
    precomputed_if_kernel(out, threshold).launch(
        grid=(1, 1, 1), block=(NUM_THREADS, 1, 1), stream=stream,
    )


# -- Test logic --------------------------------------------------------------

def run_one(threshold: int) -> dict:
    """Run both kernels and compare output.

    Returns dict with pass/fail and details.
    """
    result = {"threshold": threshold, "ok": False, "error": "", "detail": ""}

    try:
        out_inline = torch.zeros(NUM_THREADS, dtype=torch.int32, device="cuda")
        out_precomp = torch.zeros(NUM_THREADS, dtype=torch.int32, device="cuda")

        launch_inline(out_inline, threshold)
        launch_precomputed(out_precomp, threshold)
        torch.cuda.synchronize()

        expected_ones = min(threshold, NUM_THREADS)

        inline_ones = out_inline.sum().item()
        precomp_ones = out_precomp.sum().item()

        result["detail"] = (
            f"threshold={threshold}, expected_ones={expected_ones}, "
            f"inline_ones={inline_ones}, precomp_ones={precomp_ones}"
        )

        if inline_ones != expected_ones:
            result["error"] = (
                f"inline kernel wrong: got {inline_ones}, expected {expected_ones}"
            )
            return result

        if precomp_ones != expected_ones:
            result["error"] = (
                f"precomputed kernel wrong: got {precomp_ones}, expected {expected_ones} "
                f"(likely all {NUM_THREADS} — bare ast.Name not recognised as dynamic)"
            )
            return result

        result["ok"] = True

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test pre-computed if-condition in FlyDSL kernels",
    )
    parser.add_argument(
        "--threshold", type=int, default=64,
        help="Thread threshold for conditional write (default: 64)",
    )
    args = parser.parse_args()

    r = run_one(args.threshold)
    print(f"[CASE] {r['detail']}")
    if r["ok"]:
        print("[OK] inline and precomputed kernels match")
        return 0

    print(f"[FAIL] {r['error']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# pytest interface
# ---------------------------------------------------------------------------
import pytest  # noqa: E402


@pytest.mark.parametrize("threshold", [0, 1, 64, 128, 256])
def test_precomputed_if_matches_inline(threshold):
    r = run_one(threshold)
    assert r["ok"], f"precomputed if mismatch: {r['error']} ({r['detail']})"
