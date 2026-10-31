#!/usr/bin/env python3

"""
FlyDSL version of cutedsl_extra_helper.py:
Test what happens when @flyc.kernel calls an external helper function
that contains `if tid == 0:` — with and without decorator.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx


# Case 1: plain helper — no decorator, has dynamic if
def helper_with_if(tid):
    out = fx.Int32(-1)
    if tid == fx.Int32(0):
        out = fx.Int32(100)
    else:
        out = fx.Int32(200)
    return out


# Case 2: helper with @flyc.jit — gets its own AST rewrite
@flyc.jit
def helper_with_if_jit(tid):
    out = fx.Int32(-1)
    if tid == fx.Int32(0):
        out = fx.Int32(100)
    else:
        out = fx.Int32(200)
    return out


@flyc.kernel
def kernel_plain_helper():
    """Case 1: call plain helper (no decorator) — if is NOT ast-rewritten"""
    tid = fx.thread_idx.x
    v = helper_with_if(tid)
    if tid <= fx.Int32(1):
        fx.printf("plain-helper: tid={} v={}", tid, v)


@flyc.kernel
def kernel_jit_helper():
    """Case 2: call @flyc.jit helper — if IS ast-rewritten by jit"""
    tid = fx.thread_idx.x
    v = helper_with_if_jit(tid)
    if tid <= fx.Int32(1):
        fx.printf("jit-helper: tid={} v={}", tid, v)


@flyc.jit
def run_plain_helper(stream: fx.Stream = fx.Stream(None)):
    kernel_plain_helper().launch(grid=(1, 1, 1), block=(2, 1, 1), stream=stream)


@flyc.jit
def run_jit_helper(stream: fx.Stream = fx.Stream(None)):
    kernel_jit_helper().launch(grid=(1, 1, 1), block=(2, 1, 1), stream=stream)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device is required.")

    stream = torch.cuda.current_stream()

    print("=== Case 1: plain helper (no decorator) ===")
    try:
        run_plain_helper(stream)
        torch.cuda.synchronize()
        print("Case 1 completed (may be WRONG — if not rewritten)")
    except Exception as e:
        print(f"Case 1 failed: {e}")

    print()
    print("=== Case 2: @flyc.jit helper ===")
    try:
        run_jit_helper(stream)
        torch.cuda.synchronize()
        print("Case 2 completed")
    except Exception as e:
        print(f"Case 2 failed: {e}")


if __name__ == "__main__":
    main()
