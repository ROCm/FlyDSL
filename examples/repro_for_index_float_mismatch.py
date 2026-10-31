#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Repro: dynamic for loop-carried variable starts as index and becomes float.

Goal:
- Trigger FlyDSL's auto-cast branch in scf_for_dispatch:
  if got.type == index or expect_ty == index -> arith.IndexCastOp(expect_ty, got)
- When expect_ty is index but got is f32, this produces an invalid index_cast.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.jit
def repro_index_float_mismatch(n: fx.Int32):
    acc = fx.Index(0)
    for i in range(0, n):
        acc = fx.Float32(1.0)
    return acc


def main():
    n = 4
    print("[repro] running index-vs-float loop-carried mismatch")
    try:
        repro_index_float_mismatch(n)
    except Exception as e:
        print("[repro] captured expected failure:")
        print(type(e).__name__, str(e))
        return
    raise RuntimeError("Unexpected pass: this repro expects an index/float mismatch failure.")


if __name__ == "__main__":
    main()
