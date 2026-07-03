#!/usr/bin/env python3
"""Find which instructions are NOT hidden behind MFMA in a GEMM hot loop.

Model: each MFMA occupies a fixed execute window of EXEC cycles from its issue
cycle. Back-to-back MFMAs tile [c, c+EXEC) windows; while the matrix unit is busy
any co-issued scalar/VMEM/LDS op is "free". Cycles OUTSIDE the union of those
windows are EXPOSED -- the matrix unit idles there, so cyc/mfma > EXEC. We attribute
each exposed gap to the (non-MFMA) instruction at its start = what blocked the next
MFMA from issuing on time.

Input: an ATT rocprofv3 UI dispatch dir (has code.json + se*_wv*.json). Pick the
steady-state cycle window with --range (avoid prologue/tail).

Usage:
  python3 mfma_coverage.py <dispatch_dir> [--wave se0_sm0_sl0_wv0.json]
                                          [--range LO,HI] [--exec 16]

  # find a steady window first (cycles): the script prints the wave cycle span;
  # pick a mid slice spanning ~10 outer loop iterations.

Notes:
- EXEC is the MFMA execute latency, NOT the issue latency. fp4
  mfma_scale_f32_16x16x128 ~ 16; fp8 16x16x128 ~ 32. Pass --exec accordingly.
- "idle" gaps = pure latency stalls (waitcnt drain / dependency) with no issuing
  instruction; real wins come from cutting the named-instruction gaps.
"""
import argparse
import collections
import glob
import json
import os
import sys


def load(dispatch_dir, wave):
    code = json.load(open(os.path.join(dispatch_dir, "code.json")))["code"]
    if wave:
        wpath = os.path.join(dispatch_dir, wave)
    else:
        cands = sorted(glob.glob(os.path.join(dispatch_dir, "se*_wv0.json")))
        wpath = cands[0]
    wj = json.load(open(wpath))
    return code, wj["wave"]["instructions"], os.path.basename(wpath)


def op_of(code, cid):
    a = code[cid][0].strip().split() if cid < len(code) else ["?"]
    return a[0] if a else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dispatch_dir")
    ap.add_argument("--wave", default=None)
    ap.add_argument("--range", default=None, help="LO,HI cycle window (steady state)")
    ap.add_argument("--exec", type=int, default=16, dest="exec_cyc", help="MFMA execute latency")
    args = ap.parse_args()

    code, insts, wname = load(args.dispatch_dir, args.wave)
    cyc = [r[0] for r in insts]
    print(f"wave={wname}  inst cycle span {min(cyc)}..{max(cyc)}  n={len(insts)}")
    if not args.range:
        print("pass --range LO,HI (a mid steady slice, ~10 outer iters). e.g. "
              f"--range {min(cyc) + (max(cyc)-min(cyc))//4},{min(cyc) + (max(cyc)-min(cyc))//2}")
        return
    lo, hi = (int(x) for x in args.range.split(","))
    seg = sorted((r for r in insts if lo <= r[0] < hi), key=lambda r: r[0])
    span = hi - lo

    E = args.exec_cyc
    mfma = sorted(r[0] for r in seg if op_of(code, r[4]).startswith("v_mfma"))
    if not mfma:
        print("no MFMA in window")
        return

    # next_free model (matrix-unit pipeline): each MFMA occupies ONE E-cycle execute
    # slot, but slots pipeline -- consecutive MFMAs can ISSUE < E apart and still both
    # be hidden (the unit stays busy). Track next_free = when the matrix unit frees.
    #   - issue t <= next_free : hidden (co-issued in the shadow); slot advances +E
    #   - issue t >  next_free : the unit was IDLE for (t - next_free) -> EXPOSED
    # This fixes the older union-of-[issue,issue+E) model, which capped overlapping
    # windows and so mislabeled shadow-hidden loads (dense 8-cyc-apart MFMAs) as
    # exposed. Blame each exposed gap on the first non-MFMA op issuing inside it.
    next_free = mfma[0]
    gaps = []
    for t in mfma:
        if t > next_free:
            gaps.append((next_free, t))
            next_free = t + E
        else:
            next_free = next_free + E
    exp = sum(b - a for a, b in gaps)
    cov = span - exp
    print(f"\nsegment [{lo},{hi})  span={span}  mfma={len(mfma)}  exec={E}")
    print(f"MFMA-covered: {cov} ({cov*100//span}%)   EXPOSED: {exp} ({exp*100//span}%)")
    print(f"cyc/mfma = {span/len(mfma):.2f}  (floor = {E})")

    total_gap = exp or 1

    def first_non_mfma(g0, g1):
        for r in seg:
            if g0 <= r[0] < g1 and not op_of(code, r[4]).startswith("v_mfma"):
                return op_of(code, r[4])
        return "idle"

    attr = collections.Counter()
    for g0, g1 in gaps:
        attr[first_non_mfma(g0, g1)] += g1 - g0
    print(f"\n== exposed gap cycles by blocking instruction ({len(gaps)} gaps, {total_gap} cyc) ==")
    for o, c in attr.most_common(15):
        print(f"  {c:6d} cyc ({c*100//total_gap:2d}%)  {o}")


if __name__ == "__main__":
    main()
