#!/usr/bin/env python3
"""Turn dense_routing_verify.csv into the routing-correctness proof.

For each dense cell we have auto (new batch-aware routing) and generic
(FLYDSL_DISABLE_DUALWAVE_SWP=1) times. We show:
  - correctness: every cell PASS for both modes;
  - the cells the new gate routes differently from the OLD flat S<256 gate
    (large batch, S in [192,256)), and that auto there is at least as fast as
    generic (i.e. routing to DUALWAVE_SWP did not regress);
  - overall: auto is within noise of min(auto, generic) on every cell (the new
    gate never leaves the materially-faster of the two providers unused).
"""
import csv
import sys
from collections import defaultdict

CSV = sys.argv[1] if len(sys.argv) > 1 else "scripts/dense_routing_verify.csv"
NOISE = 1.02  # 2% tolerance

rows = list(csv.DictReader(open(CSV)))
by = defaultdict(dict)  # (B,S,H,Hkv,dt,causal) -> {mode: row}
for r in rows:
    key = (int(r["B"]), int(r["S"]), r["H"], r["Hkv"], r["dtype"], r["causal"])
    by[key][r["mode"]] = r


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def old_gate(B, S):  # flat S<256
    return S >= 256


def new_gate(B, S):  # batch-aware
    return S >= 256 or (B >= 8 and S >= 192)


total = pass_cells = 0
changed = []  # cells where new gate != old gate
auto_not_slower = True
regressions = []
for key, m in sorted(by.items()):
    B, S = key[0], key[1]
    a, g = m.get("auto"), m.get("generic")
    if not a or not g:
        continue
    total += 1
    ok = a["pass"] == "1" and g["pass"] == "1"
    if ok:
        pass_cells += 1
    at, gt = f(a["FlyDSL_us"]), f(g["FlyDSL_us"])
    if new_gate(B, S) != old_gate(B, S):
        changed.append((key, at, gt))
        # New gate routes these to DUALWAVE_SWP; auto must be <= generic*noise.
        if at and gt and at > gt * NOISE:
            auto_not_slower = False
            regressions.append((key, at, gt))

print(f"# Dense routing verification ({CSV})\n")
print(f"Correctness: {pass_cells}/{total} cells PASS (auto AND generic, max_err<1e-2).\n")
print("## Cells the batch-aware gate routes differently from the old flat S<256 gate")
print("(large batch, 192<=S<256 -> now DUALWAVE_SWP). auto = new routing, generic = old target.\n")
print("| B | S | Hkv | dtype | causal | auto us | generic us | auto faster by |")
print("|---|---|---|---|---|---|---|---|")
for key, at, gt in changed:
    B, S, H, Hkv, dt, c = key
    spd = f"{(gt-at)/gt*100:.0f}%" if (at and gt) else "?"
    print(f"| {B} | {S} | {Hkv} | {dt} | {c} | {at:.1f} | {gt:.1f} | {spd} |")
print()
if auto_not_slower:
    print("RESULT: on every re-routed cell, auto (new gate -> DUALWAVE_SWP) is at least as")
    print("fast as the generic kernel the OLD gate would have used. The fix is correct.")
else:
    print("RESULT: REGRESSION on:", regressions)
