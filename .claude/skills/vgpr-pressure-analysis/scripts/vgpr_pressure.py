#!/usr/bin/env python3
"""Analyze AMDGCN VGPR/AGPR register pressure from a FlyDSL ISA dump.

Two reports:
  1. High-water summary (compiler-authoritative): vgpr/agpr/sgpr counts, spill,
     accum_offset (= arch-VGPR high-water; AGPR file starts here), headroom.
  2. Per-purpose arch-VGPR breakdown: bucket each arch VGPR (v0..accum_offset-1)
     by the instruction that LAST defines it (ds_read=A/B fragment, address ALU,
     buffer_load=scale/data, cvt=epilogue, ...). Shows where the pressure goes.

Usage:
  python3 vgpr_pressure.py <dump_dir>      # dir containing *_final_isa.s
  python3 vgpr_pressure.py <file.s>        # an ISA file directly
"""
import collections
import glob
import os
import re
import sys


def find_isa(path):
    if os.path.isfile(path):
        return path
    files = glob.glob(f"{path}/**/*_final_isa.s", recursive=True)
    if not files:
        files = glob.glob(f"{path}/**/*.s", recursive=True)
    return max(files, key=os.path.getmtime) if files else None


def summary(text):
    def g(pat):
        m = re.search(pat, text)
        return m.group(1) if m else None

    vgpr = g(r"\.vgpr_count:\s*(\d+)")
    agpr = g(r"\.agpr_count:\s*(\d+)")
    sgpr = g(r"\.sgpr_count:\s*(\d+)")
    vspill = g(r"\.vgpr_spill_count:\s*(\d+)") or "0"
    sspill = g(r"\.sgpr_spill_count:\s*(\d+)") or "0"
    acc = g(r"\.amdhsa_accum_offset\s*(\d+)")
    print("== high-water (compiler-authoritative) ==")
    print(f"  vgpr_total={vgpr}  agpr={agpr}  accum_offset(arch hi-water)={acc}  sgpr={sgpr}")
    print(f"  spill: vgpr={vspill} sgpr={sspill}{'   *** SPILL ***' if vspill != '0' or sspill != '0' else ''}")
    if vgpr and vgpr.isdigit():
        print(f"  total {vgpr}/512  headroom={512 - int(vgpr)}")
    return int(acc) if acc and acc.isdigit() else (int(vgpr) if vgpr and vgpr.isdigit() else 256)


def dests(line):
    l = line.strip()
    if not l or l.startswith((";", "s_", ".")):
        return ([], None)
    m = re.match(r"(\S+)\s+(v\[(\d+):(\d+)\]|v(\d+))", l)
    if not m:
        return ([], None)
    op = m.group(1)
    rs = list(range(int(m.group(3)), int(m.group(4)) + 1)) if m.group(3) is not None else [int(m.group(5))]
    return (rs, op)


def bucket(op):
    if op.startswith("v_mfma"):
        return "MFMA dest (accum; mostly AGPR, arch only if spilled)"
    if op.startswith("ds_read"):
        return "ds_read dest (S2R A/B fragment - data)"
    if op.startswith("buffer_load_dwordx4"):
        return "buffer_load x4 (wide A/B data)"
    if op.startswith("buffer_load"):
        return "buffer_load dword (scale / narrow)"
    if op.startswith(("v_add", "v_or", "v_lshl", "v_and", "v_mul", "v_sub", "v_bitop", "v_lshr", "v_bfe")):
        return "address / index ALU"
    if op.startswith("v_cvt"):
        return "cvt (epilogue dtype convert)"
    if op.startswith(("v_mov", "v_accvgpr")):
        return "mov / accvgpr shuffle"
    return f"other ({op})"


def breakdown(text, arch_hi):
    owner = {}  # reg -> bucket of its LAST defining op (its "current" use at peak)
    for l in text.splitlines():
        rs, op = dests(l)
        if op is None:
            continue
        b = bucket(op)
        for r in rs:
            if r < arch_hi:
                owner[r] = b
    tab = collections.Counter(owner.values())
    print(f"\n== arch-VGPR breakdown (v0..v{arch_hi-1}, by last-defining op) ==")
    print(f"  {len(owner)}/{arch_hi} arch VGPRs defined in-kernel")
    print(f"  {'bucket':52s} #regs")
    for b, n in tab.most_common():
        print(f"  {b:52s} {n}")


def peak_region(text, arch_hi):
    """Show instructions touching the TOP arch VGPRs (v[hi-12..hi-1]) -- the true
    peak-pressure region (those high numbers only get allocated at the peak)."""
    lo = max(0, arch_hi - 12)
    lines = text.splitlines()
    hits = []
    for i, l in enumerate(lines):
        vs = set()
        for m in re.findall(r"v\[(\d+):(\d+)\]", l):
            vs.update(range(int(m[0]), int(m[1]) + 1))
        for m in re.findall(r"(?<![\[:0-9])v(\d+)(?![:0-9])", l):
            vs.add(int(m))
        top = sorted(r for r in vs if lo <= r < arch_hi)
        if top:
            hits.append((i, top, l.strip()[:60]))
    kind = collections.Counter(h[2].split()[0] for h in hits if h[2])
    print(f"\n== peak region (instrs touching top arch VGPRs v{lo}..v{arch_hi-1}) ==")
    print(f"  {len(hits)} instrs; op kinds: {dict(kind.most_common(6))}")
    for i, top, a in hits[:8]:
        print(f"    v{top}: {a}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fp4_dump"
    f = find_isa(path)
    if not f:
        print(f"no ISA .s under {path}")
        sys.exit(1)
    print(f"ISA: {f}")
    text = open(f).read()
    arch_hi = summary(text)
    breakdown(text, arch_hi)
    peak_region(text, arch_hi)


if __name__ == "__main__":
    main()
