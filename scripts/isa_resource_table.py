#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Summarize per-kernel ISA resource usage from a FLYDSL_DUMP_IR tree.

Delta table for codegen changes that alter address arithmetic: run a kernel
suite twice (before and after), dumping to two directories, then diff.  A
functional pass/fail does not surface a silent VGPR or spill regression, so
these numbers have to be compared explicitly.

    FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=<dir> python3 -m pytest <cases>
    python3 isa_resource_table.py <dir> --json out.json
    python3 isa_resource_table.py --diff before.json after.json

One dump directory can hold several kernels in a single ISA file, so entries are
keyed `<dump dir>::<kernel>` and every metric is scoped to that kernel: the
register counts come from that kernel's metadata entry and the instruction
counts from that kernel's body only.

--diff exits non-zero on a regression, and also whenever the comparison itself
is untrustworthy -- no kernels, a kernel on only one side, or a metric that
failed to parse.  A gate that cannot tell "no regressions" from "no data" is
worse than no gate.
"""

import argparse
import json
import os
import re
import sys

KEYS = ("vgpr", "sgpr", "spill", "scratch_store", "scratch_load", "ds_read")
WORSE_IF_UP = ("vgpr", "spill", "scratch_store", "scratch_load")

# Kernel entries in the amdhsa.kernels list start at exactly two spaces; nested
# .args entries are indented further.
_ENTRY_SPLIT = re.compile(r"^  - ", re.M)


def _int_field(text, name):
    m = re.search(r"\.%s:\s*(\d+)" % name, text)
    return int(m.group(1)) if m else None


def _kernel_body(text, name, others):
    """Instruction text of one kernel: its label up to the end of its body.

    The body ends at whichever comes first: the next `.Lfunc_endN:` or the label
    of another kernel.  Bounding by the label too means the result never depends
    on `.Lfunc_end` numbering lining up with declaration order.

    Returns None when no terminator is found.  Falling back to "rest of file"
    would silently fold the following kernels' instructions into this one's
    counts -- a wrong number rather than a visible failure.
    """
    m = re.search(r"^%s:\s*$" % re.escape(name), text, re.M)
    if not m:
        return None
    rest = text[m.end() :]

    ends = []
    fn_end = re.search(r"^\.Lfunc_end\d+:", rest, re.M)
    if fn_end:
        ends.append(fn_end.start())
    for other in others:
        nxt = re.search(r"^%s:\s*$" % re.escape(other), rest, re.M)
        if nxt:
            ends.append(nxt.start())
    return rest[: min(ends)] if ends else None


def parse_isa(path):
    with open(path) as fh:
        text = fh.read()

    _, _, meta = text.partition("amdhsa.kernels:")
    meta = meta.split(".end_amdgpu_metadata")[0]

    names, chunks = [], []
    for chunk in _ENTRY_SPLIT.split(meta)[1:]:
        m = re.search(r"\.name:\s*(\S+)", chunk)
        if not m:
            continue
        names.append(m.group(1))
        chunks.append(chunk)

    out = {}
    for name, chunk in zip(names, chunks):
        body = _kernel_body(text, name, [n for n in names if n != name])
        out[name] = {
            "vgpr": _int_field(chunk, "vgpr_count"),
            "sgpr": _int_field(chunk, "sgpr_count"),
            "spill": _int_field(chunk, "vgpr_spill_count"),
            # None rather than 0 when the body is missing, so a parse failure is
            # reported instead of silently looking like a clean kernel.
            "scratch_store": body.count("scratch_store") if body is not None else None,
            "scratch_load": body.count("scratch_load") if body is not None else None,
            "ds_read": body.count("ds_read") if body is not None else None,
        }
    return out


def collect(root):
    out = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith("final_isa.s"):
                mod = os.path.basename(dirpath)
                for kern, metrics in parse_isa(os.path.join(dirpath, f)).items():
                    # The dump dir is usually named after its only kernel; keep
                    # the qualifier only where it actually disambiguates.
                    out[kern if kern == mod else "%s::%s" % (mod, kern)] = metrics
    return out


def do_diff(before_path, after_path):
    with open(before_path) as fh:
        before = json.load(fh)
    with open(after_path) as fh:
        after = json.load(fh)

    problems = []
    if not before or not after:
        problems.append("one or both inputs contain no kernels (before=%d, after=%d)" % (len(before), len(after)))

    names = sorted(set(before) | set(after))
    hdr = "%-52s " % "kernel" + " ".join("%15s" % k for k in KEYS)
    print(hdr)
    print("-" * len(hdr))
    worsened = improved = unchanged = 0
    for n in names:
        b, x = before.get(n), after.get(n)
        if b is None or x is None:
            side = "ONLY IN BEFORE" if x is None else "ONLY IN AFTER"
            print("%-52s  %s" % (n, side))
            problems.append("%s: %s" % (n, side))
            continue
        missing = [k for k in KEYS if b.get(k) is None or x.get(k) is None]
        if missing:
            print("%-52s  UNPARSED: %s" % (n, ",".join(missing)))
            problems.append("%s: unparsed metrics %s" % (n, ",".join(missing)))
            continue
        cells, changed = [], False
        for k in KEYS:
            bv, xv = b[k], x[k]
            if bv == xv:
                cells.append("%15s" % xv)
                continue
            changed = True
            d = xv - bv
            if k in WORSE_IF_UP:
                if d > 0:
                    worsened += 1
                else:
                    improved += 1
            cells.append(("%s->%s(%+d)" % (bv, xv, d)).rjust(15))
        if changed:
            print("%-52s " % n + " ".join(cells))
        else:
            unchanged += 1

    print(
        "\n%d kernels compared; %d unchanged; worsened metrics: %d; improved: %d"
        % (len(names), unchanged, worsened, improved)
    )
    if problems:
        print("\nCOMPARISON NOT TRUSTWORTHY (%d problem(s)):" % len(problems), file=sys.stderr)
        for p in problems[:20]:
            print("  " + p, file=sys.stderr)
        return 2
    return 1 if worsened else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", nargs="?")
    ap.add_argument("--json")
    ap.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"))
    a = ap.parse_args()

    if a.diff:
        return do_diff(a.diff[0], a.diff[1])

    if not a.dump_dir:
        ap.error("dump_dir required unless --diff is used")
    data = collect(a.dump_dir)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
    hdr = "%-52s " % "kernel" + " ".join("%8s" % k for k in KEYS)
    print(hdr)
    print("-" * len(hdr))
    for n in sorted(data):
        print("%-52s " % n + " ".join("%8s" % data[n].get(k) for k in KEYS))
    print("\n%d kernels" % len(data))
    return 0 if data else 2


if __name__ == "__main__":
    sys.exit(main())
