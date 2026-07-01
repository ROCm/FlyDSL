#!/usr/bin/env python3
"""Parse FlyDSL benchmark output into normalized dashboard records.

The CI job (``scripts/run_benchmark.sh`` driven from ``.github/workflows/flydsl.yaml``)
emits two things we care about, both to stdout / the job log:

1. A raw results table -- the *current* run::

       op                     shape                              dtype            TB/s     TFLOPS
       softmax                32768x8192                         bf16            3.938          -
       gemm                   5120x5120x8320_tile64x256x128      fp8             0.430   1361.400

2. Two comparison blocks emitted by ``scripts/compare_benchmark.py``::

       === Benchmark: current vs main ===
         layernorm 32768x8192 bf16  main=  4.270 TB/s  current=  3.971 TB/s  ratio= 0.930x  delta= -0.299 ( -7.0%)
       === Benchmark: current vs v0.2.0 ===
         layernorm 32768x8192 bf16  v0.2.0=2.326 TB/s  current=  3.971 TB/s  ratio= 1.707x  delta= +1.645 (+70.7%)

When read back from a GitHub Actions job log each line is prefixed with an ISO-8601
timestamp and may contain ANSI colour codes; both are stripped here.

This module is intentionally dependency-free (stdlib only) so it runs identically in
CI, on a local MI350X box, and in unit tests.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from typing import Optional

# Default regression gate: a current-vs-main drop of this many percent (or worse)
# flags the row red. Both metrics we track (TB/s, TFLOPS) are bigger-is-better.
DEFAULT_REGRESSION_PCT = -3.0

# Stable runner -> GPU arch map (the only durable identity of each CI box).
RUNNER_ARCH = {
    "linux-flydsl-mi355-1": "gfx950",  # MI355 / MI350X
    "linux-flydsl-mi355-8": "gfx950",
    "linux-flydsl-mi325-1": "gfx942",  # MI325
    "linux-flydsl-mi325-8": "gfx942",
    "linux-flydsl-navi-2": "gfx1201",  # Navi4x (RDNA)
}

_TS_PREFIX = re.compile(r"^\S*?\d{4}-\d{2}-\d{2}T[\d:.]+Z\s")
_ANSI = re.compile(r"\x1b\[[0-9;]*m")

_HEADER = re.compile(r"^op\s+shape\s+dtype\s+TB/s\s+TFLOPS\s*$")
# A raw current-table row: 5 whitespace columns, metric cells are number | "-" | skip/missing.
_NUMCELL = r"(?:-?\d+(?:\.\d+)?|-|skip|missing|n/?a)"
_RAW_ROW = re.compile(
    rf"^(?P<op>\S+)\s+(?P<shape>\S+)\s+(?P<dtype>\S+)\s+" rf"(?P<tbps>{_NUMCELL})\s+(?P<tflops>{_NUMCELL})\s*$"
)

# The "Perf Compare (gpu us): FlyDSL vs AIter" sweep (run_compare_sweep / RDNA WMMA sweep):
#   op         shape              dtype  FlyDSL(gpu us)  AIter(gpu us)    speedup
#   wmma_gemm  256x256x256        bf16             26.3            9.8      2.69x
_AITER_HEADER = re.compile(r"^op\s+shape\s+dtype\s+FlyDSL\(gpu us\)\s+AIter\(gpu us\)\s+speedup\s*$")
_AITER_CELL = r"(?:[\d,]+(?:\.\d+)?|-|n/?a)"
_AITER_ROW = re.compile(
    rf"^(?P<op>\S+)\s+(?P<shape>\S+)\s+(?P<dtype>\S+)\s+"
    rf"(?P<fly>{_AITER_CELL})\s+(?P<ait>{_AITER_CELL})\s+(?P<sp>[\d.]+x|-)\s*$"
)

_CMP_HEADER = re.compile(r"^=== Benchmark: current vs (?P<base>\S+) ===\s*$")
# A comparison row. <base>= may be "main", a fallback like "main~2", or a tag like "v0.2.0";
# dtype may be e.g. "int4_bf16". The label allows ~ so the main~N fallback baselines parse.
_CMP_ROW = re.compile(
    r"^\s*(?P<op>\S+)\s+(?P<shape>\S+)\s+(?P<dtype>\S+)\s+"
    r"(?P<blabel>[\w][\w.~\-]*)=\s*(?P<base_val>-?\d+(?:\.\d+)?)\s+(?P<unit>TB/s|TFLOPS)\s+"
    r"current=\s*(?P<cur>-?\d+(?:\.\d+)?)\s+(?:TB/s|TFLOPS)\s+"
    r"ratio=\s*(?P<ratio>-?\d+(?:\.\d+)?)x\s+"
    r"delta=\s*(?P<delta>[-+]?\d+(?:\.\d+)?)\s+\(\s*(?P<pct>[-+]?\d+(?:\.\d+)?)%\)\s*$"
)


def clean(line: str) -> str:
    """Strip the Actions-log timestamp prefix and ANSI colour codes."""
    line = _ANSI.sub("", line)
    line = _TS_PREFIX.sub("", line)
    return line.rstrip("\n")


def _num(cell: str) -> Optional[float]:
    """Parse a numeric cell to float, or None if it isn't one.

    Thousands separators are tolerated so the AIter sweep's ``1,264.1`` cells
    parse the same way as the plain ``3.938`` cells in the raw results table.
    """
    try:
        return float(cell.replace(",", ""))
    except ValueError:
        return None


@dataclass
class Baseline:
    label: str  # "main" or the tag, e.g. "v0.2.0"
    baseline: float
    ratio: float
    delta_pct: float


@dataclass
class Record:
    op: str
    shape: str
    dtype: str
    metric: str  # "TB/s" or "TFLOPS"
    value: Optional[float]  # current value (None if skipped)
    status: str = "ok"  # ok | skip | missing
    vs_main: Optional[dict] = None
    vs_tag: Optional[dict] = None
    regression: bool = False
    extra: Optional[dict] = None  # e.g. {"flydsl_us":.., "aiter_us":..} for speedup rows
    # run-level metadata, filled by attach_meta()
    ts: Optional[str] = None
    commit: Optional[str] = None
    pr: Optional[int] = None
    run_id: Optional[int] = None
    source: str = "ci"
    runner: Optional[str] = None
    arch: Optional[str] = None


def _key(op: str, shape: str, dtype: str) -> tuple:
    return (op, shape, dtype)


def parse_log(text: str, regression_pct: float = DEFAULT_REGRESSION_PCT) -> list[Record]:
    """Parse one benchmark log (raw text, with or without log prefixes) into records.

    The *first* raw table is treated as the current run (a later table, if present, is
    the rebuilt baseline and is ignored -- the comparison blocks already carry it).
    """
    lines = [clean(line) for line in text.splitlines()]

    # --- pass 1: first raw current-results table ---
    records: dict[tuple, Record] = {}
    in_table = False
    seen_table = False
    for ln in lines:
        if _HEADER.match(ln):
            if seen_table:
                in_table = False  # second table = baseline rebuild; stop collecting
                continue
            in_table = True
            seen_table = True
            continue
        if in_table:
            m = _RAW_ROW.match(ln)
            if not m:
                if ln.strip() and not ln.lstrip().startswith("-"):
                    in_table = False  # left the table
                continue
            op, shape, dtype = m["op"], m["shape"], m["dtype"]
            tbps, tflops = _num(m["tbps"]), _num(m["tflops"])
            # Compute ops (gemm/moe/flash_attn/mla) report TFLOPS as their headline metric;
            # memory ops (softmax/layernorm/rmsnorm) leave TFLOPS as "-" and report TB/s.
            if tflops is not None:
                metric, value = "TFLOPS", tflops
            elif tbps is not None:
                metric, value = "TB/s", tbps
            else:
                metric, value = "TFLOPS", None
            status = "ok" if value is not None else ("skip" if "skip" in (m["tbps"], m["tflops"]) else "missing")
            records[_key(op, shape, dtype)] = Record(
                op=op, shape=shape, dtype=dtype, metric=metric, value=value, status=status
            )

    # --- pass 2: comparison blocks (current vs main / vs tag) ---
    current_base: Optional[str] = None
    for ln in lines:
        h = _CMP_HEADER.match(ln)
        if h:
            current_base = h["base"]  # "main" or a tag
            continue
        if current_base is None:
            continue
        m = _CMP_ROW.match(ln)
        if not m:
            continue
        op, shape, dtype = m["op"], m["shape"], m["dtype"]
        unit, cur = m["unit"], float(m["cur"])
        bl = Baseline(
            label=m["blabel"], baseline=float(m["base_val"]), ratio=float(m["ratio"]), delta_pct=float(m["pct"])
        )
        rec = records.get(_key(op, shape, dtype))
        if rec is None:
            rec = Record(op=op, shape=shape, dtype=dtype, metric=unit, value=cur, status="ok")
            records[_key(op, shape, dtype)] = rec
        # The comparison block fixes the metric (unit) and supplies the current
        # value only when the raw table didn't already have one. When the raw
        # table did report a value it wins (raw and comparison are emitted from
        # the same run, so they agree on the number).
        rec.metric = unit
        if rec.value is None:
            rec.value = cur
        # "main" and the "main~N" fallbacks (flydsl.yaml walks back when main won't build)
        # are all the main baseline; anything else (v-tags) is the tag baseline.
        is_main = current_base.startswith("main") or bl.label.startswith("main")
        if is_main:
            # label records which main was used ("main" or a "main~N" fallback) so the UI
            # can show what the comparison was actually against.
            rec.vs_main = {"label": bl.label, "baseline": bl.baseline, "ratio": bl.ratio, "delta_pct": bl.delta_pct}
            rec.regression = bl.delta_pct <= regression_pct
        else:
            rec.vs_tag = {"tag": current_base, "baseline": bl.baseline, "ratio": bl.ratio, "delta_pct": bl.delta_pct}

    return list(records.values())


def parse_aiter_compare(text: str) -> list[Record]:
    """Parse every 'FlyDSL vs AIter' compare table into speedup records.

    Produces one record per (op, shape, dtype) with ``metric="speedup"``,
    ``value`` = FlyDSL/AIter speedup (>1 means FlyDSL wins), and the raw latencies
    under ``extra``. Only the first table (the current run) is kept; the table is
    re-emitted for the main/tag baseline rebuilds and those repeats are ignored.
    """
    lines = [clean(line) for line in text.splitlines()]
    out: dict[tuple, Record] = {}
    in_table = False
    seen = False  # the table is re-emitted for the main/tag baseline rebuilds; keep only the first (current run)
    for ln in lines:
        if _AITER_HEADER.match(ln):
            if seen:
                in_table = False
                continue
            in_table = True
            seen = True
            continue
        if not in_table:
            continue
        m = _AITER_ROW.match(ln)
        if not m:
            if ln.strip() and not ln.lstrip().startswith("-") and not ln.lstrip().startswith("="):
                in_table = False
            continue
        sp = None if m["sp"] == "-" else float(m["sp"].rstrip("x"))
        out[_key(m["op"], m["shape"], m["dtype"])] = Record(
            op=m["op"],
            shape=m["shape"],
            dtype=m["dtype"],
            metric="speedup",
            value=sp,
            status="ok" if sp is not None else "missing",
            extra={"flydsl_us": _num(m["fly"]), "aiter_us": _num(m["ait"]), "baseline": "aiter"},
        )
    return list(out.values())


def attach_meta(
    records: list[Record],
    *,
    runner: str,
    commit: str | None = None,
    pr: int | None = None,
    run_id: int | None = None,
    ts: str | None = None,
    source: str = "ci",
    arch: str | None = None,
) -> list[dict]:
    """Stamp run-level metadata onto records and return them as plain dicts."""
    arch = arch or RUNNER_ARCH.get(runner, "unknown")
    out = []
    for r in records:
        r.runner, r.arch, r.commit, r.pr, r.run_id, r.ts, r.source = (runner, arch, commit, pr, run_id, ts, source)
        out.append(asdict(r))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Parse a FlyDSL benchmark log into dashboard records.")
    ap.add_argument("log", help="path to a benchmark log (use - for stdin)")
    ap.add_argument("--runner", required=True, help="CI runner name, e.g. linux-flydsl-mi355-1")
    ap.add_argument("--arch", help="override GPU arch (else inferred from --runner)")
    ap.add_argument("--commit")
    ap.add_argument("--pr", type=int)
    ap.add_argument("--run-id", type=int)
    ap.add_argument("--ts", help="ISO-8601 timestamp for this run")
    ap.add_argument("--source", default="ci", help='record source tag (default "ci")')
    ap.add_argument("--regression-pct", type=float, default=DEFAULT_REGRESSION_PCT)
    ap.add_argument("--no-aiter", action="store_true", help="skip the FlyDSL-vs-AIter speedup table")
    ap.add_argument("--out", help="write JSON array here (default stdout)")
    args = ap.parse_args(argv)

    text = sys.stdin.read() if args.log == "-" else open(args.log, encoding="utf-8", errors="replace").read()
    recs = parse_log(text, regression_pct=args.regression_pct)
    if not args.no_aiter:
        recs += parse_aiter_compare(text)
    dicts = attach_meta(
        recs,
        runner=args.runner,
        commit=args.commit,
        pr=args.pr,
        run_id=args.run_id,
        ts=args.ts,
        source=args.source,
        arch=args.arch,
    )
    payload = json.dumps(dicts, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(payload)
        print(f"wrote {len(dicts)} records -> {args.out}", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
