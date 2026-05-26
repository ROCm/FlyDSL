"""Mori-parity launch-time dtype perf sweep driver.

Sweep matrix (mori-parity, mixed dispatch / combine dtype):

  - ``max_tokens``        in {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}
  - ``dispatch_dtype``    in {fp4, fp8_ocp}
  - ``combine_dtype``     = ``bf16`` (fixed)
  - ``quant_type``        = ``none`` (no fp8_direct_cast)
  - ``use_external_inp_buf`` in {True, False}     (P2P-read toggle)
  - ``block_num``         in {64, 128, 256}
  - ``warp_per_block``    in {4, 8, 16}

11 * 2 * 2 * 3 * 3 = 396 cases.  Each case is timed in ``--mode bench
--warmup 5 --iters 5 --compare-mori`` (eager, NOT cudagraph: cudagraph
merges dispatch + combine into a single ``[GPU] dispatch+combine
(event)`` table and we want them broken out separately; ``--compare-mori``
adds a mori head-to-head section in the output).  Bandwidth uses mori's
algo-BW formula reported by ``bench_op`` on rank 0 for both impls.  Failed cases are recorded
with ``failed=True`` (no fly_disp_us / fly_comb_us numbers) so the
markdown still has the case listed.

Output:
  - ``dispatch_combine_mixed_dtype_sweep.md`` (markdown report, four
    sections grouped by ``(dispatch_dtype, p2p_read)``; each section
    contains a table indexed by ``(max_tokens, block_num, warp_per_block)``).
  - ``dispatch_combine_mixed_dtype_sweep.jsonl`` (raw per-case data,
    one JSON record per line, for ad-hoc post-processing).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEST_SCRIPT = os.path.join(ROOT, "tests", "kernels", "test_profiler_dispatch_combine.py")


BS_LIST = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
DISPATCH_DTYPES = ["fp4", "fp8_ocp"]
COMBINE_DTYPE = "bf16"
USE_EXT_LIST = [True, False]
BLOCK_NUMS = [64, 128, 256]
WARP_PER_BLOCKS = [4, 8, 16]

HIDDEN_DIM_FOR_DTYPE = {
    "fp4": 3584,
    "fp8_ocp": 7168,
    "bf16": 7168,
}

_BENCH_LINE_RE = re.compile(
    r"\[E2E\]\s+(?P<which>dispatch|combine)\s+CUDA\s+time\s*"
    r"\s+(?P<avg>[\d\.eE+-]+)"
    r"\s+(?P<min>[\d\.eE+-]+)"
    r"\s+(?P<max>[\d\.eE+-]+)"
    r"\s+(?P<bw>[\d\.eE+-]+)"
)
_TOTAL_RECV_RE = re.compile(r"per-rank\s+total_recv\s*=\s*(\d+)")


def _parse_bench_output(stdout: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    cur_section: dict[str, Any] | None = None
    for line in stdout.splitlines():
        # ``[bench] flydsl warmup`` or ``[bench] mori timing``.
        if line.lstrip().startswith("[bench] "):
            for tag in ("flydsl", "mori"):
                if line.lstrip().startswith(f"[bench] {tag} "):
                    cur_section = out.setdefault(tag, {})
                    break
        m = _BENCH_LINE_RE.search(line)
        if m is None or cur_section is None:
            continue
        rec = {
            "avg_us": float(m.group("avg")),
            "min_us": float(m.group("min")),
            "max_us": float(m.group("max")),
            "bw_gbps": float(m.group("bw")),
        }
        cur_section[m.group("which")] = rec
    tr = _TOTAL_RECV_RE.search(stdout)
    if tr:
        out["total_recv"] = int(tr.group(1))
    return out


def _build_cmd(args: argparse.Namespace, case: dict[str, Any]) -> list[str]:
    cmd = [
        sys.executable,
        TEST_SCRIPT,
        "--mode",
        "bench",
        "--compare-mori",
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--max-tokens",
        str(case["max_tokens"]),
        "--hidden-dim",
        str(case["hidden_dim"]),
        "--dtype",
        case["dispatch_dtype"],
        "--combine-dtype",
        case["combine_dtype"],
        "--num-experts-per-rank",
        str(args.num_experts_per_rank),
        "--k",
        str(args.k),
        "--block-num",
        str(case["block_num"]),
        "--warp-per-block",
        str(case["warp_per_block"]),
        "--quant-type",
        "none",
        "--output-dir",
        args.output_subdir,
    ]
    if not case["use_external_inp_buf"]:
        cmd.append("--no-external-inp-buf")
    return cmd


def _case_id(case: dict[str, Any]) -> str:
    return (
        f"bs{case['max_tokens']}_{case['dispatch_dtype']}_to_{case['combine_dtype']}"
        f"_p2p{int(not case['use_external_inp_buf'])}"
        f"_b{case['block_num']}_w{case['warp_per_block']}"
    )


def _all_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for bs in BS_LIST:
        for d in DISPATCH_DTYPES:
            for use_ext in USE_EXT_LIST:
                for bn in BLOCK_NUMS:
                    for wpb in WARP_PER_BLOCKS:
                        cases.append(
                            {
                                "max_tokens": bs,
                                "hidden_dim": HIDDEN_DIM_FOR_DTYPE[d],
                                "dispatch_dtype": d,
                                "combine_dtype": COMBINE_DTYPE,
                                "use_external_inp_buf": use_ext,
                                "block_num": bn,
                                "warp_per_block": wpb,
                            }
                        )
    return cases


def _section_key(case: dict[str, Any]) -> tuple[str, bool]:
    return (case["dispatch_dtype"], not case["use_external_inp_buf"])


def _fmt_us(rec: dict[str, Any], section: str, which: str) -> str:
    sect = rec.get(section, {})
    if not sect or which not in sect:
        return "n/a"
    return f"{sect[which]['avg_us']:.1f}"


def _fmt_bw(rec: dict[str, Any], section: str, which: str) -> str:
    sect = rec.get(section, {})
    if not sect or which not in sect:
        return "n/a"
    return f"{sect[which]['bw_gbps']:.1f}"


def _speedup(rec: dict[str, Any], which: str) -> str:
    """``mori_us / fly_us`` -- >1 means FlyDSL is faster than mori."""
    fly = rec.get("flydsl", {}).get(which, {})
    mori = rec.get("mori", {}).get(which, {})
    if not fly or not mori:
        return "n/a"
    fly_us = fly.get("avg_us", 0.0)
    mori_us = mori.get("avg_us", 0.0)
    if fly_us <= 0.0:
        return "n/a"
    return f"{mori_us / fly_us:.2f}x"


def _write_markdown(md_path: str, results: list[dict[str, Any]]) -> None:
    sections: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for rec in results:
        sections.setdefault(_section_key(rec["case"]), []).append(rec)

    n_total = len(results)
    n_failed = sum(1 for r in results if r.get("failed"))
    n_ok = n_total - n_failed
    n_with_mori = sum(1 for r in results if not r.get("failed") and r.get("mori", {}).get("dispatch") is not None)

    with open(md_path, "w") as fh:
        fh.write("# Dispatch / Combine mixed-dtype perf sweep (mori-parity)\n\n")
        fh.write(
            "Per the launch-time dtype JIT cache (mori-parity), this sweep "
            "exercises every ``(max_tokens, dispatch_dtype, p2p_read, "
            "block_num, warp_per_block)`` combination with ``combine_dtype = "
            "bf16`` and ``quant_type = none``, head-to-head FlyDSL vs mori. "
            "Times are in microseconds, averaged across all 8 ranks; "
            "bandwidth is per-rank algo-BW (mori formula) in GB/s.  "
            "``fly_d_x`` / ``fly_c_x`` columns are ``mori_us / fly_us`` -- "
            "values > 1.00x mean FlyDSL is faster.\n\n"
        )
        fh.write("Test configuration\n\n")
        fh.write("- world_size = 8 (EP=8)\n")
        fh.write("- k = 8\n")
        fh.write("- num_experts_per_rank = 32\n")
        fh.write("- max_token_type_size = auto (max of dispatch/combine elem size)\n")
        fh.write("- launch: ``--mode bench --warmup 5 --iters 5 --compare-mori`` (eager)\n\n")
        fh.write(
            f"Total cases: {n_total}; OK: {n_ok}; Failed: {n_failed}; " f"with mori head-to-head: {n_with_mori}\n\n"
        )

        for d_dtype, p2p_read in sorted(sections.keys()):
            tag = f"{d_dtype} dispatch -> bf16 combine, P2P-read = {p2p_read}"
            recs = sections[(d_dtype, p2p_read)]
            recs.sort(
                key=lambda r: (
                    r["case"]["max_tokens"],
                    r["case"]["block_num"],
                    r["case"]["warp_per_block"],
                )
            )
            fh.write(f"## {tag}\n\n")
            fh.write(
                "| max_tokens | block_num | warp_per_block "
                "| fly_d_us | mori_d_us | fly_d_x "
                "| fly_c_us | mori_c_us | fly_c_x "
                "| fly_d_GB/s | mori_d_GB/s | fly_c_GB/s | mori_c_GB/s "
                "| total_recv | status |\n"
            )
            fh.write(
                "|-----------:|----------:|---------------:"
                "|---------:|----------:|--------:"
                "|---------:|----------:|--------:"
                "|-----------:|------------:|-----------:|------------:"
                "|-----------:|:------|\n"
            )
            for r in recs:
                c = r["case"]
                if r.get("failed"):
                    fh.write(
                        f"| {c['max_tokens']} | {c['block_num']} | "
                        f"{c['warp_per_block']} | -- | -- | -- | -- | -- | -- "
                        f"| -- | -- | -- | -- | -- | "
                        f"FAIL ({r.get('reason', 'unknown')}) |\n"
                    )
                    continue
                tr = r.get("total_recv", 0)
                fh.write(
                    f"| {c['max_tokens']} | {c['block_num']} | "
                    f"{c['warp_per_block']} "
                    f"| {_fmt_us(r, 'flydsl', 'dispatch')} "
                    f"| {_fmt_us(r, 'mori', 'dispatch')} "
                    f"| {_speedup(r, 'dispatch')} "
                    f"| {_fmt_us(r, 'flydsl', 'combine')} "
                    f"| {_fmt_us(r, 'mori', 'combine')} "
                    f"| {_speedup(r, 'combine')} "
                    f"| {_fmt_bw(r, 'flydsl', 'dispatch')} "
                    f"| {_fmt_bw(r, 'mori', 'dispatch')} "
                    f"| {_fmt_bw(r, 'flydsl', 'combine')} "
                    f"| {_fmt_bw(r, 'mori', 'combine')} "
                    f"| {tr} | OK |\n"
                )
            fh.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--num-experts-per-rank", type=int, default=32)
    ap.add_argument(
        "--output-md",
        default=os.path.join(ROOT, "dispatch_combine_mixed_dtype_sweep.md"),
    )
    ap.add_argument(
        "--output-jsonl",
        default=os.path.join(ROOT, "dispatch_combine_mixed_dtype_sweep.jsonl"),
    )
    ap.add_argument(
        "--output-subdir",
        default=os.path.join(ROOT, "dispatch_profile_perf_sweep"),
        help="--output-dir forwarded to each launch (trace JSONs go here).",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-case subprocess timeout in seconds.",
    )
    ap.add_argument(
        "--per-case-log-dir",
        default=os.path.join(ROOT, "dispatch_combine_mixed_dtype_sweep_logs"),
    )
    ap.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Resume by skipping the first N cases (useful when a sub-batch crashed).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run at most N cases (0 = unlimited).",
    )
    args = ap.parse_args()

    os.makedirs(args.per_case_log_dir, exist_ok=True)

    cases = _all_cases()
    if args.start_index > 0:
        cases = cases[args.start_index :]
    if args.limit > 0:
        cases = cases[: args.limit]

    results: list[dict[str, Any]] = []
    # If the jsonl exists, load prior runs so resumed sweeps keep history.
    if os.path.exists(args.output_jsonl):
        with open(args.output_jsonl) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    t0 = time.time()
    for i, case in enumerate(cases):
        cid = _case_id(case)
        idx = args.start_index + i
        n_total = args.start_index + len(cases)
        cmd = _build_cmd(args, case)
        log_path = os.path.join(args.per_case_log_dir, f"{idx:04d}_{cid}.log")
        elapsed_so_far = time.time() - t0
        print(
            f"[sweep {idx + 1}/{n_total}] {cid}  " f"(elapsed {elapsed_so_far:.0f}s)",
            flush=True,
        )
        rec: dict[str, Any] = {"case": case, "case_id": cid}
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                env={**os.environ},
                cwd=ROOT,
            )
            stdout = proc.stdout
            stderr = proc.stderr
            with open(log_path, "w") as fh:
                fh.write(f"# cmd: {' '.join(cmd)}\n# exit={proc.returncode}\n\n")
                fh.write(stdout)
                if stderr:
                    fh.write("\n----- stderr -----\n")
                    fh.write(stderr)
            if proc.returncode != 0:
                rec["failed"] = True
                rec["reason"] = f"exit={proc.returncode}"
            else:
                parsed = _parse_bench_output(stdout)
                rec.update(parsed)
                if "flydsl" not in parsed or "dispatch" not in parsed.get("flydsl", {}):
                    rec["failed"] = True
                    rec["reason"] = "parse-failure"
        except subprocess.TimeoutExpired:
            rec["failed"] = True
            rec["reason"] = "timeout"
            with open(log_path, "w") as fh:
                fh.write(f"# cmd: {' '.join(cmd)}\n# TIMEOUT after {args.timeout}s\n")
        results.append(rec)
        with open(args.output_jsonl, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        _write_markdown(args.output_md, results)

    print(
        f"[sweep] done {len(cases)} cases in {time.time() - t0:.0f}s; " f"markdown -> {args.output_md}",
        flush=True,
    )


if __name__ == "__main__":
    main()
