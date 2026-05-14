#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-shot collector: stage1 bench (baseline + fused + complete + meta) × scenarios × world sizes.

If a full run exits non-zero or prints no JSON line, retries the same shape with
``--skip-baseline-timing`` (fresh process) so fused/complete/meta can still be recorded.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

SCENARIOS: list[tuple[str, list[str]]] = [
    ("S_default", []),
    (
        "M_batch",
        [
            "--tokens",
            "64",
            "--model-dim",
            "512",
            "--inter-dim",
            "256",
            "--experts",
            "8",
            "--topk",
            "2",
            "--tile-m",
            "32",
            "--tile-n",
            "64",
            "--tile-k",
            "256",
        ],
    ),
    (
        "D_decode",
        [
            "--tokens",
            "1",
            "--model-dim",
            "512",
            "--inter-dim",
            "256",
            "--experts",
            "8",
            "--topk",
            "2",
            "--tile-m",
            "32",
            "--tile-n",
            "64",
            "--tile-k",
            "256",
        ],
    ),
    (
        "LARGE_M_FP4benchM",
        [
            "--tokens",
            "512",
            "--model-dim",
            "4096",
            "--inter-dim",
            "256",
            "--experts",
            "32",
            "--topk",
            "8",
            "--tile-m",
            "64",
            "--tile-n",
            "64",
            "--tile-k",
            "256",
        ],
    ),
    (
        "LARGE_L_FP4benchL",
        [
            "--tokens",
            "1024",
            "--model-dim",
            "4096",
            "--inter-dim",
            "256",
            "--experts",
            "32",
            "--topk",
            "8",
            "--tile-m",
            "128",
            "--tile-n",
            "64",
            "--tile-k",
            "256",
        ],
    ),
    (
        "WIDE64",
        [
            "--tokens",
            "64",
            "--model-dim",
            "4096",
            "--inter-dim",
            "256",
            "--experts",
            "32",
            "--topk",
            "8",
            "--tile-m",
            "64",
            "--tile-n",
            "64",
            "--tile-k",
            "256",
        ],
    ),
    (
        "WIDE192",
        [
            "--tokens",
            "192",
            "--model-dim",
            "4096",
            "--inter-dim",
            "256",
            "--experts",
            "32",
            "--topk",
            "8",
            "--tile-m",
            "64",
            "--tile-n",
            "64",
            "--tile-k",
            "256",
        ],
    ),
    (
        "WIDE256",
        [
            "--tokens",
            "256",
            "--model-dim",
            "4096",
            "--inter-dim",
            "256",
            "--experts",
            "32",
            "--topk",
            "8",
            "--tile-m",
            "64",
            "--tile-n",
            "64",
            "--tile-k",
            "256",
        ],
    ),
    (
        "WIDE384",
        [
            "--tokens",
            "384",
            "--model-dim",
            "4096",
            "--inter-dim",
            "256",
            "--experts",
            "32",
            "--topk",
            "8",
            "--tile-m",
            "64",
            "--tile-n",
            "64",
            "--tile-k",
            "256",
        ],
    ),
    (
        "WIDE512",
        [
            "--tokens",
            "512",
            "--model-dim",
            "4096",
            "--inter-dim",
            "256",
            "--experts",
            "32",
            "--topk",
            "8",
            "--tile-m",
            "64",
            "--tile-n",
            "64",
            "--tile-k",
            "256",
        ],
    ),
]


def _parse_json_line(stdout: str) -> dict[str, Any] | None:
    for line in stdout.splitlines():
        if line.startswith("{") and '"world_size"' in line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return None
    return None


def _run_bench(
    *,
    nproc: int,
    extra: list[str],
    skip_baseline: bool,
    master_port: int,
    iters: int,
    warmup: int,
) -> tuple[int, dict[str, Any] | None, str]:
    bench = os.path.join(_ROOT, "tests/kernels/bench_moe_intranode_stage1_fused_vs_split_a8w4.py")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        bench,
        "--iters",
        str(iters),
        "--warmup",
        str(warmup),
        "--master-port",
        str(master_port),
    ] + list(extra)
    if skip_baseline:
        cmd.append("--skip-baseline-timing")
    env = os.environ.copy()
    env.setdefault("MORI_SHMEM_HEAP_SIZE", "16G")
    p = subprocess.run(
        cmd,
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
    )
    js = _parse_json_line(p.stdout)
    return p.returncode, js, p.stderr


def main() -> None:
    iters = int(os.environ.get("COLLECT_ITERS", "6"))
    warmup = int(os.environ.get("COLLECT_WARMUP", "3"))
    out_path = os.environ.get("COLLECT_OUT", os.path.join(_ROOT, "stage1_a8w4_perf_matrix.jsonl"))
    rows: list[dict[str, Any]] = []
    mp = 29877
    for scen, extra in SCENARIOS:
        for nproc in (1, 4, 8):
            if nproc > 8:
                continue
            epr = 0
            # experts from extra: default 8 for S_default
            ex = 8
            for i, a in enumerate(extra):
                if a == "--experts" and i + 1 < len(extra):
                    ex = int(extra[i + 1])
            if ex % nproc != 0:
                rec = {
                    "scenario": scen,
                    "world_size": nproc,
                    "skipped": True,
                    "reason": f"experts={ex} not divisible by world_size={nproc}",
                }
                rows.append(rec)
                print(json.dumps(rec, separators=(",", ":")))
                continue
            mp += 1
            tag = f"{scen}_ws{nproc}"
            print(f"# {tag} full …", file=sys.stderr, flush=True)
            rc, js, err = _run_bench(
                nproc=nproc,
                extra=extra,
                skip_baseline=False,
                master_port=mp,
                iters=iters,
                warmup=warmup,
            )
            mode = "full"
            if rc != 0 or js is None:
                print(f"# {tag} full failed rc={rc}, retry skip-baseline …", file=sys.stderr, flush=True)
                mp += 1
                rc2, js2, err2 = _run_bench(
                    nproc=nproc,
                    extra=extra,
                    skip_baseline=True,
                    master_port=mp,
                    iters=iters,
                    warmup=warmup,
                )
                mode = "skip_baseline_retry"
                rc, js, err = rc2, js2, err2
            if js is None:
                rec = {
                    "scenario": scen,
                    "world_size": nproc,
                    "collect_mode": mode,
                    "error": "no_json",
                    "returncode": rc,
                    "stderr_tail": (err or "")[-4000:],
                }
            else:
                rec = dict(js)
                rec["collect_scenario"] = scen
                rec["collect_mode"] = mode
                rec["collect_returncode"] = rc
            rows.append(rec)
            print(json.dumps(rec, separators=(",", ":")))
            sys.stdout.flush()

    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    print(f"\nWrote {len(rows)} lines to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
