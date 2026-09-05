#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Select a verifiably idle GPU and describe it.

A single instantaneous activity reading is not evidence of an idle device: a
neighbouring job between kernel launches reads 0% just as an unused device does. This
samples over a window and requires every sample to stay under the threshold, which is
the difference between a benchmark number that means something and one that does not.

Emits the HIP-visible index on stdout. Details go to stderr so the caller can use
``HIP_VISIBLE_DEVICES=$(pick_idle_gpu.py)`` directly.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time


def _amd_smi(args: list[str]) -> str:
    exe = shutil.which("amd-smi")
    if not exe:
        raise RuntimeError("amd-smi not found")
    return subprocess.run([exe, *args], check=True, capture_output=True, text=True).stdout


def enumerate_gpus() -> list[dict]:
    """Map AMD SMI enumeration to BDF, so a HIP index can be translated deterministically."""
    out = _amd_smi(["list", "--json"])
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        data = []
    gpus = []
    for entry in data:
        gpus.append(
            {
                "smi_index": entry.get("gpu"),
                "bdf": entry.get("bdf"),
                "uuid": entry.get("uuid"),
            }
        )
    if not gpus:  # older amd-smi builds without --json
        for block in re.split(r"\nGPU:\s*", "\n" + _amd_smi(["list"])):
            m_idx = re.match(r"\s*(\d+)", block)
            m_bdf = re.search(r"BDF:\s*(\S+)", block)
            if m_idx and m_bdf:
                gpus.append({"smi_index": int(m_idx.group(1)), "bdf": m_bdf.group(1), "uuid": None})
    return gpus


def _activity(smi_index: int) -> float | None:
    try:
        out = _amd_smi(["metric", "-g", str(smi_index), "-u"])
    except Exception:
        return None
    m = re.search(r"GFX_ACTIVITY:\s*(\d+)\s*%", out)
    return float(m.group(1)) if m else None


def _vram_used_mb(smi_index: int) -> float | None:
    try:
        out = _amd_smi(["metric", "-g", str(smi_index), "-m"])
    except Exception:
        return None
    m = re.search(r"(?:USED_VRAM|TOTAL_VRAM_USED):\s*(\d+)", out)
    return float(m.group(1)) if m else None


def sample_idle(gpus: list[dict], samples: int, interval: float, max_activity: float) -> list[dict]:
    """Every sample must stay under the threshold; one busy sample disqualifies a device."""
    traces: dict[int, list[float]] = {g["smi_index"]: [] for g in gpus}
    for i in range(samples):
        for g in gpus:
            a = _activity(g["smi_index"])
            traces[g["smi_index"]].append(-1.0 if a is None else a)
        if i + 1 < samples:
            time.sleep(interval)

    out = []
    for g in gpus:
        t = traces[g["smi_index"]]
        readable = [x for x in t if x >= 0]
        out.append(
            {
                **g,
                "activity_samples": t,
                "activity_max": max(readable) if readable else None,
                "vram_used_mb": _vram_used_mb(g["smi_index"]),
                "idle": bool(readable) and max(readable) <= max_activity,
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--max-activity", type=float, default=2.0, help="percent GFX activity allowed in every sample")
    ap.add_argument("--json", action="store_true", help="emit the full record instead of just the index")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    try:
        gpus = enumerate_gpus()
    except Exception as exc:
        print(f"cannot enumerate GPUs: {exc}", file=sys.stderr)
        return 2
    if not gpus:
        print("no GPUs reported by amd-smi", file=sys.stderr)
        return 2

    records = sample_idle(gpus, args.samples, args.interval, args.max_activity)
    idle = [r for r in records if r["idle"]]
    if not args.quiet:
        for r in records:
            print(
                f"  smi{r['smi_index']} bdf={r['bdf']} activity_max={r['activity_max']}% idle={r['idle']}",
                file=sys.stderr,
            )
    if not idle:
        print("no GPU stayed idle across the whole sampling window", file=sys.stderr)
        return 1

    idle.sort(key=lambda r: (r["activity_max"] if r["activity_max"] is not None else 1e9, r["smi_index"]))
    chosen = idle[0]
    if args.json:
        print(json.dumps(chosen, indent=2))
    else:
        print(chosen["smi_index"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
