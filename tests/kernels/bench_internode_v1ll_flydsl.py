#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Internode V1 LL — FlyDSL vs mori **性能**专用入口（2×GPU ``torchrun``）。

对标 intranode 的 ``tests/kernels/test_profiler_dispatch_combine.py``（bench 模式）：本脚本只做
**计时与 mori 对比**，默认**不跑** golden 精度校验（底层仍用同一 ``test_internode_v1ll_dispatch_flydsl.py``）。

典型用法（单机 8 卡常用物理 6、7；ROCm 须同时设 ``HIP_VISIBLE_DEVICES``）::

    cd FlyDSL
    CUDA_VISIBLE_DEVICES=6,7 HIP_VISIBLE_DEVICES=6,7 \\
      python tests/kernels/bench_internode_v1ll_flydsl.py

    # 只测 dispatch / combine / e2e 之一
    python tests/kernels/bench_internode_v1ll_flydsl.py --scope dispatch

    # 快速冒烟（更少 warmup/iters）
    python tests/kernels/bench_internode_v1ll_flydsl.py --quick

    # 只测 FlyDSL（跳过 mori 计时）
    python tests/kernels/bench_internode_v1ll_flydsl.py --skip-mori

    # 透传给底层测试（须放在 ``--`` 之后，避免与本脚本参数冲突）
    python tests/kernels/bench_internode_v1ll_flydsl.py -- --hidden 4096 --n-tok 32

环境提示
  - ``FLYDSL_TRACE_INTERNODE=0``（默认由本脚本设置）避免 bench 时 barrier 串行打印拖死。
  - 对称堆：由环境 ``MORI_SHMEM_HEAP_SIZE`` 决定；本脚本**不设默认 8G**。需要固定大小时可传 ``--mori-shmem-gb N``。
  - dispatch 段 FlyDSL 默认与 mori 同口径：计时循环内**仅** copy+main（warmup 前一次 reset）。**顺序**：先 FlyDSL bench 再 mori bench（避免 mori 压测后 FlyDSL 首帧 ROCm fault）。若要旧版「每轮 Python 全量清零」计时：``FLYDSL_BENCH_DISPATCH_INCLUDE_RESET=1``。
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def _test_script() -> str:
    return os.path.join(_repo_root(), "tests/kernels", "test_internode_v1ll_dispatch_flydsl.py")


def _apply_device_pair(pair: str) -> None:
    p = pair.strip()
    if p:
        os.environ["CUDA_VISIBLE_DEVICES"] = p
        os.environ["HIP_VISIBLE_DEVICES"] = p


@dataclass
class BenchExtract:
    scope: str
    mori_us: Optional[float]
    flydsl_us: Optional[float]
    ratio: Optional[float]


def _parse_bench_lines(text: str) -> Dict[str, BenchExtract]:
    """Parse rank0 ``[bench scope=*]`` lines from subprocess output."""
    # [bench scope=dispatch] mori EpDispatch...: 53.99 us/iter per GPU
    mori_re = re.compile(
        r"\[bench scope=(\w+)\]\s*mori[^:]+:\s*([\d,]+(?:\.\d+)?)\s*us/iter",
    )
    fly_re = re.compile(
        r"\[bench scope=(\w+)\]\s*FlyDSL[^:]+:\s*([\d,]+(?:\.\d+)?)\s*us/iter",
    )
    ratio_re = re.compile(
        r"\[bench scope=(\w+)\]\s*ratio FlyDSL/mori:\s*([\d,]+(?:\.\d+)?)x",
    )

    def fnum(s: str) -> float:
        return float(s.replace(",", ""))

    out: Dict[str, BenchExtract] = {}
    for m in mori_re.finditer(text):
        sc = m.group(1)
        out.setdefault(sc, BenchExtract(sc, None, None, None)).mori_us = fnum(m.group(2))
    for m in fly_re.finditer(text):
        sc = m.group(1)
        out.setdefault(sc, BenchExtract(sc, None, None, None)).flydsl_us = fnum(m.group(2))
    for m in ratio_re.finditer(text):
        sc = m.group(1)
        out.setdefault(sc, BenchExtract(sc, None, None, None)).ratio = fnum(m.group(2))
    return out


def _run_torchrun(
    *,
    bench_scope: str,
    forward: List[str],
    env: dict,
    dry_run: bool,
    flydsl_only: bool,
) -> Tuple[int, str]:
    root = _repo_root()
    script = _test_script()
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        script,
        *([] if not flydsl_only else ["--flydsl-only"]),
        "--bench",
        "--bench-scope",
        bench_scope,
        *forward,
    ]
    print("\n" + "=" * 88, flush=True)
    print(f"[bench_internode_v1ll] scope={bench_scope}", flush=True)
    print(f"cmd: {' '.join(cmd)}", flush=True)
    print("=" * 88 + "\n", flush=True)
    if dry_run:
        return 0, ""
    proc = subprocess.Popen(
        cmd,
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    proc.wait()
    return int(proc.returncode), "".join(lines)


def _print_summary(rows: Dict[str, BenchExtract]) -> None:
    if not rows:
        print("\n[bench_internode_v1ll] (no [bench scope=...] lines parsed; try without --skip-mori)", flush=True)
        return
    print("\n" + "=" * 88, flush=True)
    print("Internode V1 LL bench summary (us/iter per GPU, rank0 parsed)")
    print("=" * 88)
    print(f"{'scope':<10s} {'mori(us)':>14s} {'FlyDSL(us)':>14s} {'FlyDSL/mori':>12s}")
    for sc in sorted(rows.keys()):
        r = rows[sc]
        m = "-" if r.mori_us is None else f"{r.mori_us:,.2f}"
        f = "-" if r.flydsl_us is None else f"{r.flydsl_us:,.2f}"
        x = "-" if r.ratio is None else f"{r.ratio:,.3f}x"
        print(f"{sc:<10s} {m:>14s} {f:>14s} {x:>12s}")
    print("=" * 88 + "\n", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Internode V1 LL FlyDSL perf vs mori (torchrun ×2 GPUs, --bench only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Extra args after -- are forwarded to test_internode_v1ll_dispatch_flydsl.py",
    )
    ap.add_argument(
        "--scope",
        choices=("all", "dispatch", "combine", "e2e"),
        default="all",
        help="Benchmark segment(s): all = dispatch, combine, e2e in order",
    )
    ap.add_argument(
        "--device-pair",
        default=os.environ.get("DEVICE_PAIR", "6,7"),
        help="Set CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES (default 6,7; empty = do not set)",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Shorter warmup/iters (8 / 64)",
    )
    ap.add_argument(
        "--skip-mori",
        action="store_true",
        help="Only time FlyDSL (--flydsl-only); summary table will miss mori/ratio lines",
    )
    ap.add_argument(
        "--mori-shmem-gb",
        type=int,
        default=None,
        metavar="N",
        help="If set, export MORI_SHMEM_HEAP_SIZE=N G (overrides env). Otherwise do not touch env.",
    )
    ap.add_argument("--bench-warmup", type=int, default=None)
    ap.add_argument("--bench-iters", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args, forward = ap.parse_known_args()

    # Allow `script.py -- --hidden 4096` (strip leading `--`)
    if forward and forward[0] == "--":
        forward = forward[1:]

    dp = (args.device_pair or "").strip()
    if dp:
        _apply_device_pair(dp)

    if args.mori_shmem_gb is not None:
        os.environ["MORI_SHMEM_HEAP_SIZE"] = f"{int(args.mori_shmem_gb)}G"

    env = os.environ.copy()
    if os.environ.get("FLYDSL_BENCH_TRACE", "0").lower() not in ("1", "true", "yes"):
        env.setdefault("FLYDSL_TRACE_INTERNODE", "0")

    if args.quick:
        warm, iters = 8, 64
    else:
        warm, iters = 20, 200
    if args.bench_warmup is not None:
        warm = args.bench_warmup
    if args.bench_iters is not None:
        iters = args.bench_iters

    base_forward = ["--bench-warmup", str(warm), "--bench-iters", str(iters), *forward]

    scopes = ["dispatch", "combine", "e2e"] if args.scope == "all" else [args.scope]

    print(
        f"[bench_internode_v1ll] repo={_repo_root()}\n"
        f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')!r} "
        f"HIP_VISIBLE_DEVICES={env.get('HIP_VISIBLE_DEVICES', '')!r}\n"
        f"MORI_SHMEM_HEAP_SIZE={env.get('MORI_SHMEM_HEAP_SIZE', '(unset)')!r}\n"
        f"scopes={scopes} warmup={warm} iters={iters} skip_mori={args.skip_mori}",
        flush=True,
    )

    merged = ""
    failed: Optional[str] = None
    for sc in scopes:
        code, text = _run_torchrun(
            bench_scope=sc,
            forward=base_forward,
            env=env,
            dry_run=args.dry_run,
            flydsl_only=args.skip_mori,
        )
        merged += text
        if code != 0 and failed is None:
            failed = sc
        if code != 0 and not args.dry_run:
            print(f"[bench_internode_v1ll] FAILED scope={sc} exit={code}", flush=True)
            sys.exit(code)

    if args.dry_run:
        return

    _print_summary(_parse_bench_lines(merged))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
