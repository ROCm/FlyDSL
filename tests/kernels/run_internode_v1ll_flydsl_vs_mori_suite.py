#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""
一键串联 Internode V1 LL（FlyDSL vs mori）的精度与性能测试。

目的
  - 功能/精度：dispatch only、combine only、dispatch+combine（与 mori golden 逐 rank 比对）
  - 性能：dispatch、combine、e2e 三段独立计时，输出与 mori 同口径的 us/iter 及比值

运行环境（单机多卡，常用两张物理卡）
  默认将 ``CUDA_VISIBLE_DEVICES`` / ``HIP_VISIBLE_DEVICES`` 设为 ``6,7``（可通过
  ``--device-pair`` 或环境变量 ``DEVICE_PAIR`` 覆盖）。ROCm 下须同时设 HIP，否则
  可见设备与 PyTorch 子进程不一致。

用法::

  cd /path/to/FlyDSL
  python tests/kernels/run_internode_v1ll_flydsl_vs_mori_suite.py

  # 快速冒烟（更少 bench 迭代）
  python tests/kernels/run_internode_v1ll_flydsl_vs_mori_suite.py --quick

  # 只跑精度或只跑性能
  python tests/kernels/run_internode_v1ll_flydsl_vs_mori_suite.py --skip-perf
  python tests/kernels/run_internode_v1ll_flydsl_vs_mori_suite.py --skip-correctness

  # 按场景分组：dispatch →（精度+性能）→ combine →（精度+性能）→ e2e →（精度+性能）
  python tests/kernels/run_internode_v1ll_flydsl_vs_mori_suite.py --by-scope

  # 列出将要执行的命令（不执行）
  python tests/kernels/run_internode_v1ll_flydsl_vs_mori_suite.py --dry-run

底层单次调用等价于::

  CUDA_VISIBLE_DEVICES=6,7 MORI_SHMEM_HEAP_SIZE=16G \\
    torchrun --standalone --nproc_per_node=2 \\
    tests/kernels/test_internode_v1ll_dispatch_flydsl.py [args]

实现细节见 ``tests/kernels/test_internode_v1ll_dispatch_flydsl.py``（mori 配置
``InterNodeV1LL``、block_num=256、rdma_block_num=128、``gpu_per_node=1`` 与 2-rank
等价于 2 节点拓扑）。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    extra: List[str]


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def _test_script() -> str:
    return os.path.join(_repo_root(), "tests/kernels", "test_internode_v1ll_dispatch_flydsl.py")


def _apply_device_pair(pair: str) -> None:
    p = pair.strip()
    if p:
        os.environ["CUDA_VISIBLE_DEVICES"] = p
        os.environ["HIP_VISIBLE_DEVICES"] = p


def _run_one(
    *,
    dry_run: bool,
    extra_env: dict,
    forward_args: List[str],
    scenario: Scenario,
) -> int:
    root = _repo_root()
    script = _test_script()
    env = os.environ.copy()
    env.setdefault("MORI_SHMEM_HEAP_SIZE", "16G")
    if os.environ.get("FLYDSL_SUITE_TRACE", "0").lower() not in ("1", "true", "yes"):
        env.setdefault("FLYDSL_TRACE_INTERNODE", "0")
    env.update(extra_env)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        script,
        *scenario.extra,
        *forward_args,
    ]
    print("\n" + "=" * 88, flush=True)
    print(f"[suite] scenario: {scenario.name}", flush=True)
    print(f"        {scenario.description}", flush=True)
    print(f"        cmd: {' '.join(cmd)}", flush=True)
    print("=" * 88 + "\n", flush=True)
    if dry_run:
        return 0
    proc = subprocess.run(
        cmd,
        cwd=root,
        env=env,
        text=True,
    )
    return int(proc.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="Internode V1 LL FlyDSL vs mori 套件（2×GPU torchrun）")
    ap.add_argument(
        "--device-pair",
        default=os.environ.get("DEVICE_PAIR", "6,7"),
        help="物理 GPU 序号，写入 CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES（默认 6,7；空字符串表示不改）",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="更少 bench 迭代、沿用较小默认形状（仍可被显式参数覆盖）",
    )
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    ap.add_argument("--skip-correctness", action="store_true", help="跳过三段精度测试")
    ap.add_argument("--skip-perf", action="store_true", help="跳过三段性能测试")
    ap.add_argument(
        "--by-scope",
        action="store_true",
        help=(
            "按 dispatch / combine / e2e 分组：每个 scope 先跑精度再跑性能（默认顺序为三段精度后三段性能）"
        ),
    )
    ap.add_argument(
        "--mori-shmem-gb",
        type=int,
        default=8,
        help=(
            "MORI_SHMEM_HEAP_SIZE（GiB）；默认 8。对称堆为单块 hipMalloc，需留约 1.5GiB 余量给 "
            "PyTorch/NCCL/JIT；低于 6 在满卡或碎片重时 combine 段易出现 Memory access fault，建议 6–8。"
        ),
    )
    args, forward = ap.parse_known_args()

    dp = (args.device_pair or "").strip()
    if dp:
        _apply_device_pair(dp)
    if int(args.mori_shmem_gb) < 6:
        print(
            "[suite] warning: --mori-shmem-gb < 6 can fault on busy GPUs (need contiguous VRAM for "
            "heap + headroom). Prefer 6–8 GiB if you see low-address GPU page faults in combine.",
            flush=True,
        )
    os.environ["MORI_SHMEM_HEAP_SIZE"] = f"{int(args.mori_shmem_gb)}G"

    if args.quick:
        forward_base = ["--bench-warmup", "8", "--bench-iters", "64"]
    else:
        forward_base = ["--bench-warmup", "20", "--bench-iters", "200"]

    # User forward overrides (e.g. --hidden 4096 --n-tok 32) append after forward_base
    forward_all = forward_base + list(forward)

    correctness = [
        Scenario(
            "corr_dispatch",
            "精度：仅 dispatch（tok/wts/idx vs mori，行序无关排序后比对）",
            ["--check", "dispatch", "--no-combine"],
        ),
        Scenario(
            "corr_combine",
            "精度：仅 combine 输出（要求 total_recv 与 mori 一致）",
            ["--check", "combine"],
        ),
        Scenario(
            "corr_both",
            "精度：dispatch + combine",
            ["--check", "both"],
        ),
    ]
    perf = [
        Scenario(
            "perf_dispatch",
            "性能：mori dispatch vs FlyDSL reset+copy+main",
            ["--bench", "--bench-scope", "dispatch"],
        ),
        Scenario(
            "perf_combine",
            "性能：每轮 redispatch（不计时）后仅测 combine 段",
            ["--bench", "--bench-scope", "combine"],
        ),
        Scenario(
            "perf_e2e",
            "性能：单步 dispatch+combine 整段",
            ["--bench", "--bench-scope", "e2e"],
        ),
    ]

    todo: List[Scenario] = []
    if args.by_scope:
        for c, p in zip(correctness, perf):
            if not args.skip_correctness:
                todo.append(c)
            if not args.skip_perf:
                todo.append(p)
    else:
        if not args.skip_correctness:
            todo.extend(correctness)
        if not args.skip_perf:
            todo.extend(perf)
    if not todo:
        print("Nothing to run: both correctness and perf skipped.", file=sys.stderr)
        sys.exit(2)

    print(
        f"[suite] repo={_repo_root()}\n"
        f"[suite] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')!r} "
        f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES','')!r}\n"
        f"[suite] MORI_SHMEM_HEAP_SIZE={os.environ.get('MORI_SHMEM_HEAP_SIZE','')!r}\n"
        f"[suite] scenarios={len(todo)} dry_run={args.dry_run}",
        flush=True,
    )

    failed: Optional[str] = None
    for sc in todo:
        code = _run_one(
            dry_run=args.dry_run,
            extra_env={},
            forward_args=forward_all,
            scenario=sc,
        )
        if code != 0 and failed is None:
            failed = sc.name
        if code != 0 and not args.dry_run:
            print(f"[suite] FAILED: {sc.name} (exit {code})", flush=True)
            sys.exit(code)

    if args.dry_run:
        print("[suite] dry-run finished.")
        return

    print(
        "\n"
        + "=" * 88
        + "\n[suite] All selected scenarios finished OK.\n"
        + "=" * 88
        + "\n"
        "提示：性能数字在子进程 stdout 的 ``[bench scope=...]`` 行；"
        "若需完整日志可单独 torchrun 单一场景或对子进程重定向到文件。\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
