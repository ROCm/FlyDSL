#!/usr/bin/env python3
"""
Benchmark FlyDSL vs mori ``EpDispatchCombineOp`` for InterNodeV1LL (dispatch + combine round-trip).

Uses the same distributed + shmem setup as parity tests. Intended for 8 GPUs on one node.

**Steady-state timing (default):** ``--pre-warmup`` runs many round-trips *before* measurement to
flush FlyDSL JIT; inner ``--warmup`` is only a short micro-warmup around the CUDA timer window.

::

    cd FlyDSL
    PYTHONPATH=python:.. torchrun --standalone --nproc_per_node=8 \\
      tests/kernels/bench_dispatch_combine_internode_v1ll_flydsl_vs_mori.py \\
      --pre-warmup 24 --warmup 3 --repeat 128

Optional: also time dispatch-only (each iteration ends with ``op.reset()``)::

    ... same torchrun ... --dispatch-only

Debug (every rank prints + barriers; use to see **where** a hang happens)::

    PYTHONUNBUFFERED=1 ... --debug

Quick exit after two smoke round-trips (no timed loop)::

    ... --smoke-only

Environment:
    MORI_SHMEM_HEAP_SIZE  (default in script: 8G for quick runs; increase for large configs)
"""
from __future__ import annotations

import argparse
from datetime import datetime
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import mori.shmem as ms

from kernels.dispatch_combine_internode_v1ll_op import (
    FlyDSLDispatchCombineInterNodeV1LLConfig,
    FlyDSLDispatchCombineInterNodeV1LLOp,
)
from mori.ops.dispatch_combine import (
    EpDispatchCombineConfig,
    EpDispatchCombineKernelType,
    EpDispatchCombineOp,
)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pre-warmup",
        type=int,
        default=24,
        help="Per-path rounds before timing: run FlyDSL round-trip this many times, sync, "
        "then mori round-trip the same count (avoids interleaving which can stall). "
        "Use 0 to skip (JIT may leak into timed region). Ignored with --smoke-only.",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Micro-warmup iterations inside the timed CUDA-event window (after pre-warmup)",
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=128,
        help="Timed iterations per path (per-iter CUDA events → mean/median/stdev)",
    )
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--cur-tok", type=int, default=8, help="Tokens per rank this step")
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--max-tok", type=int, default=16)
    p.add_argument("--experts-per-rank", type=int, default=4)
    p.add_argument("--block-num", type=int, default=8)
    p.add_argument("--rdma-block-num", type=int, default=4)
    p.add_argument("--warp-per-block", type=int, default=2)
    p.add_argument("--dispatch-only", action="store_true", help="Benchmark dispatch + reset only")
    p.add_argument(
        "--debug",
        action="store_true",
        help="Per-rank phase logs + dist.barrier after each phase (hang diagnosis)",
    )
    p.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run smoke FlyDSL + mori round-trips then exit (no timed benchmark)",
    )
    p.add_argument(
        "--cold-smoke",
        action="store_true",
        help="Before pre-warmup: one cold FlyDSL + one cold mori round-trip with dispatch/combine split "
        "(for diagnostics; adds ~1s JIT noise before steady-state bench)",
    )
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def _info0(msg: str, *, rank: int) -> None:
    if rank == 0:
        print(f"[bench] {msg}", flush=True)


def _log(msg: str, *, rank: int, world_size: int, debug: bool) -> None:
    if debug:
        print(f"[bench rank={rank}/{world_size}] {msg}", flush=True)
    elif rank == 0:
        print(f"[bench] {msg}", flush=True)


def _barrier(
    tag: str,
    *,
    rank: int,
    world_size: int,
    debug: bool,
) -> None:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _log(f"sync enter: {tag}", rank=rank, world_size=world_size, debug=debug)
    dist.barrier()
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1e3
    _log(f"sync leave: {tag} ({dt_ms:.2f} ms wall)", rank=rank, world_size=world_size, debug=debug)


def _require_torchrun_8():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws != 8:
        raise SystemExit(f"Need WORLD_SIZE=8 (got {ws}). Run: torchrun --nproc_per_node=8 ...")
    if "LOCAL_RANK" not in os.environ:
        raise SystemExit("Expected torchrun (LOCAL_RANK missing)")


def _all_gather_float(rank: int, world_size: int, dev: torch.device, x: float) -> list[float]:
    t = torch.tensor([x], dtype=torch.float32, device=dev)
    out = [torch.zeros(1, dtype=torch.float32, device=dev) for _ in range(world_size)]
    dist.all_gather(out, t)
    return [float(o.item()) for o in out]


def _summarize_cross_rank(xs: list[float]) -> str:
    return f"min={min(xs):.4f}  mean={sum(xs)/len(xs):.4f}  max={max(xs):.4f}"


def _bench_cuda_ms_detailed(
    *,
    warmup: int,
    repeat: int,
    fn,
) -> dict[str, float]:
    """Per-iteration CUDA event timing after micro-warmup; returns ms stats for ``fn``."""
    # Sync after each warmup iter: without this, consecutive distributed GPU ops can reorder
    # across ranks and deadlock (warmup>=2 used to hang; timed loop always synced per iter).
    # CPU barrier too: per-process CUDA sync does not order peer ranks.
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()
        dist.barrier()
    times_ms: list[float] = []
    for _ in range(repeat):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        times_ms.append(e0.elapsed_time(e1))
    mean_ms = statistics.mean(times_ms)
    med_ms = statistics.median(times_ms)
    std_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0
    return {
        "mean": mean_ms,
        "median": med_ms,
        "stdev": std_ms,
        "min": min(times_ms),
        "max": max(times_ms),
    }


def main() -> None:
    os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", os.environ.get("MORI_SHMEM_HEAP_SIZE", "8G"))
    args = _parse_args()
    _require_torchrun_8()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device("cuda", local_rank),
    )
    import torch._C._distributed_c10d as c10d

    c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")
    dev = torch.device("cuda", local_rank)
    _barrier("after_shmem_torch_process_group_init", rank=rank, world_size=world_size, debug=args.debug)
    _info0("process_group + shmem init OK", rank=rank)

    common_kwargs = dict(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=world_size,
        hidden_dim=args.hidden_dim,
        scale_dim=0,
        scale_type_size=1,
        max_token_type_size=2,
        max_num_inp_token_per_rank=args.max_tok,
        num_experts_per_rank=args.experts_per_rank,
        num_experts_per_token=args.topk,
        warp_num_per_block=args.warp_per_block,
        block_num=args.block_num,
        max_total_recv_tokens=0,
        use_external_inp_buf=True,
        gpu_per_node=8,
        rdma_block_num=args.rdma_block_num,
        num_qp_per_pe=1,
        quant_type="none",
    )

    fly_cfg = FlyDSLDispatchCombineInterNodeV1LLConfig(**common_kwargs)
    mori_cfg = EpDispatchCombineConfig(
        **common_kwargs,
        kernel_type=EpDispatchCombineKernelType.InterNodeV1LL,
    )

    _info0("building FlyDSLDispatchCombineInterNodeV1LLOp (may JIT / alloc)...", rank=rank)
    t0 = time.perf_counter()
    fly_op = FlyDSLDispatchCombineInterNodeV1LLOp(fly_cfg, use_flydsl_copy_staging=True)
    torch.cuda.synchronize()
    _log(
        f"FlyDSL op ctor finished wall {(time.perf_counter() - t0)*1e3:.2f} ms",
        rank=rank,
        world_size=world_size,
        debug=args.debug,
    )
    _barrier("after_fly_op_ctor", rank=rank, world_size=world_size, debug=args.debug)

    _info0("building EpDispatchCombineOp (mori)...", rank=rank)
    t1 = time.perf_counter()
    mori_op = EpDispatchCombineOp(mori_cfg)
    torch.cuda.synchronize()
    _log(
        f"mori op ctor finished wall {(time.perf_counter() - t1)*1e3:.2f} ms",
        rank=rank,
        world_size=world_size,
        debug=args.debug,
    )
    _barrier("after_mori_op_ctor", rank=rank, world_size=world_size, debug=args.debug)

    g = torch.Generator(device=dev)
    g.manual_seed(args.seed + rank)
    inp = torch.randn(args.cur_tok, args.hidden_dim, dtype=torch.bfloat16, device=dev, generator=g)
    wts = torch.rand(args.cur_tok, args.topk, dtype=torch.float32, device=dev, generator=g)
    wts = wts / (wts.sum(dim=-1, keepdim=True) + 1e-6)

    pe0 = rank
    pe1 = (rank + 1) % world_size
    exp0 = pe0 * args.experts_per_rank
    exp1 = pe1 * args.experts_per_rank + 1
    idx = torch.empty(args.cur_tok, args.topk, dtype=torch.int32, device=dev)
    idx[:, 0] = exp0
    idx[:, 1] = exp1

    _barrier("inputs_ready", rank=rank, world_size=world_size, debug=args.debug)

    def fly_roundtrip():
        d = fly_op.dispatch(inp, wts, None, idx)
        fly_op.combine(d[0], None, d[3], call_reset=True)

    def mori_roundtrip():
        d = mori_op.dispatch(inp, wts, None, idx)
        mori_op.combine(d[0], None, d[3], call_reset=True)

    def fly_dispatch_only():
        fly_op.dispatch(inp, wts, None, idx)
        fly_op.reset()

    def mori_dispatch_only():
        mori_op.dispatch(inp, wts, None, idx)
        mori_op.reset()

    if args.cold_smoke:
        _log(
            "cold-smoke: FlyDSL dispatch then combine (includes JIT on first dispatch)",
            rank=rank,
            world_size=world_size,
            debug=args.debug,
        )
        t_smoke_f0 = time.perf_counter()
        d_f = fly_op.dispatch(inp, wts, None, idx)
        torch.cuda.synchronize()
        t_disp = (time.perf_counter() - t_smoke_f0) * 1e3
        fly_op.combine(d_f[0], None, d_f[3], call_reset=True)
        torch.cuda.synchronize()
        t_total = (time.perf_counter() - t_smoke_f0) * 1e3
        t_comb = t_total - t_disp
        _log(
            f"cold-smoke FlyDSL total {t_total:.2f} ms (dispatch {t_disp:.2f} + combine {t_comb:.2f})",
            rank=rank,
            world_size=world_size,
            debug=args.debug,
        )
        _barrier("after_cold_smoke_fly", rank=rank, world_size=world_size, debug=args.debug)

        _log("cold-smoke: mori dispatch then combine", rank=rank, world_size=world_size, debug=args.debug)
        t_smoke_m0 = time.perf_counter()
        d_m = mori_op.dispatch(inp, wts, None, idx)
        torch.cuda.synchronize()
        md_disp = (time.perf_counter() - t_smoke_m0) * 1e3
        mori_op.combine(d_m[0], None, d_m[3], call_reset=True)
        torch.cuda.synchronize()
        mt_total = (time.perf_counter() - t_smoke_m0) * 1e3
        md_comb = mt_total - md_disp
        _log(
            f"cold-smoke mori total {mt_total:.2f} ms (dispatch {md_disp:.2f} + combine {md_comb:.2f})",
            rank=rank,
            world_size=world_size,
            debug=args.debug,
        )
        _barrier("after_cold_smoke_mori", rank=rank, world_size=world_size, debug=args.debug)

    if args.smoke_only and not args.cold_smoke:
        _info0("smoke-only: one FlyDSL + mori round-trip each (add --cold-smoke for dispatch/combine split)", rank=rank)
        t_a = time.perf_counter()
        fly_roundtrip()
        torch.cuda.synchronize()
        tf = (time.perf_counter() - t_a) * 1e3
        t_b = time.perf_counter()
        mori_roundtrip()
        torch.cuda.synchronize()
        tm = (time.perf_counter() - t_b) * 1e3
        if rank == 0:
            print(f"[bench] smoke-only wall: FlyDSL {tf:.2f} ms, mori {tm:.2f} ms", flush=True)
        _barrier("after_smoke_only_minimal", rank=rank, world_size=world_size, debug=args.debug)

    if args.smoke_only:
        if rank == 0:
            print("[bench] --smoke-only: exiting before pre-warmup / timed loops.", flush=True)
        dist.destroy_process_group()
        return

    if args.pre_warmup > 0:
        _info0(
            f"pre-warmup: same CUDA-event path as timed bench — "
            f"FlyDSL repeat={args.pre_warmup} then mori repeat={args.pre_warmup} "
            f"(micro_warmup={max(1, args.warmup)}; stats discarded)...",
            rank=rank,
        )
        # Same helper as timed bench; warmup must sync per iter (see _bench_cuda_ms_detailed).
        t_pw = time.perf_counter()
        mw = max(1, args.warmup)
        _barrier("before_pre_warmup_fly", rank=rank, world_size=world_size, debug=args.debug)
        _ = _bench_cuda_ms_detailed(warmup=mw, repeat=args.pre_warmup, fn=fly_roundtrip)
        _barrier("after_pre_warmup_fly", rank=rank, world_size=world_size, debug=args.debug)
        _barrier("before_pre_warmup_mori", rank=rank, world_size=world_size, debug=args.debug)
        _ = _bench_cuda_ms_detailed(warmup=mw, repeat=args.pre_warmup, fn=mori_roundtrip)
        _barrier("after_pre_warmup_mori", rank=rank, world_size=world_size, debug=args.debug)
        if rank == 0:
            print(
                f"[bench] pre-warmup done: wall {((time.perf_counter() - t_pw) * 1e3):.2f} ms "
                f"(not part of score)",
                flush=True,
            )

    _log(
        f"timed FlyDSL: micro-warmup={args.warmup} repeat={args.repeat} (per-iter CUDA events)",
        rank=rank,
        world_size=world_size,
        debug=args.debug,
    )
    _barrier("before_bench_fly", rank=rank, world_size=world_size, debug=args.debug)
    fly_stats = _bench_cuda_ms_detailed(warmup=args.warmup, repeat=args.repeat, fn=fly_roundtrip)
    _log(
        f"FlyDSL this-rank: mean={fly_stats['mean']:.4f} med={fly_stats['median']:.4f} "
        f"std={fly_stats['stdev']:.4f} ms/iter",
        rank=rank,
        world_size=world_size,
        debug=args.debug,
    )
    _barrier("after_bench_fly", rank=rank, world_size=world_size, debug=args.debug)

    _log(
        f"timed mori: micro-warmup={args.warmup} repeat={args.repeat}",
        rank=rank,
        world_size=world_size,
        debug=args.debug,
    )
    _barrier("before_bench_mori", rank=rank, world_size=world_size, debug=args.debug)
    mori_stats = _bench_cuda_ms_detailed(warmup=args.warmup, repeat=args.repeat, fn=mori_roundtrip)
    _log(
        f"mori this-rank: mean={mori_stats['mean']:.4f} med={mori_stats['median']:.4f} "
        f"std={mori_stats['stdev']:.4f} ms/iter",
        rank=rank,
        world_size=world_size,
        debug=args.debug,
    )
    _barrier("after_bench_mori", rank=rank, world_size=world_size, debug=args.debug)

    fly_mean_list = _all_gather_float(rank, world_size, dev, fly_stats["mean"])
    fly_med_list = _all_gather_float(rank, world_size, dev, fly_stats["median"])
    mori_mean_list = _all_gather_float(rank, world_size, dev, mori_stats["mean"])
    mori_med_list = _all_gather_float(rank, world_size, dev, mori_stats["median"])

    lines = []
    if rank == 0:
        lines.append(
            f"InterNodeV1LL steady-state round-trip (dispatch+combine, call_reset=True)"
        )
        lines.append(
            f"  config: hidden={args.hidden_dim} cur_tok={args.cur_tok} topk={args.topk} "
            f"pre_warmup={args.pre_warmup} micro_warmup={args.warmup} repeat={args.repeat}"
        )
        lines.append(
            f"  FlyDSL  mean ms/iter (per rank): {_summarize_cross_rank(fly_mean_list)}"
        )
        lines.append(
            f"  FlyDSL  median ms/iter (per rank): {_summarize_cross_rank(fly_med_list)}"
        )
        lines.append(
            f"  mori    mean ms/iter (per rank): {_summarize_cross_rank(mori_mean_list)}"
        )
        lines.append(
            f"  mori    median ms/iter (per rank): {_summarize_cross_rank(mori_med_list)}"
        )
        ratio_mean = (sum(fly_mean_list) / len(fly_mean_list)) / max(
            sum(mori_mean_list) / len(mori_mean_list), 1e-9
        )
        lines.append(f"  ratio (FlyDSL/mori) cross-rank avg mean: {ratio_mean:.4f}x")
        lines.append(
            f"  (this rank 0 local) FlyDSL mean={fly_stats['mean']:.4f} med={fly_stats['median']:.4f} "
            f"std={fly_stats['stdev']:.4f} min={fly_stats['min']:.4f} max={fly_stats['max']:.4f}"
        )
        lines.append(
            f"  (this rank 0 local) mori   mean={mori_stats['mean']:.4f} med={mori_stats['median']:.4f} "
            f"std={mori_stats['stdev']:.4f} min={mori_stats['min']:.4f} max={mori_stats['max']:.4f}"
        )

    if args.dispatch_only:
        _barrier("before_dispatch_only_bench", rank=rank, world_size=world_size, debug=args.debug)
        fd_st = _bench_cuda_ms_detailed(warmup=args.warmup, repeat=args.repeat, fn=fly_dispatch_only)
        _barrier("after_fly_dispatch_only_bench", rank=rank, world_size=world_size, debug=args.debug)
        md_st = _bench_cuda_ms_detailed(warmup=args.warmup, repeat=args.repeat, fn=mori_dispatch_only)
        _barrier("after_mori_dispatch_only_bench", rank=rank, world_size=world_size, debug=args.debug)
        fd_m_list = _all_gather_float(rank, world_size, dev, fd_st["mean"])
        md_m_list = _all_gather_float(rank, world_size, dev, md_st["mean"])
        if rank == 0:
            lines.append("")
            lines.append(
                f"Dispatch-only (+ reset): micro_warmup={args.warmup} repeat={args.repeat}"
            )
            lines.append(f"  FlyDSL  mean ms/iter (per rank): {_summarize_cross_rank(fd_m_list)}")
            lines.append(f"  mori    mean ms/iter (per rank): {_summarize_cross_rank(md_m_list)}")
            r2 = (sum(fd_m_list) / len(fd_m_list)) / max(sum(md_m_list) / len(md_m_list), 1e-9)
            lines.append(f"  ratio (FlyDSL/mori) cross-rank avg mean: {r2:.4f}x")

    if rank == 0:
        lines.append(f"  completed_at: {datetime.now().isoformat(timespec='seconds')}")
        print("\n".join(lines))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
