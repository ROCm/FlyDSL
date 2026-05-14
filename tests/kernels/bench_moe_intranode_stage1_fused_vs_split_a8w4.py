#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Benchmark intranode A8W4 stage1: **baseline** vs **fused** vs **complete JIT**.

**Baseline (融合前)** — 与 ``kernels/moe_intranode_metadata_stage1_a8w4.py`` 描述一致:

* ``make_metadata_dispatch_recv_meta_jit``（intranode metadata 单 kernel）
* ``compile_peer_gather_mxfp8_rows``（独立 peer gather）
* ``kernels.mixed_moe_gemm_2stage.compile_mixed_moe_gemm1``（2-stage 里 **stage1 / 前半** MFMA 路径）

**Fused (融合后)** — ``compile_fused_dispatch_gather_gemm1_a8w4``：bench 内按 ``world_size > 1`` **自动**
打开 ``intranode_peer_gather``（多卡：GEMM 前内联 ``recv_meta`` + P2P 行 gather + barrier）；单卡编译关闭该前导以省延迟。
**无需**用户传 CLI 或感知单/多卡。

**Complete** — ``compile_complete_intranode_fused_stage1_a8w4``：与上面 fused 使用相同的 ``intranode_peer_gather``（bench 里由 ``FLYDSL_BENCH_FORCE_NO_PEER_GATHER`` 统一控制）。

Uses the same MoE routing / MX-FP8 + MX-FP4 tensors as ``tests/kernels/test_moe_gemm.py`` (A8W4 path).

**Multi-GPU:** after each rank clears dispatch state with GPU ``fill_``, the script calls
``torch.cuda.synchronize()`` before ``mori.shmem_barrier_all()`` so symmetric buffers are
consistent across PEs; otherwise ranks can deadlock or diverge. After FlyDSL JIT on each rank,
``torch.distributed.barrier()`` ensures no rank enters dry-run / shmem barriers while others
are still compiling (otherwise the bench can hang indefinitely).

四卡物理编号为 4–7 时，可限定可见设备::

    HIP_VISIBLE_DEVICES=4,5,6,7 MORI_SHMEM_HEAP_SIZE=16G torchrun --standalone --nproc_per_node=4 \\
        tests/kernels/bench_moe_intranode_stage1_fused_vs_split_a8w4.py \\
        --experts 8 --tokens 4 --model-dim 256 --inter-dim 128 --trace

Single process (1 GPU)::

    MORI_SHMEM_HEAP_SIZE=16G torchrun --standalone --nproc_per_node=1 \\
        tests/kernels/bench_moe_intranode_stage1_fused_vs_split_a8w4.py

Multi-GPU (one PE per GPU; each rank times its local path, rank0 prints JSON summary)::

    MORI_SHMEM_HEAP_SIZE=16G torchrun --standalone --nproc_per_node=8 \\
        tests/kernels/bench_moe_intranode_stage1_fused_vs_split_a8w4.py \\
        --experts 8 --tokens 4 --model-dim 256 --inter-dim 128

First run may spend many minutes in FlyDSL JIT; rank0 prints ``[bench]`` lines on stderr when
compile and dry-run finish. Use ``--trace`` for **every rank** monotonic timestamps around JIT,
``dist.barrier``, ``shmem_barrier_all``, and timing phases (pin which rank blocks).

Plain ``python`` (without ``torchrun``) is not supported: mori symmetric metadata dispatch requires
``torch.distributed`` + ``mori.shmem`` initialization (same as intranode tests).

If a multi-GPU run wedges (GPU kernel or barrier), children may stay alive until you kill them, e.g.
``pkill -f bench_moe_intranode_stage1_fused_vs_split_a8w4`` (or ``kill`` the ``torchrun`` parent PID).

``--skip-baseline-timing``: skip compiling/timing the baseline (peer_gather + mixed_moe_gemm1) path; baseline
ms is ``null`` in JSON. Use a **fresh** ``torchrun`` after a baseline GPU fault to still measure fused/complete/meta
(see ``tests/kernels/collect_stage1_a8w4_perf_matrix.py``).

Diagnostic: set ``FLYDSL_BENCH_FORCE_NO_PEER_GATHER=1`` to compile the fused GEMM **without** the
in-kernel peer-gather prologue (wrong semantics for multi-GPU MoE, but confirms hangs are in that
prologue when the hang disappears).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_FLYDSL_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
if _FLYDSL_ROOT not in sys.path:
    sys.path.insert(0, _FLYDSL_ROOT)

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "16G")


@dataclass
class BenchResult:
    world_size: int
    rank: int
    tokens: int
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    tile_m: int
    tile_n: int
    tile_k: int
    iters: int
    warmup: int
    # Intranode dispatch + peer_gather + mixed_moe_gemm_2stage stage1 (融合前)
    ms_baseline_meta_gather_mixed_gemm1: float
    # metadata + fused GEMM1 (world_size>1: recv_meta P2P gather in-kernel; single PE omits prologue)
    ms_fused_meta_gemm: float
    ms_complete_pipeline: float
    ms_meta_only: float


def _setup_dist(rank: int, world_size: int, master_port: int) -> int:
    import mori.shmem as ms
    import torch.distributed as dist

    if "LOCAL_RANK" not in os.environ:
        os.environ.update(
            {
                "LOCAL_RANK": str(rank),
                "RANK": str(rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(master_port),
            }
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    import torch

    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            rank=rank,
            world_size=world_size,
            device_id=dev,
        )
    import torch._C._distributed_c10d as c10d

    c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")
    return local_rank


def _cleanup() -> None:
    import mori.shmem as ms
    import torch.distributed as dist

    try:
        ms.shmem_finalize()
    except Exception:
        pass
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def _prepare_a8w4(
    *,
    dev: torch.device,
    tokens: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    seed: int,
) -> dict[str, Any]:
    import torch
    from tests.kernels.utils import fp4_utils
    from tests.utils import shuffle_weight

    import tests.kernels.test_moe_gemm as tmg

    torch.manual_seed(seed)
    x_fp32 = torch.randn((tokens, model_dim), device=dev, dtype=torch.float32)
    w1_fp32 = torch.randn((experts, 2 * inter_dim, model_dim), device=dev, dtype=torch.float32)
    topk_ids = torch.stack(
        [
            torch.arange(topk, device=dev, dtype=torch.int32) + ((t * topk) % experts)
            for t in range(tokens)
        ]
    ) % experts
    topk_weights = torch.full((tokens, topk), 1.0 / topk, device=dev, dtype=torch.float32)
    (
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        _sorted_size,
        blocks,
    ) = tmg.build_routing_buffers(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        experts=experts,
        model_dim=model_dim,
        tile_m=tile_m,
        moe_sort_mode="torch",
    )
    x_q, scale_x = tmg._per_1x32_mxfp8_quant(x_fp32)
    w1_flat = w1_fp32.view(experts * (2 * inter_dim), model_dim)
    w1_fp4, w1_scale_raw = tmg._per_1x32_fp4_quant(w1_flat)
    w1_shuffled = shuffle_weight(w1_fp4.view(torch.float4_e2m1fn_x2))
    w_kernel = w1_shuffled.view(torch.uint8).contiguous()
    scale_w1_1d = fp4_utils.e8m0_shuffle(w1_scale_raw).view(torch.uint8).contiguous()
    scale_x_1d = (
        fp4_utils.moe_mxfp4_sort(
            scale_x[:tokens, :].view(tokens, 1, -1),
            sorted_ids=sorted_token_ids,
            num_valid_ids=num_valid_ids,
            token_num=tokens,
            block_size=tile_m,
        )
        .view(torch.uint8)
        .contiguous()
    )
    x_bytes = x_q.view(torch.uint8).contiguous().view(tokens, -1)
    sorted_weights_1d = sorted_weights.contiguous().view(-1)
    out = torch.empty((tokens, topk, inter_dim), device=dev, dtype=torch.float16)
    bias_dummy = torch.empty((0,), device=dev, dtype=torch.float32)
    out_scale_dummy = torch.empty((0,), device=dev, dtype=torch.uint8)
    return {
        "x_bytes": x_bytes,
        "w_kernel": w_kernel,
        "scale_x_1d": scale_x_1d,
        "scale_w1_1d": scale_w1_1d,
        "sorted_token_ids": sorted_token_ids,
        "sorted_expert_ids": sorted_expert_ids,
        "sorted_weights_1d": sorted_weights_1d,
        "num_valid_ids": num_valid_ids,
        "out": out,
        "bias_dummy": bias_dummy,
        "out_scale_dummy": out_scale_dummy,
        "blocks": int(blocks),
    }


def _reset_dispatch_state(
    *,
    dest_pe_ctr: torch.Tensor,
    disp_bar: torch.Tensor,
    total_recv: torch.Tensor,
    dest_tok_map: torch.Tensor,
    shmem_recv_num: torch.Tensor,
    shmem_tok_off: torch.Tensor,
    shmem_tis: torch.Tensor,
    shmem_recv_meta: torch.Tensor,
    sentinel: int,
    peer_bar: torch.Tensor,
) -> None:
    dest_pe_ctr.fill_(0)
    disp_bar.fill_(0)
    total_recv.fill_(0)
    dest_tok_map.fill_(sentinel)
    shmem_recv_num.fill_(0)
    shmem_tok_off.fill_(0)
    shmem_tis.fill_(0)
    shmem_recv_meta.fill_(0)
    peer_bar.fill_(0)


def _bench_loop(
    *,
    fn,
    iters: int,
    warmup: int,
    dev: torch.device,
    trace_log: Optional[Callable[[str], None]] = None,
) -> float:
    import torch

    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    for wi in range(warmup):
        if trace_log is not None and wi == 0:
            trace_log("warmup[0] fn() enter")
        fn()
        if trace_log is not None and wi == 0:
            trace_log("warmup[0] fn() leave")
    torch.cuda.synchronize()
    e0.record()
    for ti in range(iters):
        if trace_log is not None and ti == 0:
            trace_log("timed[0] fn() enter")
        fn()
        if trace_log is not None and ti == 0:
            trace_log("timed[0] fn() leave")
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / max(1, iters)


def main() -> None:
    import mori.shmem as ms
    import torch
    import torch.distributed as dist
    import flydsl.expr as fx

    from kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm1
    from kernels.moe_fused_complete_intranode_a8w4 import (
        compile_complete_intranode_fused_stage1_a8w4,
    )
    from kernels.moe_fused_dispatch_gather_gemm1_a8w4 import (
        compile_fused_dispatch_gather_gemm1_a8w4,
    )
    from kernels.moe_metadata_dispatch_recv_meta import make_metadata_dispatch_recv_meta_jit
    from kernels.moe_peer_gather_mxfp8_a8w4 import compile_peer_gather_mxfp8_rows
    from kernels.recv_meta_a8w4 import RECV_META_ROW_BYTES

    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=4)
    p.add_argument("--model-dim", type=int, default=256)
    p.add_argument("--inter-dim", type=int, default=128)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--tile-m", type=int, default=32)
    p.add_argument("--tile-n", type=int, default=64)
    p.add_argument("--tile-k", type=int, default=256)
    p.add_argument("--max-tok-per-rank", type=int, default=0, help="metadata buffer; default=max(16,tokens)")
    p.add_argument("--dispatch-blocks", type=int, default=1)
    p.add_argument("--dispatch-wpb", type=int, default=1)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--master-port", type=int, default=29877)
    p.add_argument("--json-out", type=str, default="", help="optional path to write JSON lines (rank0 only)")
    p.add_argument(
        "--trace",
        action="store_true",
        help="stderr: every rank logs monotonic timestamps (JIT/barriers/timing); use to find hangs",
    )
    p.add_argument(
        "--skip-baseline-timing",
        action="store_true",
        help="do not compile or time baseline (peer_gather + mixed_moe_gemm1); baseline ms is null in JSON. "
        "Use a fresh torchrun after a baseline GPU fault to still measure fused/complete/meta.",
    )
    args = p.parse_args()

    if "RANK" not in os.environ:
        print(
            "ERROR: launch with torchrun, e.g.\n"
            "  MORI_SHMEM_HEAP_SIZE=16G torchrun --standalone --nproc_per_node=1 "
            f"{__file__}",
            file=sys.stderr,
        )
        raise SystemExit(2)

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    trace = bool(args.trace)
    bench_t0 = time.monotonic()

    def _info(msg: str) -> None:
        if rank != 0:
            return
        dt = time.monotonic() - bench_t0
        print(f"[bench +{dt:8.3f}s] {msg}", file=sys.stderr, flush=True)

    def _trace_evt(msg: str) -> None:
        if not trace:
            return
        dt = time.monotonic() - bench_t0
        print(
            f"[bench +{dt:8.3f}s] r{rank:2d}/{world_size} {msg}",
            file=sys.stderr,
            flush=True,
        )

    def _loop_trace(phase: str) -> Optional[Callable[[str], None]]:
        if not trace:
            return None

        def _log(msg: str) -> None:
            _trace_evt(f"{phase}: {msg}")

        return _log

    local_rank = _setup_dist(rank, world_size, int(args.master_port))
    dev = torch.device("cuda", local_rank)
    _trace_evt(f"dist+mori+cuda ready (local_rank={local_rank})")
    if rank == 0 and not trace:
        _info("tip: re-run with --trace if a rank hangs (per-rank JIT / barrier lines)")

    tokens = int(args.tokens)
    run_tokens = max(tokens, world_size)
    model_dim = int(args.model_dim)
    inter_dim = int(args.inter_dim)
    experts = int(args.experts)
    topk = int(args.topk)
    tile_m = int(args.tile_m)
    tile_n = int(args.tile_n)
    tile_k = int(args.tile_k)

    if experts % world_size != 0:
        raise SystemExit(f"experts={experts} must be divisible by world_size={world_size}")
    if model_dim % 64 != 0 or model_dim % tile_k != 0:
        raise SystemExit("require model_dim % 64 == 0 and model_dim % tile_k == 0")
    if inter_dim % tile_n != 0:
        raise SystemExit("require inter_dim % tile_n == 0")

    epr = experts // world_size
    mtpr = int(args.max_tok_per_rank) if int(args.max_tok_per_rank) > 0 else max(16, run_tokens)
    max_recv = world_size * mtpr
    sentinel = world_size * max_recv
    cur_tok = run_tokens
    block_num = int(args.dispatch_blocks)
    dwpb = int(args.dispatch_wpb)

    tensors = _prepare_a8w4(
        dev=dev,
        tokens=run_tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        # Same seed on every rank so MoE routing / padded blocks match across GPUs
        # (avoids asymmetric GEMM grids and fused intranode barrier skew).
        seed=int(args.seed),
    )
    x_bytes = tensors["x_bytes"]
    w_kernel = tensors["w_kernel"]
    sx = tensors["scale_x_1d"]
    sw = tensors["scale_w1_1d"]
    st = tensors["sorted_token_ids"]
    se = tensors["sorted_expert_ids"]
    sw1d = tensors["sorted_weights_1d"]
    nv = tensors["num_valid_ids"]
    out = tensors["out"]
    bias_d = tensors["bias_dummy"]
    osd = tensors["out_scale_dummy"]
    blocks = tensors["blocks"]

    _info("host tensors ready; building shmem + P2P tables…")
    _trace_evt("host tensors ready")

    x_flat = x_bytes.reshape(-1).contiguous()
    x_sym = ms.mori_shmem_create_tensor((int(x_flat.numel()),), torch.uint8)
    x_sym.copy_(x_flat)
    torch.cuda.synchronize()
    _trace_evt("dry x_sym: cuda sync done, enter shmem_barrier_all (post x copy)")
    ms.shmem_barrier_all()
    _trace_evt("dry x_sym: leave shmem_barrier_all")
    x_stage = x_sym.view(run_tokens, -1)

    shmem_tok_off = ms.mori_shmem_create_tensor((1,), torch.int32)
    shmem_recv_num = ms.mori_shmem_create_tensor((world_size,), torch.int32)
    shmem_tis = ms.mori_shmem_create_tensor((max_recv,), torch.int32)
    shmem_recv_meta = ms.mori_shmem_create_tensor((max_recv * RECV_META_ROW_BYTES,), torch.int8)
    _trace_evt("recv_meta shmem alloc ok, enter shmem_barrier_all")
    ms.shmem_barrier_all()
    _trace_evt("leave shmem_barrier_all (recv_meta buffers)")

    dest_pe_ctr = torch.zeros(world_size, dtype=torch.int32, device=dev)
    disp_bar = torch.zeros(1, dtype=torch.int32, device=dev)
    total_recv = torch.zeros(1, dtype=torch.int32, device=dev)
    dest_tok_map = torch.full((mtpr * topk,), sentinel, dtype=torch.int32, device=dev)
    p2p_tok_off = torch.zeros(world_size, dtype=torch.int64, device=dev)
    p2p_tis = torch.zeros(world_size, dtype=torch.int64, device=dev)
    p2p_recv_num = torch.zeros(world_size, dtype=torch.int64, device=dev)
    p2p_recv_meta = torch.zeros(world_size, dtype=torch.int64, device=dev)
    p2p_x = torch.zeros(world_size, dtype=torch.int64, device=dev)
    for pe in range(world_size):
        p2p_tok_off[pe] = ms.shmem_ptr_p2p(shmem_tok_off.data_ptr(), rank, pe)
        p2p_tis[pe] = ms.shmem_ptr_p2p(shmem_tis.data_ptr(), rank, pe)
        p2p_recv_num[pe] = ms.shmem_ptr_p2p(shmem_recv_num.data_ptr(), rank, pe)
        p2p_recv_meta[pe] = ms.shmem_ptr_p2p(shmem_recv_meta.data_ptr(), rank, pe)
        p2p_x[pe] = ms.shmem_ptr_p2p(x_stage.data_ptr(), rank, pe)

    idx = torch.zeros((mtpr, topk), dtype=torch.int32, device=dev)
    wts = torch.zeros((mtpr, topk), dtype=torch.float32, device=dev)
    for t in range(cur_tok):
        for k in range(topk):
            # All-to-all friendly: every destination PE appears in routing so metadata
            # handshake (recv_num per src_pe) does not block on a missing sender.
            slot = t * topk + k
            dest_pe = (rank + slot) % world_size
            idx[t, k] = int(dest_pe * epr + (slot % max(1, epr)))
            wts[t, k] = 1.0 / float(topk)

    _info("FlyDSL JIT: metadata_dispatch → peer_gather → mixed_moe_gemm1 → fused_gemm1 → complete …")
    _trace_evt("JIT enter make_metadata_dispatch_recv_meta_jit")
    meta = make_metadata_dispatch_recv_meta_jit(
        rank=rank,
        npes=world_size,
        experts_per_rank=epr,
        experts_per_token=topk,
        max_tok_per_rank=mtpr,
        block_num=block_num,
        warp_num_per_block=dwpb,
    )
    _trace_evt("JIT leave make_metadata_dispatch_recv_meta_jit")
    skip_bt = bool(args.skip_baseline_timing)
    if not skip_bt:
        _trace_evt("JIT enter compile_peer_gather_mxfp8_rows")
        gather = compile_peer_gather_mxfp8_rows(model_dim=model_dim, max_recv_cap=max_recv)
        _trace_evt("JIT leave compile_peer_gather_mxfp8_rows")
        # Baseline stage1 MFMA: real mixed_moe_gemm_2stage.compile_mixed_moe_gemm1 (not vendored fused wrapper).
        _trace_evt("JIT enter compile_mixed_moe_gemm1")
        mixed_moe_gemm1 = compile_mixed_moe_gemm1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage1=False,
            a_dtype="fp8",
            b_dtype="fp4",
            out_dtype="f16",
            act="silu",
        )
        _trace_evt("JIT leave compile_mixed_moe_gemm1")
    else:
        gather = None  # type: ignore[assignment]
        mixed_moe_gemm1 = None  # type: ignore[assignment]
        if rank == 0:
            _info("JIT: skipping baseline (peer_gather + mixed_moe_gemm1) per --skip-baseline-timing")
    _peer_gather_on = (world_size > 1) and os.environ.get(
        "FLYDSL_BENCH_FORCE_NO_PEER_GATHER", ""
    ).strip() != "1"
    _trace_evt("JIT enter compile_fused_dispatch_gather_gemm1_a8w4")
    gemm_fused = compile_fused_dispatch_gather_gemm1_a8w4(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=False,
        intranode_peer_gather=_peer_gather_on,
    )
    _trace_evt("JIT leave compile_fused_dispatch_gather_gemm1_a8w4")
    _trace_evt("JIT enter compile_complete_intranode_fused_stage1_a8w4")
    complete = compile_complete_intranode_fused_stage1_a8w4(
        rank=rank,
        npes=world_size,
        experts_per_rank=epr,
        experts_per_token=topk,
        max_tok_per_rank=mtpr,
        dispatch_block_num=block_num,
        dispatch_warp_num_per_block=dwpb,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=False,
        intranode_peer_gather=_peer_gather_on,
    )
    _trace_evt("JIT leave compile_complete_intranode_fused_stage1_a8w4")

    # JIT duration differs per rank; without a CPU barrier, fast ranks enter dry-run /
    # shmem_barrier_all while slow ranks are still compiling → apparent hang.
    _trace_evt("post-JIT: cuda.synchronize() enter")
    torch.cuda.synchronize()
    _trace_evt("post-JIT: cuda.synchronize() leave")
    _info("post-JIT GPU sync done; dist.barrier (wait for slowest rank JIT)…")
    _trace_evt("dist.barrier enter")
    dist.barrier()
    _trace_evt("dist.barrier leave")

    peer_bar = torch.zeros(1, dtype=torch.int32, device=dev)
    stream = fx.Stream(None)

    _info("FlyDSL JIT compile done on all ranks; dry-run metadata…")

    def _launch_meta() -> None:
        meta(
            fx.Int64(idx.data_ptr()),
            fx.Int64(wts.data_ptr()),
            fx.Int64(shmem_tok_off.data_ptr()),
            fx.Int64(shmem_recv_num.data_ptr()),
            fx.Int64(dest_pe_ctr.data_ptr()),
            fx.Int64(disp_bar.data_ptr()),
            fx.Int64(dest_tok_map.data_ptr()),
            fx.Int64(total_recv.data_ptr()),
            fx.Int64(p2p_tok_off.data_ptr()),
            fx.Int64(p2p_tis.data_ptr()),
            fx.Int64(p2p_recv_num.data_ptr()),
            fx.Int64(p2p_recv_meta.data_ptr()),
            fx.Int32(cur_tok),
            stream=stream,
        )

    def _launch_gather(tr: int) -> None:
        if gather is None:
            raise RuntimeError("internal: _launch_gather called with baseline skipped")
        gather(
            fx.Int64(shmem_recv_meta.data_ptr()),
            fx.Int64(p2p_x.data_ptr()),
            fx.Int64(x_stage.data_ptr()),
            fx.Int32(tr),
            fx.Int32(model_dim),
            fx.Int32(world_size),
            stream=stream,
        )

    def _launch_fused_gemm(tr: int) -> None:
        gemm_fused(
            out,
            x_stage,
            w_kernel,
            sx,
            sw,
            st,
            se,
            sw1d,
            nv,
            bias_d,
            osd,
            fx.Int32(run_tokens),
            fx.Int32(inter_dim * 2),
            fx.Int32(model_dim),
            fx.Int32(blocks),
            fx.Int64(shmem_recv_meta.data_ptr()),
            fx.Int64(p2p_x.data_ptr()),
            fx.Int64(x_stage.data_ptr()),
            fx.Int32(tr),
            fx.Int32(world_size),
            fx.Int64(peer_bar.data_ptr()),
            stream,
        )

    def _launch_mixed_moe_gemm1() -> None:
        if mixed_moe_gemm1 is None:
            raise RuntimeError("internal: _launch_mixed_moe_gemm1 called with baseline skipped")
        mixed_moe_gemm1(
            out,
            x_stage,
            w_kernel,
            sx,
            sw,
            st,
            se,
            sw1d,
            nv,
            bias_d,
            osd,
            fx.Int32(run_tokens),
            fx.Int32(inter_dim * 2),
            fx.Int32(model_dim),
            fx.Int32(blocks),
            stream=stream,
        )

    def _sync_tr() -> int:
        _trace_evt("_sync_tr: cuda.synchronize enter")
        torch.cuda.synchronize()
        _trace_evt("_sync_tr: shmem_barrier (pre-read total_recv) enter")
        ms.shmem_barrier_all()
        _trace_evt("_sync_tr: shmem_barrier (pre-read total_recv) leave")
        tr = int(total_recv.item())
        _trace_evt("_sync_tr: shmem_barrier (post-read total_recv) enter")
        ms.shmem_barrier_all()
        _trace_evt("_sync_tr: shmem_barrier (post-read total_recv) leave")
        return tr

    # Dry-run: establish recv row count for gather + fused GEMM (must match each iteration).
    _info("dry-run: reset + metadata launch + sync total_recv…")
    _trace_evt("dry-run: _reset_dispatch_state")
    _reset_dispatch_state(
        dest_pe_ctr=dest_pe_ctr,
        disp_bar=disp_bar,
        total_recv=total_recv,
        dest_tok_map=dest_tok_map,
        shmem_recv_num=shmem_recv_num,
        shmem_tok_off=shmem_tok_off,
        shmem_tis=shmem_tis,
        shmem_recv_meta=shmem_recv_meta,
        sentinel=sentinel,
        peer_bar=peer_bar,
    )
    _trace_evt("dry-run: cuda sync before shmem (post-reset)")
    torch.cuda.synchronize()
    _trace_evt("dry-run: shmem_barrier (post-reset) enter")
    ms.shmem_barrier_all()
    _trace_evt("dry-run: shmem_barrier (post-reset) leave")
    _trace_evt("dry-run: _launch_meta")
    _launch_meta()
    _trace_evt("dry-run: cuda sync after _launch_meta")
    torch.cuda.synchronize()
    tr_ref = _sync_tr()
    tr_use = int(tr_ref)
    _info(f"dry-run total_recv_ref={tr_ref}; timing loops (iters={args.iters}, warmup={args.warmup})…")
    _trace_evt(f"dry-run done tr_use={tr_use}")

    def run_baseline_intranode_mixed_stage1_once() -> None:
        _reset_dispatch_state(
            dest_pe_ctr=dest_pe_ctr,
            disp_bar=disp_bar,
            total_recv=total_recv,
            dest_tok_map=dest_tok_map,
            shmem_recv_num=shmem_recv_num,
            shmem_tok_off=shmem_tok_off,
            shmem_tis=shmem_tis,
            shmem_recv_meta=shmem_recv_meta,
            sentinel=sentinel,
            peer_bar=peer_bar,
        )
        torch.cuda.synchronize()
        ms.shmem_barrier_all()
        _launch_meta()
        torch.cuda.synchronize()
        _launch_gather(tr_use)
        _launch_mixed_moe_gemm1()
        torch.cuda.synchronize()
        ms.shmem_barrier_all()

    def run_fused_once() -> None:
        _reset_dispatch_state(
            dest_pe_ctr=dest_pe_ctr,
            disp_bar=disp_bar,
            total_recv=total_recv,
            dest_tok_map=dest_tok_map,
            shmem_recv_num=shmem_recv_num,
            shmem_tok_off=shmem_tok_off,
            shmem_tis=shmem_tis,
            shmem_recv_meta=shmem_recv_meta,
            sentinel=sentinel,
            peer_bar=peer_bar,
        )
        torch.cuda.synchronize()
        ms.shmem_barrier_all()
        _launch_meta()
        torch.cuda.synchronize()
        peer_bar.fill_(0)
        torch.cuda.synchronize()
        _launch_fused_gemm(tr_use)
        torch.cuda.synchronize()
        ms.shmem_barrier_all()

    def run_complete_jit_once_v2() -> None:
        _reset_dispatch_state(
            dest_pe_ctr=dest_pe_ctr,
            disp_bar=disp_bar,
            total_recv=total_recv,
            dest_tok_map=dest_tok_map,
            shmem_recv_num=shmem_recv_num,
            shmem_tok_off=shmem_tok_off,
            shmem_tis=shmem_tis,
            shmem_recv_meta=shmem_recv_meta,
            sentinel=sentinel,
            peer_bar=peer_bar,
        )
        torch.cuda.synchronize()
        ms.shmem_barrier_all()
        peer_bar.fill_(0)
        torch.cuda.synchronize()
        complete(
            fx.Int64(idx.data_ptr()),
            fx.Int64(wts.data_ptr()),
            fx.Int64(shmem_tok_off.data_ptr()),
            fx.Int64(shmem_recv_num.data_ptr()),
            fx.Int64(dest_pe_ctr.data_ptr()),
            fx.Int64(disp_bar.data_ptr()),
            fx.Int64(dest_tok_map.data_ptr()),
            fx.Int64(total_recv.data_ptr()),
            fx.Int64(p2p_tok_off.data_ptr()),
            fx.Int64(p2p_tis.data_ptr()),
            fx.Int64(p2p_recv_num.data_ptr()),
            fx.Int64(p2p_recv_meta.data_ptr()),
            fx.Int32(cur_tok),
            fx.Int64(shmem_recv_meta.data_ptr()),
            fx.Int64(p2p_x.data_ptr()),
            fx.Int64(x_stage.data_ptr()),
            fx.Int32(tr_use),
            fx.Int64(peer_bar.data_ptr()),
            out,
            x_stage,
            w_kernel,
            sx,
            sw,
            st,
            se,
            sw1d,
            nv,
            bias_d,
            osd,
            fx.Int32(run_tokens),
            fx.Int32(inter_dim * 2),
            fx.Int32(model_dim),
            fx.Int32(blocks),
            stream=stream,
        )
        torch.cuda.synchronize()
        ms.shmem_barrier_all()

    def run_meta_only() -> None:
        _reset_dispatch_state(
            dest_pe_ctr=dest_pe_ctr,
            disp_bar=disp_bar,
            total_recv=total_recv,
            dest_tok_map=dest_tok_map,
            shmem_recv_num=shmem_recv_num,
            shmem_tok_off=shmem_tok_off,
            shmem_tis=shmem_tis,
            shmem_recv_meta=shmem_recv_meta,
            sentinel=sentinel,
            peer_bar=peer_bar,
        )
        torch.cuda.synchronize()
        ms.shmem_barrier_all()
        _launch_meta()
        torch.cuda.synchronize()
        ms.shmem_barrier_all()

    def _phase_cpu_barrier(phase: str) -> None:
        """Align ranks between timing phases (avoids shmem_barrier skew / straggler desync)."""
        torch.cuda.synchronize()
        dist.barrier()
        _trace_evt(f"post-phase dist.barrier ok ({phase})")

    def _bench_phase_ms(label: str, fn_run: Callable[[], None], *, fatal: bool) -> tuple[float, bool]:
        try:
            ms = _bench_loop(
                fn=fn_run,
                iters=int(args.iters),
                warmup=int(args.warmup),
                dev=dev,
                trace_log=_loop_trace(label),
            )
            return ms, True
        except Exception as e:
            if rank == 0:
                print(f"[bench] {label} timing FAILED: {e}", file=sys.stderr, flush=True)
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            if fatal:
                raise
            return float("nan"), False

    _info("timing: baseline (meta+gather+mixed_moe_gemm1) …")
    _trace_evt("timing: baseline _bench_loop enter")
    if skip_bt:
        ms_baseline, baseline_ok = float("nan"), False
        if rank == 0:
            _info("timing: baseline skipped (--skip-baseline-timing)")
    else:
        ms_baseline, baseline_ok = _bench_phase_ms(
            "baseline", run_baseline_intranode_mixed_stage1_once, fatal=True
        )
        if baseline_ok:
            _trace_evt(f"timing: baseline _bench_loop leave ms/iter={ms_baseline:.6f}")
            _info(f"timing: baseline done ({ms_baseline:.4f} ms/iter local)")
    _phase_cpu_barrier("after_baseline")
    _info("timing: fused (meta+fused_dispatch_gather GEMM1) …")
    _trace_evt("timing: fused _bench_loop enter")
    ms_fused, fused_ok = _bench_phase_ms("fused", run_fused_once, fatal=False)
    if fused_ok:
        _trace_evt(f"timing: fused _bench_loop leave ms/iter={ms_fused:.6f}")
        _info(f"timing: fused done ({ms_fused:.4f} ms/iter local)")
    _phase_cpu_barrier("after_fused")
    _info("timing: complete JIT pipeline …")
    _trace_evt("timing: complete _bench_loop enter")
    ms_complete, complete_ok = _bench_phase_ms("complete", run_complete_jit_once_v2, fatal=False)
    if complete_ok:
        _trace_evt(f"timing: complete _bench_loop leave ms/iter={ms_complete:.6f}")
        _info(f"timing: complete done ({ms_complete:.4f} ms/iter local)")
    _phase_cpu_barrier("after_complete")
    _info("timing: meta-only …")
    _trace_evt("timing: meta_only _bench_loop enter")
    ms_meta, meta_ok = _bench_phase_ms("meta_only", run_meta_only, fatal=False)
    if meta_ok:
        _trace_evt(f"timing: meta_only _bench_loop leave ms/iter={ms_meta:.6f}")
        _info(f"timing: meta-only done ({ms_meta:.4f} ms/iter local)")
    _phase_cpu_barrier("after_meta_only")

    def _tensor_ms(ok: bool, ms: float) -> torch.Tensor:
        v = float(ms) if ok and math.isfinite(ms) else -1.0
        return torch.tensor([v], dtype=torch.float32, device=dev)

    _trace_evt("dist.all_reduce(MAX) timings enter")
    # Use -1 sentinel when a phase failed so MAX across ranks does not look like a valid time.
    t_baseline = _tensor_ms(baseline_ok, ms_baseline)
    t_fused = _tensor_ms(fused_ok, ms_fused)
    t_comp = _tensor_ms(complete_ok, ms_complete)
    t_meta = _tensor_ms(meta_ok, ms_meta)
    dist.all_reduce(t_baseline, op=dist.ReduceOp.MAX)
    dist.all_reduce(t_fused, op=dist.ReduceOp.MAX)
    dist.all_reduce(t_comp, op=dist.ReduceOp.MAX)
    dist.all_reduce(t_meta, op=dist.ReduceOp.MAX)
    _trace_evt("dist.all_reduce(MAX) timings leave")

    tb = float(t_baseline.item())
    baseline_ms_out = tb if tb >= 0.0 and math.isfinite(tb) else float("nan")
    tf = float(t_fused.item())
    fused_ms_out = tf if tf >= 0.0 and math.isfinite(tf) else float("nan")
    tc = float(t_comp.item())
    complete_ms_out = tc if tc >= 0.0 and math.isfinite(tc) else float("nan")
    tm = float(t_meta.item())
    meta_ms_out = tm if tm >= 0.0 and math.isfinite(tm) else float("nan")

    br = BenchResult(
        world_size=world_size,
        rank=rank,
        tokens=run_tokens,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        iters=int(args.iters),
        warmup=int(args.warmup),
        ms_baseline_meta_gather_mixed_gemm1=baseline_ms_out,
        ms_fused_meta_gemm=fused_ms_out,
        ms_complete_pipeline=complete_ms_out,
        ms_meta_only=meta_ms_out,
    )

    if rank == 0:
        def _fmt_ms(ms: float) -> str:
            if math.isfinite(ms):
                return f"{ms:.4f}"
            return "n/a"

        d = asdict(br)
        for k in (
            "ms_baseline_meta_gather_mixed_gemm1",
            "ms_fused_meta_gemm",
            "ms_complete_pipeline",
            "ms_meta_only",
        ):
            if not math.isfinite(float(d[k])):
                d[k] = None
        if d["ms_baseline_meta_gather_mixed_gemm1"] is None:
            d["ms_split_meta_gather_gemm"] = None
        else:
            # Deprecated JSON key (older bench used vendored fused GEMM with gather=False).
            d["ms_split_meta_gather_gemm"] = float(d["ms_baseline_meta_gather_mixed_gemm1"])
        d["note"] = (
            "Times are MAX across ranks (ms/iter). "
            "baseline = make_metadata_dispatch_recv_meta_jit + compile_peer_gather_mxfp8_rows "
            "+ mixed_moe_gemm_2stage.compile_mixed_moe_gemm1 (stage1 only); "
            "fused = metadata + compile_fused_dispatch_gather_gemm1_a8w4 "
            "(intranode_peer_gather when world_size>1); "
            "complete = single FlyDSL JIT meta+fused GEMM1."
        )
        d["total_recv_ref"] = int(tr_ref)
        d["skip_baseline_timing"] = bool(skip_bt)
        line = json.dumps(d, separators=(",", ":"))
        print(line)
        bl_s = _fmt_ms(br.ms_baseline_meta_gather_mixed_gemm1)
        fu_s = _fmt_ms(br.ms_fused_meta_gemm)
        co_s = _fmt_ms(br.ms_complete_pipeline)
        me_s = _fmt_ms(br.ms_meta_only)
        sp_s = (
            f"{br.ms_baseline_meta_gather_mixed_gemm1 / max(br.ms_fused_meta_gemm, 1e-9):.3f}x"
            if math.isfinite(br.ms_baseline_meta_gather_mixed_gemm1)
            and math.isfinite(br.ms_fused_meta_gemm)
            else "n/a"
        )
        print(
            f"\nSummary (max across {world_size} ranks, ms/iter, iters={args.iters}):\n"
            f"  meta_only                                      {me_s}\n"
            f"  baseline (meta+gather+mixed_moe_gemm1 stage1) {bl_s}\n"
            f"  fused (meta+fused_dispatch_gather GEMM1)     {fu_s}\n"
            f"  complete JIT                                   {co_s}\n"
            f"  speedup baseline/fused                         {sp_s}\n"
        )
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                f.write(line + "\n")

    _cleanup()


if __name__ == "__main__":
    main()
