#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Minimal EAGER-loop driver for ATT tracing the emajor megastage1 kernel (8-rank).

Builds ExpS1 with FUSED_MEGA_EMAJOR=1 (+ optional SPLIT), runs forward N times eagerly (no
CUDAGraph, so rocprofv3 kernel_iteration_range can pick individual dispatches).  Low-memory
(local weights only, ~1GB).  Env: TRACE_ITERS (default 8), plus FUSED_MEGA_* knobs.

  FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 FUSED_MEGA_EMAJOR=1 rocprofv3 -i in.yaml -- \
    torchrun --standalone --nproc_per_node=8 megamoeexp/trace_drv.py
"""

from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import tests.kernels.test_moe_gemm as tmg  # noqa: E402

try:
    import aiter.ops.quant as _aq  # noqa: E402

    if not hasattr(_aq, "per_1x32_mx_quant_hip"):

        def _shim(x, quant_dtype=None, scale_type=None):
            t = str(quant_dtype).lower()
            return (tmg._per_1x32_fp4_quant if ("fp4" in t or "f4" in t) else tmg._per_1x32_mxfp8_quant)(x.contiguous())

        _aq.per_1x32_mx_quant_hip = _shim
except Exception:  # noqa: BLE001
    pass

import mori.shmem as ms  # noqa: E402

from megamoeexp.bench_megamoeexp import NETWORKS, _qact, _setup_dist  # noqa: E402
from tests.kernels.utils import fp4_utils  # noqa: E402
from tests.utils import shuffle_weight  # noqa: E402


def main():
    net_name = os.environ.get("PROBE_NET", "v4_pro")
    quant = os.environ.get("PROBE_QUANT", "a8w4")
    tokens = int(os.environ.get("PROBE_TOKENS", "64"))
    iters = int(os.environ.get("TRACE_ITERS", "8"))

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = _setup_dist(rank, world, 29981)
    dev = torch.device("cuda", local_rank)

    net = NETWORKS[net_name]
    model_dim, inter_dim, experts = net["model_dim"], net["inter_dim"], net["experts"]
    topk = int(net["topk"])
    epr = experts // world
    mtpr = 16
    while mtpr < tokens:
        mtpr <<= 1
    is_fp4 = quant == "a4w4"

    init_scale = float(model_dim) ** -0.25
    _wfp4 = torch.randint(0, 256, (epr * (2 * inter_dim), model_dim // 2), dtype=torch.uint8, device=dev)
    w1 = shuffle_weight(_wfp4.view(torch.float4_e2m1fn_x2)).view(torch.uint8).contiguous()
    _wsc = torch.randint(0, 256, (epr * (2 * inter_dim), model_dim // 32), dtype=torch.uint8, device=dev)
    w1s = fp4_utils.e8m0_shuffle(_wsc).view(torch.uint8).contiguous()
    del _wfp4, _wsc
    torch.cuda.empty_cache()

    x_bf16 = (torch.randn((tokens, model_dim), device=dev, dtype=torch.float32) * init_scale).to(torch.bfloat16)
    torch.manual_seed(1 + rank * 101)
    topk_ids = torch.stack([torch.randperm(experts, device=dev)[:topk] for _ in range(tokens)]).to(torch.int32)
    wts = torch.full((tokens, topk), 1.0 / topk, device=dev, dtype=torch.float32)
    x_q, x_sc = _qact(x_bf16, is_fp4)

    from megamoeexp.fused_moe_megakernel import FusedMoEMegaStage1 as ExpS1

    op = ExpS1(
        rank=rank,
        world_size=world,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        quant=quant,
        w1=w1,
        w1_scale=w1s,
        max_tok_per_rank=mtpr,
        network=net_name,
        out_dtype="auto",
    )

    def _fwd():
        op.forward(x_q, wts, x_sc, topk_ids)

    eager = os.environ.get("TRACE_GRAPH", "1") in ("0", "", "false", "False")
    # Dispatch accounting (for ATT kernel_iteration_range).  CUDAGraph capture does NOT dispatch.
    #   eager warmup: EW dispatches ; replay warmup: RW dispatches ; traced replays: `iters`.
    EW, RW = 2, 2
    trace_lo = EW + RW + 1
    if rank == 0:
        mode = "eager" if eager else "graph"
        print(
            f"[trace] net={net_name} bs={tokens} frontier={op._frontier} split={getattr(op,'_split',False)} "
            f"mode={mode} iters={iters} trace_range=[{trace_lo},{trace_lo + iters - 1}]",
            flush=True,
        )
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    if eager:
        for _ in range(EW + RW + iters):
            _fwd()
            torch.cuda.synchronize()
            ms.shmem_barrier_all()
    else:
        for _ in range(EW):  # eager JIT warmup (dispatch 1..EW)
            _fwd()
        torch.cuda.synchronize()
        ms.shmem_barrier_all()
        cap = torch.cuda.Stream()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=cap):  # capture: 0 dispatch
            _fwd()
        # per-replay barrier keeps all 8 ranks in lockstep so peer P2P data is present when rank 0
        # (ATT-slowed) consumes -> minimal frontier spin -> trace reflects real compute, not skew.
        for _ in range(RW + iters):  # dispatch EW+1 ..
            ms.shmem_barrier_all()
            g.replay()
            torch.cuda.synchronize()
            ms.shmem_barrier_all()
    if rank == 0:
        print("[trace] done", flush=True)
    # rank 0 is wrapped by rocprofv3: must exit cleanly (normal interpreter shutdown) so the
    # rocprofiler library destructor flushes the trace.  os._exit() would skip that flush.
    # Other ranks hard-exit to avoid nccl/shmem teardown hangs.
    sys.stdout.flush()
    sys.stderr.flush()
    if rank != 0:
        os._exit(0)


if __name__ == "__main__":
    main()
