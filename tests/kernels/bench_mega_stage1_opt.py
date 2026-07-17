#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""MoE stage-1 megakernel (``FusedMoEMegaStage1``) optimization A/B: CUDAGraph device time for

    baseline (b_nt=0, dedup=off)  vs  +b_nt  vs  +dedup  vs  +both

across a batch-size sweep, on the SAME op (``kernels/fused_moe_megakernel.py``).  Reports each
variant's CUDAGraph time and speedup over baseline, plus a bit-exactness check (max|Δ| vs baseline
output) — both b_nt (a B-load cache hint) and dedup (a data-movement change, same math) must be
numerically identical to baseline.

Run (8x MI355X)::

  MORI_SHMEM_HEAP_SIZE=40G BS=256 torchrun --standalone --nproc_per_node=8 \
    tests/kernels/bench_mega_stage1_opt.py --network v4_pro --quant a8w4
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import flydsl.expr as fx  # noqa: E402

from kernels.fused_moe_megakernel import FusedMoEMegaStage1, _default_b_nt  # noqa: E402
from tests.kernels.bench_moe_intranode_stage1_groupgemm import (  # noqa: E402
    NETWORKS, _prepare, _setup_dist,
)


def _cudagraph_time_ms(op, x, wts, scales, ids, *, warmup=10, iters=50):
    """Median-ish CUDAGraph device time (ms) of one op.forward, synced across ranks."""
    for _ in range(3):
        op.forward(x, wts, scales, ids)
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        op.forward(x, wts, scales, ids)
    s.synchronize()
    with torch.cuda.graph(g, stream=s):
        op.forward(x, wts, scales, ids)
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters, g


def _build(net, quant, mtpr, rank, world, w1, w1s, *, b_nt, dedup):
    return FusedMoEMegaStage1(
        rank=rank, world_size=world, model_dim=net["model_dim"], inter_dim=net["inter_dim"],
        experts=net["experts"], topk=net["topk"], quant=quant, w1=w1, w1_scale=w1s,
        max_tok_per_rank=mtpr, b_nt=b_nt, dedup=dedup)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--network", default="v4_pro", choices=list(NETWORKS))
    p.add_argument("--quant", default="a8w4", choices=["a8w4", "a4w4"])
    p.add_argument("--bs-list", default="32,64,128,256,512")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--master-port", type=int, default=29931)
    args = p.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = _setup_dist(rank, world, args.master_port)
    dev = torch.device("cuda", local_rank)
    net = NETWORKS[args.network]
    experts, epr = net["experts"], net["experts"] // world

    bs_list = [int(b) for b in args.bs_list.split(",") if b.strip()]
    results = {}
    for bs in bs_list:
        mtpr = max(16, bs)
        # Heuristic b_nt (deployed config): _default_b_nt(mtpr) -> 3 (streaming) for mtpr<=128,
        # else 0 (cache).  So at bs=512 the +b_nt/+both variants use b_nt=0 (== baseline b_nt), i.e.
        # the heuristic avoids the large-batch streaming regression.
        hb = _default_b_nt(mtpr)
        variants = [("baseline",    dict(b_nt=0, dedup=False)),
                    ("+b_nt(heur)", dict(b_nt=hb, dedup=False)),
                    ("+dedup",      dict(b_nt=0, dedup=True)),
                    ("+both(heur)", dict(b_nt=hb, dedup=True))]
        T = _prepare(dev, quant=args.quant, tokens=bs, model_dim=net["model_dim"],
                     inter_dim=net["inter_dim"], experts=experts, topk=net["topk"],
                     seed=args.seed, rank=rank, world=world)
        w_kernel, scale_w1_1d = T["w_kernel"], T["scale_w1_1d"]
        _wpe = w_kernel.numel() // experts
        _spe = scale_w1_1d.numel() // experts
        w1 = w_kernel.reshape(-1)[rank * epr * _wpe:(rank + 1) * epr * _wpe].contiguous()
        w1s = scale_w1_1d.reshape(-1)[rank * epr * _spe:(rank + 1) * epr * _spe].contiguous()
        x = T["x_payload"] if args.quant == "a8w4" else T["x_payload"]
        x_arg = x.view(torch.uint8) if args.quant == "a4w4" else x
        scales = T["scale_mx_u8"].view(torch.uint8)
        wts, ids = T["wts"], T["topk_ids"]

        ref_out = None
        row = {}
        for name, cfg in variants:
            op = _build(net, args.quant, mtpr, rank, world, w1, w1s, **cfg)
            ms_, _g = _cudagraph_time_ms(op, x_arg, wts, scales, ids,
                                         warmup=args.warmup, iters=args.iters)
            # correctness: compare this variant's out to baseline out (bit-exact expected).
            out = op._out.detach().clone().float()
            if ref_out is None:
                ref_out = out
                maxabs = 0.0
            else:
                maxabs = (out - ref_out).abs().max().item()
            row[name] = (ms_, maxabs)
            del op, _g
            torch.cuda.synchronize()
        results[bs] = row
        if rank == 0:
            base = row["baseline"][0]
            print(f"\n==== {args.network} {args.quant} bs={bs} (per-rank={bs}) ====", flush=True)
            print(f"  {'variant':10s} {'ms':>9s} {'speedup':>9s} {'max|Δ|':>10s}", flush=True)
            for name, _c in variants:
                ms_, mx = row[name]
                print(f"  {name:10s} {ms_:9.4f} {base/ms_:8.3f}x {mx:10.2e}", flush=True)

    if rank == 0:
        print("\n\n#### SUMMARY (speedup over baseline) ####", flush=True)
        hdr = f"{'bs':>6s} " + " ".join(f"{n:>10s}" for n, _ in variants)
        print(hdr, flush=True)
        for bs in bs_list:
            base = results[bs]["baseline"][0]
            cells = " ".join(f"{base/results[bs][n][0]:9.3f}x" for n, _ in variants)
            print(f"{bs:6d} {cells}", flush=True)
        print("\n#### device time (ms) ####", flush=True)
        print(hdr, flush=True)
        for bs in bs_list:
            cells = " ".join(f"{results[bs][n][0]:10.4f}" for n, _ in variants)
            print(f"{bs:6d} {cells}", flush=True)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
