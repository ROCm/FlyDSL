#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Compare production MegaMoE v1 and experimental v2 stage1 on identical EP inputs."""

from __future__ import annotations

import argparse
import math
import os

import mori.shmem as ms
import torch
import torch.distributed as dist

from kernels.mega_moe import MegaMoE
from kernels.mega_moe.mega_moe_exp import MegaMoEV2
from tests.kernels.mega_moe_exp.test_mega_moe_v2 import _chunked_fp4_quant, _cleanup, _setup_dist
from tests.kernels.utils import gemm_common_utils

MODEL_DIM = 7168
INTER_DIM = 3072
EXPERTS = 384
TOPK = 6


def _all_value(dev, value, op):
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=dev)
    dist.all_reduce(tensor, op=op)
    return float(tensor.item())


def _make_inputs(dev, rank, world, tokens, seed):
    epr = EXPERTS // world
    init_scale = float(MODEL_DIM) ** -0.25
    torch.manual_seed(seed + rank * 101)
    x = (torch.randn(tokens, MODEL_DIM, dtype=torch.float32, device=dev) * init_scale).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(EXPERTS, device=dev)[:TOPK] for _ in range(tokens)]).to(torch.int32)
    weights = torch.full((tokens, TOPK), 1.0 / TOPK, dtype=torch.float32, device=dev)

    torch.manual_seed(seed + 10000 + rank)
    w1_f32 = torch.randn(epr, 2 * INTER_DIM, MODEL_DIM, dtype=torch.float32, device=dev) * init_scale
    w1_q, w1_scale = _chunked_fp4_quant(w1_f32.view(epr * 2 * INTER_DIM, MODEL_DIM))
    w1 = (
        gemm_common_utils.shuffle_weight_w4(
            w1_q.view(epr, 2 * INTER_DIM, MODEL_DIM // 2), NLane=16, gate_up=True, moe_gemm=True
        )
        .view(torch.uint8)
        .contiguous()
    )
    w1s = (
        gemm_common_utils.shuffle_scale_w4(
            w1_scale.view(epr * 2 * INTER_DIM, MODEL_DIM // 32), experts_cnt=epr, gate_up=True
        )
        .view(torch.uint8)
        .contiguous()
    )
    del w1_f32, w1_q, w1_scale
    torch.cuda.empty_cache()
    return x.contiguous(), weights, ids.contiguous(), w1, w1s


def _make_moe(cls, rank, world, mtpr, tune_tokens, w1, w1s):
    dummy = torch.empty(1, dtype=torch.uint8, device=w1.device)
    return cls(
        rank=rank,
        world_size=world,
        model_dim=MODEL_DIM,
        inter_dim=INTER_DIM,
        experts=EXPERTS,
        topk=TOPK,
        quant="a8w4",
        w1=w1,
        w1_scale=w1s,
        w2=dummy,
        w2_scale=dummy,
        max_tok_per_rank=mtpr,
        tune_tokens=tune_tokens,
        enable_fused_stage1=True,
        enable_fused_stage2=True,
    )


def _scale_rows(scale_buf, rows):
    scale_cols = (INTER_DIM // 32 + 7) // 8 * 8
    cols = torch.arange(INTER_DIM // 32, dtype=torch.int64, device=rows.device)
    d0, d1, d2 = rows >> 5, (rows >> 4) & 1, rows & 15
    d3, d4, d5 = cols >> 3, (cols >> 2) & 1, cols & 3
    offsets = (
        d0[:, None] * (scale_cols * 32)
        + d3[None, :] * 256
        + d5[None, :] * 64
        + d2[:, None] * 4
        + d4[None, :] * 2
        + d1[:, None]
    )
    return scale_buf[offsets].clone()


def _canonical_v1(moe):
    nvalid = int(moe._s1_nv.view(-1)[0].item())
    compact_rows = torch.arange(nvalid, dtype=torch.int64, device=moe.dev)
    src = moe._s1_sti[:nvalid]
    tile_m = int(moe.sort_block_m)
    experts = moe._s1_se_atom[: (nvalid + tile_m - 1) // tile_m].repeat_interleave(tile_m)[:nvalid] + moe.rank * moe.epr
    valid = ((src & 0x00FFFFFF) < moe.max_recv) & ((src >> 24) < moe.topk)
    compact_rows, src, experts = compact_rows[valid], src[valid], experts[valid]
    logical_rows = (src & 0x00FFFFFF).to(torch.int64) * moe.topk + (src >> 24).to(torch.int64)
    fp8 = moe._s1_out.view(-1, INTER_DIM)[logical_rows].clone()
    scales = _scale_rows(moe._s1_osd, compact_rows)
    return _sort_routes(moe, src, experts, fp8, scales)


def _canonical_v2(moe):
    op = moe._s1_op
    nvalid = int(op.num_valid.view(-1)[0].item())
    tiles = nvalid // moe.sort_block_m
    trb = op.tile_row_base[:tiles].to(torch.int64)
    fixed_rows = (trb[:, None] + torch.arange(moe.sort_block_m, dtype=torch.int64, device=moe.dev)[None, :]).reshape(-1)
    src = op.srcmap_em[fixed_rows]
    experts = op.sorted_expert_ids[:tiles].repeat_interleave(moe.sort_block_m)
    valid = ((src & 0x00FFFFFF) < moe.max_recv) & ((src >> 24) < moe.topk)
    compact_rows = torch.arange(nvalid, dtype=torch.int64, device=moe.dev)[valid]
    fp8 = moe._s1_out.view(-1, INTER_DIM)[compact_rows].clone()
    scales = _scale_rows(moe._s1_osd, compact_rows)
    return _sort_routes(moe, src[valid], experts[valid], fp8, scales)


def _sort_routes(moe, src, experts, fp8, scales):
    src_global = (src & 0x00FFFFFF).to(torch.int64)
    slot = (src >> 24).to(torch.int64)
    key = experts.to(torch.int64) * (moe.world_size * moe.mtpr * moe.topk)
    key = key + src_global * moe.topk + slot
    order = torch.argsort(key)
    return key[order], fp8[order], scales[order]


def _dequant(fp8, scales):
    return (
        fp8.float()
        .view(-1, INTER_DIM // 32, 32)
        .mul(torch.pow(2.0, scales.float() - 127.0)[:, :, None])
        .reshape(-1, INTER_DIM)
    )


def _time_graph(fn, iters):
    fn()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=capture_stream):
        fn()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = _setup_dist(rank, world, 29921)
    dev = torch.device("cuda", local_rank)
    tokens = int(args.tokens)
    mtpr = max(16, tokens)
    if EXPERTS % world != 0:
        raise ValueError(f"experts={EXPERTS} must divide world={world}")

    x, weights, ids, w1, w1s = _make_inputs(dev, rank, world, tokens, args.seed)
    v1 = _make_moe(MegaMoE, rank, world, mtpr, tokens, w1, w1s)
    v2 = _make_moe(MegaMoEV2, rank, world, mtpr, tokens, w1, w1s)
    xq, scales = v1.quantize(x)

    def run_v1():
        v1._run_fused_stage1(xq, weights, scales, ids)

    def run_v2():
        v2._run_fused_stage1(xq, weights, scales, ids)

    run_v1()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    key1, fp81, scale1 = _canonical_v1(v1)
    run_v2()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    key2, fp82, scale2 = _canonical_v2(v2)

    keys_match = key1.shape == key2.shape and torch.equal(key1, key2)
    if keys_match:
        fp8_mismatch = (fp81.view(torch.uint8) != fp82.view(torch.uint8)).float().mean().item()
        scale_mismatch = (scale1 != scale2).float().mean().item()
        out1, out2 = _dequant(fp81, scale1), _dequant(fp82, scale2)
        rel_l2 = (torch.norm(out1 - out2) / torch.norm(out1)).item()
    else:
        fp8_mismatch = scale_mismatch = rel_l2 = float("inf")

    v1_ms = _time_graph(run_v1, int(args.iters))
    v2_ms = _time_graph(run_v2, int(args.iters))
    keys_ok = _all_value(dev, 1.0 if keys_match else 0.0, dist.ReduceOp.MIN) == 1.0
    fp8_max = _all_value(dev, fp8_mismatch, dist.ReduceOp.MAX)
    scale_max = _all_value(dev, scale_mismatch, dist.ReduceOp.MAX)
    rel_max = _all_value(dev, rel_l2, dist.ReduceOp.MAX)
    v1_mean = _all_value(dev, v1_ms, dist.ReduceOp.SUM) / world
    v2_mean = _all_value(dev, v2_ms, dist.ReduceOp.SUM) / world
    v1_max = _all_value(dev, v1_ms, dist.ReduceOp.MAX)
    v2_max = _all_value(dev, v2_ms, dist.ReduceOp.MAX)
    ok = keys_ok and math.isfinite(rel_max) and rel_max < 0.05

    if rank == 0:
        print(
            f"[STAGE1-COMPARE] bs={tokens} {'PASS' if ok else 'FAIL'} "
            f"keys={keys_ok} fp8_mismatch={fp8_max:.3e} scale_mismatch={scale_max:.3e} "
            f"dequant_relL2={rel_max:.3e} v1_ms={v1_mean:.4f}/{v1_max:.4f} "
            f"v2_ms={v2_mean:.4f}/{v2_max:.4f} speedup={v1_mean / v2_mean:.3f}",
            flush=True,
        )
    _cleanup()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
