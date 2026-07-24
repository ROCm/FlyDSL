#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Compare production MegaMoE v1 and experimental v2 stage1 on identical EP inputs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import mori.shmem as ms
import torch
import torch.distributed as dist

import flydsl.expr as fx
from kernels.common.tensor_shim import _run_compiled
from kernels.mega_moe import MegaMoE
from kernels.mega_moe.mega_moe_exp import MegaMoEV2
from kernels.mega_moe.mega_moe_exp.mega_moe_stage1 import compile_mega_moe_stage1
from tests.kernels.mega_moe_exp.test_mega_moe_v2 import (
    _chunked_fp4_quant,
    _cleanup,
    _setup_dist,
)
from tests.kernels.utils import gemm_common_utils

MODEL_DIM = 7168
INTER_DIM = 3072
EXPERTS = int(os.environ.get("MEGA_MOE_EXPERTS", "384"))
TOPK = 6
CALIBRATION_PATH = (
    Path(__file__).resolve().parents[3]
    / "kernels/comm/mega_moe_tuning_config"
    / "flydsl_gfx950_mi355x_MegaStage1V2_ep4.json"
)


class _Stage1OnlyMegaMoE(MegaMoE):
    def _build_fused_stage2(self, **kwargs):
        del kwargs


class _Stage1OnlyMegaMoEV2(MegaMoEV2):
    def _build_fused_stage2(self, **kwargs):
        del kwargs


def _all_value(dev, value, op):
    tensor = torch.tensor([float(value)], dtype=torch.float64, device=dev)
    dist.all_reduce(tensor, op=op)
    return float(tensor.item())


def _make_inputs(dev, rank, world, tokens, seed):
    epr = EXPERTS // world
    init_scale = float(MODEL_DIM) ** -0.25
    torch.manual_seed(seed + rank * 101)
    x = (torch.randn(tokens, MODEL_DIM, dtype=torch.float32, device=dev) * init_scale).to(torch.bfloat16)
    ids = (
        torch.stack([torch.randperm(EXPERTS, device=dev)[:TOPK] for _ in range(tokens)]).to(torch.int32)
        if tokens
        else torch.empty((0, TOPK), dtype=torch.int32, device=dev)
    )
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
    return x.contiguous(), weights, ids.contiguous(), w1.reshape(-1).contiguous(), w1s.reshape(-1).contiguous()


def _quantize_input(moe, x):
    if x.shape[0]:
        return moe.quantize(x)
    return (
        torch.empty((0, MODEL_DIM), dtype=torch.float8_e4m3fn, device=x.device),
        torch.zeros(1, dtype=torch.int32, device=x.device),
    )


def _make_moe(
    cls,
    rank,
    world,
    mtpr,
    tune_tokens,
    w1,
    w1s,
    v2_tile_m_values=(32,),
    v2_dispatch_cu=None,
    v2_grid_mult=None,
):
    dummy = torch.empty(1, dtype=torch.uint8, device=w1.device)
    kwargs = dict(
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
    if issubclass(cls, MegaMoEV2):
        kwargs["stage1_tile_m_values"] = tuple(v2_tile_m_values)
        if v2_dispatch_cu is not None:
            kwargs["stage1_dispatch_cu"] = v2_dispatch_cu
        if v2_grid_mult is not None:
            kwargs["stage1_grid_mult"] = v2_grid_mult
    return cls(**kwargs)


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


def _canonical_v2(moe, output):
    if output is moe._s1_small_output:
        op = moe._s1_small_op
        tile_m = 32
    elif output is moe._s1_output:
        op = moe._s1_op
        tile_m = int(moe._s1_active_tile_m)
    else:
        raise ValueError("stage1 output does not belong to this MegaMoEV2 instance")
    nvalid = int(op.num_valid.view(-1)[0].item())
    tiles = nvalid // tile_m
    trb = op.tile_row_base[:tiles].to(torch.int64)
    fixed_rows = (trb[:, None] + torch.arange(tile_m, dtype=torch.int64, device=moe.dev)[None, :]).reshape(-1)
    src = op.srcmap_em[fixed_rows]
    experts = op.sorted_expert_ids[:tiles].repeat_interleave(tile_m)
    valid = ((src & 0x00FFFFFF) < moe.max_recv) & ((src >> 24) < moe.topk)
    compact_rows = torch.arange(nvalid, dtype=torch.int64, device=moe.dev)[valid]
    fp8 = output.a2.view(-1, INTER_DIM)[compact_rows].clone()
    scales = _scale_rows(output.a2_scale, compact_rows)
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


_CONFIG_KEYS = (
    "sort_block_m",
    "tile_n",
    "tile_k",
    "num_waves",
    "wgm",
    "grid_mult",
    "sched_nmajor",
    "pipe_weights",
    "mfma_amajor",
    "swizzle_a",
    "num_dispatch_cu",
    "tune_use_xcd",
    "use_tile_resource",
    "waves_per_eu_hint",
)


def _load_calibrated_config(path, tokens, world, experts):
    data = json.loads(Path(path).read_text())
    for row in data["megastage1_v2"]:
        row_ep_size = int(row.get("ep_size", data["ep_size"]))
        if (
            row["scope"] == "stage1_only"
            and int(row["num_tokens"]) == int(tokens)
            and int(row["max_tokens_per_rank"]) == int(tokens)
            and row_ep_size == int(world)
        ):
            if int(row["total_experts"]) != int(experts):
                continue
            return {key: row[key] for key in _CONFIG_KEYS}
    raise ValueError(f"no exact stage1 calibration for EP{world}/E{experts}/BS{tokens}")


def _make_fixed_v2_runner(moe, xq, weights, scales, ids, tokens, config):
    launch = compile_mega_moe_stage1(
        model_dim=MODEL_DIM,
        inter_dim=INTER_DIM,
        rank=moe.rank,
        experts_per_rank=moe.epr,
        fuse_npes=moe.world_size,
        fuse_topk=moe.topk,
        fuse_cap=moe._s1_cap,
        fuse_mtpr=moe.mtpr,
        fuse_scale_dim=moe._s1_scale_dim,
        sort_block_m=int(config["sort_block_m"]),
        tile_n=int(config["tile_n"]),
        tile_k=int(config["tile_k"]),
        num_waves=int(config["num_waves"]),
        grid_mult=int(config["grid_mult"]),
        wgm=int(config["wgm"]),
        sched_nmajor=bool(config["sched_nmajor"]),
        pipe_weights=bool(config["pipe_weights"]),
        mfma_amajor=bool(config["mfma_amajor"]),
        swizzle_a=bool(config["swizzle_a"]),
        use_xcd=moe._s1_use_xcd and bool(config["tune_use_xcd"]),
        use_tile_resource=bool(config["use_tile_resource"]),
        waves_per_eu_hint=int(config["waves_per_eu_hint"]),
        num_cu=moe._s1_num_cu,
        num_dispatch_cu=int(config["num_dispatch_cu"]),
    )
    op = moe._s1_op

    def run():
        _run_compiled(
            launch,
            moe._s1_out,
            moe._s1_rx,
            moe._s1_w1,
            moe._s1_scale_i32,
            moe._s1_w1_scale,
            op.tile_row_base,
            op.sorted_expert_ids,
            op.num_valid,
            moe._s1_osd,
            fx.Int32(moe._s1_nvm),
            fx.Int64(moe._s1_disp.data_ptr()),
            fx.Int32(tokens),
            fx.Int64(xq.data_ptr()),
            fx.Int64(ids.data_ptr()),
            fx.Int64(weights.data_ptr()),
            fx.Int64(scales.data_ptr()),
            fx.Int64(moe._s1_epoch_parity.data_ptr()),
            fx.Int64(moe._s1_epoch_expected.data_ptr()),
            fx.Stream(torch.cuda.current_stream()),
        )
        moe._s1_active_tile_m = int(config["sort_block_m"])
        return moe._s1_output

    return run


def _sample_indices(rows, count, device):
    if rows <= count:
        return torch.arange(rows, dtype=torch.int64, device=device)
    return torch.linspace(0, rows - 1, count, dtype=torch.int64, device=device)


def _run_reference_mode(args, rank, world, dev, x, weights, ids, w1, w1s):
    reference_dir = Path(args.reference_dir)
    if rank == 0:
        reference_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    if args.reference_role == "write-v1":
        moe = _make_moe(_Stage1OnlyMegaMoE, rank, world, args.tokens, args.tokens, w1, w1s)
        xq, scales = _quantize_input(moe, x)
        moe._run_fused_stage1(xq, weights, scales, ids)
        torch.cuda.synchronize()
        ms.shmem_barrier_all()
        keys, fp8, out_scales = _canonical_v1(moe)
        sample = _sample_indices(keys.numel(), args.sample_rows, dev)
        torch.save(
            {
                "keys": keys.cpu(),
                "sample": sample.cpu(),
                "fp8": fp8[sample].cpu(),
                "scales": out_scales[sample].cpu(),
            },
            reference_dir / f"rank{rank}.pt",
        )
        if rank == 0:
            print(f"[STAGE1-REFERENCE] rows={keys.numel()} samples={sample.numel()}", flush=True)
        return 0

    config = _load_calibrated_config(args.calibration_json, args.tokens, world, EXPERTS)
    moe = _make_moe(
        _Stage1OnlyMegaMoEV2,
        rank,
        world,
        args.tokens,
        args.tokens,
        w1,
        w1s,
        v2_tile_m_values=(int(config["sort_block_m"]),),
        v2_dispatch_cu=int(config["num_dispatch_cu"]),
        v2_grid_mult=int(config["grid_mult"]),
    )
    xq, scales = _quantize_input(moe, x)
    if args.vary_routing:
        for replay in range(1, args.eager_replays):
            replay_ids = ids.roll(replay, dims=0).contiguous()
            _make_fixed_v2_runner(
                moe,
                xq,
                weights,
                scales,
                replay_ids,
                args.tokens,
                config,
            )()
    output = _make_fixed_v2_runner(moe, xq, weights, scales, ids, args.tokens, config)()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    keys, fp8, out_scales = _canonical_v2(moe, output)
    reference = torch.load(reference_dir / f"rank{rank}.pt", map_location="cpu", weights_only=False)
    keys_match = keys.shape == reference["keys"].shape and torch.equal(keys.cpu(), reference["keys"])
    sample = reference["sample"].to(dev)
    reference_out = _dequant(reference["fp8"].to(dev), reference["scales"].to(dev))
    actual_out = _dequant(fp8[sample], out_scales[sample])
    rel_l2 = (torch.norm(actual_out - reference_out) / torch.norm(reference_out)).item()
    keys_ok = _all_value(dev, 1.0 if keys_match else 0.0, dist.ReduceOp.MIN) == 1.0
    rel_max = _all_value(dev, rel_l2, dist.ReduceOp.MAX)
    if rank == 0:
        print(f"[STAGE1-VALIDATE] keys={keys_ok} sampled_dequant_relL2={rel_max:.8e}", flush=True)
    return 0 if keys_ok and rel_max < 0.05 else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--v2-tile-m-values", default="32")
    parser.add_argument("--v2-dispatch-cu", type=int)
    parser.add_argument("--v2-grid-mult", type=int)
    parser.add_argument("--perf-only", choices=("v1", "v2"))
    parser.add_argument("--eager-only", action="store_true")
    parser.add_argument("--eager-replays", type=int, default=1)
    parser.add_argument("--vary-routing", action="store_true")
    parser.add_argument("--reference-role", choices=("write-v1", "check-v2"))
    parser.add_argument("--reference-dir", default="/tmp/mega_moe_stage1_reference")
    parser.add_argument("--sample-rows", type=int, default=1024)
    parser.add_argument("--calibration-json", default=str(CALIBRATION_PATH))
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if rank == 0:
        print("[STAGE1-COMPARE] initializing distributed + MORI", flush=True)
    local_rank = _setup_dist(rank, world, 29921)
    dev = torch.device("cuda", local_rank)
    if rank == 0:
        print("[STAGE1-COMPARE] distributed setup ready", flush=True)
    tokens = int(args.tokens)
    mtpr = max(16, tokens, int(args.max_tokens))
    if mtpr & (mtpr - 1):
        raise ValueError(f"max tokens must be a power of two, got {mtpr}")
    if EXPERTS % world != 0:
        raise ValueError(f"experts={EXPERTS} must divide world={world}")

    x, weights, ids, w1, w1s = _make_inputs(dev, rank, world, tokens, args.seed)
    if rank == 0:
        print("[STAGE1-COMPARE] inputs ready", flush=True)
    if args.reference_role is not None:
        try:
            return _run_reference_mode(args, rank, world, dev, x, weights, ids, w1, w1s)
        finally:
            _cleanup()

    v2_tile_m_values = tuple(int(value) for value in args.v2_tile_m_values.split(","))
    if args.perf_only is not None:
        if rank == 0:
            print(f"[STAGE1-COMPARE] building {args.perf_only}", flush=True)
        cls = MegaMoE if args.perf_only == "v1" else MegaMoEV2
        moe = _make_moe(
            cls,
            rank,
            world,
            mtpr,
            tokens,
            w1,
            w1s,
            v2_tile_m_values=v2_tile_m_values,
            v2_dispatch_cu=args.v2_dispatch_cu,
            v2_grid_mult=args.v2_grid_mult,
        )
        xq, scales = _quantize_input(moe, x)
        if rank == 0:
            print(f"[STAGE1-CONFIG] mode={args.perf_only} config={moe._s1cfg}", flush=True)

        def run(route_ids=ids):
            moe._run_fused_stage1(xq, weights, scales, route_ids)

        if args.eager_only:
            if rank == 0:
                print(f"[STAGE1-COMPARE] eager smoke {args.perf_only}", flush=True)
            for replay in range(args.eager_replays):
                route_ids = ids.roll(replay, dims=0).contiguous() if args.vary_routing and ids.shape[0] else ids
                run(route_ids)
            torch.cuda.synchronize()
            ms.shmem_barrier_all()
            if rank == 0:
                print(f"[STAGE1-SMOKE] mode={args.perf_only} bs={tokens} PASS", flush=True)
            _cleanup()
            return 0

        if rank == 0:
            print(f"[STAGE1-COMPARE] timing {args.perf_only}", flush=True)
        local_ms = _time_graph(run, int(args.iters))
        mean_ms = _all_value(dev, local_ms, dist.ReduceOp.SUM) / world
        max_ms = _all_value(dev, local_ms, dist.ReduceOp.MAX)
        if rank == 0:
            print(
                f"[STAGE1-PERF] mode={args.perf_only} bs={tokens} " f"mean_ms={mean_ms:.4f} max_ms={max_ms:.4f}",
                flush=True,
            )
        _cleanup()
        return 0

    if rank == 0:
        print("[STAGE1-COMPARE] building v1 + v2", flush=True)
    v1 = _make_moe(MegaMoE, rank, world, mtpr, tokens, w1, w1s)
    v2 = _make_moe(
        MegaMoEV2,
        rank,
        world,
        mtpr,
        tokens,
        w1,
        w1s,
        v2_tile_m_values=v2_tile_m_values,
        v2_dispatch_cu=args.v2_dispatch_cu,
        v2_grid_mult=args.v2_grid_mult,
    )
    xq, scales = _quantize_input(v1, x)
    if rank == 0:
        print("[STAGE1-COMPARE] operators ready", flush=True)

    def run_v1():
        v1._run_fused_stage1(xq, weights, scales, ids)

    def run_v2():
        return v2._run_fused_stage1(xq, weights, scales, ids)

    if rank == 0:
        print("[STAGE1-COMPARE] checking v1 output", flush=True)
    run_v1()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    key1, fp81, scale1 = _canonical_v1(v1)
    if rank == 0:
        print("[STAGE1-COMPARE] checking v2 output", flush=True)
    v2_output = run_v2()
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    key2, fp82, scale2 = _canonical_v2(v2, v2_output)

    keys_match = key1.shape == key2.shape and torch.equal(key1, key2)
    if keys_match:
        fp8_mismatch = (fp81.view(torch.uint8) != fp82.view(torch.uint8)).float().mean().item()
        scale_mismatch = (scale1 != scale2).float().mean().item()
        out1, out2 = _dequant(fp81, scale1), _dequant(fp82, scale2)
        rel_l2 = (torch.norm(out1 - out2) / torch.norm(out1)).item()
    else:
        fp8_mismatch = scale_mismatch = rel_l2 = float("inf")

    if rank == 0:
        print("[STAGE1-COMPARE] timing v1", flush=True)
    v1_ms = _time_graph(run_v1, int(args.iters))
    if rank == 0:
        print("[STAGE1-COMPARE] timing v2", flush=True)
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
