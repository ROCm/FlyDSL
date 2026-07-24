#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Collective compact-stage1 autotune and fixed-calibration benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from kernels.mega_moe.mega_moe_exp import MegaMoEV2
from tests.kernels.mega_moe_exp.compare_stage1_v1_v2 import (
    EXPERTS,
    INTER_DIM,
    MODEL_DIM,
    TOPK,
    _all_value,
    _cleanup,
    _load_calibrated_config,
    _make_fixed_v2_runner,
    _make_inputs,
    _setup_dist,
    _time_graph,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs-list", default="8192")
    parser.add_argument("--max-batch", type=int, default=0)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rep", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default="/tmp/mega_moe_stage1_v2_autotune.json")
    parser.add_argument("--cache-dir", default="/tmp/mega_moe_stage1_v2_autotune_cache")
    parser.add_argument(
        "--calibration-json",
        help="benchmark exact checked-in calibration instead of invoking autotune",
    )
    args = parser.parse_args()

    batches = tuple(int(value) for value in args.bs_list.split(",") if value.strip())
    if not batches or min(batches) <= 0:
        raise ValueError("--bs-list must contain positive batch sizes")
    max_tokens = max(max(batches), int(args.max_batch))
    os.environ["FLYDSL_AUTOTUNE_CACHE_DIR"] = args.cache_dir

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = _setup_dist(rank, world, 29951)
    dev = torch.device("cuda", local_rank)
    if EXPERTS % world:
        raise ValueError(f"experts={EXPERTS} must divide world={world}")

    try:
        x, weights, ids, w1, w1s = _make_inputs(dev, rank, world, max_tokens, args.seed)
        dummy = torch.empty(1, dtype=torch.uint8, device=dev)
        moe = MegaMoEV2(
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
            max_tok_per_rank=max_tokens,
            tune_tokens=max_tokens,
            enable_fused_stage1=True,
            enable_fused_stage2=True,
            stage1_tile_m_values=(32, 64, 128),
        )
        if min(batches) <= moe._s1_small_max_tokens:
            raise ValueError(
                "autotune_stage1.py only benchmarks compact stage1; "
                f"all batches must exceed {moe._s1_small_max_tokens}"
            )
        moe._s1_mega.warmup = int(args.warmup)
        moe._s1_mega.rep = int(args.rep)
        xq, scales = moe.quantize(x)
        results = {}

        for tokens in batches:
            xq_i = xq[:tokens].contiguous()
            scales_i = scales[:tokens].contiguous()
            weights_i = weights[:tokens].contiguous()
            ids_i = ids[:tokens].contiguous()

            if args.calibration_json:
                config = _load_calibrated_config(
                    args.calibration_json,
                    tokens,
                    world,
                    EXPERTS,
                )
                run = _make_fixed_v2_runner(
                    moe,
                    xq_i,
                    weights_i,
                    scales_i,
                    ids_i,
                    tokens,
                    config,
                )
            else:

                def run():
                    return moe._run_fused_stage1(xq_i, weights_i, scales_i, ids_i)

                config = None

            local_ms = _time_graph(run, int(args.iters))
            mean_ms = _all_value(dev, local_ms, dist.ReduceOp.SUM) / world
            max_ms = _all_value(dev, local_ms, dist.ReduceOp.MAX)
            if config is None:
                config = moe._s1_mega.last_config.to_dict()
            results[str(tokens)] = {
                "mean_ms": mean_ms,
                "max_ms": max_ms,
                "config": config,
            }
            if rank == 0:
                print(
                    f"[STAGE1-AUTOTUNE] bs={tokens} mean_ms={mean_ms:.4f} " f"max_ms={max_ms:.4f} config={config}",
                    flush=True,
                )
                output = Path(args.json_out)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(results, indent=2, sort_keys=True))
        return 0
    finally:
        _cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
