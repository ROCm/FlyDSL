#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tune PA metadata grid multipliers and persist winners to CSV.

Example:
  python3 scripts/tune_pa_metadata_grid.py \
    --shape 16,4096,1,per_tensor \
    --shape 81,8192,4,per_token
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flydsl.runtime.device import get_rocm_arch
from kernels.attention.pa_decode_fp8 import get_pa_metadata, pa_decode_ps_launch
from kernels.attention.pa_metadata_tuning import (
    PA_METADATA_TUNING_CSV,
    write_pa_metadata_tuning_rows,
)


def _parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",")]
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _parse_shape(value: str) -> tuple[int, int, int, bool]:
    try:
        batch_size, context_length, query_length, quant_mode = value.split(",")
        per_token_kv = {"per_tensor": False, "per_token": True}[quant_mode]
        parsed = (int(batch_size), int(context_length), int(query_length), per_token_kv)
    except (KeyError, TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("shape must be BATCH,CONTEXT,QUERY_LENGTH,per_tensor|per_token") from error
    if any(item < 1 for item in parsed[:3]):
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return parsed


def _measure_candidate(
    *,
    batch_size: int,
    context_length: int,
    query_length: int,
    per_token_kv: bool,
    grid_multiplier: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    warmup: int,
    iterations: int,
    rounds: int,
    device: torch.device,
) -> tuple[float, int]:
    pages_per_sequence = (context_length + block_size - 1) // block_size
    num_blocks = batch_size * pages_per_sequence
    context_lengths = torch.full((batch_size,), context_length, dtype=torch.int32, device=device)
    query = torch.zeros(
        (batch_size * query_length, num_query_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    kv_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device=device) * pages_per_sequence
    kv_page_indices = torch.arange(num_blocks, dtype=torch.int32, device=device)
    key = torch.zeros(
        (num_blocks, num_kv_heads, head_dim // 16, block_size, 16),
        dtype=torch.float8_e4m3fnuz,
        device=device,
    )
    value = torch.ones(
        (num_blocks, num_kv_heads, block_size // 16, head_dim, 16),
        dtype=torch.float8_e4m3fnuz,
        device=device,
    )
    if per_token_kv:
        key_scale = torch.ones((num_blocks, num_kv_heads, block_size), dtype=torch.float32, device=device)
        value_scale = torch.ones_like(key_scale)
    else:
        key_scale = torch.ones((1,), dtype=torch.float32, device=device)
        value_scale = torch.ones((1,), dtype=torch.float32, device=device)

    metadata = get_pa_metadata(
        query,
        key,
        context_lengths,
        kv_indptr,
        num_query_heads,
        num_kv_heads,
        per_token_kv=per_token_kv,
        grid_multiplier=grid_multiplier,
    )
    output = torch.empty_like(query)

    def launch():
        pa_decode_ps_launch(
            output,
            query,
            key,
            value,
            context_lengths,
            kv_page_indices,
            kv_indptr,
            1.0 / head_dim**0.5,
            key_scale=key_scale,
            value_scale=value_scale,
            metadata=metadata,
        )

    for _ in range(warmup):
        launch()
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iterations):
            launch()
    torch.cuda.synchronize(device)

    timings = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000.0 / iterations)

    if not torch.allclose(
        output.float(),
        torch.ones_like(output, dtype=torch.float32),
        atol=2e-2,
        rtol=2e-2,
    ):
        raise RuntimeError(f"grid_multiplier={grid_multiplier} failed correctness")
    return statistics.median(timings), num_blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", action="append", type=_parse_shape, required=True)
    parser.add_argument("--candidates", type=_parse_int_list, default=[1, 2, 3])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-query-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=PA_METADATA_TUNING_CSV)
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations < 1 or args.rounds < 1:
        parser.error("warmup must be non-negative; iterations and rounds must be positive")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    arch = get_rocm_arch()
    num_cu = torch.cuda.get_device_properties(device).multi_processor_count
    tuned_rows = []

    for batch_size, context_length, query_length, per_token_kv in args.shape:
        timings = {}
        num_blocks = 0
        for grid_multiplier in args.candidates:
            latency_us, num_blocks = _measure_candidate(
                batch_size=batch_size,
                context_length=context_length,
                query_length=query_length,
                per_token_kv=per_token_kv,
                grid_multiplier=grid_multiplier,
                num_query_heads=args.num_query_heads,
                num_kv_heads=args.num_kv_heads,
                head_dim=args.head_dim,
                block_size=args.block_size,
                warmup=args.warmup,
                iterations=args.iterations,
                rounds=args.rounds,
                device=device,
            )
            timings[grid_multiplier] = latency_us
            print(
                f"B={batch_size} C={context_length} Q={query_length} "
                f"per_token={per_token_kv} grid={grid_multiplier}: {latency_us:.3f} us"
            )

        best_grid = min(timings, key=timings.get)
        best_latency = timings[best_grid]
        print(f"  winner: grid={best_grid}, {best_latency:.3f} us")
        tuned_rows.append(
            {
                "arch": arch,
                "num_cu": num_cu,
                "batch_size": batch_size,
                "num_blocks": num_blocks,
                "context_length": context_length,
                "query_length": query_length,
                "per_token_kv": per_token_kv,
                "num_query_heads": args.num_query_heads,
                "num_kv_heads": args.num_kv_heads,
                "head_dim": args.head_dim,
                "block_size": args.block_size,
                "grid_multiplier": best_grid,
                "latency_us": best_latency,
            }
        )
        torch.cuda.empty_cache()

    write_pa_metadata_tuning_rows(tuned_rows, args.output)
    print(f"Wrote {len(tuned_rows)} tuned configuration(s) to {args.output}")


if __name__ == "__main__":
    main()
