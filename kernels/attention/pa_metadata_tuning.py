# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Runtime lookup and offline tuning for PA metadata grid multipliers.

Run a fresh search with:
  FLYDSL_AUTOTUNE=1 FLYDSL_AUTOTUNE_CONFIG_DIR=/path/to/configs \
    python3 -m kernels.attention.pa_metadata_tuning \
    --shape 81,8192,4,per_token
"""

from __future__ import annotations

import argparse
import inspect
import statistics
from collections.abc import Callable, Iterable

from flydsl.autotune import Autotuner, Config

PA_METADATA_GRID_KEY = [
    "batch_size",
    "num_blocks",
    "query_length",
    "per_token_kv",
    "num_cu",
    "num_query_heads",
    "num_kv_heads",
    "head_dim",
    "block_size",
]


def run_pa_metadata_grid_config(
    batch_size,
    num_blocks,
    query_length,
    per_token_kv,
    num_cu,
    num_query_heads,
    num_kv_heads,
    head_dim,
    block_size,
    device_tensor,
    runner,
    grid_multiplier,
):
    del device_tensor
    if runner is None:
        raise RuntimeError("runner is required when benchmarking a grid config")
    return runner(grid_multiplier)


def make_pa_metadata_grid_autotuner(
    *,
    candidates: Iterable[int] = (1, 2, 3),
    warmup: int = 5,
    rep: int = 3,
    do_bench: Callable | None = None,
) -> Autotuner:
    kwargs = {
        "fn": run_pa_metadata_grid_config,
        "configs": [Config(grid_multiplier=int(candidate)) for candidate in candidates],
        "key": PA_METADATA_GRID_KEY,
        "warmup": warmup,
        "rep": rep,
        "do_bench_fn": do_bench,
    }
    if "artifact_name" in inspect.signature(Autotuner).parameters:
        kwargs["artifact_name"] = "pa_metadata_grid"
    return Autotuner(
        **kwargs,
    )


_RUNTIME_TUNER = make_pa_metadata_grid_autotuner(warmup=0, rep=1)


def lookup_pa_metadata_grid_multiplier(
    *,
    num_cu: int,
    batch_size: int,
    num_blocks: int,
    query_length: int,
    per_token_kv: bool,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    device_tensor,
) -> int | None:
    args = (
        batch_size,
        num_blocks,
        query_length,
        per_token_kv,
        num_cu,
        num_query_heads,
        num_kv_heads,
        head_dim,
        block_size,
        device_tensor,
        None,
    )
    get_cached_config = getattr(_RUNTIME_TUNER, "get_cached_config", None)
    if get_cached_config is not None:
        config = get_cached_config(*args)
    else:
        config = _RUNTIME_TUNER.cache.get(_RUNTIME_TUNER._make_key(args, {}))
    if config is None:
        return None
    return int(config.kwargs["grid_multiplier"])


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


def _tune_shape(
    *,
    batch_size: int,
    context_length: int,
    query_length: int,
    per_token_kv: bool,
    candidates: list[int],
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    warmup: int,
    iterations: int,
    rounds: int,
    device,
):
    import torch

    from kernels.attention.pa_decode_fp8 import get_pa_metadata, pa_decode_ps_launch

    pages_per_sequence = (context_length + block_size - 1) // block_size
    num_blocks = batch_size * pages_per_sequence
    num_cu = torch.cuda.get_device_properties(device).multi_processor_count
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

    output = torch.empty_like(query)
    graphs = {}
    metadata_resources = []
    for grid_multiplier in candidates:
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
        metadata_resources.append(metadata)

        def launch(metadata=metadata):
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

        launch()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(iterations):
                launch()
        graphs[grid_multiplier] = graph
    torch.cuda.synchronize(device)

    active_grid = {"value": None}
    timings_us = {}

    def replay_grid(grid_multiplier):
        active_grid["value"] = grid_multiplier
        graphs[grid_multiplier].replay()
        return grid_multiplier

    def graph_bench(call, warmup, rep):
        for _ in range(warmup):
            call()
        torch.cuda.synchronize(device)
        samples_ms = []
        for _ in range(rep):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            call()
            end.record()
            end.synchronize()
            samples_ms.append(start.elapsed_time(end) / iterations)
        latency_ms = statistics.median(samples_ms)
        timings_us[active_grid["value"]] = latency_ms * 1000.0
        return latency_ms

    tuner = make_pa_metadata_grid_autotuner(
        candidates=candidates,
        warmup=warmup,
        rep=rounds,
        do_bench=graph_bench,
    )
    tuner_args = (
        batch_size,
        num_blocks,
        query_length,
        per_token_kv,
        num_cu,
        num_query_heads,
        num_kv_heads,
        head_dim,
        block_size,
        query,
        replay_grid,
    )
    tuner(*tuner_args)
    get_cached_config = getattr(tuner, "get_cached_config", None)
    if get_cached_config is not None:
        best_config = get_cached_config(*tuner_args)
    else:
        best_config = tuner.cache.get(tuner._make_key(tuner_args, {}))
    best_grid = int(best_config.kwargs["grid_multiplier"])
    torch.cuda.synchronize(device)

    if not torch.allclose(
        output.float(),
        torch.ones_like(output, dtype=torch.float32),
        atol=2e-2,
        rtol=2e-2,
    ):
        raise RuntimeError(f"grid_multiplier={best_grid} failed correctness")
    persistent_config_path = getattr(tuner, "persistent_config_path", None)
    config_path = persistent_config_path(*tuner_args) if persistent_config_path is not None else tuner._cache_file
    return best_grid, timings_us.get(best_grid), config_path


def main() -> None:
    import torch

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
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations < 1 or args.rounds < 1:
        parser.error("warmup must be non-negative; iterations and rounds must be positive")

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    config_path = None

    for batch_size, context_length, query_length, per_token_kv in args.shape:
        best_grid, best_latency, config_path = _tune_shape(
            batch_size=batch_size,
            context_length=context_length,
            query_length=query_length,
            per_token_kv=per_token_kv,
            candidates=args.candidates,
            num_query_heads=args.num_query_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            block_size=args.block_size,
            warmup=args.warmup,
            iterations=args.iterations,
            rounds=args.rounds,
            device=device,
        )
        if best_latency is None:
            print(f"  cached winner: grid={best_grid}")
        else:
            print(f"  winner: grid={best_grid}, {best_latency:.3f} us")
        torch.cuda.empty_cache()

    print(f"Autotune config: {config_path}")


if __name__ == "__main__":
    main()
