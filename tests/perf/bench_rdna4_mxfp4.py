# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Benchmark or pre-calibrate RDNA4 MXFP4 GEMMs with hot/rotating weights.

Example from the repository root:
    python tests/perf/bench_rdna4_mxfp4.py --m 1 8 128 --nk 5120x8704 --tune
"""

import argparse
import json
import math
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch

from flydsl.kernels.gemm.rdna4_mxfp4 import choose_config, gemm, prepare_weights, tune_gemm


def graph_time(function, calls):
    graph = torch.cuda.CUDAGraph()
    function()
    with torch.cuda.graph(graph):
        function()
    for _ in range(3):
        graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(15):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / calls)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", nargs="+", type=int, default=[1, 128, 8192])
    parser.add_argument("--nk", nargs="+", default=["5120x17408"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cache-policy", choices=["auto", "hot", "rotating"], default="auto")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.calibrate_only and not args.tune:
        parser.error("--calibrate-only requires --tune")
    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(1)
    device = f"cuda:{args.gpu}"
    results = []
    for dimensions in args.nk:
        n, k = map(int, dimensions.lower().split("x"))
        generator = torch.Generator(device=device).manual_seed(42)
        packed = torch.randint(0, 256, (n, k // 2), device=device, dtype=torch.uint8, generator=generator)
        # Cover varied lookup indices while retaining normal E4M3 folding.
        scales = torch.randint(124, 136, (n, k // 32), device=device, dtype=torch.uint8, generator=generator)
        weights = prepare_weights(packed, scales)
        assert weights.mode == "e4m3" and weights.normal_only
        for m in args.m:
            x = torch.randn(m, k, device=device, generator=generator).to(torch.float8_e4m3fn)
            xs = torch.ones(m, device=device)
            started = time.monotonic()
            config = (
                tune_gemm(x, weights, xs, cache_policy=args.cache_policy)
                if args.tune
                else choose_config(m, n, k, weights.mode, cache_policy=args.cache_policy)
            )
            calibration_seconds = time.monotonic() - started if args.tune else 0
            out = gemm(x, weights, xs, config=config)
            row = dict(
                shape=[m, n, k],
                config=config,
                mode=weights.mode,
                normal_only=weights.normal_only,
                calibration_seconds=calibration_seconds,
                cache_policy=args.cache_policy,
                gpu=args.gpu,
                device=torch.cuda.get_device_name(),
                arch=torch.cuda.get_device_properties(args.gpu).gcnArchName,
            )
            if not args.calibrate_only:
                copies = max(8, math.ceil(256 * 2**20 / weights.storage_bytes))
                pool = [weights] + [
                    replace(
                        weights,
                        packed=weights.packed.clone(),
                        scales=weights.scales.clone(),
                        base_exponents=weights.base_exponents.clone(),
                    )
                    for _ in range(copies - 1)
                ]

                def hot(x=x, weights=weights, xs=xs, out=out, config=config):
                    for _ in range(8):
                        gemm(x, weights, xs, out=out, config=config)

                def rotating(x=x, pool=pool, xs=xs, out=out, config=config):
                    for weight in pool:
                        gemm(x, weight, xs, out=out, config=config)

                row["hot_us"] = graph_time(hot, 8)
                row["rotating_us"] = graph_time(rotating, copies)
                traffic = weights.storage_bytes + m * k + 4 * m + 2 * m * n
                row["rotating_effective_gb_s"] = traffic / row["rotating_us"] / 1000
                row["rotating_logical_tflops"] = 2 * m * n * k / row["rotating_us"] / 1e6
                del pool, hot, rotating
            results.append(row)
            print(json.dumps(row), flush=True)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(results, indent=2) + "\n")
            del out, x, xs
        del packed, scales, weights
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
