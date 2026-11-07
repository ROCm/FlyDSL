# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Offline MXFP4 comparison; results never override normal dispatch.

    python tests/perf/bench_rdna4_mxfp4.py --m 1 8 128 --nk 5120x8704
    python tests/perf/bench_rdna4_mxfp4.py --configs candidates.json --output timings.json

``--configs`` accepts a JSON list of explicit register/hybrid configurations.
The analytical choice is measured too. Reuse a choice with ``gemm(config=...)``.
"""

import argparse
import json
import random
import statistics
from functools import partial
from pathlib import Path

import torch

from flydsl.kernels.gemm.rdna4_mxfp4 import choose_config, gemm, get_device_capabilities, prepare_weights


def capture(function, calls, stream):
    graph = torch.cuda.CUDAGraph()
    events = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(2)]
    with torch.cuda.graph(graph, stream=stream):
        events[0].record()
        for _ in range(calls):
            function()
        events[1].record()
    return graph, events, calls


def check(actual, expected):
    error = ((actual.float() - expected).square().mean() / expected.square().mean().clamp_min(1e-30)).sqrt()
    if not actual.isfinite().all() or float(error) > 0.005:
        raise AssertionError(f"relative RMS error: {float(error)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", nargs="+", type=int, default=[1, 128, 8192])
    parser.add_argument("--nk", nargs="+", default=["5120x17408"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cache-policy", choices=["auto", "hot", "rotating"], default="auto")
    parser.add_argument("--configs", type=Path, help="Offline candidate list")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    extras = json.loads(args.configs.read_text()) if args.configs else []
    if not isinstance(extras, list) or not all(isinstance(c, dict) for c in extras):
        parser.error("--configs must contain a list of configuration dictionaries")
    torch.cuda.set_device(args.gpu)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = f"cuda:{args.gpu}"
    capacities = get_device_capabilities(args.gpu)
    stream = torch.cuda.Stream(device=args.gpu)
    flush = torch.ones(256 * 1024**2, dtype=torch.uint8, device=device)
    results = []
    for dimensions in args.nk:
        n, k = map(int, dimensions.lower().split("x"))
        generator = torch.Generator(device=device).manual_seed(42)
        packed = torch.randint(0, 256, (n, k // 2), device=device, dtype=torch.uint8, generator=generator)
        scales = torch.randint(124, 132, (n, k // 32), device=device, dtype=torch.uint8, generator=generator)
        weights = prepare_weights(packed, scales)
        cc = torch.linspace(0, n - 1, min(64, n), device=device).long().unique()
        codes = torch.stack((packed[cc] & 15, packed[cc] >> 4), -1).flatten(-2).long()
        lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device=device)
        decoded = lut[codes] * torch.exp2(scales[cc].float() - 127).repeat_interleave(32, -1)
        for m in args.m:
            x = torch.randn(m, k, device=device, generator=generator).to(torch.float8_e4m3fn)
            xs = torch.ones(m, device=device)
            rr = torch.linspace(0, m - 1, min(32, m), device=device).long().unique()
            expected = ((x.float()[rr] @ decoded.T) * xs[rr, None]).to(torch.bfloat16).float()
            config = choose_config(m, n, k, weights.mode, cache_policy=args.cache_policy, **capacities)
            configs = list({json.dumps(c, sort_keys=True): c for c in [config, *extras]}.values())
            out = torch.empty(m, n, device=device, dtype=torch.bfloat16)
            # Conditioning reads run outside GPU events; activations stay warm.
            conditioning = torch.cuda.CUDAGraph()
            with torch.cuda.graph(conditioning, stream=stream):
                flush.sum(dtype=torch.int32)
                x.view(torch.uint8).sum(dtype=torch.int32)
                xs.sum()
            trials = []
            for choice in configs:
                gemm(x, weights, xs, out=out, config=choice)
                torch.cuda.synchronize(args.gpu)
                check(out[rr[:, None], cc[None, :]], expected)
                launch = partial(gemm, x, weights, xs, out=out, config=choice)
                graphs = {
                    "hot": capture(launch, 8 if m < 128 else 2, stream),
                    "rotating": capture(launch, 1, stream),
                }
                x.view(torch.uint8).bitwise_xor_(128)
                for graph, _, _ in graphs.values():
                    out.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize(args.gpu)
                    check(out[rr[:, None], cc[None, :]], -expected)
                x.view(torch.uint8).bitwise_xor_(128)
                trials.append((choice, graphs, {"hot": [], "rotating": []}))
            rng = random.Random(42)
            for _ in range(16):
                order = [(trial, regime) for trial in trials for regime in ("hot", "rotating")]
                rng.shuffle(order)
                for (_, graphs, samples), regime in order:
                    graph, events, calls = graphs[regime]
                    (conditioning if regime == "rotating" else graph).replay()
                    graph.replay()
                    torch.cuda.synchronize(args.gpu)
                    samples[regime].append(events[0].elapsed_time(events[1]) * 1000 / calls)
            for choice, _, samples in trials:
                medians = {name: statistics.median(values) for name, values in samples.items()}
                traffic = weights.storage_bytes + m * k + 4 * m + 2 * m * n
                row = dict(
                    shape=[m, n, k],
                    config=choice,
                    analytical=choice == config,
                    mode=weights.mode,
                    cache_policy=args.cache_policy,
                    gpu=args.gpu,
                    capacities=capacities,
                    samples_us=samples,
                    **{name + "_us": value for name, value in medians.items()},
                    cold_protocol="read_evicted_weights_warm_activations",
                    rotating_effective_gb_s=traffic / medians["rotating"] / 1000,
                    rotating_logical_tflops=2 * m * n * k / medians["rotating"] / 1e6,
                )
                results.append(row)
                print(json.dumps(row), flush=True)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(results, indent=2) + "\n")
            del trials, graphs, conditioning, launch, x, xs, out
        del weights, packed, scales, decoded
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
