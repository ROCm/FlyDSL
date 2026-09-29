# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Single-GPU correctness and uninstrumented graph A/B for tile-level producer/consumer fusion."""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from kernels.monokernel.config import KIMI_K3_CONFIG
from kernels.monokernel.packing import pack_a16w4_scale, pack_a16w4_weight
from kernels.monokernel.reference import situ
from kernels.monokernel.formats import dequantize_mxfp4
from kernels.kimi_k3_monokernel.moe import kimi_k3_mxfp4_gemm1, kimi_k3_mxfp4_gemm2
from kernels.kimi_k3_monokernel.routed_pipeline import RoutedTilePipeline


def relative_l2(got, expected):
    return float((got.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--producers", type=int, default=None)
    parser.add_argument("--up-tile", type=int, default=32)
    parser.add_argument("--up-order", choices=("sample", "interleaved"), default="sample")
    parser.add_argument("--soft-profile", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--prefetch", type=int, default=None)
    parser.add_argument("--route-prefetch", action="store_true")
    parser.add_argument("--grid", type=int, default=256)
    parser.add_argument("--down-tile", type=int, choices=(0, 16, 32, 64), default=0)
    parser.add_argument("--sample-group", type=int, choices=(0, 1, 2, 4), default=0)
    parser.add_argument("--handoff", choices=("direct", "wave"), default="wave")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--trace-file")
    parser.add_argument("--overlap", action="store_true", help="all tokens choose the same 16 experts")
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    cpu_gen = torch.Generator().manual_seed(args.seed)
    s, h, inter, ne, topk, bm = args.samples, 3584, 384, 896, 16, 16
    config = KIMI_K3_CONFIG
    activation = dict(situ_beta=config.situ_beta, situ_linear_beta=config.situ_linear_beta)
    x = torch.randn(s, h, generator=gen, device=device, dtype=torch.bfloat16)
    ug_raw = torch.randint(0, 256, (ne, 2 * inter, h // 2), generator=gen, dtype=torch.uint8, device=device)
    dn_raw = torch.randint(0, 256, (ne, h, inter // 2), generator=gen, dtype=torch.uint8, device=device)
    ug_s = torch.full((ne, 2 * inter, h // 32), 118, dtype=torch.uint8, device=device)
    dn_s = torch.full((ne, h, inter // 32), 119, dtype=torch.uint8, device=device)
    ug, dn = pack_a16w4_weight(ug_raw), pack_a16w4_weight(dn_raw)
    ug_scale, dn_scale = pack_a16w4_scale(ug_s), pack_a16w4_scale(dn_s)
    ids = torch.stack([torch.randperm(ne, generator=cpu_gen)[:topk] for _ in range(s)])
    if args.overlap:
        ids[:] = ids[0].clone()
    probs = torch.rand(s, topk, generator=cpu_gen).softmax(-1)
    max_sorted = s * topk * bm
    eids_cpu = ids.unique(sorted=True)
    tids_cpu = torch.full((max_sorted,), s, dtype=torch.int32)
    weights_cpu = torch.zeros(max_sorted)
    positions = []
    for block, expert in enumerate(eids_cpu.tolist()):
        routes = (ids == expert).nonzero().tolist()
        for row, (token, slot) in enumerate(routes):
            position = block * bm + row
            positions.append((position, token, slot, expert))
            tids_cpu[position] = token | (slot << 24)
            weights_cpu[position] = probs[token, slot]
    eids = torch.zeros(s * topk, dtype=torch.int32, device=device)
    eids[: len(eids_cpu)] = eids_cpu.to(device=device, dtype=torch.int32)
    tids, weights = tids_cpu.to(device), weights_cpu.to(device)
    valid = torch.tensor([len(eids_cpu) * bm, s], dtype=torch.int32, device=device)
    mid = torch.empty(max_sorted, inter, dtype=torch.bfloat16, device=device)
    out = torch.empty(s, h, dtype=torch.bfloat16, device=device)
    step = torch.zeros(1, dtype=torch.int32, device=device)
    ids_gpu = ids.to(device=device, dtype=torch.int32)
    probs_gpu = probs.to(device)
    partial = torch.empty_like(out)
    reduced = torch.empty_like(out)
    print("BUILD pipeline", flush=True)
    pipeline = RoutedTilePipeline(
        s,
        ug_raw,
        ug_s,
        dn_raw,
        dn_s,
        producers=args.producers,
        up_tile=args.up_tile,
        up_order=args.up_order,
        soft_profile=args.soft_profile,
        route_prefetch=args.route_prefetch,
        prefetch=args.prefetch,
        trace=args.trace,
        handoff=args.handoff,
        sample_group=args.sample_group,
        grid=args.grid,
        down_tile=args.down_tile,
    )

    def baseline(layer=0):
        out.zero_()
        kimi_k3_mxfp4_gemm1(x, ug, ug_scale, eids, valid, tids, mid, samples=s, **activation)
        kimi_k3_mxfp4_gemm2(mid, dn, dn_scale, eids, valid, tids, weights, out, samples=s, max_sorted=max_sorted)

    def candidate(layer=0):
        pipeline(x, ids_gpu, probs_gpu, partial, reduced, step, layer)

    def candidate_mid():
        values = pipeline.mid[..., 0].contiguous().view(torch.bfloat16).reshape(s, topk, inter)
        return torch.stack([values[token, slot] for _, token, slot, _ in positions])

    index = torch.tensor([position for position, *_ in positions], device=device)
    print("RUN baseline", flush=True)
    baseline()
    expected_mid, expected_out = mid[index].clone(), out.clone()
    print("RUN pipeline (JIT)", flush=True)
    candidate()
    torch.cuda.synchronize()
    print("RUN pipeline complete", flush=True)
    result = dict(
        samples=s,
        seed=args.seed,
        producers=pipeline.producers,
        up_tile=args.up_tile,
        up_order=args.up_order,
        soft_profile=args.soft_profile,
        route_prefetch=args.route_prefetch,
        prefetch=pipeline.prefetch,
        trace=args.trace,
        handoff=args.handoff,
        sample_group=args.sample_group,
        grid=args.grid,
        down_tile=pipeline.down_tile,
        overlap=args.overlap,
        active_experts=len(eids_cpu),
    )
    result["mid_vs_staged_rel_l2"] = relative_l2(candidate_mid(), expected_mid)
    result["output_vs_staged_rel_l2"] = relative_l2(reduced, expected_out)
    # Independent FP32 reference catches shared ABI/layout errors. The final
    # staged scatter uses BF16 atomics, so compare its numerical error to the retained
    # staged implementation as well as an absolute tolerance.
    golden_mid = []
    golden_out = torch.zeros(s, h, device=device)
    for _, token, slot, expert in positions:
        u = dequantize_mxfp4(ug_raw[expert], ug_s[expert]) @ x[token].float()
        m = situ(u, beta=config.situ_beta, linear_beta=config.situ_linear_beta).to(torch.bfloat16)
        golden_mid.append(m)
        d = dequantize_mxfp4(dn_raw[expert], dn_s[expert]) @ m.float()
        golden_out[token].add_(d * float(probs[token, slot]))
    result["mid_reference_rel_l2"] = relative_l2(candidate_mid(), torch.stack(golden_mid))
    result["staged_reference_rel_l2"] = relative_l2(expected_out, golden_out)
    result["fused_reference_rel_l2"] = relative_l2(reduced, golden_out)
    print(json.dumps(result), flush=True)
    assert result["mid_vs_staged_rel_l2"] < 2e-3, result
    assert result["mid_reference_rel_l2"] < 2e-3, result
    assert (
        max(result["output_vs_staged_rel_l2"], result["fused_reference_rel_l2"], result["staged_reference_rel_l2"])
        < 1e-2
    ), result

    if args.trace:
        timeline = pipeline.timeline.cpu()
        up_tasks = s * topk * (inter // args.up_tile)
        up, down = timeline[:up_tasks], timeline[up_tasks:]
        trace_result = {
            "first_up": int(up[:, 0].min()),
            "last_up": int(up[:, 2].max()),
            "first_down_math": int(down[:, 2].min()),
            "last_down_math": int(down[:, 3].max()),
            "first_tp": int(down[:, 4].min()),
            "last_tp": int(down[:, 5].max()),
        }
        trace_result["ug_down_overlap"] = trace_result["first_down_math"] < trace_result["last_up"]
        trace_result["down_tp_overlap"] = trace_result["first_tp"] < trace_result["last_down_math"]
        print(json.dumps(trace_result), flush=True)
        if args.trace_file:
            with open(args.trace_file, "w") as f:
                json.dump({"result": result, "summary": trace_result, "timeline": timeline.tolist()}, f)
        return

    # Distinct epoch per graph layer; changing payloads checks tag reuse.
    graphs = {}
    runs = {"staged": baseline, "fused": candidate}
    for name, run in runs.items():
        step.add_(1)
        run()
        step.add_(1)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for layer in range(16):
                run(layer)
            step.add_(1)
        graphs[name] = graph
    # Change payload between replays to exercise generation-tag reuse. Compare
    # valid intermediate rows and the final output of each graph.
    for _ in range(3):
        x.mul_(0.9)
        graphs["staged"].replay()
        expected_mid, expected_out = mid[index].clone(), out.clone()
        reduced.fill_(float("nan"))
        graphs["fused"].replay()
        torch.cuda.synchronize()
        assert relative_l2(candidate_mid(), expected_mid) < 2e-3
        assert relative_l2(reduced, expected_out) < 1e-2
    result["graph_replay_correct"] = True
    timing = {name: [] for name in graphs}
    for _ in range(10):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    for rep in range(args.repeats):
        order = list(graphs) if rep % 2 == 0 else list(reversed(graphs))
        for name in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            timing[name].append(start.elapsed_time(end) * 1000 / 16)
    result.update({f"{name}_median_us": statistics.median(values) for name, values in timing.items()})
    result["speedup"] = result["staged_median_us"] / result["fused_median_us"]
    result["timings_us"] = timing
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
