import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


UP_AITER = Path("/root/up-aiter")
sys.path.insert(0, str(UP_AITER))

import aiter  # noqa: E402
from aiter import QuantType, dtypes  # noqa: E402
from aiter.ops.flydsl.kernels.kimi_fp4_moe_16384 import (  # noqa: E402
    EXPERTS,
    INTER_DIM,
    MODEL_DIM,
    STAGE1_KERNEL,
    STAGE2_KERNEL,
    TOKEN,
    TOPK,
    run_kimi_fp4_flydsl_moe_16384,
)
from aiter.ops.shuffle import shuffle_weight  # noqa: E402
from aiter.utility.fp4_utils import e8m0_shuffle  # noqa: E402


M = TOKEN


@dataclass(frozen=True)
class KimiShape:
    experts: int = EXPERTS
    model_dim: int = MODEL_DIM
    inter_dim: int = INTER_DIM
    topk: int = TOPK


SHAPE = KimiShape()


def build_flydsl_weights(shape: KimiShape, device: torch.device, seed: int = 0):
    torch.manual_seed(seed)
    quant = aiter.get_torch_quant(QuantType.per_1x32)

    w1 = torch.randn(
        (shape.experts, 2 * shape.inter_dim, shape.model_dim),
        dtype=dtypes.bf16,
        device=device,
    ) / 10
    w2 = torch.randn(
        (shape.experts, shape.model_dim, shape.inter_dim),
        dtype=dtypes.bf16,
        device=device,
    ) / 10

    w1_q, w1_scale = quant(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_scale = quant(w2, quant_dtype=dtypes.fp4x2)

    return {
        "w1": shuffle_weight(w1_q.view(torch.float4_e2m1fn_x2)).view(w1_q.dtype),
        "w2": shuffle_weight(w2_q.view(torch.float4_e2m1fn_x2)).view(w2_q.dtype),
        "w1_scale": e8m0_shuffle(w1_scale).view(
            shape.experts, 2 * shape.inter_dim, shape.model_dim // 32
        ),
        "w2_scale": e8m0_shuffle(w2_scale).view(
            shape.experts, shape.model_dim, shape.inter_dim // 32
        ),
    }


def build_inputs(shape: KimiShape, device: torch.device, seed: int = 1):
    torch.manual_seed(seed)
    hidden = torch.randn((M, shape.model_dim), dtype=dtypes.bf16, device=device) / 10

    routed_experts = shape.experts - 1
    routed_topk = shape.topk - 1
    shared_id = shape.experts - 1

    gen = torch.Generator(device=device).manual_seed(seed)
    bias = torch.randn(routed_experts, generator=gen, device=device) * 0.5
    scores = torch.randn(M, routed_experts, generator=gen, device=device) + bias
    routed_weight, routed_ids = torch.topk(scores.softmax(-1), routed_topk, dim=-1)

    shared_ids = torch.full((M, 1), shared_id, device=device, dtype=routed_ids.dtype)
    shared_weight = torch.ones((M, 1), device=device, dtype=routed_weight.dtype)

    topk_ids = torch.cat([shared_ids, routed_ids], dim=1).to(torch.int32)
    topk_weight = torch.cat([shared_weight, routed_weight], dim=1).to(torch.float32)
    return hidden, topk_ids, topk_weight


def run_flydsl_16384(hidden, topk_ids, topk_weight, weights):
    return run_kimi_fp4_flydsl_moe_16384(
        hidden,
        weights["w1"],
        w2=weights["w2"],
        topk_ids=topk_ids,
        topk_weight=topk_weight,
        w1_scale=weights["w1_scale"],
        w2_scale=weights["w2_scale"].view(dtypes.fp8_e8m0),
    )


def bench_cudagraph(fn, warmup: int, graph_iters: int, measure: int):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_iters):
            fn()

    graph.replay()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(measure)]
    totals = []
    for i in range(measure):
        starts[i].record()
        graph.replay()
        ends[i].record()
    torch.cuda.synchronize()
    for start, end in zip(starts, ends):
        totals.append(start.elapsed_time(end) * 1e3)
    totals.sort()
    return totals[len(totals) // 2] / graph_iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--graph-iters", type=int, default=5)
    parser.add_argument("--measure", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"M={M} NE={SHAPE.experts} H={SHAPE.model_dim} INTER={SHAPE.inter_dim} TOPK={SHAPE.topk}")
    print(f"stage1={STAGE1_KERNEL}")
    print(f"stage2={STAGE2_KERNEL}")

    weights = build_flydsl_weights(SHAPE, device)
    hidden, topk_ids, topk_weight = build_inputs(SHAPE, device)

    def fn():
        return run_flydsl_16384(hidden, topk_ids, topk_weight, weights)

    out = fn()
    torch.cuda.synchronize()
    if not torch.isfinite(out).all().item():
        raise RuntimeError("FlyDSL output has non-finite values")

    us = bench_cudagraph(fn, args.warmup, args.graph_iters, args.measure)
    print(f"flydsl_us={us:.1f}")


if __name__ == "__main__":
    main()
