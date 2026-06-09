import os
os.system("rm -rf ~/.flydsl")
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
UP_AITER = Path("/root/up-aiter")
BUILD_PYTHON_PACKAGES = THIS_DIR / "build-fly" / "python_packages"
if BUILD_PYTHON_PACKAGES.exists():
    sys.path.insert(0, str(BUILD_PYTHON_PACKAGES))
sys.path.insert(0, str(UP_AITER))
sys.path.insert(0, str(THIS_DIR))

import aiter  # noqa: E402
from aiter import ActivationType, QuantType, dtypes  # noqa: E402
from aiter.fused_moe import fused_moe  # noqa: E402
from aiter.ops.shuffle import shuffle_scale, shuffle_weight  # noqa: E402
from aiter.utility.fp4_utils import e8m0_shuffle  # noqa: E402
from kimi_fp4_moe_16384 import (  # noqa: E402
    EXPERTS,
    INTER_DIM,
    MODEL_DIM,
    STAGE1_KERNEL,
    STAGE2_KERNEL,
    TOKEN,
    TOPK,
    run_kimi_fp4_flydsl_moe_16384,
)
from kimi_fp4_moe_16384_opt import (  # noqa: E402
    MXFP4_STAGE1_KERNEL,
    MXFP4_STAGE2_KERNEL,
    run_kimi_fp4_flydsl_atomic_stage2_16384,
    run_kimi_fp4_flydsl_mxfp4_sort_16384,
    run_kimi_fp4_mxfp4_moe_16384_aiter_ref,
    run_kimi_fp4_mxfp4_moe_16384_opt,
    run_kimi_fp4_mxfp4_moe_16384_opt_gemm1,
    run_kimi_fp4_mxfp4_moe_16384_opt_gemm2,
)


M = TOKEN


@dataclass(frozen=True)
class KimiShape:
    experts: int = EXPERTS
    model_dim: int = MODEL_DIM
    inter_dim: int = INTER_DIM
    topk: int = TOPK


SHAPE = KimiShape()


def _mark_flydsl_shuffled(weights):
    for t in weights.values():
        t.is_shuffled = True
    return weights


def _mark_mxfp4_shuffled(weights):
    for t in weights.values():
        t.is_shuffled = True
        t.shuffle_kind = "mxfp4_moe"
    return weights


def build_weights(shape: KimiShape, device: torch.device, seed: int = 0):
    torch.manual_seed(seed)
    quant = aiter.get_torch_quant(QuantType.per_1x32)
    ne, h, inter = shape.experts, shape.model_dim, shape.inter_dim

    w1 = torch.randn((ne, 2 * inter, h), dtype=dtypes.bf16, device=device) / 10
    w2 = torch.randn((ne, h, inter), dtype=dtypes.bf16, device=device) / 10
    w1_q, w1_scale = quant(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_scale = quant(w2, quant_dtype=dtypes.fp4x2)

    flydsl = _mark_flydsl_shuffled(
        {
            "w1": shuffle_weight(w1_q.view(torch.float4_e2m1fn_x2)).view(w1_q.dtype),
            "w2": shuffle_weight(w2_q.view(torch.float4_e2m1fn_x2)).view(w2_q.dtype),
            "w1_scale": e8m0_shuffle(w1_scale).view(ne, 2 * inter, h // 32),
            "w2_scale": e8m0_shuffle(w2_scale).view(ne, h, inter // 32),
        }
    )

    mxfp4 = _mark_mxfp4_shuffled(
        {
            "w1": shuffle_weight(
                w1_q,
                layout=(16, 16),
                is_guinterleave=True,
                gate_up=True,
            ).view(w1_q.dtype),
            "w2": shuffle_weight(
                w2_q,
                layout=(16, 16),
                is_guinterleave=True,
                gate_up=False,
            ).view(w2_q.dtype),
            "w1_scale": shuffle_scale(
                w1_scale.view(ne * 2 * inter, h // 32),
                experts_cnt=ne,
                is_guinterleave=True,
                gate_up=True,
            ).view(ne, 2 * inter, h // 32),
            "w2_scale": shuffle_scale(
                w2_scale.view(ne * h, inter // 32),
                experts_cnt=ne,
                is_guinterleave=True,
                gate_up=False,
            ).view(ne, h, inter // 32),
        }
    )

    return {"flydsl": flydsl, "mxfp4": mxfp4}


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


def run_aiter_moe(hidden, topk_ids, topk_weight, weights):
    return fused_moe(
        hidden,
        weights["w1"],
        weights["w2"],
        topk_weight,
        topk_ids,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        w1_scale=weights["w1_scale"],
        w2_scale=weights["w2_scale"],
    )


def run_local_kimi(hidden, topk_ids, topk_weight, weights):
    return run_kimi_fp4_flydsl_moe_16384(
        hidden,
        weights["w1"],
        w2=weights["w2"],
        topk_ids=topk_ids,
        topk_weight=topk_weight,
        w1_scale=weights["w1_scale"],
        w2_scale=weights["w2_scale"],
    )


def run_local_kimi_mxfp4_sort(hidden, topk_ids, topk_weight, weights):
    return run_kimi_fp4_flydsl_mxfp4_sort_16384(
        hidden,
        weights["w1"],
        w2=weights["w2"],
        topk_ids=topk_ids,
        topk_weight=topk_weight,
        w1_scale=weights["w1_scale"],
        w2_scale=weights["w2_scale"],
    )


def run_local_kimi_atomic_stage2(hidden, topk_ids, topk_weight, weights):
    return run_kimi_fp4_flydsl_atomic_stage2_16384(
        hidden,
        weights["w1"],
        w2=weights["w2"],
        topk_ids=topk_ids,
        topk_weight=topk_weight,
        w1_scale=weights["w1_scale"],
        w2_scale=weights["w2_scale"],
    )


def run_local_mxfp4_opt(hidden, topk_ids, topk_weight, weights):
    return run_kimi_fp4_mxfp4_moe_16384_opt(
        hidden,
        weights["w1"],
        w2=weights["w2"],
        topk_ids=topk_ids,
        topk_weight=topk_weight,
        w1_scale=weights["w1_scale"],
        w2_scale=weights["w2_scale"],
    )


def run_local_mxfp4_opt_gemm2(hidden, topk_ids, topk_weight, weights):
    return run_kimi_fp4_mxfp4_moe_16384_opt_gemm2(
        hidden,
        weights["w1"],
        w2=weights["w2"],
        topk_ids=topk_ids,
        topk_weight=topk_weight,
        w1_scale=weights["w1_scale"],
        w2_scale=weights["w2_scale"],
    )


def run_local_mxfp4_opt_gemm1(hidden, topk_ids, topk_weight, weights):
    return run_kimi_fp4_mxfp4_moe_16384_opt_gemm1(
        hidden,
        weights["w1"],
        w2=weights["w2"],
        topk_ids=topk_ids,
        topk_weight=topk_weight,
        w1_scale=weights["w1_scale"],
        w2_scale=weights["w2_scale"],
    )


def run_local_mxfp4_aiter_ref(hidden, topk_ids, topk_weight, weights):
    return run_kimi_fp4_mxfp4_moe_16384_aiter_ref(
        hidden,
        weights["w1"],
        w2=weights["w2"],
        topk_ids=topk_ids,
        topk_weight=topk_weight,
        w1_scale=weights["w1_scale"],
        w2_scale=weights["w2_scale"],
    )


def make_runners(hidden, topk_ids, topk_weight, weights):
    return {
        "aiter_moe": lambda: run_aiter_moe(
            hidden,
            topk_ids,
            topk_weight,
            weights["flydsl"],
        ),
        "aiter_mxfp4_moe": lambda: run_aiter_moe(
            hidden,
            topk_ids,
            topk_weight,
            weights["mxfp4"],
        ),
        "local_kimi_fp4": lambda: run_local_kimi(
            hidden,
            topk_ids,
            topk_weight,
            weights["flydsl"],
        ),
        "local_kimi_fp4_mxfp4_sort": lambda: run_local_kimi_mxfp4_sort(
            hidden,
            topk_ids,
            topk_weight,
            weights["flydsl"],
        ),
        "local_kimi_fp4_atomic_stage2": lambda: run_local_kimi_atomic_stage2(
            hidden,
            topk_ids,
            topk_weight,
            weights["flydsl"],
        ),
        "local_mxfp4_aiter_ref": lambda: run_local_mxfp4_aiter_ref(
            hidden,
            topk_ids,
            topk_weight,
            weights["mxfp4"],
        ),
        "local_mxfp4_opt": lambda: run_local_mxfp4_opt(
            hidden,
            topk_ids,
            topk_weight,
            weights["mxfp4"],
        ),
        "local_mxfp4_opt_gemm1": lambda: run_local_mxfp4_opt_gemm1(
            hidden,
            topk_ids,
            topk_weight,
            weights["mxfp4"],
        ),
        "local_mxfp4_opt_gemm2": lambda: run_local_mxfp4_opt_gemm2(
            hidden,
            topk_ids,
            topk_weight,
            weights["mxfp4"],
        ),
    }


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


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().reshape(-1),
        b.float().reshape(-1),
        dim=0,
    ).item()


def max_abs(a, b):
    return (a.float() - b.float()).abs().max().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--graph-iters", type=int, default=5)
    parser.add_argument("--measure", type=int, default=5)
    parser.add_argument(
        "--runners",
        default=None,
        help="comma-separated runner names; defaults to all runners",
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"M={M} NE={SHAPE.experts} H={SHAPE.model_dim} "
        f"INTER={SHAPE.inter_dim} TOPK={SHAPE.topk}"
    )
    print(f"stage1={STAGE1_KERNEL}")
    print(f"stage2={STAGE2_KERNEL}")
    print(f"mxfp4_stage1={MXFP4_STAGE1_KERNEL}")
    print(f"mxfp4_stage2={MXFP4_STAGE2_KERNEL}")

    weights = build_weights(SHAPE, device)
    hidden, topk_ids, topk_weight = build_inputs(SHAPE, device)

    runners = make_runners(hidden, topk_ids, topk_weight, weights)
    if args.runners:
        requested = [item.strip() for item in args.runners.split(",") if item.strip()]
        unknown = [name for name in requested if name not in runners]
        if unknown:
            raise ValueError(f"unknown runners: {', '.join(unknown)}")
        runners = {name: runners[name] for name in requested}

    outputs = {}
    for name, fn in runners.items():
        out = fn()
        torch.cuda.synchronize()
        if not torch.isfinite(out).all().item():
            raise RuntimeError(f"{name} output has non-finite values")
        outputs[name] = out

    check_specs = [
        ("aiter_mxfp4_vs_aiter", "aiter_mxfp4_moe", "aiter_moe"),
        ("local_vs_aiter", "local_kimi_fp4", "aiter_moe"),
        ("local_mxfp4_sort_vs_aiter", "local_kimi_fp4_mxfp4_sort", "aiter_moe"),
        ("local_atomic_stage2_vs_aiter", "local_kimi_fp4_atomic_stage2", "aiter_moe"),
        ("local_mxfp4_ref_vs_aiter_mxfp4", "local_mxfp4_aiter_ref", "aiter_mxfp4_moe"),
        ("local_mxfp4_opt_vs_aiter_mxfp4", "local_mxfp4_opt", "aiter_mxfp4_moe"),
        ("local_mxfp4_opt_gemm1_vs_aiter_mxfp4", "local_mxfp4_opt_gemm1", "aiter_mxfp4_moe"),
        ("local_mxfp4_opt_gemm2_vs_aiter_mxfp4", "local_mxfp4_opt_gemm2", "aiter_mxfp4_moe"),
    ]
    for name, lhs_name, rhs_name in check_specs:
        if lhs_name not in outputs or rhs_name not in outputs:
            continue
        lhs = outputs[lhs_name]
        rhs = outputs[rhs_name]
        print(
            f"{name}_cos={cosine(lhs, rhs):.6f} "
            f"{name}_max_abs={max_abs(lhs, rhs):.6f}"
        )

    for name, fn in runners.items():
        us = bench_cudagraph(fn, args.warmup, args.graph_iters, args.measure)
        print(f"{name}_us={us:.1f}")


if __name__ == "__main__":
    main()
