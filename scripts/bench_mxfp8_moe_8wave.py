#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""MiniMax M3 MXFP8 prefill reproduction and grouped GEMM benchmark.

HIP_VISIBLE_DEVICES=0 python scripts/bench_mxfp8_moe_8wave.py --aiter --check

Routing and weight packing are prepared once. ``pipeline_us`` includes gather /
quantize, fused GEMM1 + SwiGLU, intermediate quantize, GEMM2 and weighted reduction.
``sorted_pipeline_us`` additionally includes GPU sorting and route conversion.
Workspace sizing is done once for the fixed benchmark input, outside timing.
Timings are sums of GPU kernel durations, excluding host dispatch and packing.
``--profile-stage 1|2`` launches only that prepared GEMM, without a nested profiler.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import flydsl.compiler as flyc  # noqa: E402
from kernels.moe.moe_sorting_kernel import moe_sorting_flydsl, moe_sorting_get_workspace_size  # noqa: E402
from kernels.moe.mxfp8_moe_8wave import (  # noqa: E402
    compile_mxfp8_moe_gemm_8w,
    compile_mxfp8_moe_quant,
    compile_mxfp8_moe_reduce,
    compile_mxfp8_moe_unpack_routes,
)
from tests.kernels.test_mxfp8_moe_8wave import prepare_weights, quantize_mxfp8  # noqa: E402
from tests.kernels.utils.gemm_common_utils import e8m0_to_f32  # noqa: E402
from tests.test_common import run_perftest  # noqa: E402


def make_routing(topk_ids, weights, experts):
    """Prepare GPU sorting and maps, with fixed-input workspace sizing."""
    tokens, topk = topk_ids.shape
    device = topk_ids.device
    ids = topk_ids.int()
    max_rows = (tokens * topk + experts * 255 + 255) // 256 * 256
    packed = torch.empty(max_rows, device=device, dtype=torch.int32)
    sorted_weights = torch.empty(max_rows, device=device, dtype=torch.float32)
    expert_ids = torch.empty(max_rows // 256, device=device, dtype=torch.int32)
    num_valid = torch.empty(2, device=device, dtype=torch.int32)
    empty_output = torch.empty(0, device=device, dtype=torch.bfloat16)
    workspace = torch.empty(
        moe_sorting_get_workspace_size(tokens, experts, topk, 256), device=device, dtype=torch.int32
    )

    def sort():
        moe_sorting_flydsl(
            ids, weights, packed, sorted_weights, expert_ids, num_valid, empty_output, experts, 256, workspace=workspace
        )

    sort()
    rows = int(num_valid[0])
    row_map = torch.empty(rows, device=device, dtype=torch.int32)
    inverse = torch.empty(tokens * topk, device=device, dtype=torch.int32)
    unpack = bind(
        compile_mxfp8_moe_unpack_routes(topk=topk), packed, row_map, inverse, rows, tokens, torch.cuda.current_stream()
    )
    unpack()

    def route():
        sort()
        unpack()

    return row_map, inverse, expert_ids[: rows // 256], route


def bind(builder, *args):
    compiled = flyc.compile(builder, *args)
    return lambda: compiled(*args)


def check_reference(x, w1, w2, topk_ids, weights, output, sample_tokens):
    """Independent dequantize / torch matmul reference for complete routed output."""
    sample_tokens = min(sample_tokens, x.shape[0])
    sample_ids = torch.linspace(0, x.shape[0] - 1, sample_tokens, device=x.device, dtype=torch.long)
    xq, sx = quantize_mxfp8(x[sample_ids])
    xf = xq.float() * e8m0_to_f32(sx).repeat_interleave(32, -1)
    topk = topk_ids.shape[1]
    inter = w2.shape[-1]
    routes = topk_ids[sample_ids]
    act = torch.empty(sample_tokens, topk, inter, device=x.device, dtype=torch.bfloat16)
    for e in range(w1.shape[0]):
        rows, slots = torch.where(routes == e)
        if rows.numel():
            q, s = quantize_mxfp8(w1[e])
            wf = q.float() * e8m0_to_f32(s).repeat_interleave(32, -1)
            gate, up = (xf[rows] @ wf.T).chunk(2, -1)
            gate = gate.clamp(max=7)
            act[rows, slots] = (gate * torch.sigmoid(1.702 * gate) * (up.clamp(-7, 7) + 1)).bfloat16()
    aq, sa = quantize_mxfp8(act)
    af = aq.float() * e8m0_to_f32(sa).repeat_interleave(32, -1)
    partial = torch.empty(sample_tokens, topk, x.shape[-1], device=x.device, dtype=torch.bfloat16)
    for e in range(w2.shape[0]):
        rows, slots = torch.where(routes == e)
        if rows.numel():
            q, s = quantize_mxfp8(w2[e])
            wf = q.float() * e8m0_to_f32(s).repeat_interleave(32, -1)
            partial[rows, slots] = (af[rows, slots] @ wf.T).bfloat16()
    ref = (partial.float() * weights[sample_ids, :, None]).sum(1).bfloat16().float()
    actual = output[sample_ids].float()
    assert output.isfinite().all()
    diff = float((actual - ref).square().sum() / (actual.square() + ref.square()).sum())
    assert diff < 2e-4, f"quantized MoE reference logits diff: {diff}"
    return {
        "reference_tokens": sample_tokens,
        "reference_sampling": "uniform over full sequence",
        "quantized_reference_logits_diff": diff,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--inter-dim", type=int, default=384, choices=(384, 768))
    parser.add_argument(
        "--config", type=Path, default=ROOT.parent / "aiter/aiter/configs/model_configs/minimax_m3_mxfp8_tuned_fmoe.csv"
    )
    parser.add_argument("--aiter", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--reference-tokens", type=int, default=128)
    parser.add_argument("--swizzle", type=int, default=1)
    parser.add_argument("--stage2-swizzle", type=int, default=3)
    parser.add_argument("--profile-stage", type=int, choices=(1, 2))
    parser.add_argument("--sweep-swizzle", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.config.open() as f:
        row = next(
            r for r in csv.DictReader(f) if int(r["token"]) == args.tokens and int(r["inter_dim"]) == args.inter_dim
        )
    m, h, inter, experts, topk = (int(row[k]) for k in ("token", "model_dim", "inter_dim", "expert", "topk"))
    os.environ["AITER_CONFIG_FMOE"] = str(args.config.resolve())
    torch.manual_seed(42)
    device = "cuda"
    x = torch.randn(m, h, device=device, dtype=torch.bfloat16)
    # Match test_moe_2stage's random BF16 input / weights / router score.
    w1 = torch.randn(experts, 2 * inter, h, device=device, dtype=torch.bfloat16)
    w2 = torch.randn(experts, h, inter, device=device, dtype=torch.bfloat16)
    score = torch.randn(m, experts, device=device, dtype=torch.bfloat16)
    selected, topk_ids = score.float().topk(topk, dim=-1)
    weights = selected.softmax(-1)
    row_map, inverse, eids, route = make_routing(topk_ids, weights, experts)
    rows = row_map.numel()
    kp = (inter + 255) // 256 * 256
    b1, sb1, _ref = prepare_weights(w1, 1)
    del _ref
    b2, sb2, _ref = prepare_weights(w2, 2)
    del _ref
    aq1 = torch.empty(m, h, device=device, dtype=torch.int8)
    sa1 = torch.full((rows, h // 32), 127, device=device, dtype=torch.uint8)
    act = torch.empty(rows, inter, device=device, dtype=torch.bfloat16)
    aq2 = torch.empty(rows, kp, device=device, dtype=torch.int8)
    sa2 = torch.empty(rows, kp // 32, device=device, dtype=torch.uint8)
    partial = torch.empty(rows, h, device=device, dtype=torch.bfloat16)
    output = torch.empty_like(x)
    stream = torch.cuda.current_stream()
    quant1 = bind(
        compile_mxfp8_moe_quant(K=h, gather=False, scatter_scale_topk=topk),
        x.flatten(),
        aq1.flatten(),
        sa1.flatten(),
        inverse,
        m,
        stream,
    )
    stage1 = bind(
        compile_mxfp8_moe_gemm_8w(K=h, stage=1, xcd_swizzle=args.swizzle, gather_a=True),
        aq1.flatten(),
        b1.flatten(),
        act.flatten(),
        sa1.flatten(),
        sb1.flatten(),
        eids,
        row_map,
        rows,
        2 * inter,
        stream,
    )
    quant2 = bind(
        compile_mxfp8_moe_quant(K=inter, gather=False),
        act.flatten(),
        aq2.flatten(),
        sa2.flatten(),
        row_map,
        rows,
        stream,
    )
    stage2 = bind(
        compile_mxfp8_moe_gemm_8w(K=kp, logical_k=inter, stage=2, xcd_swizzle=args.stage2_swizzle),
        aq2.flatten(),
        b2.flatten(),
        partial.flatten(),
        sa2.flatten(),
        sb2.flatten(),
        eids,
        row_map,
        rows,
        h,
        stream,
    )
    reduce = bind(
        compile_mxfp8_moe_reduce(N=h, topk=topk),
        partial.flatten(),
        output.flatten(),
        inverse,
        weights.flatten(),
        m,
        stream,
    )
    stages = dict(gather_quant=quant1, stage1=stage1, intermediate_quant=quant2, stage2=stage2, reduce=reduce)

    def forward():
        for fn in stages.values():
            fn()
        return output

    if args.profile_stage:
        if args.profile_stage == 1:
            quant1()
        else:
            act.normal_()
            quant2()
        for _ in range(5):
            stages[f"stage{args.profile_stage}"]()
        torch.cuda.synchronize()
        return

    forward()
    torch.cuda.synchronize()
    if args.sweep_swizzle:
        for stage, k, lk, n, a, b, c, sa, sb in (
            (1, h, h, 2 * inter, aq1, b1, act, sa1, sb1),
            (2, kp, inter, h, aq2, b2, partial, sa2, sb2),
        ):
            for sw in (0, 1, 2, 3, 4, 6, 8, 16):
                fn = bind(
                    compile_mxfp8_moe_gemm_8w(K=k, logical_k=lk, stage=stage, xcd_swizzle=sw, gather_a=stage == 1),
                    a.flatten(),
                    b.flatten(),
                    c.flatten(),
                    sa.flatten(),
                    sb.flatten(),
                    eids,
                    row_map,
                    rows,
                    n,
                    stream,
                )
                _, us = run_perftest(fn, num_iters=20, num_warmup=3)
                print(
                    json.dumps(dict(stage=stage, swizzle=sw, us=us, useful_tflops=2 * m * topk * n * lk / us / 1e6)),
                    flush=True,
                )
        return

    result = dict(
        tokens=m,
        hidden_dim=h,
        inter_dim=inter,
        experts=experts,
        topk=topk,
        sorted_rows=rows,
        padded_k=kp,
        gpu=torch.cuda.get_device_name(),
        swizzle=args.swizzle,
        stage2_swizzle=args.stage2_swizzle,
        seed=42,
        routing_in_pipeline=False,
    )
    if args.check:
        result.update(check_reference(x, w1, w2, topk_ids, weights, output, args.reference_tokens))
    for name, fn in stages.items():
        _, us = run_perftest(fn, num_iters=20, num_warmup=3)
        result[name + "_us"] = us
    _, result["pipeline_us"] = run_perftest(forward, num_iters=20, num_warmup=3)

    def sorted_forward():
        route()
        return forward()

    _, result["sorting_and_maps_us"] = run_perftest(route, num_iters=20, num_warmup=3)
    _, result["sorted_pipeline_us"] = run_perftest(sorted_forward, num_iters=20, num_warmup=3)
    result["timing"] = "sum of GPU kernel durations; preallocated workspace; fixed input"
    result["sorted_pipeline_useful_tflops"] = 6 * m * topk * h * inter / result["sorted_pipeline_us"] / 1e6
    result["stage1_useful_tflops"] = 4 * m * topk * h * inter / result["stage1_us"] / 1e6
    result["stage2_useful_tflops"] = 2 * m * topk * h * inter / result["stage2_us"] / 1e6
    result["pipeline_useful_tflops"] = 6 * m * topk * h * inter / result["pipeline_us"] / 1e6

    if args.check:
        # Re-check after repeatedly rebuilding routing during the timed path.
        sorted_forward()
        sample_ids = torch.linspace(0, m - 1, min(args.reference_tokens, m), device=x.device, dtype=torch.long)
        snapshot = output[sample_ids].clone()
        for _ in range(3):
            sorted_forward()
            assert torch.equal(output[sample_ids], snapshot), "MoE output changed across identical launches"
        result.update(check_reference(x, w1, w2, topk_ids, weights, output, args.reference_tokens))

    if args.aiter:
        import importlib

        import aiter
        from aiter.ops.quant import per_1x32_f8_scale_f8_quant
        from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
        from aiter.utility import fp4_utils

        module = importlib.import_module("aiter.fused_moe")
        wq1, ws1 = per_1x32_f8_scale_f8_quant(w1, scale_type=aiter.dtypes.fp8_e8m0)
        wq2, ws2 = per_1x32_f8_scale_f8_quant(w2, scale_type=aiter.dtypes.fp8_e8m0)
        wq1, wq2 = shuffle_weight_a16w4(wq1, 16, True), shuffle_weight_a16w4(wq2, 16, False)
        ws1 = shuffle_scale_a16w4(ws1, experts, True)
        ws2 = fp4_utils.e8m0_shuffle(ws2)

        topk_ids_i32 = topk_ids.int()

        def baseline():
            return module.fused_moe(
                x,
                wq1,
                wq2,
                weights,
                topk_ids_i32,
                w1_scale=ws1,
                w2_scale=ws2,
                quant_type=aiter.QuantType.per_1x32,
                activation=aiter.ActivationType.Swiglu,
                doweight_stage1=False,
                gate_mode="interleave",
            )

        calls = []
        module.kernel_bench_callable = calls
        try:
            # Atomic stage-2 microbenchmarks mutate the original output buffer.
            baseline_output = baseline().clone()
        finally:
            module.kernel_bench_callable = None
        for name, fn in calls:
            _, result["aiter_" + name + "_us"] = run_perftest(fn, num_iters=20, num_warmup=3)
        _, result["aiter_e2e_us"] = run_perftest(baseline, num_iters=20, num_warmup=3)
        a, b = output.float(), baseline_output.float()
        result["aiter_output_logits_diff"] = float((a - b).square().sum() / (a.square() + b.square()).sum())
    rendered = json.dumps(result, indent=2)
    print(rendered, flush=True)
    if args.output:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
