# SPDX-License-Identifier: Apache-2.0
"""Check and benchmark the complete DSV4-Pro layer-forward interface.

Like the K3 PR's tool, this runs one prepared layer per TP rank. The initial
DSV4 implementation uses native ATOM indexer/attention/mHC and a FlyDSL MoE.
Its comparison splits MoE into five stages, retaining the same layer inputs.
"""

import argparse
import json
import math
import os
from pathlib import Path

from ..config import validate_shape
from .compare import bench_graph, error_metrics


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--layer-idx", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-lens", "--samples", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--tp", type=int, choices=(1, 2, 4, 8), default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--replays", type=int, default=3)
    p.add_argument(
        "--baseline-repeats",
        type=int,
        default=2,
        help="same-input staged graph runs per replay, to record native drift",
    )
    p.add_argument("--check", action="store_true", help="checks always run, including before benchmarking")
    p.add_argument("--bench", action="store_true")
    p.add_argument(
        "--profile-launches",
        action="store_true",
        help="record actual device kernels for one replay per path; diagnostic even on failed cases",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--graph-iters", type=int, default=10)
    p.add_argument("--max-nrmse", type=float, default=0.015)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    for seq in a.seq_lens:
        try:
            validate_shape(a.batch_size, seq)
        except ValueError as exc:
            p.error(str(exc))
    if not 0 <= a.layer_idx <= 60:
        p.error("--layer-idx must be in 0..60")
    if not math.isfinite(a.max_nrmse) or not 0 < a.max_nrmse < 1:
        p.error("--max-nrmse must be finite and in (0,1)")
    for name in ("replays", "baseline_repeats", "warmup", "repeats", "graph_iters"):
        if getattr(a, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    return a


def main(argv=None):
    args = parse_args(argv)
    os.environ["ATOM_DSV4_MONOKERNEL"] = "1"
    os.environ.setdefault("AITER_BF16_FP8_MOE_BOUND", "0")
    os.environ.setdefault("ATOM_MOE_GU_ITLV", "1")
    import torch
    import torch.distributed as dist
    from aiter.dist.parallel_state import graph_capture
    from atom.models.deepseek_v4 import HCState

    from ..atom_layer import AtomLayerModule
    from ..checkpoint import load_moe

    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("LOCAL_RANK", "0"))
    if world != args.tp:
        raise ValueError(f"use torchrun with WORLD_SIZE={args.tp}")
    torch.cuda.set_device(rank)
    if world > 1:
        dist.init_process_group("nccl")
    print(f"rank={rank}: loading complete layer {args.layer_idx}", flush=True)
    cfg, weights, shared, table = load_moe(args.checkpoint, args.layer_idx, world, rank, "cuda")
    fixture = AtomLayerModule(args.checkpoint, args.layer_idx, weights, shared, table, world, rank)
    fields = ("residual", "post_mix", "comb_mix", "x_prev")
    initial_cache = [t.clone() for t in fixture.cache_tensors]

    def reset_cache():
        for dst, src in zip(fixture.cache_tensors, initial_cache):
            dst.copy_(src)

    def copy_state(state):
        return HCState(
            **{k: getattr(state, k).clone() if getattr(state, k) is not None else None for k in fields},
            res_preshuffle=state.res_preshuffle,
        )

    results = []
    for seq in args.seq_lens:
        token_ids = torch.zeros(seq, dtype=torch.int64, device="cuda")
        md, ctx = fixture.metadata(seq, token_ids)
        # Include the delayed previous sublayer, the state handed to every
        # backbone layer after layer zero. Preserve its FP32 mixing matrices.
        state = HCState(torch.zeros(seq, 4, cfg.hidden, dtype=torch.bfloat16, device="cuda"))
        if args.layer_idx:
            state.x_prev = torch.zeros(seq, cfg.hidden, dtype=torch.bfloat16, device="cuda")
            state.post_mix = torch.ones(seq, 4, device="cuda")
            state.comb_mix = torch.eye(4, device="cuda").expand(seq, -1, -1).contiguous()

        def call(unfused=False):
            if unfused:
                return fixture.block.mono_kernel_forward(state, ctx.positions, unfused=True)
            return fixture.block(state, ctx.positions)

        # Rejections must happen before indexer/compressor/cache side effects.
        before_guard = [t.clone() for t in fixture.cache_tensors]
        real_bs = ctx.running_bs
        ctx.running_bs = 2
        try:
            try:
                fixture.block.mono_kernel_forward(state, ctx.positions)
            except ValueError:
                pass
            else:
                raise AssertionError("multi-request layer call was not rejected")
        finally:
            ctx.running_bs = real_bs
        if any(not torch.equal(a, b) for a, b in zip(before_guard, fixture.cache_tensors)):
            raise AssertionError("unsupported call modified a layer cache")
        call()
        call(True)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with graph_capture() as capture:
            with torch.cuda.graph(graph, stream=capture.stream):
                captured = call()
        staged_graph = torch.cuda.CUDAGraph()
        with graph_capture() as capture:
            with torch.cuda.graph(staged_graph, stream=capture.stream):
                captured_staged = call(True)
        checks = []
        previous_output = None
        for replay in range(args.replays):
            torch.manual_seed(args.seed + seq * 1000 + replay)
            state.residual.normal_(0, 0.2)
            if state.x_prev is not None:
                state.x_prev.normal_(0, 0.2)
            token_ids.random_(0, 129280)
            reset_cache()
            graph.replay()
            mono = copy_state(captured)
            if previous_output is not None and torch.equal(previous_output, mono.x_prev):
                raise AssertionError("changed input did not change the captured layer output")
            previous_output = mono.x_prev.clone()
            cache = [t.clone() for t in fixture.cache_tensors]
            cache_fields = {k: v.clone() for k, v in fixture.cache_observation(md, ctx)[0].items()}
            reset_cache()
            staged_graph.replay()
            staged = copy_state(captured_staged)
            metrics = {k: error_metrics(getattr(mono, k), getattr(staged, k)) for k in fields}
            cache_changed = [int((a != b).sum()) for a, b in zip(cache, fixture.cache_tensors)]
            cache_field_errors = {}
            fields_now, numerical_regions = fixture.cache_observation(md, ctx)
            for key, actual in fields_now.items():
                expected = cache_fields[key]
                finite = torch.isfinite(actual) & torch.isfinite(expected)
                cache_field_errors[key] = {
                    "changed_values": int((actual != expected).sum()),
                    "changed_bytes": int(
                        (actual.contiguous().view(torch.uint8) != expected.contiguous().view(torch.uint8)).sum()
                    ),
                    "nonfinite_mismatch": int((~finite & (actual != expected)).sum()),
                    "finite_error": error_metrics(actual[finite], expected[finite]) if finite.any() else None,
                }
            # Native split-K drift reaches FP32 compressor rings and their
            # newly quantized rows. Compare logical values at the same NRMSE
            # limit; require all bytes outside these exact regions to match.
            outside_state = []
            for expected, actual in zip(cache, fixture.cache_tensors):
                changed = actual != expected
                for field in numerical_regions:
                    if field.untyped_storage().data_ptr() == actual.untyped_storage().data_ptr():
                        if not field.is_contiguous():
                            raise AssertionError("unexpected numerical cache field layout")
                        offset = field.storage_offset() * field.element_size()
                        changed[offset : offset + field.numel() * field.element_size()] = False
                outside_state.append(int(changed.sum()))
            fields_passed = all(
                value["nonfinite_mismatch"] == 0
                and (value["finite_error"] is None or value["finite_error"]["nrmse"] <= args.max_nrmse)
                for value in cache_field_errors.values()
            )
            check = {
                "replay": replay,
                "state": metrics,
                "cache_changed_bytes": cache_changed,
                "output_rms": float(mono.x_prev.float().square().mean().sqrt()),
                "cache_fields": cache_field_errors,
                "cache_changed_bytes_outside_numerical_regions": outside_state,
                "passed": all(m["nrmse"] <= args.max_nrmse for m in metrics.values())
                and not any(outside_state)
                and fields_passed
                and mono.res_preshuffle == staged.res_preshuffle,
            }
            # Diagnose the native attention/compressor independently: these
            # are identical staged graphs on identical inputs and cache state.
            # Their drift never excuses a failed mono/staged comparison.
            baseline_fields = {k: v.clone() for k, v in fields_now.items()}
            baseline_drift = []
            for _ in range(args.baseline_repeats - 1):
                reset_cache()
                staged_graph.replay()
                repeated_fields, _ = fixture.cache_observation(md, ctx)
                field_drift = {}
                for k, value in repeated_fields.items():
                    expected = baseline_fields[k]
                    finite = torch.isfinite(value) & torch.isfinite(expected)
                    field_drift[k] = {
                        "nonfinite_mismatch": int((~finite & (value != expected)).sum()),
                        "finite_error": error_metrics(value[finite], expected[finite]) if finite.any() else None,
                    }
                baseline_drift.append(
                    {
                        "state": {k: error_metrics(getattr(captured_staged, k), getattr(staged, k)) for k in fields},
                        "cache_fields": field_drift,
                    }
                )
            check["baseline_repeat_drift"] = baseline_drift
            checks.append(check)
            print(f"rank={rank} seq={seq}: {json.dumps(check)}", flush=True)
        case = {
            "seq_len": seq,
            "shape_guard_before_cache_write": True,
            "checks": checks,
            "passed": all(c["passed"] for c in checks),
        }
        # Every rank must agree before entering collective graph capture.
        passed = torch.tensor(int(case["passed"]), device="cuda")
        if world > 1:
            dist.all_reduce(passed, op=dist.ReduceOp.MIN)
        if args.bench and passed.item():
            case["mono"] = bench_graph(call, args, atom_collectives=True)
            case["staged"] = bench_graph(lambda: call(True), args, atom_collectives=True)
        if args.profile_launches:
            case["launch_profiles"] = {}
            for mode, captured_graph in (("mono", graph), ("staged", staged_graph)):
                reset_cache()
                torch.cuda.synchronize()
                if world > 1:
                    dist.barrier()
                # One replay only: profiler setup, cache reset and TP barrier
                # are outside the interval. Count device kernel events, not
                # the single hipGraphLaunch host API call.
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                ) as prof:
                    captured_graph.replay()
                    torch.cuda.synchronize()
                trace_path = args.output.with_name(f"{args.output.stem}.rank{rank}.seq{seq}.{mode}.trace.json")
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                prof.export_chrome_trace(str(trace_path))
                events = json.loads(trace_path.read_text())["traceEvents"]
                kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
                names = {}
                for event in kernels:
                    name = event["name"]
                    names[name] = names.get(name, 0) + 1
                if not kernels:
                    raise RuntimeError(f"profiler recorded no device kernels; inspect {trace_path}")
                case["launch_profiles"][mode] = {
                    "kernel_count": len(kernels),
                    "kernels": names,
                    "trace": str(trace_path),
                }
        results.append(case)
    result = {
        "component": "layer",
        "model": "DSV4-Pro",
        "layer": args.layer_idx,
        "attention_kind": cfg.attention_kind(args.layer_idx),
        "indexer_enabled": fixture.block.attn.indexer is not None and not fixture.block.attn.skip_topk,
        "tp": world,
        "rank": rank,
        "seed": args.seed,
        "cases": results,
        "max_nrmse": args.max_nrmse,
        "cache_comparison": "logical compressor/quantized values: same NRMSE limit; all other bytes: exact",
        "metadata": "native ATOM decode capture fixture, private initially zero cache",
        "launch_contract": "multiple launches per layer forward",
        "compressor_projection": "mono adapter scoped Torch BF16 RNE",
        "routed_mid_contract": "FP32 SwiGLU directly to per32 FP8, native fused exponent",
        "baseline": "native attention/mHC plus five-stage FlyDSL MoE",
    }
    result["comparison_execution"] = "CUDA Graph for both mono and staged"
    result["baseline_repeats"] = args.baseline_repeats
    result["bf16_gemm_config_override"] = os.environ.get("AITER_CONFIG_GEMM_BF16")
    if world > 1:
        ranks = [None] * world
        dist.all_gather_object(ranks, result)
        if rank == 0:
            result["all_ranks_passed"] = all(c["passed"] for r in ranks for c in r["cases"])
            result["tp_summary"] = [
                {
                    "seq_len": c["seq_len"],
                    **{
                        mode: max(r["cases"][i][mode]["median_us"] for r in ranks)
                        for mode in ("mono", "staged")
                        if mode in c
                    },
                }
                for i, c in enumerate(results)
            ]
    path = args.output.with_name(args.output.stem + f".rank{rank}" + args.output.suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    fixture.close()
    if dist.is_initialized():
        dist.destroy_process_group()
    if not all(c["passed"] for c in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
