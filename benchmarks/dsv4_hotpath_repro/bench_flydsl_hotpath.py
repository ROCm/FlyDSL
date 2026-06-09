#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""FlyDSL-only hotpath replay for the DeepSeek-V4-Pro c=256 regression.

This benchmark intentionally does not import AITER or ATOM.  It builds small
FlyDSL kernels with AITER-like launcher signatures, then replays the same
high-frequency host call pattern seen in the non-DPA c=256 trace.

The measured signal is host-side launch overhead after compilation:
JitFunction/cache-key/TensorAdaptor/CallState behavior, not model math.
"""

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

import flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx

try:
    from flydsl.runtime.device import get_rocm_arch
except Exception:  # pragma: no cover - older FlyDSL variants may not expose it
    get_rocm_arch = None


BLOCK_DIM = 256
VEC_WIDTH = 4
TILE_ELEMS = BLOCK_DIM * VEC_WIDTH

PROFILE_CALLS = {
    # Rank-local mean calls per prefill window from the c=256 trace.
    # These counts are used to reproduce the same host-launch pressure without
    # requiring AITER, ATOM, model weights, or a serving process.
    "dsv4-c256": {
        "qk": 81,
        "fused_compress": 80,
        "hca_compress": 41,
        "hca_scatter": 41,
        "moe": 0,
    }
}


@flyc.kernel
def _pointer_add_kernel(
    a: fx.Pointer,
    b: fx.Pointer,
    out: fx.Pointer,
    n: fx.Int32,
):
    idx = fx.block_idx.x * fx.block_dim.x + fx.thread_idx.x
    if idx < n:
        out[idx] = a[idx] + b[idx]


@flyc.jit
def launch_qk_norm_rope_quant_like(
    q_in: fx.Pointer,
    kv_in: fx.Pointer,
    q_weight: fx.Tensor,
    kv_weight: fx.Tensor,
    cos_cache: fx.Tensor,
    sin_cache: fx.Tensor,
    positions: fx.Pointer,
    q_out: fx.Pointer,
    kv_out: fx.Pointer,
    q_scale: fx.Pointer,
    kv_scale: fx.Pointer,
    kv_in_row_stride: fx.Int32,
    num_tokens: fx.Int32,
    block_dim: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    grid_x = (num_tokens + block_dim - 1) // block_dim
    _pointer_add_kernel(q_in, kv_in, q_out, num_tokens).launch(
        grid=(grid_x, 1, 1),
        block=(block_dim, 1, 1),
        stream=stream,
    )


@flyc.kernel
def _tensor_add_kernel(
    a: fx.Tensor,
    b: fx.Tensor,
    out: fx.Tensor,
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x
    tile_elems = block_dim * vec_width

    ta = fx.logical_divide(a, fx.make_layout(tile_elems, 1))
    tb = fx.logical_divide(b, fx.make_layout(tile_elems, 1))
    tout = fx.logical_divide(out, fx.make_layout(tile_elems, 1))
    ta = fx.slice(ta, (None, bid))
    tb = fx.slice(tb, (None, bid))
    tout = fx.slice(tout, (None, bid))

    ta = fx.logical_divide(ta, fx.make_layout(vec_width, 1))
    tb = fx.logical_divide(tb, fx.make_layout(vec_width, 1))
    tout = fx.logical_divide(tout, fx.make_layout(vec_width, 1))

    copy_bits = vec_width * 32
    copy_atom = fx.make_copy_atom(fx.UniversalCopy(copy_bits), fx.Float32)

    ra = fx.make_rmem_tensor(vec_width, fx.Float32)
    rb = fx.make_rmem_tensor(vec_width, fx.Float32)
    rout = fx.make_rmem_tensor(vec_width, fx.Float32)

    fx.copy_atom_call(copy_atom, fx.slice(ta, (None, tid)), ra)
    fx.copy_atom_call(copy_atom, fx.slice(tb, (None, tid)), rb)
    fx.memref_store_vec(fx.arith.addf(fx.memref_load_vec(ra), fx.memref_load_vec(rb)), rout)
    fx.copy_atom_call(copy_atom, rout, fx.slice(tout, (None, tid)))


@flyc.jit
def launch_fused_compress_attn_like(
    kv_in: fx.Tensor,
    kv_in_row_stride: fx.Int32,
    score_in: fx.Tensor,
    score_in_row_stride: fx.Int32,
    plan: fx.Tensor,
    kv_state: fx.Tensor,
    kv_state_slot_stride: fx.Int32,
    kv_state_pos_stride: fx.Int32,
    score_state: fx.Tensor,
    score_state_slot_stride: fx.Int32,
    score_state_pos_stride: fx.Int32,
    state_slot_mapping: fx.Tensor,
    ape: fx.Tensor,
    rms_weight: fx.Tensor,
    cos_cache: fx.Tensor,
    sin_cache: fx.Tensor,
    kv_cache: fx.Tensor,
    kv_cache_block_stride: fx.Int32,
    kv_cache_token_stride: fx.Int32,
    cache_scale: fx.Tensor,
    cache_scale_block_stride: fx.Int32,
    block_table: fx.Tensor,
    block_table_seq_stride: fx.Int32,
    plan_capacity: fx.Int32,
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    grid_x = (plan_capacity + block_dim * vec_width - 1) // (block_dim * vec_width)
    _tensor_add_kernel(kv_in, score_in, kv_state, block_dim, vec_width).launch(
        grid=(grid_x, 1, 1),
        block=(block_dim, 1, 1),
        stream=stream,
    )


@flyc.jit
def launch_hca_compress_forward_like(
    kv_in: fx.Tensor,
    kv_in_row_stride: fx.Int32,
    score_in: fx.Tensor,
    score_in_row_stride: fx.Int32,
    plan: fx.Tensor,
    kv_state: fx.Tensor,
    kv_state_slot_stride: fx.Int32,
    kv_state_pos_stride: fx.Int32,
    score_state: fx.Tensor,
    score_state_slot_stride: fx.Int32,
    score_state_pos_stride: fx.Int32,
    state_slot_mapping: fx.Tensor,
    ape: fx.Tensor,
    kv_compressed: fx.Tensor,
    kv_compressed_row_stride: fx.Int32,
    plan_capacity: fx.Int32,
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    grid_x = (plan_capacity + block_dim * vec_width - 1) // (block_dim * vec_width)
    _tensor_add_kernel(kv_in, score_in, kv_compressed, block_dim, vec_width).launch(
        grid=(grid_x, 1, 1),
        block=(block_dim, 1, 1),
        stream=stream,
    )


@flyc.jit
def launch_hca_norm_rope_scatter_like(
    kv_compressed: fx.Tensor,
    kv_compressed_row_stride: fx.Int32,
    plan: fx.Tensor,
    rms_weight: fx.Tensor,
    cos_cache: fx.Tensor,
    sin_cache: fx.Tensor,
    kv_cache: fx.Tensor,
    kv_cache_block_stride: fx.Int32,
    kv_cache_token_stride: fx.Int32,
    block_table: fx.Tensor,
    block_table_seq_stride: fx.Int32,
    plan_capacity: fx.Int32,
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    grid_x = (plan_capacity + block_dim * vec_width - 1) // (block_dim * vec_width)
    _tensor_add_kernel(kv_compressed, kv_cache, kv_cache, block_dim, vec_width).launch(
        grid=(grid_x, 1, 1),
        block=(block_dim, 1, 1),
        stream=stream,
    )


@flyc.jit
def launch_moe_like(
    hidden: fx.Tensor,
    weight0: fx.Tensor,
    weight1: fx.Tensor,
    expert_ids: fx.Tensor,
    topk_weight: fx.Tensor,
    out: fx.Tensor,
    num_tokens: fx.Int32,
    block_dim: fx.Constexpr[int],
    vec_width: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    grid_x = (num_tokens + block_dim * vec_width - 1) // (block_dim * vec_width)
    _tensor_add_kernel(hidden, weight0, out, block_dim, vec_width).launch(
        grid=(grid_x, 1, 1),
        block=(block_dim, 1, 1),
        stream=stream,
    )


@dataclass
class LauncherSpec:
    name: str
    fn: Callable
    args: tuple
    args_without_stream: tuple


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = int(math.ceil((pct / 100.0) * len(ordered))) - 1
    return ordered[min(max(idx, 0), len(ordered) - 1)]


def _bench_host(fn: Callable, calls: int, warmup_calls: int) -> dict:
    for _ in range(warmup_calls):
        fn()
    torch.cuda.synchronize()

    samples = []
    t0 = time.perf_counter_ns()
    for _ in range(calls):
        s = time.perf_counter_ns()
        fn()
        e = time.perf_counter_ns()
        samples.append((e - s) / 1000.0)
    t1 = time.perf_counter_ns()
    torch.cuda.synchronize()
    t2 = time.perf_counter_ns()

    return {
        "calls": calls,
        "host_wall_total_ms": (t1 - t0) / 1e6,
        "host_wall_us_per_call": (t1 - t0) / calls / 1000.0 if calls else float("nan"),
        "host_return_mean_us": statistics.mean(samples) if samples else float("nan"),
        "host_return_p50_us": statistics.median(samples) if samples else float("nan"),
        "host_return_p95_us": _percentile(samples, 95.0),
        "host_return_p99_us": _percentile(samples, 99.0),
        "host_return_max_us": max(samples) if samples else float("nan"),
        "sync_tail_ms": (t2 - t1) / 1e6,
    }


def _bench_gpu_event(fn: Callable, calls: int, warmup_calls: int) -> dict:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup_calls):
        fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(calls):
        fn()
    end.record()
    torch.cuda.synchronize()
    total_us = start.elapsed_time(end) * 1000.0
    return {
        "calls": calls,
        "gpu_event_total_us": total_us,
        "gpu_event_us_per_call": total_us / calls if calls else float("nan"),
    }


def _git_head(path: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", path, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def _make_ptr(dtype, tensor):
    return flyc.from_c_void_p(dtype, tensor.data_ptr())


def build_specs(tokens: int, stream_arg) -> dict[str, LauncherSpec]:
    n = max(1, tokens)
    tensor_elems = _round_up(max(tokens, TILE_ELEMS), TILE_ELEMS)
    max_pos = 8192

    q = torch.randn(n, device="cuda", dtype=torch.float32)
    kv = torch.randn(n, device="cuda", dtype=torch.float32)
    q_out = torch.empty_like(q)
    kv_out = torch.empty_like(kv)
    q_scale = torch.empty(n, device="cuda", dtype=torch.float32)
    kv_scale = torch.empty(n, device="cuda", dtype=torch.float32)
    positions = torch.arange(n, device="cuda", dtype=torch.int32)

    q_weight = torch.randn(16, 512, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(512, device="cuda", dtype=torch.bfloat16)
    cos_cache = torch.randn(max_pos, 32, device="cuda", dtype=torch.bfloat16)
    sin_cache = torch.randn(max_pos, 32, device="cuda", dtype=torch.bfloat16)

    qk_args = (
        _make_ptr(fx.Float32, q),
        _make_ptr(fx.Float32, kv),
        q_weight,
        kv_weight,
        cos_cache,
        sin_cache,
        _make_ptr(fx.Int32, positions),
        _make_ptr(fx.Float32, q_out),
        _make_ptr(fx.Float32, kv_out),
        _make_ptr(fx.Float32, q_scale),
        _make_ptr(fx.Float32, kv_scale),
        512,
        n,
        BLOCK_DIM,
        stream_arg,
    )

    kv_in = torch.randn(tensor_elems, device="cuda", dtype=torch.float32)
    score_in = torch.randn(tensor_elems, device="cuda", dtype=torch.float32)
    plan = torch.arange(tensor_elems, device="cuda", dtype=torch.int32)
    kv_state = torch.empty(tensor_elems, device="cuda", dtype=torch.float32)
    score_state = torch.empty(tensor_elems, device="cuda", dtype=torch.float32)
    state_slot_mapping = torch.arange(tensor_elems, device="cuda", dtype=torch.int32)
    ape = torch.randn(tensor_elems, device="cuda", dtype=torch.float32)
    rms_weight = torch.randn(tensor_elems, device="cuda", dtype=torch.float32)
    kv_cache = torch.empty(tensor_elems, device="cuda", dtype=torch.float32)
    cache_scale = torch.empty(tensor_elems, device="cuda", dtype=torch.float32)
    block_table = torch.arange(tensor_elems, device="cuda", dtype=torch.int32)
    kv_compressed = torch.empty(tensor_elems, device="cuda", dtype=torch.float32)

    fused_args = (
        kv_in,
        512,
        score_in,
        128,
        plan,
        kv_state,
        2048,
        512,
        score_state,
        1024,
        256,
        state_slot_mapping,
        ape,
        rms_weight,
        cos_cache,
        sin_cache,
        kv_cache,
        4096,
        256,
        cache_scale,
        64,
        block_table,
        512,
        tensor_elems,
        BLOCK_DIM,
        VEC_WIDTH,
        stream_arg,
    )

    hca_compress_args = (
        kv_in,
        512,
        score_in,
        128,
        plan,
        kv_state,
        2048,
        512,
        score_state,
        1024,
        256,
        state_slot_mapping,
        ape,
        kv_compressed,
        512,
        tensor_elems,
        BLOCK_DIM,
        VEC_WIDTH,
        stream_arg,
    )

    hca_scatter_args = (
        kv_compressed,
        512,
        plan,
        rms_weight,
        cos_cache,
        sin_cache,
        kv_cache,
        4096,
        256,
        block_table,
        512,
        tensor_elems,
        BLOCK_DIM,
        VEC_WIDTH,
        stream_arg,
    )

    moe_args = (
        kv_in,
        score_in,
        ape,
        state_slot_mapping,
        rms_weight,
        kv_cache,
        tensor_elems,
        BLOCK_DIM,
        VEC_WIDTH,
        stream_arg,
    )

    return {
        "qk": LauncherSpec("qk", launch_qk_norm_rope_quant_like, qk_args, qk_args[:-1]),
        "fused_compress": LauncherSpec(
            "fused_compress",
            launch_fused_compress_attn_like,
            fused_args,
            fused_args[:-1],
        ),
        "hca_compress": LauncherSpec(
            "hca_compress",
            launch_hca_compress_forward_like,
            hca_compress_args,
            hca_compress_args[:-1],
        ),
        "hca_scatter": LauncherSpec(
            "hca_scatter",
            launch_hca_norm_rope_scatter_like,
            hca_scatter_args,
            hca_scatter_args[:-1],
        ),
        "moe": LauncherSpec("moe", launch_moe_like, moe_args, moe_args[:-1]),
    }


def _make_invokers(spec: LauncherSpec, stream_arg, call_style: str) -> dict[str, Callable]:
    invokers = {}
    if call_style in ("positional", "both"):
        invokers["jit_positional"] = lambda spec=spec: spec.fn(*spec.args)
    if call_style in ("keyword-stream", "both"):
        invokers["jit_keyword_stream"] = lambda spec=spec: spec.fn(*spec.args_without_stream, stream=stream_arg)
    return invokers


def _compile_spec(spec: LauncherSpec) -> tuple[Callable | None, float | None, str | None]:
    t0 = time.perf_counter_ns()
    try:
        compiled = flyc.compile(spec.fn, *spec.args)
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        return compiled, (t1 - t0) / 1000.0, None
    except Exception as exc:
        return None, None, repr(exc)


def bench_spec(
    spec: LauncherSpec,
    calls: int,
    warmup_calls: int,
    stream_arg,
    call_style: str,
    gpu_event_calls: int,
) -> dict:
    compiled, compile_once_us, compile_error = _compile_spec(spec)
    result = {
        "name": spec.name,
        "calls": calls,
        "compile_once_us": compile_once_us,
        "compile_error": compile_error,
        "paths": {},
    }
    if compile_error:
        return result

    invokers = _make_invokers(spec, stream_arg, call_style)
    if compiled is not None:
        invokers["compiled_positional"] = lambda compiled=compiled, spec=spec: compiled(*spec.args)

    for path_name, fn in invokers.items():
        host = _bench_host(fn, calls, warmup_calls)
        gpu = _bench_gpu_event(fn, min(calls, gpu_event_calls), min(warmup_calls, gpu_event_calls))
        merged = dict(host)
        merged.update(gpu)
        result["paths"][path_name] = merged

    return result


def bench_mixed(
    specs: dict[str, LauncherSpec],
    calls_by_kernel: dict[str, int],
    warmup_windows: int,
    stream_arg,
    call_style: str,
    use_compiled: bool,
) -> dict:
    compiled = {}
    if use_compiled:
        for name, spec in specs.items():
            if calls_by_kernel.get(name, 0) <= 0:
                continue
            compiled_fn, _, err = _compile_spec(spec)
            if err:
                return {"error": f"compile failed for {name}: {err}"}
            compiled[name] = compiled_fn

    def call_one(name: str):
        spec = specs[name]
        if use_compiled:
            compiled[name](*spec.args)
        elif call_style == "keyword-stream":
            spec.fn(*spec.args_without_stream, stream=stream_arg)
        else:
            spec.fn(*spec.args)

    warmup_counts = {name: count * warmup_windows for name, count in calls_by_kernel.items()}
    for name, count in warmup_counts.items():
        for _ in range(count):
            call_one(name)
    torch.cuda.synchronize()

    total_calls = sum(calls_by_kernel.values())
    t0 = time.perf_counter_ns()
    for name, count in calls_by_kernel.items():
        for _ in range(count):
            call_one(name)
    t1 = time.perf_counter_ns()
    torch.cuda.synchronize()
    t2 = time.perf_counter_ns()

    path = "compiled_positional" if use_compiled else f"jit_{call_style}"
    return {
        "path": path,
        "calls_by_kernel": calls_by_kernel,
        "total_calls": total_calls,
        "host_wall_total_ms": (t1 - t0) / 1e6,
        "host_wall_us_per_call": (t1 - t0) / total_calls / 1000.0 if total_calls else float("nan"),
        "sync_tail_ms": (t2 - t1) / 1e6,
    }


def collect_env(label: str | None) -> dict:
    flydsl_root = str(Path(flydsl.__file__).resolve().parents[1])
    return {
        "label": label,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda": getattr(torch.version, "cuda", None),
        "torch_hip": getattr(torch.version, "hip", None),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "rocm_arch": get_rocm_arch() if get_rocm_arch is not None else None,
        "flydsl_file": str(Path(flydsl.__file__).resolve()),
        "flydsl_version": getattr(flydsl, "__version__", None),
        "flydsl_git_head": _git_head(flydsl_root),
        "flydsl_runtime_cache_dir": os.environ.get("FLYDSL_RUNTIME_CACHE_DIR"),
        "flydsl_runtime_enable_cache": os.environ.get("FLYDSL_RUNTIME_ENABLE_CACHE"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=None, help="Label stored in the JSON result.")
    parser.add_argument("--case", default="dsv4-c256", choices=sorted(PROFILE_CALLS))
    parser.add_argument("--tokens", type=int, default=991)
    parser.add_argument("--windows", type=int, default=16)
    parser.add_argument("--warmup-windows", type=int, default=1)
    parser.add_argument("--kernels", default="qk,fused_compress,hca_compress,hca_scatter")
    parser.add_argument("--include-moe", action="store_true")
    parser.add_argument("--call-style", choices=["positional", "keyword-stream", "both"], default="both")
    parser.add_argument("--gpu-event-calls", type=int, default=200)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA/ROCm GPU is not available.", file=sys.stderr)
        return 2

    stream_arg = torch.cuda.current_stream().cuda_stream
    specs = build_specs(args.tokens, stream_arg)
    profile = dict(PROFILE_CALLS[args.case])
    if args.include_moe:
        profile["moe"] = 16

    wanted = [k.strip() for k in args.kernels.split(",") if k.strip()]
    if args.include_moe and "moe" not in wanted:
        wanted.append("moe")
    unknown = sorted(set(wanted) - set(specs))
    if unknown:
        raise ValueError(f"unknown kernels: {unknown}")

    calls_by_kernel = {
        name: profile.get(name, 0) * args.windows
        for name in wanted
        if profile.get(name, 0) > 0
    }
    warmup_by_kernel = {
        name: profile.get(name, 0) * args.warmup_windows
        for name in wanted
        if profile.get(name, 0) > 0
    }

    result = {
        "env": collect_env(args.label),
        "profile": {
            "case": args.case,
            "tokens": args.tokens,
            "windows": args.windows,
            "warmup_windows": args.warmup_windows,
            "source": "rank-local c=256 trace call counts",
            "calls_per_window": profile,
            "calls_by_kernel": calls_by_kernel,
        },
        "per_kernel": {},
        "mixed_replay": {},
    }

    print(f"FlyDSL: {result['env']['flydsl_file']}")
    print(f"GPU: {result['env']['gpu']}  arch={result['env']['rocm_arch']}")
    print(f"calls_by_kernel: {calls_by_kernel}")

    for name in wanted:
        calls = calls_by_kernel.get(name, 0)
        if calls <= 0:
            continue
        print(f"\n[{name}] calls={calls}")
        spec_result = bench_spec(
            specs[name],
            calls,
            warmup_by_kernel.get(name, 0),
            stream_arg,
            args.call_style,
            args.gpu_event_calls,
        )
        result["per_kernel"][name] = spec_result
        for path_name, metrics in spec_result.get("paths", {}).items():
            print(
                f"  {path_name:<22s} "
                f"host={metrics['host_wall_us_per_call']:.2f} us/call "
                f"gpu={metrics['gpu_event_us_per_call']:.2f} us/call"
            )

    if calls_by_kernel:
        mixed_style = "keyword-stream" if args.call_style in ("keyword-stream", "both") else "positional"
        result["mixed_replay"]["jit"] = bench_mixed(
            specs,
            calls_by_kernel,
            args.warmup_windows,
            stream_arg,
            mixed_style,
            use_compiled=False,
        )
        result["mixed_replay"]["compiled"] = bench_mixed(
            specs,
            calls_by_kernel,
            args.warmup_windows,
            stream_arg,
            mixed_style,
            use_compiled=True,
        )
        print("\n[mixed]")
        for name, metrics in result["mixed_replay"].items():
            if "error" in metrics:
                print(f"  {name}: {metrics['error']}")
            else:
                print(
                    f"  {name:<8s} {metrics['path']:<24s} "
                    f"host={metrics['host_wall_us_per_call']:.2f} us/call "
                    f"total={metrics['host_wall_total_ms']:.2f} ms"
                )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {out}")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
