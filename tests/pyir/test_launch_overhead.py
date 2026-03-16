#!/usr/bin/env python3
"""Compare compiled IR & actual launch overhead: simple kernel vs PA-like.

Measures the raw C-level func_exe() call time (excluding Python dispatch)
and the full JitFunction.__call__ dispatch time.

Run with:
    pytest tests/pyir/test_launch_overhead.py -s
"""

import ctypes
import inspect
import os
import time

import pytest
import torch

os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "1")

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.jit_argument import convert_to_jit_arguments
from flydsl.compiler.jit_function import _ensure_stream_arg
from flydsl.compiler.protocol import fly_pointers

DEV = "cuda"
BF16 = torch.bfloat16


# ── Minimal GPU kernel using the proper @flyc.kernel API ──

@flyc.kernel
def _noop_kernel(A: fx.Tensor, block_dim: fx.Constexpr[int]):
    tid = fx.thread_idx.x


def _dump_artifact_ir(jf, label):
    """Dump compiled IR from a JitFunction's cached artifact."""
    if not hasattr(jf, "_mem_cache") or not jf._mem_cache:
        print(f"[{label}] No cached artifact found")
        return None
    key = list(jf._mem_cache.keys())[0]
    artifact = jf._mem_cache[key]
    ir_text = artifact._ir_text
    print(f"\n{'=' * 70}")
    print(f"COMPILED IR for [{label}]")
    print(f"{'=' * 70}")
    print(ir_text)
    print(f"{'=' * 70}\n")
    return ir_text


def _measure_func_exe(jf, args, label, n=500):
    """Measure the raw C-level launch time (func_exe call only)."""
    key = list(jf._mem_cache.keys())[0]
    artifact = jf._mem_cache[key]
    artifact._ensure_engine()

    sig = inspect.signature(jf.func)
    bound = sig.bind(*args)
    bound.apply_defaults()
    _ctx = ir.Context()
    _ctx.load_all_available_dialects()
    with _ctx:
        _, jit_args, _, _ = convert_to_jit_arguments(sig, bound)
        _ensure_stream_arg(jit_args)
        all_c_ptrs = []
        for arg in jit_args:
            all_c_ptrs.extend(fly_pointers(arg))

    func_ptr = artifact._engine.raw_lookup(artifact._entry)
    func_exe = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(func_ptr)
    packed = artifact._packer.pack(all_c_ptrs)

    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        func_exe(packed)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000)

    torch.cuda.synchronize()
    times = times[20:]
    avg = sum(times) / len(times)
    p50 = sorted(times)[len(times) // 2]
    p99 = sorted(times)[int(len(times) * 0.99)]
    print(f"[{label}] raw func_exe: avg={avg:.1f}us  p50={p50:.1f}us  p99={p99:.1f}us")
    return avg


def _measure_full_dispatch(jf, args, label, n=500):
    """Measure the full JitFunction.__call__ time (Python dispatch + launch)."""
    for _ in range(5):
        jf(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        jf(*args)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000)

    torch.cuda.synchronize()
    times = times[20:]
    avg = sum(times) / len(times)
    p50 = sorted(times)[len(times) // 2]
    print(f"[{label}] full dispatch: avg={avg:.1f}us  p50={p50:.1f}us")
    return avg


def test_empty_kernel():
    """Empty kernel (no gpu.func, no launch) — pure host overhead."""
    a = torch.randn(128, 128, dtype=BF16, device=DEV)
    b = torch.randn(128, 128, dtype=BF16, device=DEV)

    @flyc.jit
    def k_empty(arg_a, arg_b, stream: fx.Stream = fx.Stream(None)):
        pass

    k_empty(a, b)
    torch.cuda.synchronize()
    _dump_artifact_ir(k_empty, "k_empty")
    _measure_func_exe(k_empty, (a, b), "k_empty")
    _measure_full_dispatch(k_empty, (a, b), "k_empty")


def test_single_gpu_launch():
    """Single GPU kernel launch — 1 simple kernel."""
    a = torch.randn(128, 128, dtype=BF16, device=DEV)

    @flyc.jit
    def k_gpu1(arg_a, stream: fx.Stream = fx.Stream(None)):
        _noop_kernel(arg_a, 64).launch(
            grid=(1, 1, 1), block=(64, 1, 1), stream=stream
        )

    k_gpu1(a)
    torch.cuda.synchronize()
    _dump_artifact_ir(k_gpu1, "k_gpu1")
    _measure_func_exe(k_gpu1, (a,), "k_gpu1")
    _measure_full_dispatch(k_gpu1, (a,), "k_gpu1")


def test_7arg_noop():
    """7 tensor args, no GPU launch — measures arg-packing overhead."""
    tensors = [torch.randn(128, 128, dtype=BF16, device=DEV) for _ in range(7)]

    @flyc.jit
    def k_7arg(a, b, c, d, e, f, g, stream: fx.Stream = fx.Stream(None)):
        pass

    k_7arg(*tensors)
    torch.cuda.synchronize()
    _dump_artifact_ir(k_7arg, "k_7arg_noop")
    _measure_func_exe(k_7arg, tuple(tensors), "k_7arg_noop")
    _measure_full_dispatch(k_7arg, tuple(tensors), "k_7arg_noop")


def test_7arg_gpu_launch():
    """7 tensor args, WITH GPU launch."""
    a = torch.randn(128, 128, dtype=BF16, device=DEV)

    @flyc.jit
    def k_7arg_gpu(a, b, c, d, e, f, g, stream: fx.Stream = fx.Stream(None)):
        _noop_kernel(a, 64).launch(
            grid=(1, 1, 1), block=(64, 1, 1), stream=stream
        )

    tensors = [a] + [torch.randn(128, 128, dtype=BF16, device=DEV) for _ in range(6)]
    k_7arg_gpu(*tensors)
    torch.cuda.synchronize()
    _dump_artifact_ir(k_7arg_gpu, "k_7arg_gpu")
    _measure_func_exe(k_7arg_gpu, tuple(tensors), "k_7arg_gpu")
    _measure_full_dispatch(k_7arg_gpu, tuple(tensors), "k_7arg_gpu")
