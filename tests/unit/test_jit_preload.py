# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""``JitFunction.preload()``: compile without launching, without process-wide state.

Preload used to be emulated by temporarily setting ``COMPILE_ONLY=1``. That flag
is read by every launch in the process, so a launch on another thread during the
window silently did nothing. ``preload()`` must never touch it.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.jit_function import CompiledFunction

try:
    import torch
except ImportError:
    torch = None

HAS_GPU = torch is not None and torch.cuda.is_available()
needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="requires a ROCm GPU")


@flyc.kernel
def _add_kernel(X: fx.Pointer, n: fx.Int32, delta: fx.Constexpr[float]):
    idx = fx.block_idx.x * fx.block_dim.x + fx.thread_idx.x
    if idx < n:
        X[idx] = X[idx] + delta


@flyc.jit
def add(X: fx.Pointer, n: fx.Int32, delta: fx.Constexpr[float] = 1.0, stream: fx.Stream = fx.Stream(None)):
    _add_kernel(X, n, delta).launch(grid=((n + 63) // 64, 1, 1), block=(64, 1, 1), stream=stream)


def _ptr(value=0):
    return flyc.from_c_void_p(fx.Float32, value)


@pytest.fixture(autouse=True)
def _no_compile_only(monkeypatch):
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def test_preload_does_not_dispatch_or_create_engine(monkeypatch):
    from flydsl.compiler import jit_executor, jit_function

    def boom(*_a, **_k):
        raise AssertionError("preload must not execute anything")

    monkeypatch.setattr(jit_executor, "ExecutionEngine", boom)
    monkeypatch.setattr(jit_executor.CompiledArtifact, "_ensure_engine", boom)
    monkeypatch.setattr(jit_function, "_build_call_state", boom)

    compiled = add.preload(_ptr(), 8, 3.0)
    assert isinstance(compiled, CompiledFunction)
    assert "COMPILE_ONLY" not in os.environ


@needs_gpu
def test_preload_with_live_buffer_does_not_launch():
    x = torch.zeros(256, device="cuda")
    compiled = add.preload(_ptr(x.data_ptr()), 256, 5.0)
    torch.cuda.synchronize()
    assert torch.count_nonzero(x).item() == 0

    compiled(_ptr(x.data_ptr()), 256, 5.0, fx.Stream(None))
    torch.cuda.synchronize()
    assert torch.all(x == 5.0)


@needs_gpu
def test_concurrent_preload_does_not_suppress_launches():
    """Launches on one thread stay live while other threads preload new specializations."""
    n, launches = 4096, 200
    x = torch.zeros(n, device="cuda")
    add(_ptr(x.data_ptr()), n)
    torch.cuda.synchronize()

    stop = threading.Event()
    seen_compile_only = []

    def preload_many(worker):
        y = torch.zeros(n, device="cuda")
        k = 0
        while not stop.is_set() and k < 16:
            add.preload(_ptr(y.data_ptr()), n, float(100 + worker * 16 + k))
            seen_compile_only.append(os.environ.get("COMPILE_ONLY"))
            k += 1
        torch.cuda.synchronize()
        return torch.count_nonzero(y).item()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(preload_many, w) for w in range(4)]
        for _ in range(launches):
            add(_ptr(x.data_ptr()), n)
        stop.set()
        untouched = [f.result() for f in futures]

    torch.cuda.synchronize()
    assert untouched == [0, 0, 0, 0]
    assert set(seen_compile_only) <= {None}
    assert "COMPILE_ONLY" not in os.environ
    assert torch.all(x == float(launches + 1))
