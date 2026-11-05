# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kernel-side ``_preload_compiled``: no process-wide ``COMPILE_ONLY`` toggling.

The MegaMoE, common and MegaMoE-TP shims each used to set ``COMPILE_ONLY=1``
under their own lock, so concurrent preloads could leave it set and make later
launches (e.g. the cached combine kernel) silently no-ops.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from kernels.common import tensor_shim as common_shim
from kernels.mega_moe import tensor_shim as mega_shim

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)


@flyc.kernel
def _add_kernel(X: fx.Pointer, n: fx.Int32, delta: fx.Constexpr[float]):
    idx = fx.block_idx.x * fx.block_dim.x + fx.thread_idx.x
    if idx < n:
        X[idx] = X[idx] + delta


@flyc.jit
def _add(X: fx.Pointer, n: fx.Int32, delta: fx.Constexpr[float], stream: fx.Stream):
    _add_kernel(X, n, delta).launch(grid=((n + 63) // 64, 1, 1), block=(64, 1, 1), stream=stream)


@flyc.jit
def _add_one(X: fx.Pointer, n: fx.Int32, stream: fx.Stream):
    _add_kernel(X, n, 1.0).launch(grid=((n + 63) // 64, 1, 1), block=(64, 1, 1), stream=stream)


def _ptr(t):
    return flyc.from_c_void_p(fx.Float32, t.data_ptr())


@pytest.fixture(autouse=True)
def _no_compile_only(monkeypatch):
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def test_shims_share_one_implementation():
    from kernels.mega_moe_tp import mega_moe_tp

    assert mega_shim._preload_compiled is common_shim._preload_compiled
    assert mega_moe_tp._preload_compiled is common_shim._preload_compiled


def test_preload_does_not_launch():
    x = torch.zeros(256, device="cuda")
    assert common_shim._preload_compiled(_add, _ptr(x), 256, 7.0, torch.cuda.current_stream()) is None
    torch.cuda.synchronize()
    assert torch.count_nonzero(x).item() == 0
    assert "COMPILE_ONLY" not in os.environ


def test_concurrent_preloads_keep_launches_live():
    n, launches, workers, per_worker = 4096, 200, 4, 8
    x = torch.zeros(n, device="cuda")
    stream = torch.cuda.current_stream()

    def preload_many(worker):
        y = torch.zeros(n, device="cuda")
        shim = mega_shim if worker % 2 else common_shim
        for k in range(per_worker):
            shim._preload_compiled(_add, _ptr(y), n, float(1000 + worker * per_worker + k), stream)
        torch.cuda.synchronize()
        return torch.count_nonzero(y).item()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(preload_many, w) for w in range(workers)]
        for _ in range(launches):
            common_shim._run_compiled(_add_one, _ptr(x), n, stream)
        untouched = [f.result() for f in futures]

    torch.cuda.synchronize()
    assert untouched == [0] * workers
    assert "COMPILE_ONLY" not in os.environ
    assert torch.all(x == float(launches))

    # A first-time compile after the preloads (e.g. the lazily built combine kernel) must still launch.
    z = torch.zeros(n, device="cuda")
    common_shim._run_compiled(_add, _ptr(z), n, 3.0, stream)
    torch.cuda.synchronize()
    assert torch.all(z == 3.0)
