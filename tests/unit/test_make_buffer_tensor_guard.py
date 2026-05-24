#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Guard against the silent-OOB pattern in ``make_buffer_tensor``.

``fx.rocdl.make_buffer_tensor(tensor, max_size=False)`` used to auto-derive
the descriptor size from ``cosize(layout) * elem_bytes``, baking the static
shape into IR.  The JIT cache key does not include shape by default, so a
kernel compiled for shape A is silently reused for shape B with A's
``num_records`` baked in -- producing OOB reads (returns 0) and dropped
writes (the PR #551 fp8gemm regression mode).

The guard now requires callers to pass ``num_records_bytes`` whenever
``max_size=False``.  Either pass an explicit byte count computed from
runtime tensor extents, or use ``max_size=True`` for the safe coarse path.
"""

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx


@pytest.fixture(autouse=True)
def _no_disk_cache(monkeypatch):
    """The guard fires at trace time; the JIT disk cache would short-circuit
    re-trace and mask the guard.  Disable it per-test."""
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")


def _make_launch(max_size, num_records_bytes=None):
    """Build a launch wrapper around ``make_buffer_tensor`` with the given
    arguments.  Returns ``launch_fn``."""

    if num_records_bytes is None:

        @flyc.kernel
        def probe(x: fx.Tensor):
            fx.rocdl.make_buffer_tensor(x, max_size=max_size)

    else:

        @flyc.kernel
        def probe(x: fx.Tensor):
            fx.rocdl.make_buffer_tensor(x, max_size=max_size, num_records_bytes=num_records_bytes)

    @flyc.jit
    def launch(x: fx.Tensor):
        probe(x).launch(grid=(1, 1, 1), block=(1, 1, 1))

    return launch


def test_max_size_false_without_num_records_bytes_raises(monkeypatch):
    """The unsafe auto-derive path must raise a clear ValueError at trace time."""
    monkeypatch.setenv("COMPILE_ONLY", "1")
    launch = _make_launch(max_size=False, num_records_bytes=None)
    with pytest.raises(Exception) as exc:
        launch(torch.empty((4, 8), dtype=torch.float32))
    assert "num_records_bytes" in str(exc.value), str(exc.value)


def test_max_size_true_default_works(monkeypatch):
    """Safe coarse path (the default) is unaffected."""
    monkeypatch.setenv("COMPILE_ONLY", "1")
    launch = _make_launch(max_size=True)
    launch(torch.empty((4, 8), dtype=torch.float32))


def test_explicit_num_records_bytes_int_works(monkeypatch):
    """Caller-provided int byte count is the preferred robust path."""
    monkeypatch.setenv("COMPILE_ONLY", "1")
    launch = _make_launch(max_size=False, num_records_bytes=4 * 8 * 4)
    launch(torch.empty((4, 8), dtype=torch.float32))


def test_max_size_false_with_dynamic_layout_auto_derives(monkeypatch):
    """When the tensor layout is dynamic (``mark_layout_dynamic``), the
    historical ``cosize(layout) * elem_bytes`` auto-derive is safe -- it
    produces a runtime expression that adapts to the actual tensor extent
    -- so no explicit ``num_records_bytes`` is required."""
    monkeypatch.setenv("COMPILE_ONLY", "1")
    launch = _make_launch(max_size=False, num_records_bytes=None)
    t = flyc.from_dlpack(torch.empty((4, 8), dtype=torch.float32)).mark_layout_dynamic(leading_dim=1, divisibility=1)
    launch(t)
