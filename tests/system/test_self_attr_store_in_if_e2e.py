#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""End-to-end guard: a method kernel that stores through a ``self`` attribute
inside a dynamic ``if`` must compile and run.

The AST rewriter collects the variables written inside a control-flow region so
it can thread them through the scf.if/for/while result values. A store like
``self.buf[i] = x`` propagates the "store" context down to the base ``Name`` of
the attribute (``self``), which used to make the rewriter treat ``self`` as a
written region arg. At trace time the dispatcher then tried to carry ``self`` --
a plain Python object, not an MLIR value -- through the scf.if results and raised
``TypeError: state variable 'self' is ... not an MLIR Value``.

``self`` is now excluded from the collected write args (it stays reachable inside
the region via closure), so the kernel compiles and launches. This is the
system-level counterpart of the unit guard in
``tests/unit/test_if_dispatch_paths.py::test_collect_assigned_vars_excludes_self``.

Note: branch-gating of side-effecting stores inside a dynamic ``if`` is a
separate, pre-existing limitation, so this test intentionally does not assert
per-thread gated values -- only that the kernel compiles, runs, and writes.
"""

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


class _SelfStoreKernel:
    """Kernel object whose ``self`` holds the output tensor and writes to it
    through a ``self`` attribute inside a dynamic branch."""

    @flyc.kernel
    def run(self, Out: fx.Tensor, threshold: fx.Int32, block_dim: fx.Constexpr[int]):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        gid = bid * block_dim + tid
        self.out = Out
        if tid < threshold:
            # Store through a ``self`` attribute inside a dynamic branch: the
            # pattern that used to force ``self`` into the region write args and
            # crash at trace time.
            self.out[gid] = fx.Float32(2.0)

    @flyc.jit
    def launch(self, Out: fx.Tensor, n: fx.Int32, block_dim: fx.Constexpr[int], stream: fx.Stream = fx.Stream(None)):
        grid_x = (n + block_dim - 1) // block_dim
        self.run(Out, n, block_dim).launch(grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream)


def test_self_attr_store_in_dynamic_if_end_to_end(monkeypatch):
    """``self.<attr>[i] = v`` inside a dynamic if compiles and runs instead of
    raising ``TypeError: state variable 'self' ... not an MLIR Value``."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device required")

    # Avoid compile-cache hits so the dynamic dispatch is exercised here.
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

    block_dim = 64
    threshold = 32
    n = block_dim
    out = torch.zeros(n, device="cuda", dtype=torch.float32)
    t_out = flyc.from_torch_tensor(out).mark_layout_dynamic(leading_dim=0)

    # Before the fix this raised at trace time; now it compiles and launches.
    _SelfStoreKernel().launch(t_out, n, threshold)
    torch.cuda.synchronize()

    # The branch body ran and wrote through `self.out`, so the store landed.
    assert bool((out == 2.0).any()), "kernel did not write through self.out inside the dynamic if"
