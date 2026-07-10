#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Regression test for issue #734.

A ``Vector`` with a multi-dim logical shape (e.g. ``(4, 1)``) carried across a
runtime ``for`` loop used to lose its shape: the loop-carried value was rebuilt
from the bare ``scf.for`` block-argument ``ir.Value`` (a flat ``vector<Nxf32>``),
collapsing ``(4, 1)`` -> ``(4,)``. The next ``vec_sum += vec`` then broadcast
``(4,) + (4, 1)`` to ``(4, 4)`` (``vector<16xf32>``), which fails to match the
loop's ``vector<4xf32>`` iter_arg type and aborts compilation.

The fix preserves the exemplar instance's shape/dtype when reconstructing a
loop-carried Vector. This test confirms a multi-dim Vector survives a dynamic
for loop end-to-end and accumulates correctly.
"""

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)


@flyc.kernel
def _k_vec_carry(Out: fx.Tensor, n: fx.Int32):
    tid = fx.thread_idx.x
    # Loop-carried Vector with a *multi-dim* shape (4, 1) -- the shape that the
    # bug collapsed to (4,). Each element starts at 1.0.
    vec_sum = fx.Vector.filled((4, 1), 1.0, fx.Float32)
    for _ in range(1, n):  # dynamic bound -> scf.for, so vec_sum is loop-carried
        vec_sum += fx.Vector.filled((4, 1), 1.0, fx.Float32)
    # After 1 init + (n-1) adds, every element == n; reduce over the 4 lanes -> 4*n.
    s = vec_sum.reduce(fx.vector.ReductionOp.ADD)
    rsrc = buffer_ops.create_buffer_resource(Out)
    buffer_ops.buffer_store(s, rsrc, tid)


@flyc.jit
def _j_vec_carry(Out: fx.Tensor, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
    _k_vec_carry(Out, n).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)


def test_vector_carry_shape_preserved():
    """vec_sum=(4,1) ones; for _ in range(1,n): vec_sum += (4,1) ones; reduce -> 4*n."""
    N = 5
    out = torch.zeros(1, device="cuda", dtype=torch.float32)
    t_out = flyc.from_torch_tensor(out).mark_layout_dynamic(leading_dim=0, divisibility=1)
    _j_vec_carry(t_out, fx.Int32(N))
    torch.cuda.synchronize()
    assert out.item() == pytest.approx(4 * N)  # 20.0
