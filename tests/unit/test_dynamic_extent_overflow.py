#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""The size of a dynamic multi-dimensional layout must not wrap at 2**31 (#1176).

Each dimension of the (4, S1) tensor fits in i32, but 4 * S1 = 2**31 does not.
With the size computed in i32 it wrapped negative, the coordinate decomposition
in logical_divide divided by it, and every lane wrote to lane * 16 -- in bounds,
so nothing flagged it. The correct targets are a permutation of 0..63.
"""

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import Vector as Vec

try:
    import torch
except ImportError:
    torch = None

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]
if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)

S0, ROW_STRIDE, LANES = 4, 16, 64


@flyc.kernel(known_block_size=[LANES, 1, 1])
def _scatter_ones(out: fx.Tensor):
    lanes = fx.logical_divide(fx.rocdl.make_buffer_tensor(out, max_size=False), fx.make_layout(1, 1))
    reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
    fx.memref_store_vec(Vec.filled(1, 1.0, fx.Float32), reg)
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    lane = fx.thread_idx.x % LANES
    fx.copy(atom, reg, fx.slice(lanes, (None, fx.Int32(lane))))


@flyc.jit
def _launch(out: fx.Tensor, stream: fx.Stream):
    _scatter_ones(out).launch(grid=(1, 1, 1), block=(LANES, 1, 1), stream=stream)


def _written_offsets(s1):
    # Rows overlap (stride 16 < s1), so the storage only has to span one row.
    storage = torch.zeros((S0 - 1) * ROW_STRIDE + s1, dtype=torch.float32, device="cuda")
    out = storage.as_strided((S0, s1), (ROW_STRIDE, 1))
    _launch(out, torch.cuda.current_stream())
    torch.cuda.synchronize()
    return torch.nonzero(storage).flatten().tolist()


@pytest.mark.parametrize(
    "s1",
    [
        pytest.param(1000, id="small"),
        pytest.param(2**29 - 1, id="size_below_2^31", marks=pytest.mark.large_shape),
        pytest.param(2**29, id="size_2^31", marks=pytest.mark.large_shape),
    ],
)
def test_lane_scatter_follows_layout(s1):
    # Lane l has coordinate (l % 4, l // 4) -> offset (l % 4) * 16 + l // 4.
    assert _written_offsets(s1) == list(range(LANES))
