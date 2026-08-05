#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Nested ``add_offset`` chains on shared (LDS) pointers must read the same elements
however the offsets are spelled: constant inner, runtime inner, or pre-merged.
"""

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx

try:
    import torch
except ImportError:
    torch = None

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

BLOCK_DIM = 256
NUM_STAGES = 4
NUM_BLOCKS = 8

# Per-thread LDS read index. Coprime with BLOCK_DIM, so a base/offset split that
# loses the runtime or the compile-time part reads a different element and the
# reference comparison fails instead of aliasing back onto the identity.
LANE_STRIDE = 7


def _stage_sum_body(A, C, block_dim, num_stages, read_ptr):
    """Stage every thread's slice of A into LDS, then reduce across stages.

    ``read_ptr(s_ptr, stage_elems, lane)`` builds the LDS element address for
    stage ``s``; the spelling of that address is what each kernel varies.
    """
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    tile = block_dim * num_stages

    a_ptr = fx.get_iter(A)
    c_ptr = fx.get_iter(C)
    s_ptr = fx.SharedAllocator().allocate(fx.Array[fx.Float32, tile]).peek().ptr

    for s in fx.range_constexpr(num_stages):
        s_ptr[s * block_dim + tid] = a_ptr[bid * tile + s * block_dim + tid]
    fx.barrier()

    lane = (tid * LANE_STRIDE) % block_dim
    acc = fx.Float32(0.0)
    for s in fx.range_constexpr(num_stages):
        acc = acc + read_ptr(s_ptr, s * block_dim, lane).load()
    c_ptr[bid * block_dim + tid] = acc


@flyc.kernel
def lds_chain_const_inner_kernel(
    A: fx.Tensor,
    C: fx.Tensor,
    block_dim: fx.Constexpr[int],
    num_stages: fx.Constexpr[int],
):
    def read_ptr(s_ptr, stage_elems, lane):
        stage = fx.add_offset(s_ptr, fx.make_int_tuple(stage_elems))  # compile-time, inner
        return fx.add_offset(stage, fx.make_int_tuple(lane))  # runtime, outer

    _stage_sum_body(A, C, block_dim, num_stages, read_ptr)


@flyc.kernel
def lds_chain_runtime_inner_kernel(
    A: fx.Tensor,
    C: fx.Tensor,
    block_dim: fx.Constexpr[int],
    num_stages: fx.Constexpr[int],
):
    def read_ptr(s_ptr, stage_elems, lane):
        base = fx.add_offset(s_ptr, fx.make_int_tuple(lane))  # runtime, inner
        return fx.add_offset(base, fx.make_int_tuple(stage_elems))  # compile-time, outer

    _stage_sum_body(A, C, block_dim, num_stages, read_ptr)


@flyc.kernel
def lds_flat_offset_kernel(
    A: fx.Tensor,
    C: fx.Tensor,
    block_dim: fx.Constexpr[int],
    num_stages: fx.Constexpr[int],
):
    def read_ptr(s_ptr, stage_elems, lane):
        return fx.add_offset(s_ptr, fx.make_int_tuple(lane + stage_elems))  # already merged

    _stage_sum_body(A, C, block_dim, num_stages, read_ptr)


def _launcher(kernel):
    @flyc.jit
    def launch(
        A: fx.Tensor,
        C: fx.Tensor,
        num_blocks: fx.Constexpr[int],
        block_dim: fx.Constexpr[int],
        num_stages: fx.Constexpr[int],
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(A, C, block_dim, num_stages).launch(grid=(num_blocks, 1, 1), block=(block_dim, 1, 1), stream=stream)

    return launch


def _reference(a):
    """out[b, t] = sum_s A[b, s, (t * LANE_STRIDE) % BLOCK_DIM]"""
    staged = a.reshape(NUM_BLOCKS, NUM_STAGES, BLOCK_DIM).sum(dim=1)
    lane = (torch.arange(BLOCK_DIM, device=a.device) * LANE_STRIDE) % BLOCK_DIM
    return staged[:, lane].reshape(-1)


@pytest.mark.parametrize(
    "kernel",
    [lds_chain_const_inner_kernel, lds_chain_runtime_inner_kernel, lds_flat_offset_kernel],
    ids=["const_inner", "runtime_inner", "flat"],
)
def test_lds_nested_add_offset_addressing(kernel):
    """Every spelling of the LDS address chain must read the same elements."""
    torch.manual_seed(0)
    a = torch.randn(NUM_BLOCKS * NUM_STAGES * BLOCK_DIM, dtype=torch.float32, device="cuda")
    c = torch.zeros(NUM_BLOCKS * BLOCK_DIM, dtype=torch.float32, device="cuda")

    _launcher(kernel)(a, c, NUM_BLOCKS, BLOCK_DIM, NUM_STAGES, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    torch.testing.assert_close(c, _reference(a), atol=1e-5, rtol=1e-5)
