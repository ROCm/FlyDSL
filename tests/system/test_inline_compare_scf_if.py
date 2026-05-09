#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""
Inline Compare — AST rewriter generates scf.IfOp correctly for
``if tid < threshold: buffer_ops.buffer_store(arith.constant(1), rsrc, tid)``.

The compare ``tid < threshold`` is between two MLIR runtime values
(threadIdx.x and a kernel argument), so _could_be_dynamic returns True
and visit_If rewrites it into scf_if_dispatch → scf.IfOp with no
live-out (no results, no yield).
"""

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


def test_inline_compare_buffer_store_no_liveout(monkeypatch):
    """tid < threshold → scf.IfOp (no results), body does buffer_store."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device required")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

    BLOCK = 64

    @flyc.kernel
    def conditionalStoreKernel(
        Out: fx.Tensor,
        threshold: fx.Int32,
        block_dim: fx.Constexpr[int],
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        gid = bid * block_dim + tid

        rsrc = buffer_ops.create_buffer_resource(Out)

        if tid < threshold:
            buffer_ops.buffer_store(arith.constant(1.0, type=fx.T.f32()), rsrc, gid)

    @flyc.jit
    def conditionalStore(
        Out: fx.Tensor,
        threshold: fx.Int32,
        n: fx.Int32,
        block_dim: fx.Constexpr[int],
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_x = (n + block_dim - 1) // block_dim
        conditionalStoreKernel(Out, threshold, block_dim).launch(
            grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream
        )

    size = BLOCK * 4
    threshold = BLOCK // 2

    out = torch.zeros(size, device="cuda", dtype=torch.float32)
    t_out = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=0, divisibility=4)

    conditionalStore(t_out, threshold, size, BLOCK)
    torch.cuda.synchronize()

    ref = torch.zeros(size, device="cuda", dtype=torch.float32)
    for blk in range(size // BLOCK):
        for t in range(BLOCK):
            if t < threshold:
                ref[blk * BLOCK + t] = 1.0

    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    print(f"[PASS] test_no_liveout: out[:{threshold}] all 1.0, out[{threshold}:] all 0.0")
    print(f"  shape={out.shape}, threshold={threshold}, matched {size} elements")


def test_inline_compare_buffer_store_with_liveout(monkeypatch):
    """tid < threshold → scf.IfOp WITH results (live-out variable)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device required")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

    BLOCK = 64
    VEC = 4

    @flyc.kernel
    def liveoutIfKernel(
        A: fx.Tensor,
        C: fx.Tensor,
        threshold: fx.Int32,
        block_dim: fx.Constexpr[int],
        vec_width: fx.Constexpr[int],
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        tile_elems = block_dim * vec_width

        A_buf = fx.rocdl.make_buffer_tensor(A)
        C_buf = fx.rocdl.make_buffer_tensor(C)

        tA = fx.logical_divide(A_buf, fx.make_layout(tile_elems, 1))
        tC = fx.logical_divide(C_buf, fx.make_layout(tile_elems, 1))

        tA = fx.slice(tA, (None, bid))
        tC = fx.slice(tC, (None, bid))

        tA = fx.logical_divide(tA, fx.make_layout(vec_width, 1))
        tC = fx.logical_divide(tC, fx.make_layout(vec_width, 1))

        reg_ty = fx.MemRefType.get(
            fx.T.f32(), fx.LayoutType.get(vec_width, 1), fx.AddressSpace.Register
        )
        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)

        rA = fx.memref_alloca(reg_ty, fx.make_layout(vec_width, 1))
        rC = fx.memref_alloca(reg_ty, fx.make_layout(vec_width, 1))

        fx.copy_atom_call(copy_atom, fx.slice(tA, (None, tid)), rA)
        vA = fx.memref_load_vec(rA)
        print(f"  rA type:  {rA.type}")
        print(f"  vA type:  {vA.type}")
        vOut = vA

        if tid < threshold:
            vOut = fx.arith.addf(vA, vA)
        else:
            vOut = fx.arith.negf(vA)

        fx.memref_store_vec(vOut, rC)
        fx.copy_atom_call(copy_atom, rC, fx.slice(tC, (None, tid)))

    @flyc.jit
    def liveoutIf(
        A: fx.Tensor,
        C,
        threshold: fx.Int32,
        n: fx.Int32,
        block_dim: fx.Constexpr[int],
        vec_width: fx.Constexpr[int],
        stream: fx.Stream = fx.Stream(None),
    ):
        tile_elems = block_dim * vec_width
        grid_x = (n + tile_elems - 1) // tile_elems
        liveoutIfKernel(A, C, threshold, block_dim, vec_width).launch(
            grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream
        )

    num_blocks = 4
    size = BLOCK * VEC * num_blocks
    threshold = BLOCK // 2

    a = torch.randn(size, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)
    t_a = flyc.from_dlpack(a).mark_layout_dynamic(leading_dim=0, divisibility=VEC)

    liveoutIf(t_a, c, threshold, size, BLOCK, VEC)
    torch.cuda.synchronize()

    a3 = a.view(num_blocks, BLOCK, VEC)
    tid_idx = torch.arange(BLOCK, device="cuda").view(1, BLOCK, 1)
    ref = torch.where(tid_idx < threshold, a3 + a3, -a3)
    ref = ref.reshape(-1)

    torch.testing.assert_close(c, ref, rtol=1e-5, atol=1e-5)
    max_diff = (c - ref).abs().max().item()
    print(f"[PASS] test_with_liveout (inline compare): max_diff={max_diff:.2e}, threshold={threshold}")
    print(f"  shape={c.shape}, tid<{threshold} -> 2*a, tid>={threshold} -> -a")


def test_inline_compare_buffer_store_with_liveout_flag(monkeypatch):
    """flag = tid < threshold; if flag: → scf.IfOp WITH results (live-out, flag variant)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device required")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

    BLOCK = 64
    VEC = 4

    @flyc.kernel
    def liveoutIfFlagKernel(
        A: fx.Tensor,
        C: fx.Tensor,
        threshold: fx.Int32,
        block_dim: fx.Constexpr[int],
        vec_width: fx.Constexpr[int],
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        tile_elems = block_dim * vec_width

        A_buf = fx.rocdl.make_buffer_tensor(A)
        C_buf = fx.rocdl.make_buffer_tensor(C)

        tA = fx.logical_divide(A_buf, fx.make_layout(tile_elems, 1))
        tC = fx.logical_divide(C_buf, fx.make_layout(tile_elems, 1))

        tA = fx.slice(tA, (None, bid))
        tC = fx.slice(tC, (None, bid))

        tA = fx.logical_divide(tA, fx.make_layout(vec_width, 1))
        tC = fx.logical_divide(tC, fx.make_layout(vec_width, 1))

        reg_ty = fx.MemRefType.get(
            fx.T.f32(), fx.LayoutType.get(vec_width, 1), fx.AddressSpace.Register
        )
        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)

        rA = fx.memref_alloca(reg_ty, fx.make_layout(vec_width, 1))
        rC = fx.memref_alloca(reg_ty, fx.make_layout(vec_width, 1))

        fx.copy_atom_call(copy_atom, fx.slice(tA, (None, tid)), rA)
        vA = fx.memref_load_vec(rA)
        vOut = vA

        flag = tid < threshold

        if flag:
            vOut = fx.arith.addf(vA, vA)
        else:
            vOut = fx.arith.negf(vA)

        fx.memref_store_vec(vOut, rC)
        fx.copy_atom_call(copy_atom, rC, fx.slice(tC, (None, tid)))

    @flyc.jit
    def liveoutIfFlag(
        A: fx.Tensor,
        C,
        threshold: fx.Int32,
        n: fx.Int32,
        block_dim: fx.Constexpr[int],
        vec_width: fx.Constexpr[int],
        stream: fx.Stream = fx.Stream(None),
    ):
        tile_elems = block_dim * vec_width
        grid_x = (n + tile_elems - 1) // tile_elems
        liveoutIfFlagKernel(A, C, threshold, block_dim, vec_width).launch(
            grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream
        )

    num_blocks = 4
    size = BLOCK * VEC * num_blocks
    threshold = BLOCK // 2

    a = torch.randn(size, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)
    t_a = flyc.from_dlpack(a).mark_layout_dynamic(leading_dim=0, divisibility=VEC)

    liveoutIfFlag(t_a, c, threshold, size, BLOCK, VEC)
    torch.cuda.synchronize()

    a3 = a.view(num_blocks, BLOCK, VEC)
    tid_idx = torch.arange(BLOCK, device="cuda").view(1, BLOCK, 1)
    ref = torch.where(tid_idx < threshold, a3 + a3, -a3)
    ref = ref.reshape(-1)

    torch.testing.assert_close(c, ref, rtol=1e-5, atol=1e-5)
    max_diff = (c - ref).abs().max().item()
    print(f"[PASS] test_with_liveout (flag variant): max_diff={max_diff:.2e}, threshold={threshold}")
    print(f"  shape={c.shape}, flag=tid<{threshold} -> 2*a, else -> -a")
