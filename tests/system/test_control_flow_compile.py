#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import flydsl.compiler as flyc
import flydsl.expr as fx
import pytest
import torch


def test_control_flow_kernel_snippet_compiles_without_error(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required for control-flow compile coverage test")

    @flyc.kernel
    def vecAbsKernel(
        A: fx.Tensor,
        C: fx.Tensor,
        block_dim: fx.Constexpr[int],
        vec_width: fx.Constexpr[int],
        print_debug: fx.Constexpr[bool] = True,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        if print_debug and bid == 0 and tid <= 2:
            fx.printf("[kernel] bid={}, tid={}", bid, tid)

    @flyc.jit
    def vecAbs(
        A: fx.Tensor,
        C,
        n: fx.Int32,
        const_n: fx.Constexpr[int],
        block_dim: fx.Constexpr[int],
        vec_width: fx.Constexpr[int],
        stream: fx.Stream = fx.Stream(None),
    ):
        tile_elems = block_dim * vec_width
        grid_x = (n + tile_elems - 1) // tile_elems
        vecAbsKernel(A, C, block_dim, vec_width).launch(
            grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream
        )

    monkeypatch.setenv("FLYDSL_COMPILE_ONLY", "1")
    threads = 64
    vec = 4
    size = threads * vec
    a = torch.randn(size, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)
    t_a = flyc.from_dlpack(a).mark_layout_dynamic(leading_dim=0, divisibility=vec)
    vecAbs(t_a, c, size, size, threads, vec)


def test_control_flow_dynamic_if_end_to_end_numeric(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required for dynamic if end-to-end test")
    # Avoid compile-cache hits so dynamic dispatch is exercised in this test process.
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

    @flyc.kernel
    def dynamicIfKernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        block_dim: fx.Constexpr[int],
        vec_width: fx.Constexpr[int],
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        tile_elems = block_dim * vec_width

        A = fx.rocdl.make_buffer_tensor(A)
        B = fx.rocdl.make_buffer_tensor(B)
        C = fx.rocdl.make_buffer_tensor(C)

        tA = fx.logical_divide(A, fx.make_layout(tile_elems, 1))
        tB = fx.logical_divide(B, fx.make_layout(tile_elems, 1))
        tC = fx.logical_divide(C, fx.make_layout(tile_elems, 1))

        tA = fx.slice(tA, (None, bid))
        tB = fx.slice(tB, (None, bid))
        tC = fx.slice(tC, (None, bid))

        tA = fx.logical_divide(tA, fx.make_layout(vec_width, 1))
        tB = fx.logical_divide(tB, fx.make_layout(vec_width, 1))
        tC = fx.logical_divide(tC, fx.make_layout(vec_width, 1))

        reg_ty = fx.MemRefType.get(
            fx.T.f32(), fx.LayoutType.get(vec_width, 1), fx.AddressSpace.Register
        )
        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)

        rA = fx.memref_alloca(reg_ty, fx.make_layout(vec_width, 1))
        rB = fx.memref_alloca(reg_ty, fx.make_layout(vec_width, 1))
        rC = fx.memref_alloca(reg_ty, fx.make_layout(vec_width, 1))

        fx.copy_atom_call(copy_atom, fx.slice(tA, (None, tid)), rA)
        fx.copy_atom_call(copy_atom, fx.slice(tB, (None, tid)), rB)

        vA = fx.memref_load_vec(rA)
        vB = fx.memref_load_vec(rB)
        vOut = fx.arith.addf(vA, vB)

        # Runtime branch (tid/bid come from GPU execution), so this should lower to dynamic scf.if.
        if (tid % 2) == 0:
            vOut = fx.arith.addf(vOut, vA)
        else:
            vOut = fx.arith.subf(vOut, vB)

        if (bid % 2) == 0:
            vOut = fx.arith.addf(vOut, vB)
        else:
            vOut = fx.arith.subf(vOut, vA)

        fx.memref_store_vec(vOut, rC)
        fx.copy_atom_call(copy_atom, rC, fx.slice(tC, (None, tid)))

    @flyc.jit
    def dynamicIfVec(
        A: fx.Tensor,
        B: fx.Tensor,
        C,
        n: fx.Int32,
        block_dim: fx.Constexpr[int],
        vec_width: fx.Constexpr[int],
        stream: fx.Stream = fx.Stream(None),
    ):
        tile_elems = block_dim * vec_width
        grid_x = (n + tile_elems - 1) // tile_elems
        dynamicIfKernel(A, B, C, block_dim, vec_width).launch(
            grid=(grid_x, 1, 1), block=(block_dim, 1, 1), stream=stream
        )

    block_dim = 64
    vec_width = 4
    num_blocks = 5
    size = block_dim * vec_width * num_blocks

    a = torch.randn(size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)

    t_a = flyc.from_dlpack(a).mark_layout_dynamic(leading_dim=0, divisibility=vec_width)
    dynamicIfVec(t_a, b, c, size, block_dim, vec_width)
    torch.cuda.synchronize()

    a3 = a.view(num_blocks, block_dim, vec_width)
    b3 = b.view(num_blocks, block_dim, vec_width)
    tid = torch.arange(block_dim, device="cuda").view(1, block_dim, 1)
    bid = torch.arange(num_blocks, device="cuda").view(num_blocks, 1, 1)

    ref = a3 + b3
    ref = torch.where((tid % 2) == 0, ref + a3, ref - b3)
    ref = torch.where((bid % 2) == 0, ref + b3, ref - a3)
    ref = ref.reshape(-1)

    torch.testing.assert_close(c, ref, rtol=1e-5, atol=1e-5)
