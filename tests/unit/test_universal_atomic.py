#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tests for UniversalAtomic CopyOp — reduceAdd and atomicMax scenarios."""

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx

try:
    import torch
except ImportError:
    torch = None

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available", allow_module_level=True)


@flyc.kernel
def reduce_add_kernel(
    A: fx.Tensor,
    Out: fx.Tensor,
    block_dim: fx.Constexpr[int],
    syncscope: fx.Constexpr[str],
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    tA = fx.logical_divide(A, fx.make_layout(block_dim, 1))
    tA = fx.slice(tA, (None, bid))
    tA = fx.logical_divide(tA, fx.make_layout(1, 1))

    loadAtom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    atomicAtom = fx.make_copy_atom(
        fx.UniversalAtomic(fx.AtomicOp.Add, fx.Float32, syncscope=syncscope),
        fx.Float32,
    )

    rA = fx.make_rmem_tensor(1, fx.Float32)
    fx.copy_atom_call(loadAtom, fx.slice(tA, (None, tid)), rA)

    tOut = fx.logical_divide(Out, fx.make_layout(1, 1))
    tOut = fx.slice(tOut, (None, fx.Int32(0)))
    tOut = fx.logical_divide(tOut, fx.make_layout(1, 1))
    fx.copy_atom_call(atomicAtom, rA, fx.slice(tOut, (None, fx.Int32(0))))


@flyc.jit
def reduce_add(
    A: fx.Tensor,
    Out,
    n: fx.Int32,
    const_n: fx.Constexpr[int],
    block_dim: fx.Constexpr[int],
    syncscope: fx.Constexpr[str],
    stream: fx.Stream = fx.Stream(None),
):
    grid_x = (n + block_dim - 1) // block_dim
    reduce_add_kernel(A, Out, block_dim, syncscope).launch(
        grid=(grid_x, 1, 1),
        block=(block_dim, 1, 1),
        stream=stream,
    )


@pytest.mark.parametrize(
    "syncscope",
    [
        fx.SyncScope.System,
        fx.SyncScope.SingleThread,
        fx.rocdl.SyncScope.Agent,
        fx.rocdl.SyncScope.Workgroup,
        fx.rocdl.SyncScope.Wavefront,
    ],
)
def test_reduce_add_atomic(syncscope):
    BLOCK_DIM = 64
    N = BLOCK_DIM * 4

    a_dev = torch.ones(N, device="cuda", dtype=torch.float32)
    out_dev = torch.zeros(1, device="cuda", dtype=torch.float32)

    stream = torch.cuda.Stream()
    tA = flyc.from_torch_tensor(a_dev).mark_layout_dynamic(leading_dim=0, divisibility=1)
    reduce_add(tA, out_dev, N, N, BLOCK_DIM, syncscope, stream=stream)
    torch.cuda.synchronize()

    expected = a_dev.sum().item()
    actual = out_dev.item()
    print(f"[scope={syncscope!r}] Expected: {expected}, Got: {actual}")
    assert abs(actual - expected) < 1e-3, f"scope={syncscope!r} reduceAdd mismatch: expected {expected}, got {actual}"


@flyc.kernel
def reduce_max_kernel(
    A: fx.Tensor,
    Out: fx.Tensor,
    block_dim: fx.Constexpr[int],
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    tA = fx.logical_divide(A, fx.make_layout(block_dim, 1))
    tA = fx.slice(tA, (None, bid))
    tA = fx.logical_divide(tA, fx.make_layout(1, 1))

    loadAtom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    atomicAtom = fx.make_copy_atom(fx.UniversalAtomic(fx.AtomicOp.Max, fx.Float32), fx.Float32)

    rA = fx.make_rmem_tensor(1, fx.Float32)
    fx.copy_atom_call(loadAtom, fx.slice(tA, (None, tid)), rA)

    tOut = fx.logical_divide(Out, fx.make_layout(1, 1))
    tOut = fx.slice(tOut, (None, fx.Int32(0)))
    tOut = fx.logical_divide(tOut, fx.make_layout(1, 1))
    fx.copy_atom_call(atomicAtom, rA, fx.slice(tOut, (None, fx.Int32(0))))


@flyc.jit
def reduce_max(
    A: fx.Tensor,
    Out,
    n: fx.Int32,
    const_n: fx.Constexpr[int],
    block_dim: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    grid_x = (n + block_dim - 1) // block_dim
    reduce_max_kernel(A, Out, block_dim).launch(
        grid=(grid_x, 1, 1),
        block=(block_dim, 1, 1),
        stream=stream,
    )


def test_reduce_max_atomic():
    BLOCK_DIM = 64
    N = BLOCK_DIM * 4

    a_dev = torch.arange(N, device="cuda", dtype=torch.float32)
    out_dev = torch.full((1,), float("-inf"), device="cuda", dtype=torch.float32)

    stream = torch.cuda.Stream()
    tA = flyc.from_torch_tensor(a_dev).mark_layout_dynamic(leading_dim=0, divisibility=1)
    reduce_max(tA, out_dev, N, N, BLOCK_DIM, stream=stream)
    torch.cuda.synchronize()

    expected = a_dev.max().item()
    actual = out_dev.item()
    print(f"Expected: {expected}, Got: {actual}")
    assert abs(actual - expected) < 1e-3, f"atomicMax mismatch: expected {expected}, got {actual}"


if __name__ == "__main__":
    for _scope in [
        fx.SyncScope.System,
        fx.SyncScope.SingleThread,
        fx.rocdl.SyncScope.Agent,
        fx.rocdl.SyncScope.Workgroup,
        fx.rocdl.SyncScope.Wavefront,
    ]:
        test_reduce_add_atomic(_scope)
    test_reduce_max_atomic()
    print("ALL PASSED")
