#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tests for the bare ``fx.llvm`` atomics."""

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
def bare_atomic_add_kernel(
    A: fx.Pointer,
    Out: fx.Pointer,
    block_dim: fx.Constexpr[int],
    syncscope: fx.Constexpr[str],
):
    idx = fx.block_idx.x * block_dim + fx.thread_idx.x
    val = (A + idx).load()
    fx.atomic_add(Out, val, syncscope=syncscope)


@flyc.jit
def bare_atomic_add(
    A: fx.Pointer,
    Out: fx.Pointer,
    n: fx.Int32,
    block_dim: fx.Constexpr[int],
    syncscope: fx.Constexpr[str],
    stream: fx.Stream = fx.Stream(None),
):
    grid_x = (n + block_dim - 1) // block_dim
    bare_atomic_add_kernel(A, Out, block_dim, syncscope).launch(
        grid=(grid_x, 1, 1),
        block=(block_dim, 1, 1),
        stream=stream,
    )


@pytest.mark.parametrize("syncscope", [fx.SyncScope.System, fx.rocdl.SyncScope.Agent])
def test_bare_atomic_add(syncscope):
    BLOCK_DIM = 64
    N = BLOCK_DIM * 4

    a_dev = torch.ones(N, device="cuda", dtype=torch.float32)
    out_dev = torch.zeros(1, device="cuda", dtype=torch.float32)

    stream = torch.cuda.Stream()
    pA = flyc.from_c_void_p(fx.Float32, a_dev.data_ptr())
    pOut = flyc.from_c_void_p(fx.Float32, out_dev.data_ptr())
    bare_atomic_add(pA, pOut, N, BLOCK_DIM, syncscope, stream=stream)
    torch.cuda.synchronize()

    assert abs(out_dev.item() - float(N)) < 1e-3, f"expected {N}, got {out_dev.item()}"


@flyc.kernel
def bare_atomic_ticket_kernel(
    Counter: fx.Pointer,
    Out: fx.Pointer,
    block_dim: fx.Constexpr[int],
):
    idx = fx.block_idx.x * block_dim + fx.thread_idx.x
    ticket = fx.atomic_add(Counter, fx.Int32(1))
    (Out + idx).store(ticket)


@flyc.jit
def bare_atomic_ticket(
    Counter: fx.Pointer,
    Out: fx.Pointer,
    n: fx.Int32,
    block_dim: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    grid_x = (n + block_dim - 1) // block_dim
    bare_atomic_ticket_kernel(Counter, Out, block_dim).launch(
        grid=(grid_x, 1, 1),
        block=(block_dim, 1, 1),
        stream=stream,
    )


def test_bare_atomic_add_return():
    BLOCK_DIM = 64
    N = BLOCK_DIM * 4

    counter_dev = torch.zeros(1, device="cuda", dtype=torch.int32)
    out_dev = torch.zeros(N, device="cuda", dtype=torch.int32)

    stream = torch.cuda.Stream()
    pCounter = flyc.from_c_void_p(fx.Int32, counter_dev.data_ptr())
    pOut = flyc.from_c_void_p(fx.Int32, out_dev.data_ptr())
    bare_atomic_ticket(pCounter, pOut, N, BLOCK_DIM, stream=stream)
    torch.cuda.synchronize()

    assert counter_dev.cpu().item() == N
    # Every thread must observe a distinct pre-update value, i.e. a permutation of [0, N).
    tickets = out_dev.cpu().sort().values
    assert torch.equal(tickets, torch.arange(N, dtype=torch.int32))


@flyc.kernel
def bare_atomic_unsigned_min_kernel(
    A: fx.Pointer,
    Out: fx.Pointer,
    block_dim: fx.Constexpr[int],
):
    idx = fx.block_idx.x * block_dim + fx.thread_idx.x
    value = (A + idx).load(fx.Uint32)
    fx.atomic_min(Out, value)


@flyc.jit
def bare_atomic_unsigned_min(
    A: fx.Pointer,
    Out: fx.Pointer,
    n: fx.Int32,
    block_dim: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    grid_x = (n + block_dim - 1) // block_dim
    bare_atomic_unsigned_min_kernel(A, Out, block_dim).launch(
        grid=(grid_x, 1, 1),
        block=(block_dim, 1, 1),
        stream=stream,
    )


def test_bare_atomic_unsigned_min():
    values = torch.tensor([-1, -(2**31), 7, 23], device="cuda", dtype=torch.int32)
    out_dev = torch.full((1,), -1, device="cuda", dtype=torch.int32)

    stream = torch.cuda.Stream()
    pA = flyc.from_c_void_p(fx.Uint32, values.data_ptr())
    pOut = flyc.from_c_void_p(fx.Uint32, out_dev.data_ptr())
    bare_atomic_unsigned_min(pA, pOut, values.numel(), values.numel(), stream=stream)
    torch.cuda.synchronize()

    # Interpreted as uint32, the inputs are [2**32 - 1, 2**31, 7, 23].
    assert out_dev.item() == 7


@flyc.kernel
def bare_atomic_fminmax_kernel(
    A: fx.Pointer,
    Out: fx.Pointer,
    block_dim: fx.Constexpr[int],
    take_max: fx.Constexpr[bool],
):
    idx = fx.block_idx.x * block_dim + fx.thread_idx.x
    val = (A + idx).load()
    if take_max:
        fx.atomic_fmax(Out, val)
    else:
        fx.atomic_fmin(Out, val)


@flyc.jit
def bare_atomic_fminmax(
    A: fx.Pointer,
    Out: fx.Pointer,
    n: fx.Int32,
    block_dim: fx.Constexpr[int],
    take_max: fx.Constexpr[bool],
    stream: fx.Stream = fx.Stream(None),
):
    grid_x = (n + block_dim - 1) // block_dim
    bare_atomic_fminmax_kernel(A, Out, block_dim, take_max).launch(
        grid=(grid_x, 1, 1),
        block=(block_dim, 1, 1),
        stream=stream,
    )


@pytest.mark.parametrize("take_max", [True, False])
@pytest.mark.parametrize(
    "sign",
    ["mixed", "positive", "negative"],
)
def test_bare_atomic_float_minmax(take_max, sign):
    """The integer-atomic fmax / fmin must order negatives and positives alike."""
    BLOCK_DIM = 64
    N = BLOCK_DIM * 4

    values = (torch.arange(N, dtype=torch.float32) - N // 2) / 8.0
    if sign == "positive":
        values = values.abs() + 1.0
    elif sign == "negative":
        values = -(values.abs() + 1.0)

    a_dev = values.cuda()
    init = float("-inf") if take_max else float("inf")
    out_dev = torch.full((1,), init, device="cuda", dtype=torch.float32)

    stream = torch.cuda.Stream()
    pA = flyc.from_c_void_p(fx.Float32, a_dev.data_ptr())
    pOut = flyc.from_c_void_p(fx.Float32, out_dev.data_ptr())
    bare_atomic_fminmax(pA, pOut, N, BLOCK_DIM, take_max, stream=stream)
    torch.cuda.synchronize()

    expected = values.max().item() if take_max else values.min().item()
    got = out_dev.cpu().item()
    assert got == expected, f"expected {expected}, got {got}"


if __name__ == "__main__":
    test_bare_atomic_add(fx.SyncScope.System)
    test_bare_atomic_add(fx.rocdl.SyncScope.Agent)
    test_bare_atomic_add_return()
    test_bare_atomic_unsigned_min()
    for _take_max in (True, False):
        for _sign in ("mixed", "positive", "negative"):
            test_bare_atomic_float_minmax(_take_max, _sign)
    print("ALL PASSED")
