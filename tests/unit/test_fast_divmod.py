#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tests for magic-number division: fastdivmod_magic / fast_divmod / FastDivmod.

The host magic math is checked in pure Python (L0). The device path is checked
by running a kernel that compares fast_divmod against native ``//`` and ``%``
for a runtime divisor (L2).
"""

from __future__ import annotations

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.numeric import FastDivmod, fast_divmod, fastdivmod_magic

try:
    import torch
except ImportError:
    torch = None


DIVISORS = [1, 2, 3, 7, 8, 127, 128, 768, 1000, 1024, 12289, 32000, 65536, 128256]
DIVIDENDS = [0, 1, 2, 5, 255, 256, 1023, 100000, 998244353, (1 << 31) - 1]


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize("divisor", DIVISORS)
def test_fastdivmod_magic_matches_floordiv(divisor):
    magic, shift = fastdivmod_magic(divisor)
    assert magic <= (1 << 32)
    for n in DIVIDENDS:
        q = (n * magic) >> (32 + shift)
        assert q == n // divisor, f"d={divisor} n={n}: got {q}, want {n // divisor}"
        assert n - q * divisor == n % divisor


@pytest.mark.l0_backend_agnostic
def test_fastdivmod_magic_rejects_out_of_range():
    with pytest.raises(ValueError):
        fastdivmod_magic(0)
    with pytest.raises(ValueError):
        fastdivmod_magic(1 << 31)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("divisor", [3, 7, 768, 32000, 128256])
def test_fast_divmod_device_matches_native(divisor):
    BLOCK = 256
    NBLOCKS = 64
    P = BLOCK * NBLOCKS

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def kernel(Q: fx.Tensor, R: fx.Tensor, d: fx.Int32, magic: fx.Int64, shift: fx.Int32):
        g = fx.block_idx.x * BLOCK + fx.thread_idx.x
        n = fx.Int32(fx.Uint32(g) * fx.Uint32(2654435761) & fx.Uint32(0x7FFFFFFF))
        q, r = fast_divmod(n, d, magic, shift)
        fx.memref_store(q, Q, g)
        fx.memref_store(r, R, g)

    @flyc.jit
    def launch(
        Q: fx.Tensor, R: fx.Tensor, d: fx.Int32, magic: fx.Int64, shift: fx.Int32, stream: fx.Stream = fx.Stream(None)
    ):
        kernel(Q, R, d, magic, shift).launch(grid=(NBLOCKS, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    magic, shift = fastdivmod_magic(divisor)
    q = torch.zeros(P, dtype=torch.int32, device="cuda")
    r = torch.zeros(P, dtype=torch.int32, device="cuda")
    launch(q, r, divisor, magic, shift, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    g = torch.arange(P, dtype=torch.int64, device="cuda")
    n = ((g * 2654435761) & 0x7FFFFFFF).to(torch.int64)
    assert torch.equal(q.to(torch.int64), n // divisor)
    assert torch.equal(r.to(torch.int64), n % divisor)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
def test_fastdivmod_class_constant_divisor():
    BLOCK = 128
    P = BLOCK * 8
    DIV = 768

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def kernel(Q: fx.Tensor, R: fx.Tensor):
        g = fx.block_idx.x * BLOCK + fx.thread_idx.x
        n = fx.Int32(fx.Uint32(g) * fx.Uint32(2654435761) & fx.Uint32(0x7FFFFFFF))
        fdm = FastDivmod(DIV)
        q, r = fdm.divmod(n)
        fx.memref_store(q, Q, g)
        fx.memref_store(r, R, g)

    @flyc.jit
    def launch(Q: fx.Tensor, R: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(Q, R).launch(grid=(P // BLOCK, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    q = torch.zeros(P, dtype=torch.int32, device="cuda")
    r = torch.zeros(P, dtype=torch.int32, device="cuda")
    launch(q, r, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    g = torch.arange(P, dtype=torch.int64, device="cuda")
    n = ((g * 2654435761) & 0x7FFFFFFF).to(torch.int64)
    assert torch.equal(q.to(torch.int64), n // DIV)
    assert torch.equal(r.to(torch.int64), n % DIV)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("divisor", [1, 3, 768, 128256])
def test_fastdivmod_dynamic_divisor(divisor):
    """FastDivmod built from a runtime divisor *inside* the kernel.

    Checks correctness only. This is deliberately not the pattern to copy: the
    magic derivation emits a 64-bit divide per thread (195 instructions on
    gfx1150 against 53 for a native divmod). Building it in the launch wrapper
    instead, as test_fastdivmod_value_protocol_kernel_arg does, keeps the divide
    out of the kernel and costs 27.
    """
    BLOCK = 256
    P = BLOCK * 32

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def kernel(Q: fx.Tensor, R: fx.Tensor, d: fx.Int32):
        g = fx.block_idx.x * BLOCK + fx.thread_idx.x
        n = fx.Int32(fx.Uint32(g) * fx.Uint32(2654435761) & fx.Uint32(0x7FFFFFFF))
        q, r = FastDivmod(d).divmod(n)
        fx.memref_store(q, Q, g)
        fx.memref_store(r, R, g)

    @flyc.jit
    def launch(Q: fx.Tensor, R: fx.Tensor, d: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        kernel(Q, R, d).launch(grid=(P // BLOCK, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    q = torch.zeros(P, dtype=torch.int32, device="cuda")
    r = torch.zeros(P, dtype=torch.int32, device="cuda")
    launch(q, r, divisor, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    g = torch.arange(P, dtype=torch.int64, device="cuda")
    n = ((g * 2654435761) & 0x7FFFFFFF).to(torch.int64)
    assert torch.equal(q.to(torch.int64), n // divisor)
    assert torch.equal(r.to(torch.int64), n % divisor)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
def test_fastdivmod_value_protocol_kernel_arg():
    """FastDivmod crosses the host/device boundary as a kernel argument.

    The launch wrapper builds it from a runtime divisor and passes the instance
    itself; the (magic, shift, divisor) leaves are extracted/reconstructed by the
    value protocol.
    """
    BLOCK = 256
    P = BLOCK * 32
    DIV = 768

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def kernel(Q: fx.Tensor, R: fx.Tensor, fdm: FastDivmod):
        g = fx.block_idx.x * BLOCK + fx.thread_idx.x
        n = fx.Int32(fx.Uint32(g) * fx.Uint32(2654435761) & fx.Uint32(0x7FFFFFFF))
        q, r = fdm.divmod(n)
        fx.memref_store(q, Q, g)
        fx.memref_store(r, R, g)

    @flyc.jit
    def launch(Q: fx.Tensor, R: fx.Tensor, d: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        kernel(Q, R, FastDivmod(d)).launch(grid=(P // BLOCK, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    q = torch.zeros(P, dtype=torch.int32, device="cuda")
    r = torch.zeros(P, dtype=torch.int32, device="cuda")
    launch(q, r, DIV, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    g = torch.arange(P, dtype=torch.int64, device="cuda")
    n = ((g * 2654435761) & 0x7FFFFFFF).to(torch.int64)
    assert torch.equal(q.to(torch.int64), n // DIV)
    assert torch.equal(r.to(torch.int64), n % DIV)
