#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tests for fx.rocdl stochastic rounding + philox RNG.

The rounding math is checked exactly in pure Python (L0): over the full 16-bit
random range the number of round-ups equals the discarded low 16 bits, which is
what makes stochastic rounding unbiased. The device path and the RNG are checked
on GPU (L2): outputs land on the two nearest bf16 values with the right up
probability, and the empirical mean matches the input.
"""

from __future__ import annotations

import numpy as np
import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx

try:
    import torch
except ImportError:
    torch = None


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize("x", [1.0, 1.001953125, 3.14159, 1e-3, float("inf"), float("-inf")])
def test_stochastic_round_bf16_unbiased(x):
    u = int(np.array([x], dtype=np.float32).view(np.uint32)[0])
    low16 = u & 0xFFFF
    base = u >> 16
    r = np.arange(0, 1 << 16, dtype=np.uint64)
    truncated = ((np.uint64(u) + r) >> np.uint64(16)).astype(np.uint64)
    # exactly `low16` of the 65536 random draws round up
    up_count = int(np.sum(truncated != np.uint64(base)))
    assert up_count == low16
    # every result is one of the two nearest bf16 values
    assert set(truncated.tolist()) <= {base, base + 1}


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("frac", [0.0, 0.5, 0.9])
def test_stochastic_round_bf16_device(frac):
    BLOCK = 256
    NBLOCKS = 4096
    P = BLOCK * NBLOCKS

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def kernel(Out: fx.Tensor, x: fx.Float32, seed: fx.Int32):
        g = fx.block_idx.x * BLOCK + fx.thread_idx.x
        r = fx.rocdl.philox_4x32(fx.Uint32(g), fx.Uint32(seed))[0]
        fx.memref_store(fx.rocdl.stochastic_round_bf16(x, r), Out, g)

    @flyc.jit
    def launch(Out: fx.Tensor, x: fx.Float32, seed: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        kernel(Out, x, seed).launch(grid=(NBLOCKS, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    step = 2.0**-7  # bf16 step above 1.0
    x = 1.0 + frac * step
    lo, hi = 1.0, 1.0 + step

    out = torch.zeros(P, dtype=torch.bfloat16, device="cuda")
    launch(out, x, 1234, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    f = out.float()

    assert torch.all((f == lo) | (f == hi)), "outputs must be one of the two nearest bf16 values"
    up_prob = (f == hi).float().mean().item()
    assert abs(up_prob - frac) < 0.01, f"up_prob {up_prob} far from frac {frac}"
    assert abs(f.mean().item() - x) < 1e-4


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
def test_philox_4x32_uniform():
    BLOCK = 256
    NBLOCKS = 1024
    P = BLOCK * NBLOCKS

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def kernel(Out: fx.Tensor, seed: fx.Int32):
        g = fx.block_idx.x * BLOCK + fx.thread_idx.x
        fx.memref_store(fx.Int32(fx.rocdl.philox_4x32(fx.Uint32(g), fx.Uint32(seed))[0]), Out, g)

    @flyc.jit
    def launch(Out: fx.Tensor, seed: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        kernel(Out, seed).launch(grid=(NBLOCKS, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    out = torch.zeros(P, dtype=torch.int32, device="cuda")
    launch(out, 7, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    u = out.cpu().numpy().astype(np.uint32).astype(np.float64) / 2**32
    assert abs(u.mean() - 0.5) < 0.01, f"mean {u.mean()} not uniform"
    # counter-based PRNG on distinct counters must not collide in bulk
    assert np.unique(out.cpu().numpy()).size > int(0.999 * P)
