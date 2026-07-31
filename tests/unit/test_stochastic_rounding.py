#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tests for target-neutral RNG and stochastic BF16 conversion.

The rounding math is checked exactly in pure Python (L0): over the full 16-bit
random range the number of round-ups equals the discarded low 16 bits, which is
what makes stochastic rounding unbiased. Device tests check the public paths,
special values, Philox reference output, reproducibility, and uniformity.
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


_PHILOX_ZERO_VECTOR = (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)


@pytest.mark.l0_backend_agnostic
def test_philox_4x32_reference_vector():
    words = fx.random.philox_4x32(fx.Uint32(0), fx.Uint32(0))
    assert tuple(int(word.value) for word in words) == _PHILOX_ZERO_VECTOR
    repeated = fx.random.philox_4x32(fx.Uint32(0), fx.Uint32(0))
    different_seed = fx.random.philox_4x32(fx.Uint32(0), fx.Uint32(1))
    assert tuple(int(word.value) for word in repeated) == _PHILOX_ZERO_VECTOR
    assert tuple(int(word.value) for word in different_seed) != _PHILOX_ZERO_VECTOR


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize(
    "x",
    [1.0, 1.001953125, 3.14159, 1e-3, -1.001953125, -3.14159, float("inf"), float("-inf")],
)
def test_cvt_f32_to_bf16_sr_unbiased(x):
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
def test_cvt_f32_to_bf16_sr_device(frac):
    BLOCK = 256
    NBLOCKS = 4096
    P = BLOCK * NBLOCKS

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def kernel(Out: fx.Tensor, x: fx.Float32, seed: fx.Int32):
        g = fx.block_idx.x * BLOCK + fx.thread_idx.x
        r = fx.Uint32(g) * fx.Uint32(1664525) + fx.Uint32(seed)
        fx.memref_store(fx.random.cvt_f32_to_bf16_sr(x, r), Out, g)

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
def test_cvt_f32_to_bf16_sr_special_values_device():
    BLOCK = 1

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def kernel(Out: fx.Tensor, x: fx.Float32, seed: fx.Int32):
        g = fx.block_idx.x * BLOCK + fx.thread_idx.x
        r = fx.Uint32(g) * fx.Uint32(1664525) + fx.Uint32(seed)
        fx.memref_store(fx.random.cvt_f32_to_bf16_sr(x, r), Out, g)

    @flyc.jit
    def launch(Out: fx.Tensor, x: fx.Float32, seed: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        kernel(Out, x, seed).launch(grid=(1, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    def run(x, seed):
        out = torch.empty(BLOCK, dtype=torch.bfloat16, device="cuda")
        launch(out, x, seed, stream=torch.cuda.Stream())
        torch.cuda.synchronize()
        return out

    for x in (float("inf"), float("-inf"), float("nan")):
        first = run(x, 1)
        second = run(x, 2)
        assert torch.equal(first.view(torch.int16), second.view(torch.int16))
        if np.isnan(x):
            assert torch.isnan(first).all()
        else:
            assert torch.isinf(first).all()
            assert (first < 0).all().item() == (x < 0)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
def test_philox_4x32_reference_vector_device():
    @flyc.kernel(known_block_size=[1, 1, 1])
    def kernel(Out: fx.Tensor):
        words = fx.random.philox_4x32(fx.Uint32(0), fx.Uint32(0))
        fx.memref_store(fx.Int32(words[0]), Out, 0)
        fx.memref_store(fx.Int32(words[1]), Out, 1)
        fx.memref_store(fx.Int32(words[2]), Out, 2)
        fx.memref_store(fx.Int32(words[3]), Out, 3)

    @flyc.jit
    def launch(Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(Out).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

    out = torch.empty(4, dtype=torch.int32, device="cuda")
    launch(out, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    assert tuple(int(value) & 0xFFFFFFFF for value in out.cpu().tolist()) == _PHILOX_ZERO_VECTOR


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
        fx.memref_store(fx.Int32(fx.random.philox_4x32(fx.Uint32(g), fx.Uint32(seed))[0]), Out, g)

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
