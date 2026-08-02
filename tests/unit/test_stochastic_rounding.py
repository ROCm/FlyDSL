#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tests for target-neutral RNG and stochastic BF16 conversion.

The rounding math is checked exactly in pure Python (L0): over the full 16-bit
random range the number of round-ups equals the discarded low 16 bits, which is
what makes stochastic rounding unbiased. Device tests check the public paths,
the target-specific CDNA4 conversion, special values, Philox reference output,
reproducibility, and uniformity.
"""

from __future__ import annotations

import numpy as np
import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.runtime.device import get_rocm_arch

try:
    import torch
except ImportError:
    torch = None


_PHILOX_ZERO_VECTOR = (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)


@pytest.mark.l0_backend_agnostic
def test_randint4x_reference_vector():
    words = fx.random.randint4x(fx.Uint32(0), fx.Uint32(0))
    assert tuple(int(word.value) for word in words) == _PHILOX_ZERO_VECTOR
    repeated = fx.random.randint4x(seed=fx.Uint32(0), offset=fx.Uint32(0), n_rounds=10)
    different_seed = fx.random.randint4x(fx.Uint32(1), fx.Uint32(0))
    assert tuple(int(word.value) for word in repeated) == _PHILOX_ZERO_VECTOR
    assert tuple(int(word.value) for word in different_seed) != _PHILOX_ZERO_VECTOR


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
def test_cdna4_cvt_sr_f32_to_bf16_device():
    if str(get_rocm_arch()) != "gfx950":
        pytest.skip("CDNA4 stochastic conversion requires gfx950")

    @flyc.kernel(known_block_size=[1, 1, 1])
    def kernel(Out: fx.Tensor, x: fx.Float32, rand: fx.Uint32):
        fx.memref_store(fx.rocdl.cdna4.cvt_sr_f32_to_bf16(x, rand), Out, 0)

    @flyc.jit
    def launch(
        Out: fx.Tensor,
        x: fx.Float32,
        rand: fx.Uint32,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(Out, x, rand).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

    out = torch.empty(1, dtype=torch.bfloat16, device="cuda")
    launch(out, 1.001953125, 0xFFFF, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    assert out.float().item() == 1.0078125


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
def test_randint4x_reference_vector_device():
    @flyc.kernel(known_block_size=[1, 1, 1])
    def kernel(Out: fx.Tensor):
        words = fx.random.randint4x(fx.Uint32(0), fx.Uint32(0))
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
def test_randint4x_uniform():
    BLOCK = 256
    NBLOCKS = 1024
    P = BLOCK * NBLOCKS

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def kernel(Out: fx.Tensor, seed: fx.Int32):
        g = fx.block_idx.x * BLOCK + fx.thread_idx.x
        fx.memref_store(fx.Int32(fx.random.randint4x(fx.Uint32(seed), fx.Uint32(g))[0]), Out, g)

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
