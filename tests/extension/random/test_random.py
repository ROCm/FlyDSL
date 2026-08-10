#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tests for target-neutral RNG."""

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
_PHILOX64_ZERO_VECTOR = (0x16554D9ECA36314C, 0xDB20FE9D672D0FDC, 0xD7E772CEE186176B, 0x7E68B68AEC7BA23B)
_UNIFORM_SCALE = np.float32(4.6566127342e-10)
_UNIFORM64_SCALE = np.float32(1.0842020432385337e-19)


def _philox_reference(seed, offsets, n_rounds=10):
    """Numpy Philox 4x32 reference, vectorized over *offsets*.

    Mirrors Triton's splitting: the seed widens to 64 bits (so a negative one
    sign-extends) and feeds both key words, and an offset wider than 32 bits
    fills the second counter word.
    """
    offsets = np.asarray(offsets)
    if offsets.dtype.itemsize > 4:
        wide = offsets.astype(np.uint64)
        c0 = wide.astype(np.uint32)
        c1 = (wide >> np.uint64(32)).astype(np.uint32)
    else:
        c0 = offsets.astype(np.uint32)
        c1 = np.zeros_like(c0)
    c2 = np.zeros_like(c0)
    c3 = np.zeros_like(c0)
    seed64 = np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
    k0 = np.full_like(c0, seed64.astype(np.uint32))
    k1 = np.full_like(c0, (seed64 >> np.uint64(32)).astype(np.uint32))
    for _ in range(n_rounds):
        prod_b = c2.astype(np.uint64) * np.uint64(0xCD9E8D57)
        prod_a = c0.astype(np.uint64) * np.uint64(0xD2511F53)
        next_c0 = (prod_b >> np.uint64(32)).astype(np.uint32) ^ c1 ^ k0
        next_c2 = (prod_a >> np.uint64(32)).astype(np.uint32) ^ c3 ^ k1
        c0, c1 = next_c0, prod_b.astype(np.uint32)
        c2, c3 = next_c2, prod_a.astype(np.uint32)
        k0 = k0 + np.uint32(0x9E3779B9)
        k1 = k1 + np.uint32(0xBB67AE85)
    return c0, c1, c2, c3


def _philox64_reference(counter, key, n_rounds=10):
    """Scalar Philox 4x64 reference using Triton's constants and round order."""
    mask = (1 << 64) - 1
    c0, c1, c2, c3 = (int(word) & mask for word in counter)
    k0, k1 = (int(word) & mask for word in key)
    for _ in range(n_rounds):
        prod_b = c2 * 0xCA5A826395121157
        prod_a = c0 * 0xD2E7470EE14C6C93
        c0 = ((prod_b >> 64) ^ c1 ^ k0) & mask
        c2 = ((prod_a >> 64) ^ c3 ^ k1) & mask
        c1 = prod_b & mask
        c3 = prod_a & mask
        k0 = (k0 + 0x9E3779B97F4A7C15) & mask
        k1 = (k1 + 0xBB67AE8584CAA73B) & mask
    return c0, c1, c2, c3


def _uniform_reference(words):
    """Numpy version of Triton's ``uint_to_uniform_float``."""
    signed = np.asarray(words, dtype=np.uint32).view(np.int32)
    folded = np.where(signed < 0, (-signed.astype(np.int64) - 1).astype(np.int32), signed)
    return folded.astype(np.float32) * _UNIFORM_SCALE


def _normal_pair_reference(u1, u2):
    """Numpy version of Triton's ``pair_uniform_to_normal`` (Box-Muller)."""
    u1 = np.maximum(np.float32(1.0e-7), u1)
    theta = np.float32(6.283185307179586) * u2
    r = np.sqrt(np.float32(-2.0) * np.log(u1))
    return r * np.cos(theta), r * np.sin(theta)


@pytest.mark.l0_backend_agnostic
def test_random_public_api():
    assert fx.random.__all__ == ["randint", "randint4x", "rand", "rand4x", "randn", "randn4x"]
    assert not hasattr(fx.random, "philox")


@pytest.mark.l0_backend_agnostic
def test_randint4x_reference_vector():
    words = fx.random.randint4x(fx.Uint32(0), fx.Uint32(0))
    assert tuple(int(word.value) for word in words) == _PHILOX_ZERO_VECTOR
    repeated = fx.random.randint4x(seed=fx.Uint32(0), offset=fx.Uint32(0), n_rounds=10)
    different_seed = fx.random.randint4x(fx.Uint32(1), fx.Uint32(0))
    assert tuple(int(word.value) for word in repeated) == _PHILOX_ZERO_VECTOR
    assert tuple(int(word.value) for word in different_seed) != _PHILOX_ZERO_VECTOR


@pytest.mark.l0_backend_agnostic
def test_philox_4x64_matches_triton_reference():
    """The internal 64-bit path uses Triton's Philox 4x64 constants and key layout."""
    zero = fx.Uint64(0)
    words = fx.random.universal.philox(zero, zero, zero, zero, zero)
    assert tuple(int(word.value) for word in words) == _PHILOX64_ZERO_VECTOR
    assert all(word.dtype is fx.Uint64 for word in words)

    counter = (0xFFFFFFFFFFFFFFFF, 0x123456789ABCDEF0, 0x0FEDCBA987654321, 7)
    seed = 0x89ABCDEF01234567
    expected = _philox64_reference(counter, (seed, 0))
    actual = fx.random.universal.philox(fx.Uint64(seed), *(fx.Uint64(word) for word in counter))
    assert tuple(int(word.value) for word in actual) == expected

    key = (seed, 0x76543210FEDCBA98)
    expected = _philox64_reference(counter, key)
    actual = fx.random.universal.philox_impl(
        *(fx.Uint64(word) for word in counter),
        *(fx.Uint64(word) for word in key),
    )
    assert tuple(int(word.value) for word in actual) == expected


@pytest.mark.l1a_compile_no_target_dialect
def test_philox_4x64_dynamic_path_builds_widening_ir():
    @flyc.jit
    def body(seed: fx.Uint64, c0: fx.Uint64):
        zero = fx.Uint64(0)
        words = fx.random.universal.philox(seed, c0, zero, zero, zero)
        for word in words:
            word.ir_value()
        fx.random.universal.uint_to_uniform_float(words[0]).ir_value()

    body(7, 11)
    source = body._last_compiled[1].source_ir
    assert "arith.muli" in source
    assert "i128" in source
    assert "arith.sitofp" in source


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize("word", [0, 1, 2**63 - 1, 2**63, 2**64 - 1])
def test_uint64_to_uniform_float_matches_triton(word):
    signed = word if word < 2**63 else word - 2**64
    folded = -signed - 1 if signed < 0 else signed
    expected = np.float32(folded) * _UNIFORM64_SCALE
    actual = fx.random.universal.uint_to_uniform_float(fx.Uint64(word))
    assert actual.dtype is fx.Float32
    assert float(actual.value) == float(expected)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
def test_randint4x_reference_vector_device():
    @flyc.kernel(known_block_size=[1, 1, 1])
    def kernel(Out: fx.Tensor):
        words = fx.random.randint4x(fx.Uint32(0), fx.Uint32(0))
        Out[0] = words[0]
        Out[1] = words[1]
        Out[2] = words[2]
        Out[3] = words[3]

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
def test_philox_4x64_reference_vector_device():
    @flyc.kernel(known_block_size=[1, 1, 1])
    def kernel(Out: fx.Tensor, seed: fx.Uint64):
        zero = fx.Uint64(0)
        words = fx.random.universal.philox(seed, seed, zero, zero, zero)
        for i in fx.range_constexpr(4):
            Out[i] = words[i]

    @flyc.jit
    def launch(Out: fx.Tensor, seed: fx.Uint64, stream: fx.Stream = fx.Stream(None)):
        kernel(Out, seed).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

    out = torch.empty(4, dtype=torch.int64, device="cuda")
    launch(out, 0, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    assert tuple(int(value) & 0xFFFFFFFFFFFFFFFF for value in out.cpu().tolist()) == _PHILOX64_ZERO_VECTOR


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
        Out[g] = fx.random.randint4x(fx.Uint32(seed), fx.Uint32(g))[0]

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


@pytest.mark.l0_backend_agnostic
def test_scalar_ops_match_philox_reference():
    """``randint`` / ``rand`` / ``rand4x`` agree with the numpy reference."""
    for seed, offset in ((0, 0), (7, 12345), (0x9E3779B9, 0xFFFFFFF0)):
        words = [int(w[0]) for w in _philox_reference(seed, [offset])]
        assert tuple(int(w.value) for w in fx.random.randint4x(fx.Uint32(seed), fx.Uint32(offset))) == tuple(words)
        assert int(fx.random.randint(fx.Uint32(seed), fx.Uint32(offset)).value) == words[0]

        expected = [float(u) for u in _uniform_reference(words)]
        uniforms = [float(u.value) for u in fx.random.rand4x(fx.Uint32(seed), fx.Uint32(offset))]
        assert uniforms == pytest.approx(expected, rel=1e-6)
        assert float(fx.random.rand(fx.Uint32(seed), fx.Uint32(offset)).value) == pytest.approx(expected[0], rel=1e-6)
        assert all(0.0 <= u < 1.0 for u in uniforms)


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="requires GPU")
def test_rand4x_randn4x_match_reference():
    """The float entry points reproduce Triton's uniform / Box-Muller pipeline."""
    BLOCK = 256
    NBLOCKS = 16
    P = BLOCK * NBLOCKS
    SEED = 7

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def kernel(Uniform: fx.Tensor, Normal: fx.Tensor, Scalar: fx.Tensor, seed: fx.Int32):
        g = fx.block_idx.x * BLOCK + fx.thread_idx.x
        key = fx.Uint32(seed)
        offset = fx.Uint32(g)
        uniforms = fx.random.rand4x(key, offset)
        normals = fx.random.randn4x(key, offset)
        for i in fx.range_constexpr(4):
            Uniform[g * 4 + i] = uniforms[i]
            Normal[g * 4 + i] = normals[i]
        Scalar[g * 2] = fx.random.rand(key, offset)
        Scalar[g * 2 + 1] = fx.random.randn(key, offset)

    @flyc.jit
    def launch(
        Uniform: fx.Tensor,
        Normal: fx.Tensor,
        Scalar: fx.Tensor,
        seed: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(Uniform, Normal, Scalar, seed).launch(grid=(NBLOCKS, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    uniform = torch.zeros(P * 4, dtype=torch.float32, device="cuda")
    normal = torch.zeros(P * 4, dtype=torch.float32, device="cuda")
    scalar = torch.zeros(P * 2, dtype=torch.float32, device="cuda")
    launch(uniform, normal, scalar, SEED, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    uniform = uniform.cpu().numpy().reshape(P, 4)
    normal = normal.cpu().numpy().reshape(P, 4)
    scalar = scalar.cpu().numpy().reshape(P, 2)

    words = _philox_reference(SEED, np.arange(P, dtype=np.uint32))
    ref_uniform = np.stack([_uniform_reference(w) for w in words], axis=1)
    n0, n1 = _normal_pair_reference(ref_uniform[:, 0], ref_uniform[:, 1])
    n2, n3 = _normal_pair_reference(ref_uniform[:, 2], ref_uniform[:, 3])
    ref_normal = np.stack([n0, n1, n2, n3], axis=1)

    np.testing.assert_allclose(uniform, ref_uniform, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(normal, ref_normal, rtol=1e-3, atol=1e-4)
    # the single-draw entry points share the stream with the 4x ones
    np.testing.assert_array_equal(scalar[:, 0], uniform[:, 0])
    np.testing.assert_allclose(scalar[:, 1], normal[:, 0], rtol=1e-6, atol=1e-7)

    assert uniform.min() >= 0.0 and uniform.max() < 1.0
    assert abs(uniform.mean() - 0.5) < 0.01, f"uniform mean {uniform.mean()}"
    assert abs(normal.mean()) < 0.02, f"normal mean {normal.mean()}"
    assert abs(normal.std() - 1.0) < 0.02, f"normal std {normal.std()}"


@pytest.mark.l0_backend_agnostic
def test_wide_and_signed_inputs_follow_triton_splitting():
    """A 64-bit offset fills counter word 1 and a wide seed fills key word 1."""
    cases = [
        (fx.Int32(-5), fx.Uint32(11), -5, np.array([11], dtype=np.uint32)),
        (fx.Uint64(0x123456789ABCDEF), fx.Uint32(3), 0x123456789ABCDEF, np.array([3], dtype=np.uint32)),
        (fx.Int32(7), fx.Uint64(2**32 + 3), 7, np.array([2**32 + 3], dtype=np.uint64)),
        (fx.Uint64(2**40 + 9), fx.Uint64(2**33 + 1), 2**40 + 9, np.array([2**33 + 1], dtype=np.uint64)),
    ]
    for seed, offset, ref_seed, ref_offsets in cases:
        expected = tuple(int(w[0]) for w in _philox_reference(ref_seed, ref_offsets))
        assert tuple(int(w.value) for w in fx.random.randint4x(seed, offset)) == expected

    # a signed seed sign-extends into the high key word, so it is a different
    # stream from the unsigned seed with the same low 32 bits
    signed = fx.random.randint4x(fx.Int32(-5), fx.Uint32(11))
    unsigned = fx.random.randint4x(fx.Uint32(0xFFFFFFFB), fx.Uint32(11))
    assert tuple(int(w.value) for w in signed) != tuple(int(w.value) for w in unsigned)


@pytest.mark.l0_backend_agnostic
def test_32bit_inputs_keep_the_high_words_zero():
    """Widening support must not disturb the plain 32-bit stream."""
    for seed, offset in ((0, 0), (7, 12345), (0x9E3779B9, 0xFFFFFFF0)):
        words = tuple(int(w.value) for w in fx.random.randint4x(fx.Uint32(seed), fx.Uint32(offset)))
        # same draw as feeding the counter / key high words an explicit zero
        wide = tuple(int(w.value) for w in fx.random.randint4x(fx.Uint64(seed), fx.Uint64(offset)))
        assert words == wide
    assert tuple(int(w.value) for w in fx.random.randint4x(fx.Uint32(0), fx.Uint32(0))) == _PHILOX_ZERO_VECTOR
