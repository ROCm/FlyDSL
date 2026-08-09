# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Target-neutral implementations of the random library."""

from ...expr.math import cos, log, sin, sqrt
from ...expr.numeric import Float32, Int32, Int64, Uint32, Uint64, Uint128, as_numeric

__all__ = [
    "randint",
    "randint4x",
    "rand",
    "rand4x",
    "randn",
    "randn4x",
]


def _word_width(word, what):
    width = as_numeric(word).dtype.width
    if width not in (32, 64):
        raise NotImplementedError(f"{what} is only implemented for 32- or 64-bit words, got {width}")
    return width


def _offset_words(offset):
    """Split *offset* into the two low Philox counter words.

    Only an input wider than 32 bits contributes a high word; anything 32-bit
    keeps it a literal zero, which leaves the 32-bit counter untouched.
    """
    low = Uint32(offset)
    if as_numeric(offset).dtype.width > 32:
        return low, Uint32(Uint64(offset) >> Uint64(32))
    return low, Uint32(0)


def philox_impl(c0, c1, c2, c3, k0, k1, n_rounds: int = 10):
    """Run *n_rounds* Philox 4x32 or 4x64 rounds.

    The counter word width selects the variant. Each round does two widening
    multiplies and keeps both halves, matching Triton's ``umulhi`` plus wrapping
    ``mul`` operations.
    """
    if _word_width(c0, "philox") == 32:
        word_type, wide_type = Uint32, Uint64
        PHILOX_KEY_A, PHILOX_KEY_B = 0x9E3779B9, 0xBB67AE85
        PHILOX_ROUND_A, PHILOX_ROUND_B = 0xD2511F53, 0xCD9E8D57
    else:
        word_type, wide_type = Uint64, Uint128
        PHILOX_KEY_A, PHILOX_KEY_B = 0x9E3779B97F4A7C15, 0xBB67AE8584CAA73B
        PHILOX_ROUND_A, PHILOX_ROUND_B = 0xD2E7470EE14C6C93, 0xCA5A826395121157

    c0, c1, c2, c3 = word_type(c0), word_type(c1), word_type(c2), word_type(c3)
    k0, k1 = word_type(k0), word_type(k1)
    mul_a, mul_b = wide_type(PHILOX_ROUND_A), wide_type(PHILOX_ROUND_B)
    step_a, step_b = word_type(PHILOX_KEY_A), word_type(PHILOX_KEY_B)
    shift = wide_type(word_type.width)

    for _ in range(n_rounds):
        prod_b = wide_type(c2) * mul_b
        prod_a = wide_type(c0) * mul_a
        c0 = word_type(prod_b >> shift) ^ c1 ^ k0
        c2 = word_type(prod_a >> shift) ^ c3 ^ k1
        c1 = word_type(prod_b)
        c3 = word_type(prod_a)
        k0 = k0 + step_a
        k1 = k1 + step_b

    return c0, c1, c2, c3


def philox(seed, c0, c1, c2, c3, n_rounds: int = 10):
    """Key a Philox counter with *seed* and run it.

    For 32-bit counters the seed splits across both key words. For 64-bit
    counters it fills the low key word and the high key word is zero, matching
    Triton.
    """
    wide = Uint64(seed)
    if _word_width(c0, "philox") == 32:
        k0, k1 = Uint32(wide), Uint32(wide >> Uint64(32))
    else:
        k0, k1 = wide, Uint64(0)
    return philox_impl(c0, c1, c2, c3, k0, k1, n_rounds)


def randint4x(seed, offset, n_rounds: int = 10):
    """Return four Philox-generated ``Uint32`` words.

    The parameter names, order, and default round count match Triton's
    ``randint4x(seed, offset, n_rounds=10)`` API, down to how a wide or signed
    input splits across the key and counter words. The same ``(seed, offset)``
    always yields the same words, so no RNG state has to be threaded through a
    kernel.
    """
    low, high = _offset_words(offset)
    zero = Uint32(0)
    return philox(seed, low, high, zero, zero, n_rounds)


def randint(seed, offset, n_rounds: int = 10):
    """Return the first Philox-generated ``Uint32`` word for ``(seed, offset)``.

    The other three words of the draw are discarded; use :func:`randint4x` when
    four independent words per offset are wanted.
    """
    word, _, _, _ = randint4x(seed, offset, n_rounds)
    return word


def uint_to_uniform_float(word):
    """Map a random 32- or 64-bit *word* to a ``Float32`` in [0, 1).

    The word is reinterpreted as a signed integer and negatives fold onto the
    positive range, keeping the rounded result below 1.0.
    """
    if _word_width(word, "uniform conversion") == 32:
        signed_type = Int32
        UNIFORM_SCALE = 4.6566127342e-10
    else:
        signed_type = Int64
        UNIFORM_SCALE = 1.0842020432385337e-19

    signed = signed_type(word)
    folded = (signed < signed_type(0)).select(-signed - signed_type(1), signed)
    return Float32(folded) * Float32(UNIFORM_SCALE)


def rand(seed, offset, n_rounds: int = 10):
    """Return one ``Float32`` uniformly drawn from [0, 1)."""
    return uint_to_uniform_float(randint(seed, offset, n_rounds))


def rand4x(seed, offset, n_rounds: int = 10):
    """Return four ``Float32`` values uniformly drawn from [0, 1)."""
    w0, w1, w2, w3 = randint4x(seed, offset, n_rounds)
    return (
        uint_to_uniform_float(w0),
        uint_to_uniform_float(w1),
        uint_to_uniform_float(w2),
        uint_to_uniform_float(w3),
    )


def pair_uniform_to_normal(u1, u2):
    """Box-Muller transform of two uniforms into two standard normals."""
    # Box-Muller takes log(u1); clamp u1 away from zero to keep it finite.
    UNIFORM_FLOOR = 1.0e-7
    TWO_PI = 6.283185307179586
    u1 = u1.maximumf(Float32(UNIFORM_FLOOR))
    theta = Float32(TWO_PI) * u2
    r = sqrt(Float32(-2.0) * log(u1))
    return r * cos(theta), r * sin(theta)


def randn(seed, offset, n_rounds: int = 10):
    """Return one ``Float32`` drawn from the standard normal distribution."""
    w0, w1, _, _ = randint4x(seed, offset, n_rounds)
    normal, _ = pair_uniform_to_normal(uint_to_uniform_float(w0), uint_to_uniform_float(w1))
    return normal


def randn4x(seed, offset, n_rounds: int = 10):
    """Return four ``Float32`` values drawn from the standard normal distribution."""
    u0, u1, u2, u3 = rand4x(seed, offset, n_rounds)
    n0, n1 = pair_uniform_to_normal(u0, u1)
    n2, n3 = pair_uniform_to_normal(u2, u3)
    return n0, n1, n2, n3
