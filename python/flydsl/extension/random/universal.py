# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Target-neutral implementations of the random library."""

from ...expr.math import cos, log, sin, sqrt
from ...expr.numeric import Float32, Int32, Uint32, Uint64, as_numeric

__all__ = [
    "randint",
    "randint4x",
    "rand",
    "rand4x",
    "randn",
    "randn4x",
]


def _require_32bit(word, what):
    width = as_numeric(word).dtype.width
    if width != 32:
        raise NotImplementedError(f"{what} is only implemented for 32-bit words, got {width}")


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
    """Run *n_rounds* Philox 4x32 rounds over counter ``(c0..c3)`` and key ``(k0, k1)``.

    Each round does two widening 32x32 to 64 bit multiplies and keeps the high
    half, which is what Triton's ``umulhi`` computes.
    """
    _require_32bit(c0, "philox")  # TODO(64-bit): support philox 4x64 variant

    # Philox 4x32 round and key constants, as used by Triton.
    PHILOX_KEY_A = 0x9E3779B9
    PHILOX_KEY_B = 0xBB67AE85
    PHILOX_ROUND_A = 0xD2511F53
    PHILOX_ROUND_B = 0xCD9E8D57

    c0, c1, c2, c3 = Uint32(c0), Uint32(c1), Uint32(c2), Uint32(c3)
    k0, k1 = Uint32(k0), Uint32(k1)
    mul_a, mul_b = Uint64(PHILOX_ROUND_A), Uint64(PHILOX_ROUND_B)
    step_a, step_b = Uint32(PHILOX_KEY_A), Uint32(PHILOX_KEY_B)
    shift = Uint64(32)

    for _ in range(n_rounds):
        prod_b = Uint64(c2) * mul_b
        prod_a = Uint64(c0) * mul_a
        c0 = Uint32(prod_b >> shift) ^ c1 ^ k0
        c2 = Uint32(prod_a >> shift) ^ c3 ^ k1
        c1 = Uint32(prod_b)
        c3 = Uint32(prod_a)
        k0 = k0 + step_a
        k1 = k1 + step_b

    return c0, c1, c2, c3


def philox(seed, c0, c1, c2, c3, n_rounds: int = 10):
    """Key a Philox counter with *seed* and run it.

    The seed splits across both key words, so a signed seed sign-extends into
    the high one, matching Triton.
    """
    wide = Uint64(seed)
    k0, k1 = Uint32(wide), Uint32(wide >> Uint64(32))
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
    """Map a random 32-bit *word* to a ``Float32`` uniformly drawn from [0, 1).

    The word is reinterpreted as a signed integer and negatives fold onto the
    positive range, so every bit pattern maps to a distinct value below 1.0.
    """
    UNIFORM_SCALE = 4.6566127342e-10
    _require_32bit(word, "uniform conversion")

    signed = Int32(word)
    folded = (signed < Int32(0)).select(-signed - Int32(1), signed)
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
