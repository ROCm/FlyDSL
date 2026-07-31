# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Target-neutral implementations of the random library."""

from ..numeric import BFloat16, Float32, Uint16, Uint32, Uint64

__all__ = [
    "cvt_f32_to_bf16_sr",
    "philox_4x32",
]


def cvt_f32_to_bf16_sr(x, rand):
    """Round a ``Float32`` to ``BFloat16`` with stochastic rounding.

    ``rand`` is a ``Uint32`` supplying entropy. Pass a value that is uniformly
    random over all 32 bits: this implementation reads only the low 16, but a
    target override may consume different ones, so identical rounding
    decisions are only reproducible for a fixed compilation target.

    bf16 keeps the top 16 bits of the f32, so adding the random value to the
    16 discarded mantissa bits turns truncation into a round-up with
    probability equal to the fractional distance to the next bf16 value. The
    result is one of the two bf16 neighbours of ``x`` and unbiased in
    expectation. The add works on the raw sign-magnitude bits, so a round-up
    moves away from zero for both signs; zero and any exactly representable
    value are returned unchanged.

    Everything below follows from that single add-then-truncate, and differs
    from an IEEE conversion:

    * ``|x|`` above the largest bf16 finite (``0x1.FEp+127``) can carry into
      the exponent and become Inf, with probability equal to the same
      fractional distance.
    * Inf is preserved: a zero mantissa leaves nothing for the add to carry.
    * A NaN keeps only its top 7 payload bits, so every quiet NaN stays NaN.
      A NaN carrying payload solely in the discarded low 16 bits (such as the
      signaling NaN ``0x7F800001``) instead becomes Inf, unless the add
      happens to carry.
    """
    u = Float32(x).bitcast(Uint32) + (Uint32(rand) & Uint32(0xFFFF))
    return Uint16(u >> Uint32(16)).bitcast(BFloat16)


def philox_4x32(counter, key, rounds: int = 10):
    """Philox 4x32 counter-based PRNG. Returns four ``Uint32`` random words.

    ``counter`` and ``key`` are ``Uint32`` (e.g. a global element index and a
    seed). The same ``(counter, key)`` always yields the same words, so no RNG
    state has to be threaded through a kernel. Each round does two widening
    32x32 to 64 bit multiplies.
    """
    _PHILOX_ROUND_A = 0xD2511F53
    _PHILOX_ROUND_B = 0xCD9E8D57
    _PHILOX_KEY_A = 0x9E3779B9
    _PHILOX_KEY_B = 0xBB67AE85

    c0 = Uint32(counter)
    c1 = Uint32(0)
    c2 = Uint32(0)
    c3 = Uint32(0)
    k0 = Uint32(key)
    k1 = Uint32(0)

    round_a = Uint64(_PHILOX_ROUND_A)
    round_b = Uint64(_PHILOX_ROUND_B)
    key_a = Uint32(_PHILOX_KEY_A)
    key_b = Uint32(_PHILOX_KEY_B)
    shift32 = Uint64(32)

    for _ in range(rounds):
        prod_b = Uint64(c2) * round_b
        prod_a = Uint64(c0) * round_a
        c0 = Uint32(prod_b >> shift32) ^ c1 ^ k0
        c2 = Uint32(prod_a >> shift32) ^ c3 ^ k1
        c1 = Uint32(prod_b)
        c3 = Uint32(prod_a)
        k0 = k0 + key_a
        k1 = k1 + key_b

    return c0, c1, c2, c3
