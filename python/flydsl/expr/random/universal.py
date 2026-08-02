# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Target-neutral implementations of the random library."""

from ..numeric import Uint32, Uint64

__all__ = [
    "randint4x",
]


def randint4x(seed, offset, n_rounds: int = 10):
    """Return four Philox-generated ``Uint32`` words.

    The parameter names, order, and default round count match Triton's
    ``randint4x(seed, offset, n_rounds=10)`` API. The same ``(seed, offset)``
    always yields the same words, so no RNG state has to be threaded through a
    kernel. Each round does two widening 32x32 to 64 bit multiplies.
    """
    _PHILOX_ROUND_A = 0xD2511F53
    _PHILOX_ROUND_B = 0xCD9E8D57
    _PHILOX_KEY_A = 0x9E3779B9
    _PHILOX_KEY_B = 0xBB67AE85

    c0 = Uint32(offset)
    c1 = Uint32(0)
    c2 = Uint32(0)
    c3 = Uint32(0)
    k0 = Uint32(seed)
    k1 = Uint32(0)

    round_a = Uint64(_PHILOX_ROUND_A)
    round_b = Uint64(_PHILOX_ROUND_B)
    key_a = Uint32(_PHILOX_KEY_A)
    key_b = Uint32(_PHILOX_KEY_B)
    shift32 = Uint64(32)

    for _ in range(n_rounds):
        prod_b = Uint64(c2) * round_b
        prod_a = Uint64(c0) * round_a
        c0 = Uint32(prod_b >> shift32) ^ c1 ^ k0
        c2 = Uint32(prod_a >> shift32) ^ c3 ^ k1
        c1 = Uint32(prod_b)
        c3 = Uint32(prod_a)
        k0 = k0 + key_a
        k1 = k1 + key_b

    return c0, c1, c2, c3
