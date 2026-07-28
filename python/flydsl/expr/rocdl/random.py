# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Counter-based RNG and stochastic rounding for AMD targets.

Stochastic rounding does not fit the target-neutral IEEE rounding modes
(``fx.RoundingMode`` on ``.to()``, lowered through ``arith.truncf``): it needs
entropy, and the hardware form is a target-specific instruction (e.g. the CDNA
``v_cvt_*_sr`` conversions, PTX ``cvt.rs`` on NVIDIA). So it lives under the
rocdl target package rather than as a neutral rounding mode. The f32 -> bf16
path here is a portable software emulation; a CDNA hardware ``sr`` variant can
slot in alongside it later.

``philox_4x32`` is co-located as the entropy source that feeds it.
"""

from ..numeric import BFloat16, Float32, Uint16, Uint32, Uint64

__all__ = ["philox_4x32", "stochastic_round_bf16"]

_PHILOX_ROUND_A = 0xD2511F53
_PHILOX_ROUND_B = 0xCD9E8D57
_PHILOX_KEY_A = 0x9E3779B9
_PHILOX_KEY_B = 0xBB67AE85


def philox_4x32(counter, key, rounds: int = 7):
    """Philox 4x32 counter-based PRNG. Returns four ``Uint32`` random words.

    ``counter`` and ``key`` are ``Uint32`` (e.g. a global element index and a
    seed). The same ``(counter, key)`` always yields the same words, so no RNG
    state has to be threaded through a kernel. Each round does two widening
    32x32 to 64 bit multiplies.
    """
    c0 = Uint32(counter)
    c1 = Uint32(0)
    c2 = Uint32(0)
    c3 = Uint32(0)
    k0 = Uint32(key)
    k1 = Uint32(0)
    # Materialize the round multipliers, key increments and shift once instead
    # of rebuilding identical constants on every unrolled round.
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


def stochastic_round_bf16(x, rand):
    """Round a ``Float32`` to ``BFloat16`` with stochastic rounding.

    ``rand`` is a ``Uint32`` supplying entropy; its low 16 bits are used.
    bf16 keeps the top 16 bits of the f32, so adding the random value to the
    16 discarded mantissa bits turns truncation into a round-up with
    probability equal to the fractional distance to the next bf16 value.
    bf16 shares the f32 exponent field, so Inf and NaN pass through unchanged
    and no special-case handling is needed.
    """
    u = Float32(x).bitcast(Uint32) + (Uint32(rand) & Uint32(0xFFFF))
    return Uint16(u >> Uint32(16)).bitcast(BFloat16)
