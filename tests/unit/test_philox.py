# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host half of the `kernels/attention/philox.py` tests (PHX-01, PHX-02; no GPU).

The device half is `tests/kernels/attention/test_philox_device.py`. Three
references, because they fail differently: Random123's published known-answer
vectors pin the *algorithm*; the Python oracle extends that to this module's
packing of a 64-bit seed and offset; Triton, on the device side, is the contract.
"""

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]

from kernels.attention.philox import (  # noqa: E402
    DEFAULT_ROUNDS,
    PHILOX_WIDTHS,
    Philox,
    dropout_threshold,
    randoms_per_offset,
)
from tests.kernels.attention.philox_oracle import KAT_4X32_10, philox_words, ref_u32  # noqa: E402


def test_oracle_known_answers():
    """PHX-01: the oracle reproduces Random123's published Philox-4x32-10 vectors."""
    for ctr, key, want in KAT_4X32_10:
        assert philox_words(ctr, key, 32, DEFAULT_ROUNDS) == want


@pytest.mark.parametrize("width", PHILOX_WIDTHS, ids=["u32", "u64"])
def test_oracle_packing_matches_the_documented_layout(width):
    """A 64-bit seed fills the key words and a 64-bit offset the low counter words."""
    seed, off = (1 << 40) + 12345, (1 << 35) + 999
    got = ref_u32(seed, off, width)
    assert len(got) == randoms_per_offset(width)
    assert got != ref_u32(seed ^ (1 << 33), off, width), "seed above 2**32 must matter"
    assert got != ref_u32(seed, off ^ (1 << 33), width), "offset above 2**32 must matter"
    if width == 32:
        assert got == philox_words([off & 0xFFFFFFFF, off >> 32, 0, 0], [seed & 0xFFFFFFFF, seed >> 32], 32)


def test_dropout_threshold_keeps_the_right_fraction():
    """PHX-02: `p` in, an i32 threshold out, keeping `1 - p` of a uniform u32 stream."""
    for p in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        t = dropout_threshold(p)
        kept = (2**31 - 1 - t) / 2**32
        assert abs(kept - (1.0 - p)) < 1e-3, f"p={p} keeps {kept:.4f}"
    assert dropout_threshold(0.0) < -(2**31) + 2, "p=0 must keep everything"
    assert dropout_threshold(1.0) > 2**31 - 2, "p=1 must keep nothing"
    with pytest.raises(ValueError):
        dropout_threshold(1.5)


@pytest.mark.parametrize("width", PHILOX_WIDTHS, ids=["u32", "u64"])
def test_configured_object_validates(width):
    """Bad configurations fail at construction, not at trace time."""
    with pytest.raises(ValueError, match="PHILOX_WIDTH"):
        Philox(width=48)
    with pytest.raises(ValueError, match="n_rounds"):
        Philox(width=width, n_rounds=0)
    assert Philox(width=width).randoms_per_offset == randoms_per_offset(width)


@pytest.mark.parametrize("width", PHILOX_WIDTHS, ids=["u32", "u64"])
def test_span_rejects_a_partial_call(width):
    """A count that is not a whole number of calls is a caller error."""
    rng = Philox(width=width)
    with pytest.raises(ValueError, match="multiple of randoms_per_offset"):
        rng.span_u32(0, 0, rng.randoms_per_offset + 1)
