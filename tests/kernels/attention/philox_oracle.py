# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Pure-Python Philox oracle: the Random123 algorithm, independent of `kernels/attention/philox.py`.

Shared by `tests/unit/test_philox.py` (host) and `test_philox_device.py` (device).
"""

DEFAULT_ROUNDS = 10

_C = {
    32: dict(KEY_A=0x9E3779B9, KEY_B=0xBB67AE85, MUL_A=0xD2511F53, MUL_B=0xCD9E8D57, MASK=(1 << 32) - 1, BITS=32),
    64: dict(
        KEY_A=0x9E3779B97F4A7C15,
        KEY_B=0xBB67AE8584CAA73B,
        MUL_A=0xD2E7470EE14C6C93,
        MUL_B=0xCA5A826395121157,
        MASK=(1 << 64) - 1,
        BITS=64,
    ),
}

# Random123 `kat_vectors`, philox4x32 at 10 rounds: (counter, key, expected).
KAT_4X32_10 = [
    ([0, 0, 0, 0], [0, 0], [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]),
    ([0xFFFFFFFF] * 4, [0xFFFFFFFF] * 2, [0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD]),
    (
        [0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344],
        [0xA4093822, 0x299F31D0],
        [0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1],
    ),
]


def philox_words(ctr, key, width, n_rounds=DEFAULT_ROUNDS):
    """The four output words of Philox-4xW from a counter and a key (Python ints)."""
    c = _C[width]
    mask, bits = c["MASK"], c["BITS"]
    c0, c1, c2, c3 = ctr
    k0, k1 = key
    for _ in range(n_rounds):
        p0, p2 = c0, c2
        c0 = (((c["MUL_B"] * p2) >> bits) ^ c1 ^ k0) & mask
        c2 = (((c["MUL_A"] * p0) >> bits) ^ c3 ^ k1) & mask
        c1 = (c["MUL_B"] * p2) & mask
        c3 = (c["MUL_A"] * p0) & mask
        k0 = (k0 + c["KEY_A"]) & mask
        k1 = (k1 + c["KEY_B"]) & mask
    return [c0, c1, c2, c3]


def ref_u32(seed, offset, width, n_rounds=DEFAULT_ROUNDS):
    """The u32 values `philox_u32(seed, offset, width)` must give: 64-bit seed and offset packed
    into the key and the low counter words, wide lanes split low half first."""
    mask = _C[width]["MASK"]
    if width == 32:
        ctr, key = [offset & mask, (offset >> 32) & mask, 0, 0], [seed & mask, (seed >> 32) & mask]
    else:
        ctr, key = [offset & mask, 0, 0, 0], [seed & mask, 0]
    words = philox_words(ctr, key, width, n_rounds)
    if width == 32:
        return words
    out = []
    for w in words:
        out.extend((w & 0xFFFFFFFF, (w >> 32) & 0xFFFFFFFF))
    return out
