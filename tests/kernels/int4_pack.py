# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Pack signed INT4 values into the int32 byte stream the gfx120 IU4 WMMA reads.

Eight nibbles per int32, little-endian: nibble ``i`` lives at bits ``4*i``.
Logical values are in ``[-8, 7]``.
"""

from __future__ import annotations

import numpy as np


def pack_int4(logical: np.ndarray) -> np.ndarray:
    """logical ``[rows, K]`` int in ``[-8, 7]`` → stored ``[rows, K/8]`` int32."""
    rows, k = logical.shape
    if k % 8 != 0:
        raise ValueError(f"K must be a multiple of 8, got {k}")
    nibbles = logical.astype(np.int64) & 0xF
    nibbles = nibbles.reshape(rows, k // 8, 8)
    shifts = np.arange(8, dtype=np.int64) * 4
    words = (nibbles << shifts).sum(axis=-1)
    return words.astype(np.uint32).view(np.int32)


def unpack_int4(stored: np.ndarray) -> np.ndarray:
    """stored ``[rows, K/8]`` int32 → logical ``[rows, K]`` int8 in ``[-8, 7]``."""
    rows, words = stored.shape
    k = words * 8
    u = stored.view(np.uint32).astype(np.int64)
    out = np.empty((rows, k), dtype=np.int8)
    for i in range(8):
        nibble = (u >> (4 * i)) & 0xF
        nibble = np.where(nibble >= 8, nibble - 16, nibble)
        out[:, i::8] = nibble.astype(np.int8)
    return out
