"""Shared helpers for MFMA FP8 "preshuffle" kernels (tests).

`test_moe_gemm.py` uses split global loads that always issue dwordx4 (16B) loads,
but may only logically consume 8B/16B per thread depending on tiling.

This file centralizes those helpers so multiple tests can share identical logic.
"""

from __future__ import annotations

from typing import List, Optional


def compute_split_load_lens(bytes_per_thread: int, *, max_bytes_per_load: int = 16) -> List[int]:
    """Split `bytes_per_thread` into chunks with size <= `max_bytes_per_load`.

    The tests currently expect chunks to be multiples of 4 bytes (i32 lanes) and
    typically use 16B (dwordx4) and sometimes 8B (dwordx2 packed into dwordx4).
    """
    if bytes_per_thread <= 0:
        return []
    if max_bytes_per_load <= 0:
        raise ValueError("max_bytes_per_load must be > 0")
    if bytes_per_thread % 4 != 0:
        raise ValueError(f"bytes_per_thread must be multiple of 4, got {bytes_per_thread}")

    lens: List[int] = []
    remaining = int(bytes_per_thread)
    while remaining > 0:
        curr = min(remaining, int(max_bytes_per_load))
        # Keep i32 alignment.
        curr = (curr // 4) * 4
        if curr == 0:
            # Fallback: round up to 4 bytes to avoid infinite loop.
            curr = 4
        lens.append(curr)
        remaining -= curr
    return lens


def load_fp8_tile_split_dwordx4(
    buffer_ops,
    flir,
    _arith_mlir,
    *,
    rsrc,
    idx_div4,
    lens: List[int],
    i32_type,
    mask=None,
):
    """Issue split global loads for FP8 tiles using dwordx4 loads (16B).

    Returns a list of `vector<4xi32>` values, one per chunk in `lens`.
    The caller may only consume the first N i32 lanes depending on `curr_bytes`.
    """
    def _unwrap(v):
        # Mirror the helpers used across tests: unwrap wrapper objects that carry `.value` / `._value`.
        while hasattr(v, "value") or hasattr(v, "_value"):
            v = getattr(v, "_value", getattr(v, "value", v))
        return v

    parts = []
    curr_off_i32 = 0
    for curr_bytes in lens:
        base = _unwrap(idx_div4)
        if isinstance(base, int):
            base = _unwrap(flir.arith_ext.index(int(base)).value)

        # Always load 16B (vec_width=4 i32). Even for <16B logical chunks, we
        # rely on padding in the test harness to keep these loads in-bounds.
        curr_idx = base
        if curr_off_i32:
            off = _unwrap(flir.arith_ext.index(int(curr_off_i32)).value)
            curr_idx = _arith_mlir.AddIOp(curr_idx, off).result

        # `buffer_load` supports an optional predicate `mask` in our bindings.
        # If mask is false, it returns zero-initialized vector.
        val = buffer_ops.buffer_load(rsrc, curr_idx, vec_width=4, dtype=i32_type, mask=mask)
        parts.append(val)

        # Advance by the logical i32s in this chunk.
        curr_off_i32 += int(curr_bytes) // 4
    return parts


