# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Packed ``recv_meta`` row layout for intranode metadata-only MoE dispatch (A8W4 path).

Each row is **32 bytes** (8 × i32) for device-side stores / loads.

Device loads in :mod:`kernels.moe_peer_gather_mxfp8_a8w4` use **i32 word offsets** within each row:
``hdr @ +0``, ``src_token @ +1``, ``dst_slot @ +5`` (see dispatch stores in
``moe_metadata_dispatch_recv_meta``).  Words ``+2..+4`` hold routing/aux fields.

:func:`pack_recv_meta_row_i32` emits the same word order:
``(hdr, src_token, expert_id, router_w_bits, dest_encode, dst_slot, 0, 0)``.
"""

from __future__ import annotations

import struct
from typing import Tuple

RECV_META_ROW_BYTES = 32
RECV_META_NUM_I32 = RECV_META_ROW_BYTES // 4


def pack_recv_meta_row_i32(
    *,
    src_pe: int,
    kp: int,
    src_token: int,
    expert_id: int,
    router_w: float,
    dest_encode: int,
    dst_slot: int,
) -> Tuple[int, ...]:
    w_bits = struct.unpack("<I", struct.pack("<f", float(router_w)))[0]
    hdr = (kp & 0xFF) << 8 | (src_pe & 0xFF)
    return (
        int(hdr) & 0xFFFFFFFF,
        int(src_token) & 0xFFFFFFFF,
        int(expert_id) & 0xFFFFFFFF,
        int(w_bits) & 0xFFFFFFFF,
        int(dest_encode) & 0xFFFFFFFF,
        int(dst_slot) & 0xFFFFFFFF,
        0,
        0,
    )


def unpack_recv_meta_row_i32(row: Tuple[int, ...]) -> dict:
    if len(row) != RECV_META_NUM_I32:
        raise ValueError(f"expected {RECV_META_NUM_I32} i32 fields, got {len(row)}")
    hdr = row[0]
    src_pe = hdr & 0xFF
    kp = (hdr >> 8) & 0xFF
    router_w = struct.unpack("<f", struct.pack("<I", row[3] & 0xFFFFFFFF))[0]
    return {
        "src_pe": src_pe,
        "kp": kp,
        "src_token": row[1],
        "expert_id": row[2],
        "router_w": router_w,
        "dest_encode": row[4],
        "dst_slot": row[5],
    }


def sort_recv_meta_rows_by_src_pe(
    recv_flat: bytearray,
    *,
    total_recv: int,
) -> None:
    """In-place reorder the first ``total_recv`` recv_meta rows by ``(src_pe, src_token)``.

    Intended for **host-side** staging before H2D: improves locality when the fused gather
    kernel grid-strides over ``recv_slot`` order (fewer ``src_pe`` hops per CTA time-slice).

    ``recv_flat`` must hold at least ``total_recv * RECV_META_ROW_BYTES`` bytes.
    """
    if total_recv < 0:
        raise ValueError("total_recv must be non-negative")
    need = int(total_recv) * RECV_META_ROW_BYTES
    if len(recv_flat) < need:
        raise ValueError(
            f"recv_flat length {len(recv_flat)} < required {need} for total_recv={total_recv}"
        )
    if total_recv <= 1:
        return

    def _row_key(slot: int) -> tuple[int, int]:
        off = slot * RECV_META_ROW_BYTES
        i32s = struct.unpack("<8I", recv_flat[off : off + RECV_META_ROW_BYTES])
        d = unpack_recv_meta_row_i32(i32s)
        return (int(d["src_pe"]), int(d["src_token"]))

    order = list(range(int(total_recv)))
    order.sort(key=_row_key)
    if order == list(range(int(total_recv))):
        return
    tmp = bytearray(need)
    for new_i, old_i in enumerate(order):
        so = int(old_i) * RECV_META_ROW_BYTES
        dn = int(new_i) * RECV_META_ROW_BYTES
        tmp[dn : dn + RECV_META_ROW_BYTES] = recv_flat[so : so + RECV_META_ROW_BYTES]
    recv_flat[:need] = tmp


__all__ = [
    "RECV_META_ROW_BYTES",
    "RECV_META_NUM_I32",
    "pack_recv_meta_row_i32",
    "unpack_recv_meta_row_i32",
    "sort_recv_meta_rows_by_src_pe",
]
