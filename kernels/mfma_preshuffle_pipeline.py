"""Shared MFMA preshuffle helpers for preshuffle GEMM kernels.

Key primitives:
- B preshuffle layout builder (supports byte-packed element types, incl. packed int4)
- B pack load for MFMA K32 micro-steps (8B output pack; optional int4->int8 unpack)
"""

from __future__ import annotations
from dataclasses import dataclass
from flydsl._mlir import ir
from flydsl.expr.typing import T
from flydsl.expr import arith as _arith
import flydsl.expr as fx

from kernels.layout_utils import crd2idx, idx2crd, get as layout_get


def swizzle_xor16(row, col, k_blocks16):
    """XOR-with-row swizzle on the K dimension at 16B granularity.

    Computes: col XOR ((row % k_blocks16) * 16)
    """
    rem = row % k_blocks16
    return col ^ (rem * 16)


def _buffer_load_vec(buffer_ops, vector, rsrc, idx, *, elem_type, vec_elems, elem_bytes, offset_in_bytes):
    """Load vec_elems elements via buffer_load dwordx[1,2,4] + bitcast."""
    elem_size = int(elem_bytes)
    load_bytes = int(vec_elems) * elem_size
    vec_width = load_bytes // 4

    if offset_in_bytes:
        idx_i32 = idx // 4
    elif elem_bytes == 2:
        idx_i32 = (idx * 2) // 4
    else:
        idx_i32 = idx

    i32_val = buffer_ops.buffer_load(rsrc, idx_i32, vec_width=vec_width, dtype=T.i32)
    if vec_width == 1:
        i32_vec = vector.from_elements(T.vec(1, T.i32), [i32_val])
    else:
        i32_vec = i32_val
    return vector.bitcast(T.vec(int(vec_elems), elem_type), i32_vec)


@dataclass(frozen=True)
class PreshuffleBLayout:
    """Container returned by `make_preshuffle_b_layout`."""

    layout_b: object
    kpack_bytes: int


def make_preshuffle_b_layout(
    arith,
    *,
    c_n: ir.Value,
    c_k: ir.Value,
    kpack_bytes: int = 16,
    elem_bytes: int = 1,
) -> PreshuffleBLayout:
    """Build B layout matching aiter/CK preshuffle for A8 MFMA kernels."""
    if kpack_bytes not in (8, 16):
        raise ValueError(f"kpack_bytes must be 8 or 16, got {kpack_bytes!r}")

    c16 = arith.constant(16, index=True)
    c64 = arith.constant(64, index=True)
    c4 = arith.constant(4, index=True)
    c_kpack = arith.constant(kpack_bytes, index=True)

    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    c_k_bytes = c_k * arith.constant(int(elem_bytes), index=True)
    c_k0 = c_k_bytes / c64
    n0 = c_n / c16

    c_kpack_elems = c_kpack if elem_bytes == 1 else (c_kpack / arith.constant(int(elem_bytes), index=True))

    stride_nlane = c_kpack_elems
    stride_klane = c16 * stride_nlane
    stride_k0 = c4 * stride_klane
    stride_n0 = c_k0 * stride_k0

    stride_b = (stride_n0, stride_k0, stride_klane, stride_nlane, arith.constant(1, index=True))
    layout_b = fx.make_layout((n0, c_k0, c4, c16, c_kpack_elems), stride_b)
    return PreshuffleBLayout(layout_b=layout_b, kpack_bytes=kpack_bytes)


def load_b_pack_k32(
    buffer_ops,
    arith,
    vector,
    *,
    arg_b,
    b_rsrc,
    layout_b,
    base_k: ir.Value,
    ki_step: int,
    n_blk: ir.Value,
    n_intra: ir.Value,
    lane_div_16: ir.Value,
    elem_type: ir.Type,
    kpack_bytes: int = 16,
    elem_bytes: int = 1,
    unpack_int4: bool = False,
) -> ir.Value:
    """Load one B pack for one MFMA(x32) micro-step.

    Returns an i64 Value containing 8 bytes consumed by MFMA.
    """
    if kpack_bytes not in (8, 16):
        raise ValueError(f"kpack_bytes must be 8 or 16, got {kpack_bytes!r}")
    if unpack_int4 and kpack_bytes != 8:
        raise ValueError("unpack_int4 requires kpack_bytes=8 (packed int4 layout)")
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")

    c64 = arith.constant(64, index=True)
    base_k_bytes = base_k * arith.constant(int(elem_bytes), index=True)
    k0_base = base_k_bytes / c64
    k0 = k0_base + arith.constant(ki_step // 2, index=True)
    k1 = lane_div_16
    half_bytes = kpack_bytes // 2
    k2_base = arith.constant((ki_step % 2) * half_bytes, index=True)

    coord_pack = fx.make_coord(n_blk, k0, k1, n_intra, arith.constant(0, index=True))
    idx_pack = crd2idx(coord_pack, layout_b)

    if unpack_int4:
        idx_bytes = idx_pack + k2_base
        b4 = _buffer_load_vec(
            buffer_ops, vector, b_rsrc, idx_bytes,
            elem_type=elem_type, vec_elems=4, elem_bytes=1, offset_in_bytes=True,
        )
        packed32 = vector.extract(
            vector.bitcast(T.vec(1, T.i32), b4),
            static_position=[0],
            dynamic_position=[],
        )

        c_08080808 = arith.constant(0x08080808, type=T.i32)
        c_0f0f0f0f = arith.constant(0x0F0F0F0F, type=T.i32)
        c_1e = arith.constant(0x1E, type=T.i32)
        c_4_i32 = arith.constant(4, type=T.i32)

        s0 = (packed32 & c_08080808) * c_1e
        even = (packed32 & c_0f0f0f0f) | s0

        t = packed32 >> c_4_i32
        s1 = (t & c_08080808) * c_1e
        odd = (t & c_0f0f0f0f) | s1

        v2 = vector.from_elements(T.vec(2, T.i32), [even, odd])
        v64 = vector.bitcast(T.vec(1, T.i64), v2)
        return vector.extract(v64, static_position=[0], dynamic_position=[])

    vec_elems = kpack_bytes // int(elem_bytes)
    b16 = _buffer_load_vec(
        buffer_ops, vector, b_rsrc, idx_pack,
        elem_type=elem_type, vec_elems=vec_elems, elem_bytes=elem_bytes,
        offset_in_bytes=(elem_bytes == 1),
    )

    b_i32x4 = vector.bitcast(T.i32x4, b16)

    half = ki_step % 2
    if half == 0:
        d0 = vector.extract(b_i32x4, static_position=[0], dynamic_position=[])
        d1 = vector.extract(b_i32x4, static_position=[1], dynamic_position=[])
    else:
        d0 = vector.extract(b_i32x4, static_position=[2], dynamic_position=[])
        d1 = vector.extract(b_i32x4, static_position=[3], dynamic_position=[])

    v2 = vector.from_elements(T.vec(2, T.i32), [d0, d1])
    v64 = vector.bitcast(T.vec(1, T.i64), v2)
    return vector.extract(v64, static_position=[0], dynamic_position=[])


def tile_chunk_coord_i32(
    arith,
    *,
    tx_i32_base: ir.Value,
    i: int,
    total_threads: int,
    layout_tile_div4,
    chunk_i32: int = 4,
):
    """Map (thread, chunk_id) -> (row_local, col_local_i32) for X/A loads."""
    if chunk_i32 not in (1, 2, 4):
        raise ValueError(f"chunk_i32 must be one of (1,2,4), got {chunk_i32!r}")
    chunk_off_i32 = arith.constant(i * total_threads * chunk_i32, index=True)
    tile_idx_i32 = tx_i32_base + chunk_off_i32
    coord_local = idx2crd(tile_idx_i32, layout_tile_div4)
    row_local = layout_get(coord_local, 0)
    col_local_i32 = layout_get(coord_local, 1)
    return row_local, col_local_i32


def buffer_copy_gmem16_dwordx4(
    buffer_ops,
    vector,
    *,
    elem_type,
    idx_i32: ir.Value,
    rsrc,
    vec_elems: int = 16,
    elem_bytes: int = 1,
):
    """Copy 16 bytes from global memory into regs via buffer-load dwordx4 lowering."""
    if int(vec_elems) <= 0:
        raise ValueError(f"vec_elems must be > 0, got {vec_elems!r}")
    return _buffer_load_vec(
        buffer_ops, vector, rsrc, idx_i32,
        elem_type=elem_type, vec_elems=vec_elems, elem_bytes=elem_bytes,
        offset_in_bytes=False,
    )


def lds_store_16b_xor16(
    arith,
    vector,
    *,
    lds_memref,
    vec16_ty,
    layout_lds,
    row_local: ir.Value,
    col_local_i32: ir.Value,
    tx_c4: ir.Value,
    k_blocks16: ir.Value,
    lds_base: ir.Value,
    vec_part_i32x4: ir.Value,
    elem_bytes: int = 1,
):
    """Store one 16B chunk into LDS with CK-style XOR16 swizzle on the K dimension."""
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    col_local_bytes = col_local_i32 * tx_c4
    col_swz_bytes = swizzle_xor16(row_local, col_local_bytes, k_blocks16)
    col_swz = col_swz_bytes if elem_bytes == 1 else col_swz_bytes / 2
    coord_store = fx.make_coord(row_local, col_swz)
    idx0 = crd2idx(coord_store, layout_lds) + lds_base
    v16 = vector.bitcast(vec16_ty, vec_part_i32x4)
    vector.store(v16, lds_memref, [idx0])


def lds_store_8b_xor16(
    arith,
    vector,
    *,
    lds_memref,
    vec8_ty,
    layout_lds,
    row_local: ir.Value,
    col_local_i32: ir.Value,
    tx_c4: ir.Value,
    k_blocks16: ir.Value,
    lds_base: ir.Value,
    vec_part_i32x2: ir.Value,
    elem_bytes: int = 1,
):
    """Store one 8B chunk into LDS with CK-style XOR16 swizzle on the K dimension."""
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    col_local_bytes = col_local_i32 * tx_c4
    col_swz_bytes = swizzle_xor16(row_local, col_local_bytes, k_blocks16)
    col_swz = col_swz_bytes if elem_bytes == 1 else col_swz_bytes / 2
    coord_store = fx.make_coord(row_local, col_swz)
    idx0 = crd2idx(coord_store, layout_lds) + lds_base
    v8 = vector.bitcast(vec8_ty, vec_part_i32x2)
    vector.store(v8, lds_memref, [idx0])


def lds_store_4b_xor16(
    arith,
    vector,
    *,
    lds_memref,
    vec4_ty,
    layout_lds,
    row_local: ir.Value,
    col_local_i32: ir.Value,
    tx_c4: ir.Value,
    k_blocks16: ir.Value,
    lds_base: ir.Value,
    vec_part_i32x1: ir.Value,
    elem_bytes: int = 1,
):
    """Store one 4B chunk into LDS with CK-style XOR16 swizzle on the K dimension."""
    if elem_bytes not in (1, 2):
        raise ValueError(f"elem_bytes must be 1 or 2, got {elem_bytes!r}")
    col_local_bytes = col_local_i32 * tx_c4
    col_swz_bytes = swizzle_xor16(row_local, col_local_bytes, k_blocks16)
    col_swz = col_swz_bytes if elem_bytes == 1 else col_swz_bytes / 2
    coord_store = fx.make_coord(row_local, col_swz)
    idx0 = crd2idx(coord_store, layout_lds) + lds_base
    v4 = vector.bitcast(vec4_ty, vec_part_i32x1)
    vector.store(v4, lds_memref, [idx0])


def lds_load_pack_k32(
    arith,
    vector,
    *,
    lds_memref,
    layout_lds,
    k_blocks16: ir.Value,
    curr_row_a_lds: ir.Value,
    col_base: ir.Value,
    half: int,
    lds_base: ir.Value,
    ck_lds128: bool,
    vec16_ty,
    vec8_ty,
    vec2_i64_ty,
    vec1_i64_ty,
):
    """Load one i64 A-pack for an MFMA K32 micro-step from LDS."""
    col_base_swz = swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
    if ck_lds128:
        coord_a16 = fx.make_coord(curr_row_a_lds, col_base_swz)
        idx_a16 = crd2idx(coord_a16, layout_lds) + lds_base
        loaded_a16 = vector.load_op(vec16_ty, lds_memref, [idx_a16])
        a_vec128 = vector.bitcast(vec2_i64_ty, loaded_a16)
        return vector.extract(a_vec128, static_position=[half], dynamic_position=[])
    else:
        col_swizzled = col_base_swz + (half * 8)
        coord_a = fx.make_coord(curr_row_a_lds, col_swizzled)
        idx_a = crd2idx(coord_a, layout_lds) + lds_base
        loaded_a8 = vector.load_op(vec8_ty, lds_memref, [idx_a])
        a_vec64 = vector.bitcast(vec1_i64_ty, loaded_a8)
        return vector.extract(a_vec64, static_position=[0], dynamic_position=[])


__all__ = [
    "PreshuffleBLayout",
    "buffer_copy_gmem16_dwordx4",
    "lds_load_pack_k32",
    "lds_store_4b_xor16",
    "lds_store_8b_xor16",
    "lds_store_16b_xor16",
    "make_preshuffle_b_layout",
    "load_b_pack_k32",
    "swizzle_xor16",
    "tile_chunk_coord_i32",
]
