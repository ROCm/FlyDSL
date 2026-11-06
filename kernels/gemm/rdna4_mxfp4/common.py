# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared FlyDSL layouts, packed-weight conversion and magnitude helpers."""

import flydsl.expr as fx
from flydsl.expr import const_expr
from flydsl.expr.typing import Vector as Vec


def tiled_wmma(waves_m, waves_n, repeat_m, repeat_n, weight_dtype):
    atom = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 16, fx.Float8E4M3FN, fx.Float32, elem_ty_b=weight_dtype))
    return fx.make_tiled_mma(
        atom,
        fx.make_layout((waves_m, waves_n, 1), (waves_n, 1, 0)),
        permutation=(
            fx.make_layout((16, waves_m, repeat_m), (1, 16 * repeat_m, 16)),
            fx.make_layout((16, waves_n, repeat_n), (1, 16 * repeat_n, 16)),
            16,
        ),
    )


def identity_tensor(shape):
    return fx.make_view(tuple(0 for _ in shape), fx.make_identity_layout(shape))


def global_tensor(tensor, dtype, layout, byte_count):
    """A typed, bounded global tensor; its layout owns all address arithmetic."""
    storage = fx.recast_iter(dtype, fx.get_iter(tensor))
    tensor = fx.make_view(storage, layout)
    return fx.rocdl.make_buffer_tensor(tensor, num_records_bytes=byte_count)


def tensor_load(tensor, coord, dtype, width=1):
    """Copy a contiguous fragment at a logical tensor coordinate to registers."""
    if len(coord) == 1:
        coord = coord[0]
    offset = fx.Int32(fx.crd2idx(coord, tensor.layout).unpack())
    view = fx.make_view(fx.get_iter(tensor) + offset, fx.make_layout(width, 1))
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy(width * dtype.width), dtype)
    result = fx.make_rmem_tensor(width, dtype)
    fx.copy(atom, view, result)
    value = Vec(result.load())
    return dtype(value[0]) if const_expr(width == 1) else value


def expand_nibbles(packed, t0, t1, separated_nibbles):
    """The custom MXFP4 operation: eight exact FP8 bytes from one packed word."""
    packed = fx.Uint32(packed)
    even = packed & 0x0F0F0F0F
    odd = packed.shrui(fx.Uint32(4)) & 0x0F0F0F0F
    ev = fx.Uint32(fx.rocdl.perm_b32(t1, t0, even & 0x07070707)) | ((even & 0x08080808) << 4)
    od = fx.Uint32(fx.rocdl.perm_b32(t1, t0, odd & 0x07070707)) | ((odd & 0x08080808) << 4)
    if const_expr(separated_nibbles):
        return Vec.from_elements([ev, od], fx.Uint32)
    return Vec.from_elements(
        [
            fx.Uint32(fx.rocdl.perm_b32(od, ev, fx.Uint32(0x05010400))),
            fx.Uint32(fx.rocdl.perm_b32(od, ev, fx.Uint32(0x07030602))),
        ],
        fx.Uint32,
    )


def magnitude_words(shift, mode, normal_only):
    # Give code zero a temporary .5 encoding to prevent cross-byte borrow,
    # then clear byte zero after the independent exponent adjustments.
    shift = fx.Int32(shift)
    if mode == "e4m3":
        offset = fx.Uint32(shift) * fx.Uint32(0x08080808)
        lo = (fx.Uint32(0x3C383030) + offset) & fx.Uint32(0xFFFFFF00)
        hi = fx.Uint32(0x4C484440) + offset
        if not normal_only:
            lo = (shift == -6).select(fx.Uint32(0x0C080400), lo)
            hi = (shift == -6).select(fx.Uint32(0x1C181410), hi)
            lo = (shift == -7).select(fx.Uint32(0x06040200), lo)
            hi = (shift == -7).select(fx.Uint32(0x14100C08), hi)
            lo = (shift == -8).select(fx.Uint32(0x03020100), lo)
            hi = (shift == -8).select(fx.Uint32(0x0C080604), hi)
    else:
        offset = fx.Uint32(shift) * fx.Uint32(0x04040404)
        lo = (fx.Uint32(0x3E3C3838) + offset) & fx.Uint32(0xFFFFFF00)
        hi = fx.Uint32(0x46444240) + offset
        if not normal_only:
            lo = (shift == -14).select(fx.Uint32(0x06040200), lo)
            hi = (shift == -14).select(fx.Uint32(0x0E0C0A08), hi)
            lo = (shift == -15).select(fx.Uint32(0x03020100), lo)
            hi = (shift == -15).select(fx.Uint32(0x0A080604), hi)
    return lo, hi
