# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""FlyDSL tiled-WMMA layouts shared by the MXFP4 dataflow strategies.

The MMA atom owns the RDNA4 thread/value ABI. The permutation groups each
wave's repeated atoms contiguously, preserving the tuned workgroup geometry.
"""

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


def swizzled_layout(layout, swizzle):
    """Compose FlyDSL's built-in XOR swizzle with a logical tensor layout."""
    if swizzle is None:
        return layout
    return fx.make_composed_layout(fx.static(fx.SwizzleType.get(swizzle[0], swizzle[1], swizzle[2])), layout)


def global_tensor(tensor, dtype, layout, byte_count, buffer_io=True):
    """A typed, bounded global tensor; its layout owns all address arithmetic."""
    storage = fx.recast_iter(dtype, fx.get_iter(tensor))
    tensor = fx.make_view(storage, layout)
    if const_expr(buffer_io):
        tensor = fx.rocdl.make_buffer_tensor(tensor, num_records_bytes=byte_count)
    return tensor


def tensor_load(tensor, coord, dtype, width=1, buffer_io=True):
    """Copy a contiguous fragment at a logical tensor coordinate to registers."""
    if len(coord) == 1:
        coord = coord[0]
    offset = fx.Int32(fx.crd2idx(coord, tensor.layout).unpack())
    view = fx.make_view(fx.get_iter(tensor) + offset, fx.make_layout(width, 1))
    copy_op = fx.rocdl.BufferCopy if const_expr(buffer_io) else fx.UniversalCopy
    atom = fx.make_copy_atom(copy_op(width * dtype.width), dtype)
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
