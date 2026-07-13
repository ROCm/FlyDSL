# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Shared building blocks for the RMSNorm forward/backward kernels.

Holds the constants and small device-side helpers that both
``rmsnorm_kernel.py`` (forward + quant + autograd glue) and
``rmsnorm_bwd_kernel.py`` (backward) rely on, so the two kernel modules do not
have to import from each other. Keeping these here follows the topical
``*_common.py`` convention (see ``kernels/moe/moe_common.py``).
"""

import flydsl.expr as fx
from flydsl.expr import const_expr
from flydsl.expr.vector import full
from kernels.common.kernels_common import get_warp_size

KERNEL_NAME = "rmsnorm"

EPS = 1e-5

BLOCK_THREADS = 256
WARP_SIZE = get_warp_size()
VEC_WIDTH = 8

try:
    import torch
except ImportError:
    torch = None


def make_reduction_storage(red_slots: int):
    @fx.struct
    class SharedStorage:
        s_red: fx.Array[fx.Float32, red_slots, 16]
        s_red2: fx.Array[fx.Float32, red_slots, 16]

    return SharedStorage


def make_single_reduction_storage(red_slots: int):
    """One-accumulator variant of :func:`make_reduction_storage`.

    The backward kernels run a single block reduction (only ``s_red``), so they
    use this instead of the two-slot struct to avoid allocating the unused
    ``s_red2`` LDS array.
    """

    @fx.struct
    class SharedStorage:
        s_red: fx.Array[fx.Float32, red_slots, 16]

    return SharedStorage


def load_scalar(copy_atom, elem_dtype, divided_tensor, index):
    view = fx.slice(divided_tensor, (None, index))
    r = fx.make_rmem_tensor(1, elem_dtype)
    fx.copy_atom_call(copy_atom, view, r)
    return fx.memref_load_vec(r)[0]


def store_scalar(copy_atom, elem_dtype, store_dtype, divided_tensor, index, val):
    r = fx.make_rmem_tensor(1, elem_dtype)
    ts = full(1, store_dtype(val), store_dtype)
    fx.memref_store_vec(ts, r)
    view = fx.slice(divided_tensor, (None, index))
    fx.copy_atom_call(copy_atom, r, view)


def load_vec(copy_atom, vec_width, elem_dtype, div_tensor, idx):
    r = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.copy_atom_call(copy_atom, fx.slice(div_tensor, (None, idx)), r)
    return fx.memref_load_vec(r)


def store_vec(copy_atom, vec_width, elem_dtype, val, div_tensor, idx):
    r = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.memref_store_vec(val, r)
    fx.copy_atom_call(copy_atom, r, fx.slice(div_tensor, (None, idx)))


def to_elem_scalar(dtype_str: str, elem_dtype, y):
    if const_expr(dtype_str == "f32"):
        return y
    return y.to(elem_dtype)


def to_elem_vec(dtype_str: str, elem_dtype, use_hw_cvt_bf16: bool, y):
    if const_expr(dtype_str == "bf16"):
        if const_expr(use_hw_cvt_bf16):
            return y.to(elem_dtype)
        u = y.bitcast(fx.Uint32)
        upper = u >> 16
        lsb = upper & 1
        bias = lsb + 0x7FFF
        u_round = y.bitcast(fx.Uint32) + bias
        bf16_bits = u_round >> 16
        even = bf16_bits.shuffle(bf16_bits, [0, 2, 4, 6])
        odd = bf16_bits.shuffle(bf16_bits, [1, 3, 5, 7])
        odd_sh = odd << 16
        packed = even | odd_sh
        return packed.bitcast(elem_dtype)
    if const_expr(dtype_str == "f32"):
        return y
    return y.to(elem_dtype)


if torch is not None:

    def torch_dtype_to_str(dt) -> str:
        if dt == torch.float32:
            return "f32"
        if dt == torch.float16:
            return "f16"
        if dt == torch.bfloat16:
            return "bf16"
        raise ValueError(f"unsupported torch dtype: {dt}")
