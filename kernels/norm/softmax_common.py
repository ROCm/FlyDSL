# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Typed row copies shared by softmax forward and backward."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from kernels.common.kernels_common import dtype_to_elem_type


@flyc.jit
def _load_vec(row, index, width: fx.Constexpr[int], dtype_str: fx.Constexpr[str]):
    """Load a complete copy unit."""
    dtype = dtype_to_elem_type(dtype_str)
    bits = width * (32 if dtype_str == "f32" else 16)
    atom_type = (
        fx.rocdl.BufferCopy128b if bits == 128 else (fx.rocdl.BufferCopy32b if bits == 32 else fx.rocdl.BufferCopy16b)
    )
    atom = fx.make_copy_atom(atom_type(), 32 if dtype_str == "f32" else 16)
    divided = fx.logical_divide(row, fx.make_layout(width, 1))
    registers = fx.make_rmem_tensor(width, dtype)
    fx.copy(atom, fx.slice(divided, (None, index)), registers)
    return fx.memref_load_vec(registers)


@flyc.jit
def _store_vec(row, index, value, width: fx.Constexpr[int], dtype_str: fx.Constexpr[str]):
    """Store a complete copy unit."""
    dtype = dtype_to_elem_type(dtype_str)
    bits = width * (32 if dtype_str == "f32" else 16)
    atom_type = (
        fx.rocdl.BufferCopy128b if bits == 128 else (fx.rocdl.BufferCopy32b if bits == 32 else fx.rocdl.BufferCopy16b)
    )
    atom = fx.make_copy_atom(atom_type(), 32 if dtype_str == "f32" else 16)
    divided = fx.logical_divide(row, fx.make_layout(width, 1))
    registers = fx.make_rmem_tensor(width, dtype)
    fx.memref_store_vec(value.to(dtype), registers)
    fx.copy(atom, registers, fx.slice(divided, (None, index)))
