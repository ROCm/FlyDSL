# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, memref
from flydsl._mlir.dialects.arith import AtomicRMWKind
from flydsl.expr import arith, const_expr, rocdl
from flydsl.expr.typing import T

# Pointer/global-load helpers now live in mem_ops; re-exported here for back-compat.
from kernels.common.mem_ops import extract_global_ptr as extract_global_ptr
from kernels.common.mem_ops import global_load as global_load
from kernels.common.mem_ops import global_load_i32 as global_load_i32
from kernels.common.mem_ops import global_load_i64x2 as global_load_i64x2
from kernels.common.mem_ops import global_ptr_from_addr as global_ptr_from_addr


def lds_load(lds_memref, index):
    """Scalar load from an LDS memref view at a single index."""
    return memref.load(lds_memref, [index])


def lds_store(value, lds_memref, index):
    """Scalar store to an LDS memref view at a single index (memref.store arg order)."""
    memref.store(arith.unwrap(value), lds_memref, [index])


def lds_atomic_add(value, lds_memref, index):
    """LDS ds_add (atomic add) at a single index; returns the pre-add value."""
    return memref.atomic_rmw(AtomicRMWKind.addi, arith.unwrap(value), lds_memref, [index])


def rcp_f32(value):
    return rocdl.rcp(T.f32, value)


def rcp_amdgcn_scalar(value):
    raw = arith.unwrap(value) if hasattr(value, "ir_value") or hasattr(value, "type") else value
    return llvm.call_intrinsic(ir.F32Type.get(), "llvm.amdgcn.rcp.f32", [raw], [], [])


def fabs_f32(value):
    raw = arith.unwrap(value) if hasattr(value, "ir_value") or hasattr(value, "type") else value
    return llvm.call_intrinsic(ir.F32Type.get(), "llvm.fabs.f32", [raw], [], [])


def store_nt(value, ptr, *, alignment):
    """Non-temporal (streaming) store of a scalar to an llvm ptr."""
    raw = value._value if hasattr(value, "_value") else value
    llvm.StoreOp(raw, ptr, alignment=alignment, nontemporal=True)


def exp2_amdgcn_scalar(scalar_value):
    raw = (
        arith.unwrap(scalar_value)
        if hasattr(scalar_value, "ir_value") or hasattr(scalar_value, "type")
        else scalar_value
    )
    f32_ty = ir.F32Type.get()
    return llvm.call_intrinsic(f32_ty, "llvm.amdgcn.exp2.f32", [raw], [], [])


def exp2_f32_fast(value):
    from flydsl._mlir.dialects import vector as _vector_dialect

    raw = arith.unwrap(value) if hasattr(value, "ir_value") or hasattr(value, "type") else value
    ty = raw.type
    if isinstance(ty, ir.VectorType):
        n = ty.shape[0]
        elems = []
        for i in range(n):
            scalar = _vector_dialect.extract(raw, static_position=[i], dynamic_position=[])
            elems.append(exp2_amdgcn_scalar(scalar))
        return _vector_dialect.from_elements(ty, elems)
    return exp2_amdgcn_scalar(raw)


def cdiv(numer: int, denom: int) -> int:
    return (numer + denom - 1) // denom


# Alias: several kernels historically spelled this ``ceildiv``.
ceildiv = cdiv


def align_up(value: int, align: int) -> int:
    """Round *value* up to the next multiple of *align* (static ints)."""
    return ((int(value) + int(align) - 1) // int(align)) * int(align)


def pow2_shift(value: int) -> int:
    assert value > 0 and (value & (value - 1)) == 0
    return value.bit_length() - 1


def is_pow2(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def udiv_pow2(value, divisor: int):
    return value >> fx.Int32(pow2_shift(divisor))


def urem_pow2(value, divisor: int):
    return value & fx.Int32(divisor - 1)


def udiv_const(value, divisor: int):
    if const_expr(is_pow2(divisor)):
        return udiv_pow2(value, divisor)
    return value // fx.Int32(divisor)


def urem_const(value, divisor: int):
    if const_expr(is_pow2(divisor)):
        return urem_pow2(value, divisor)
    return value % fx.Int32(divisor)


def unflatten_k(k_flat, qkhe_loop: int = 2):
    n = qkhe_loop * 2
    return [[k_flat[td * n + j] for j in range(n)] for td in range(len(k_flat) // n)]


def int_to_ptr(int_value, addr_space):
    """Cast an integer SSA value to an ``!llvm.ptr<addr_space>``."""
    raw = int_value._value if hasattr(int_value, "_value") else int_value
    return llvm.inttoptr(ir.Type.parse(f"!llvm.ptr<{addr_space}>"), raw)


def idx_to_ptr(idx_value, addr_space=1):
    """Cast an index SSA value to an ``!llvm.ptr<addr_space>`` (index -> i64 -> ptr)."""
    idx_raw = idx_value._value if hasattr(idx_value, "_value") else idx_value
    return int_to_ptr(arith.index_cast(T.i64, idx_raw), addr_space)


def lds_base_index(lds_memref):
    """Aligned base address of an LDS memref view as an index value."""
    return memref.extract_aligned_pointer_as_index(lds_memref)
