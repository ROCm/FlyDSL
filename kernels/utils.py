# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, buffer_ops, const_expr, rocdl
from flydsl.expr.typing import T


def rcp_f32(value):
    return rocdl.rcp(T.f32, value)


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


def maxnumf(a, b):
    return type(a)(arith.maxnumf(arith.unwrap(a), arith.unwrap(b)))


def unflatten_k(k_flat, qkhe_loop: int = 2):
    n = qkhe_loop * 2
    return [[k_flat[td * n + j] for j in range(n)] for td in range(len(k_flat) // n)]


def extract_global_ptr(tensor):
    from flydsl._mlir.dialects import fly as _fly

    raw = tensor.ir_value() if hasattr(tensor, "ir_value") and not isinstance(tensor, ir.Value) else tensor
    ptr_type = ir.Type.parse("!llvm.ptr<1>")
    return _fly.extract_aligned_pointer_as_index(ptr_type, raw)


def global_ptr_from_addr(addr_i64):
    raw = addr_i64.ir_value() if hasattr(addr_i64, "ir_value") and not isinstance(addr_i64, ir.Value) else addr_i64
    return llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), raw).result


def global_load(global_ptr, byte_offset, result_type, *, alignment):
    ptr = buffer_ops.get_element_ptr(global_ptr, byte_offset=fx.Int64(byte_offset), elem_type=T.i8)
    return llvm.LoadOp(result_type, ptr, alignment=alignment).result


def global_load_i64x2(global_ptr, byte_offset_i64):
    return global_load(global_ptr, byte_offset_i64, T.i64x2, alignment=16)


def global_load_i32(global_ptr, elem_offset_i32):
    return global_load(global_ptr, fx.Int64(elem_offset_i32) * fx.Int64(4), T.i32, alignment=4)


def ptr8_to_v4i32(ptr8_val):
    from flydsl.expr.rocdl import _to_ir

    i128_ty = ir.IntegerType.get_signless(128)
    v4i32_ty = ir.VectorType.get([4], ir.IntegerType.get_signless(32))
    i128_val = llvm.ptrtoint(i128_ty, _to_ir(ptr8_val))
    return llvm.bitcast(v4i32_ty, i128_val)


def buffer_load(rsrc, soffset_i32, vec_width=4, *, is_scalar=False, dtype=None, cache_modifier=0):
    if not is_scalar:
        return buffer_ops.buffer_load(
            rsrc,
            soffset_i32,
            vec_width=vec_width,
            dtype=dtype,
            cache_modifier=cache_modifier,
        )

    from flydsl.expr.rocdl import _to_ir

    i32_ty = ir.IntegerType.get_signless(32)
    if vec_width == 1:
        result_type = i32_ty
        suffix = "i32"
    elif vec_width == 4:
        result_type = ir.VectorType.get([4], i32_ty)
        suffix = "v4i32"
    else:
        raise ValueError(f"buffer_load(is_scalar=True): unsupported vec_width={vec_width}")

    rsrc_v4 = ptr8_to_v4i32(rsrc)
    cache_policy = arith.constant(cache_modifier, type=T.i32)
    return llvm.call_intrinsic(
        result_type,
        f"llvm.amdgcn.s.buffer.load.{suffix}",
        [
            _to_ir(rsrc_v4),
            _to_ir(soffset_i32),
            _to_ir(cache_policy),
        ],
        [],
        [],
    )
