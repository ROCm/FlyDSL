# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared module-level helpers for the CDNA SageAttention kernel.

State-free fx-facing free functions extracted from build_sage_attn_cdna_module.
"""

import math as host_math

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as _raw

_LOG2E = host_math.log2(host_math.e)
_LDS_CAP_BYTES = 65536


def _llvm_value(value):
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    return value


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def _extract_aligned_pointer(tensor) -> ir.Value:
    return _fly.extract_aligned_pointer_as_index(_llvm_ptr_ty(), _llvm_value(tensor))


def _pointer_load(result_type: ir.Type, ptr: ir.Value) -> ir.Value:
    return _llvm.LoadOp(result_type, _llvm_value(ptr)).result


def _pointer_store(value: ir.Value, ptr: ir.Value):
    return _llvm.StoreOp(_llvm_value(value), _llvm_value(ptr))


def _f32_to_bf16_trunc(f32_raw):
    """Bitwise f32 → bf16 truncation (upper 16 bits)."""
    i32_val = arith.bitcast(T.i32, _raw(f32_raw))
    upper = arith.ShRUIOp(i32_val, arith.constant(16, type=T.i32)).result
    i16_val = arith.TruncIOp(T.i16, upper).result
    return arith.bitcast(T.bf16, i16_val)


def _i32_pair_to_i64(lo, hi):
    return ((fx.Uint64(hi) << 32) | fx.Uint64(lo)).ir_value()


def _lds_vec_load(vec_type, lds_view, offset):
    return Vec.load(vec_type, lds_view, [fx.Index(offset)])


def _lds_vec_store(vec, lds_view, offset):
    Vec(vec).store(lds_view, [fx.Index(offset)])


def _lds_vec_store_elem(elem, lds_view, offset, elem_ty):
    Vec.from_elements([elem], elem_ty).store(lds_view, [fx.Index(offset)])
