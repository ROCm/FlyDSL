# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared module-level helpers for the CDNA SageAttention kernel.

State-free MLIR-facing free functions extracted from build_sage_attn_cdna_module.
Moving them here changes nothing about the emitted IR/ISA.
"""

import math as host_math

from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith
from flydsl.expr import math as _math
from flydsl.expr.typing import T
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


def _fadd(a, b, fm):
    return arith.addf(_raw(a), _raw(b), fastmath=fm)


def _fsub(a, b, fm):
    return arith.subf(_raw(a), _raw(b), fastmath=fm)


def _fmul(a, b, fm):
    return arith.mulf(_raw(a), _raw(b), fastmath=fm)


def _fmax(a, b, fm):
    return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm).result


def _ffma(a, b, c, fm):
    """Fused a*b + c (single rounding); folds the QK descale into the exp arg."""
    return _math.fma(_raw(a), _raw(b), _raw(c), fastmath=fm)


def _sitofp(v):
    return arith.SIToFPOp(T.f32, _raw(v)).result
