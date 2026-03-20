# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""ROCDL dialect extension for ROCm/AMD GPU programming.

This module provides access to ROCm-specific GPU operations including:
- Thread/block/grid identifiers and dimensions
- Synchronization primitives (barriers, wait operations)
- Matrix multiplication acceleration (MFMA, WMMA, SMFMAC)
- Data movement and shuffle operations
- Atomic operations
- Type conversion operations
- Buffer-backed tensor creation (make_buffer_tensor)
- Copy atom types (BufferCopy)
"""

from ..._mlir.dialects.rocdl import *  # noqa: F401,F403

# Keep references to ODS-generated builders so we can wrap them without losing access.
_ods_mfma_f32_16x16x16f16 = mfma_f32_16x16x16f16
_ods_mfma_f32_16x16x16bf16_1k = globals().get("mfma_f32_16x16x16bf16_1k", None)
_ods_mfma_f32_16x16x32_fp8_fp8 = mfma_f32_16x16x32_fp8_fp8
_ods_mfma_i32_16x16x32_i8 = mfma_i32_16x16x32_i8
_ods_mfma_scale_f32_16x16x128_f8f6f4 = (
    globals().get("mfma_scale_f32_16x16x128_f8f6f4", None)
    or globals().get("mfma_scale_f32_16x16x128_f8f6f4_", None)
)
mask_mfma = 0x008
mask_vmem_rd = 0x020
mask_dsrd = 0x100
mask_dswr = 0x200

def sched_mfma(cnt):
    sched_group_barrier(mask_mfma, cnt, 0)
def sched_vmem(cnt):
    sched_group_barrier(mask_vmem_rd, cnt, 0)
def sched_dsrd(cnt):
    sched_group_barrier(mask_dsrd, cnt, 0)
def sched_dswr(cnt):
    sched_group_barrier(mask_dswr, cnt, 0)


def _unwrap_mfma_operand(v, *, loc=None):
    """MFMA operands are MLIR Values; some trailing operands are i32 flags.

    Accept Python ints and materialize them as i32 signless constants.
    """
    from flydsl._mlir.ir import IntegerType
    from .. import arith as _arith_ext

    if isinstance(v, int):
        return _arith_ext.unwrap(_arith_ext.constant(v, type=IntegerType.get_signless(32), loc=loc), loc=loc)
    return _arith_ext.unwrap(v, loc=loc)


def _split_mfma_operands(operands, *, loc=None):
    """Split [a, b, c, cbsz, abid, blgp] into (a, b, c) Values + (cbsz, abid, blgp) ints."""
    a = _unwrap_mfma_operand(operands[0], loc=loc)
    b = _unwrap_mfma_operand(operands[1], loc=loc)
    c = _unwrap_mfma_operand(operands[2], loc=loc)
    cbsz = int(operands[3]) if len(operands) > 3 else 0
    abid = int(operands[4]) if len(operands) > 4 else 0
    blgp = int(operands[5]) if len(operands) > 5 else 0
    return a, b, c, cbsz, abid, blgp


def mfma_f32_16x16x16f16(result_type, operands, *, loc=None, ip=None):
    a, b, c, cbsz, abid, blgp = _split_mfma_operands(operands, loc=loc)
    return _ods_mfma_f32_16x16x16f16(result_type, a, b, c, cbsz, abid, blgp, loc=loc, ip=ip).result


def mfma_f32_16x16x16bf16_1k(result_type, operands, *, loc=None, ip=None):
    if _ods_mfma_f32_16x16x16bf16_1k is None:
        raise AttributeError("ROCDL op not found: mfma_f32_16x16x16bf16_1k")
    a, b, c, cbsz, abid, blgp = _split_mfma_operands(operands, loc=loc)
    return _ods_mfma_f32_16x16x16bf16_1k(result_type, a, b, c, cbsz, abid, blgp, loc=loc, ip=ip).result


def mfma_f32_16x16x32_fp8_fp8(result_type, operands, *, loc=None, ip=None):
    a, b, c, cbsz, abid, blgp = _split_mfma_operands(operands, loc=loc)
    return _ods_mfma_f32_16x16x32_fp8_fp8(result_type, a, b, c, cbsz, abid, blgp, loc=loc, ip=ip).result


def mfma_i32_16x16x32_i8(result_type, operands, *, loc=None, ip=None):
    a, b, c, cbsz, abid, blgp = _split_mfma_operands(operands, loc=loc)
    return _ods_mfma_i32_16x16x32_i8(result_type, a, b, c, cbsz, abid, blgp, loc=loc, ip=ip).result


def mfma_scale_f32_16x16x128_f8f6f4(result_type, operands, *, loc=None, ip=None):
    if _ods_mfma_scale_f32_16x16x128_f8f6f4 is None:
        raise AttributeError("ROCDL op not found: mfma_scale_f32_16x16x128_f8f6f4(_)")
    a = _unwrap_mfma_operand(operands[0], loc=loc)
    b = _unwrap_mfma_operand(operands[1], loc=loc)
    c = _unwrap_mfma_operand(operands[2], loc=loc)
    cbsz = int(operands[3]) if len(operands) > 3 else 0
    blgp = int(operands[4]) if len(operands) > 4 else 0
    opselA = int(operands[5]) if len(operands) > 5 else 0
    scaleA = _unwrap_mfma_operand(operands[6], loc=loc) if len(operands) > 6 else a
    opselB = int(operands[7]) if len(operands) > 7 else 0
    scaleB = _unwrap_mfma_operand(operands[8], loc=loc) if len(operands) > 8 else b
    return _ods_mfma_scale_f32_16x16x128_f8f6f4(
        result_type, a, b, c, cbsz, blgp, opselA, scaleA, opselB, scaleB,
        loc=loc, ip=ip,
    ).result


# ── New high-level helpers from universal.py ──────────────────────────
from .universal import *  # noqa: F401,F403


# ── Wrappers: accept DSL Numeric args (fx.Int32, fx.Float32, etc.) ─────────
# ODS-generated ops require raw ir.Value. These wrappers auto-convert.

def _to_ir(v):
    """Coerce DSL Numeric to ir.Value if needed."""
    from ..._mlir import ir as _ir
    if not isinstance(v, _ir.Value) and hasattr(v, 'ir_value'):
        return v.ir_value()
    return v


def raw_ptr_buffer_atomic_fadd(vdata, rsrc, offset, soffset, aux, **kw):
    from ..._mlir.dialects.rocdl import raw_ptr_buffer_atomic_fadd as _op
    return _op(_to_ir(vdata), _to_ir(rsrc), _to_ir(offset), _to_ir(soffset), _to_ir(aux), **kw)


def raw_ptr_buffer_atomic_fmax(vdata, rsrc, offset, soffset, aux, **kw):
    from ..._mlir.dialects.rocdl import raw_ptr_buffer_atomic_fmax as _op
    return _op(_to_ir(vdata), _to_ir(rsrc), _to_ir(offset), _to_ir(soffset), _to_ir(aux), **kw)


def cvt_pk_fp8_f32(res, src_a, src_b, old, word_sel, **kw):
    from ..._mlir.dialects.rocdl import cvt_pk_fp8_f32 as _op
    return _op(res=res, src_a=_to_ir(src_a), src_b=_to_ir(src_b),
               old=_to_ir(old), word_sel=word_sel, **kw)


def raw_ptr_buffer_load_lds(rsrc, lds_ptr, size, voffset, soffset, offset, aux, **kw):
    from ..._mlir.dialects.rocdl import raw_ptr_buffer_load_lds as _op
    return _op(_to_ir(rsrc), _to_ir(lds_ptr), _to_ir(size), _to_ir(voffset),
               _to_ir(soffset), _to_ir(offset), _to_ir(aux), **kw)
