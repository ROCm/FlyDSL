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

Example:
    >>> from flydsl._mlir_helpers import rocdl
    >>> tid_x = rocdl.workitem_id_x()
    >>> rocdl.barrier()
"""

from .._mlir._mlir_libs._mlirDialectsFlyROCDL import CopyOpCDNA3BufferCopyType, MmaOpCDNA3_MFMAType
from .._mlir._mlir_libs._mlirDialectsFlyROCDL import MmaOpGFX1250_WMMAType
from .._mlir.dialects._fly_rocdl_enum_gen import AtomStateField
from .._mlir.dialects.rocdl import *  # noqa: F401,F403
from .._mlir.extras import types as T

BufferCopy = lambda bit_size: CopyOpCDNA3BufferCopyType.get(bit_size)  # noqa: E731
BufferCopy32b = lambda: CopyOpCDNA3BufferCopyType.get(32)  # noqa: E731
BufferCopy64b = lambda: CopyOpCDNA3BufferCopyType.get(64)  # noqa: E731
BufferCopy128b = lambda: CopyOpCDNA3BufferCopyType.get(128)  # noqa: E731


def MFMA(m, n, k, elem_type, elem_type_b=None, elem_type_acc=None):
    """Create an MFMA MMA atom type for CDNA3.

    Args:
        m, n, k: MFMA tile dimensions.
        elem_type: Element type (used for A, B, and accumulator if the others
                   are not specified).
        elem_type_b: Element type for B operand (defaults to elem_type).
        elem_type_acc: Element type for accumulator (defaults to elem_type).
    """
    from .._mlir import ir

    if isinstance(elem_type, type) and hasattr(elem_type, "ir_type"):
        ty = elem_type.ir_type
    elif isinstance(elem_type, ir.Type):
        ty = elem_type
    else:
        raise TypeError(f"MFMA: unsupported elem_type {elem_type}")

    ty_b = ty if elem_type_b is None else (elem_type_b.ir_type if hasattr(elem_type_b, "ir_type") else elem_type_b)
    ty_acc = (
        ty if elem_type_acc is None else (elem_type_acc.ir_type if hasattr(elem_type_acc, "ir_type") else elem_type_acc)
    )
    return MmaOpCDNA3_MFMAType.get(m, n, k, ty, ty_b, ty_acc)


def WMMA(m, n, k, elem_type, elem_type_b=None, elem_type_acc=None):
    """Create a WMMA MMA atom type for GFX1250 (wave32).

    Args:
        m, n, k: WMMA tile dimensions.
        elem_type: Element type for A operand.
        elem_type_b: Element type for B operand (defaults to elem_type).
        elem_type_acc: Element type for accumulator (defaults to elem_type).
    """
    from .._mlir import ir

    if isinstance(elem_type, type) and hasattr(elem_type, 'ir_type'):
        ty = elem_type.ir_type
    elif isinstance(elem_type, ir.Type):
        ty = elem_type
    else:
        raise TypeError(f"WMMA: unsupported elem_type {elem_type}")

    ty_b = ty if elem_type_b is None else (elem_type_b.ir_type if hasattr(elem_type_b, 'ir_type') else elem_type_b)
    ty_acc = ty if elem_type_acc is None else (elem_type_acc.ir_type if hasattr(elem_type_acc, 'ir_type') else elem_type_acc)
    return MmaOpGFX1250_WMMAType.get(m, n, k, ty, ty_b, ty_acc)


def make_buffer_tensor(memref, alignment=4, loc=None, ip=None):
    """Convert a global-address-space fly memref to a buffer_desc memref.

    Extracts the raw pointer from the input memref, builds an AMD buffer
    resource descriptor (base, stride, numRecords, flags), and wraps it
    back into a fly.memref with BufferDesc address space.
    """
    from .._mlir import ir
    from .._mlir.dialects import arith as _arith
    from .._mlir.dialects import fly
    from . import primitive as _prim
    from .meta import _to_raw_value

    raw_memref = _to_raw_value(memref)
    layout = _prim.get_layout(memref, loc=loc, ip=ip)
    elem_type = fly.MemRefType(raw_memref.type).element_type

    llvm_ptr_ty = ir.Type.parse("!llvm.ptr")
    base = fly.extract_aligned_pointer_as_index(llvm_ptr_ty, raw_memref, loc=loc, ip=ip)
    stride = _arith.ConstantOp(T.i16(), ir.IntegerAttr.get(T.i16(), 0)).result
    num_records = _arith.ConstantOp(T.i64(), ir.IntegerAttr.get(T.i64(), 0xFFFFFFFF)).result
    from .buffer_ops import _get_buffer_flags

    flags = _arith.ConstantOp(T.i32(), ir.IntegerAttr.get(T.i32(), _get_buffer_flags())).result

    bd_ptr_type = fly.PointerType.get(
        elem_type,
        address_space=int(fly.AddressSpace.BufferDesc),
        alignment=alignment,
    )
    bd_ptr = _prim.make_ptr(bd_ptr_type, [base, stride, num_records, flags], loc=loc, ip=ip)
    return _prim.make_view(bd_ptr, layout, loc=loc, ip=ip)


# Keep references to ODS-generated builders so we can wrap them without losing access.
_ods_wmma_scale_f32_16x16x128_f8f6f4 = (
    globals().get("wmma_scale_f32_16x16x128_f8f6f4", None)
)
_ods_wmma_scale_f32_32x16x128_f4 = (
    globals().get("wmma_scale_f32_32x16x128_f4", None)
)
_ods_wave_id = wave_id  # ODS: wave_id(res, ...) -> i32
_ods_cluster_workgroup_id_x = cluster_workgroup_id_x
_ods_cluster_workgroup_id_y = cluster_workgroup_id_y
_ods_cluster_workgroup_id_z = cluster_workgroup_id_z
_ods_cluster_load_async_to_lds_b8 = cluster_load_async_to_lds_b8
_ods_cluster_load_async_to_lds_b32 = cluster_load_async_to_lds_b32
_ods_cluster_load_async_to_lds_b64 = cluster_load_async_to_lds_b64
_ods_cluster_load_async_to_lds_b128 = cluster_load_async_to_lds_b128
_ods_s_wait_asynccnt = s_wait_asynccnt
_ods_mfma_f32_16x16x16f16 = mfma_f32_16x16x16f16
_ods_mfma_f32_16x16x16bf16_1k = globals().get("mfma_f32_16x16x16bf16_1k", None)
_ods_mfma_f32_16x16x32_fp8_fp8 = mfma_f32_16x16x32_fp8_fp8
_ods_mfma_i32_16x16x32_i8 = mfma_i32_16x16x32_i8
_ods_mfma_scale_f32_16x16x128_f8f6f4 = globals().get("mfma_scale_f32_16x16x128_f8f6f4", None) or globals().get(
    "mfma_scale_f32_16x16x128_f8f6f4_", None
)

# Keep ODS references for WMMA ops so we can wrap them.
_ods_wmma_f32_16x16x16_f16 = wmma_f32_16x16x16_f16
_ods_wmma_f32_16x16x16_bf16 = wmma_f32_16x16x16_bf16
_ods_wmma_f16_16x16x16_f16 = wmma_f16_16x16x16_f16
_ods_wmma_bf16_16x16x16_bf16 = wmma_bf16_16x16x16_bf16
_ods_wmma_i32_16x16x16_iu8 = wmma_i32_16x16x16_iu8
_ods_wmma_i32_16x16x16_iu4 = wmma_i32_16x16x16_iu4
_ods_wmma_f32_16x16x16_fp8_fp8 = globals().get("wmma_f32_16x16x16_fp8_fp8", None)
_ods_wmma_f32_16x16x16_fp8_bf8 = globals().get("wmma_f32_16x16x16_fp8_bf8", None)
_ods_wmma_f32_16x16x16_bf8_fp8 = globals().get("wmma_f32_16x16x16_bf8_fp8", None)
_ods_wmma_f32_16x16x16_bf8_bf8 = globals().get("wmma_f32_16x16x16_bf8_bf8", None)
_ods_wmma_i32_16x16x32_iu4 = globals().get("wmma_i32_16x16x32_iu4", None)
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

    from . import arith as _arith_ext

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
    # ODS signature: (res, a, b, c, cbsz, blgp, opselA, scaleA, opselB, scaleB)
    #   operands (Values): a, b, c, scaleA, scaleB
    #   attributes (ints): cbsz, blgp, opselA, opselB
    # Caller passes: [a, b, c, cbsz, blgp, opselA, scaleA, opselB, scaleB]
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
        result_type,
        a,
        b,
        c,
        cbsz,
        blgp,
        opselA,
        scaleA,
        opselB,
        scaleB,
        loc=loc,
        ip=ip,
    ).result


# ---------------------------------------------------------------------------
# WMMA wrappers  (Wave Matrix Multiply-Accumulate -- RDNA3/RDNA4)
#
# WMMA operands are [A, B, C] -- all MLIR Values, no integer flags.
# For IU variants the operand list is [A_sign, A, B_sign, B, C, clamp].
# For OPSEL variants (f16->f16, bf16->bf16) the list is [A, B, C, op_sel].
# ---------------------------------------------------------------------------


def _unwrap_wmma_operand(v, *, loc=None):
    """Accept Python ints (for flags like op_sel/clamp/signed) and ArithValue wrappers."""
    from flydsl._mlir.ir import IntegerType

    from . import arith as _arith_ext

    if isinstance(v, bool):
        return _arith_ext.unwrap(_arith_ext.constant(int(v), type=IntegerType.get_signless(1), loc=loc), loc=loc)
    if isinstance(v, int):
        return _arith_ext.unwrap(_arith_ext.constant(v, type=IntegerType.get_signless(32), loc=loc), loc=loc)
    return _arith_ext.unwrap(v, loc=loc)


# --- f32 output variants ---


def wmma_f32_16x16x16_f16(result_type, operands, *, loc=None, ip=None):
    """WMMA f16->f32, 16x16x16. Operands: [A, B, C]. Returns Value."""
    ops = [_unwrap_wmma_operand(v, loc=loc) for v in operands]
    return _ods_wmma_f32_16x16x16_f16(result_type, ops, loc=loc, ip=ip).result


def wmma_f32_16x16x16_bf16(result_type, operands, *, loc=None, ip=None):
    """WMMA bf16->f32, 16x16x16. Operands: [A, B, C]. Returns Value."""
    ops = [_unwrap_wmma_operand(v, loc=loc) for v in operands]
    return _ods_wmma_f32_16x16x16_bf16(result_type, ops, loc=loc, ip=ip).result


# --- fp8 variants (gfx12 / RDNA4 only) ---


def wmma_f32_16x16x16_fp8_fp8(result_type, operands, *, loc=None, ip=None):
    """WMMA fp8->f32, 16x16x16 (gfx12). Operands: [A, B, C]. Returns Value."""
    if _ods_wmma_f32_16x16x16_fp8_fp8 is None:
        raise AttributeError("ROCDL op not found: wmma_f32_16x16x16_fp8_fp8")
    ops = [_unwrap_wmma_operand(v, loc=loc) for v in operands]
    return _ods_wmma_f32_16x16x16_fp8_fp8(result_type, ops, loc=loc, ip=ip).result


def wmma_f32_16x16x16_fp8_bf8(result_type, operands, *, loc=None, ip=None):
    """WMMA fp8+bf8->f32, 16x16x16 (gfx12). Operands: [A, B, C]. Returns Value."""
    if _ods_wmma_f32_16x16x16_fp8_bf8 is None:
        raise AttributeError("ROCDL op not found: wmma_f32_16x16x16_fp8_bf8")
    ops = [_unwrap_wmma_operand(v, loc=loc) for v in operands]
    return _ods_wmma_f32_16x16x16_fp8_bf8(result_type, ops, loc=loc, ip=ip).result


def wmma_f32_16x16x16_bf8_fp8(result_type, operands, *, loc=None, ip=None):
    """WMMA bf8+fp8->f32, 16x16x16 (gfx12). Operands: [A, B, C]. Returns Value."""
    if _ods_wmma_f32_16x16x16_bf8_fp8 is None:
        raise AttributeError("ROCDL op not found: wmma_f32_16x16x16_bf8_fp8")
    ops = [_unwrap_wmma_operand(v, loc=loc) for v in operands]
    return _ods_wmma_f32_16x16x16_bf8_fp8(result_type, ops, loc=loc, ip=ip).result


def wmma_f32_16x16x16_bf8_bf8(result_type, operands, *, loc=None, ip=None):
    """WMMA bf8->f32, 16x16x16 (gfx12). Operands: [A, B, C]. Returns Value."""
    if _ods_wmma_f32_16x16x16_bf8_bf8 is None:
        raise AttributeError("ROCDL op not found: wmma_f32_16x16x16_bf8_bf8")
    ops = [_unwrap_wmma_operand(v, loc=loc) for v in operands]
    return _ods_wmma_f32_16x16x16_bf8_bf8(result_type, ops, loc=loc, ip=ip).result


# --- f16/bf16 output variants (OPSEL: operands include op_sel flag) ---


def wmma_f16_16x16x16_f16(result_type, operands, *, loc=None, ip=None):
    """WMMA f16->f16, 16x16x16. Operands: [A, B, C, op_sel]. Returns Value."""
    ops = [_unwrap_wmma_operand(v, loc=loc) for v in operands]
    return _ods_wmma_f16_16x16x16_f16(result_type, ops, loc=loc, ip=ip).result


def wmma_bf16_16x16x16_bf16(result_type, operands, *, loc=None, ip=None):
    """WMMA bf16->bf16, 16x16x16. Operands: [A, B, C, op_sel]. Returns Value."""
    ops = [_unwrap_wmma_operand(v, loc=loc) for v in operands]
    return _ods_wmma_bf16_16x16x16_bf16(result_type, ops, loc=loc, ip=ip).result


# --- Integer variants (IU: operands include sign flags and clamp) ---


def wmma_i32_16x16x16_iu8(result_type, operands, *, loc=None, ip=None):
    """WMMA int8->i32, 16x16x16. Operands: [A_sign, A, B_sign, B, C, clamp]. Returns Value."""
    ops = [_unwrap_wmma_operand(v, loc=loc) for v in operands]
    return _ods_wmma_i32_16x16x16_iu8(result_type, ops, loc=loc, ip=ip).result


def wmma_i32_16x16x16_iu4(result_type, operands, *, loc=None, ip=None):
    """WMMA int4->i32, 16x16x16. Operands: [A_sign, A, B_sign, B, C, clamp]. Returns Value."""
    ops = [_unwrap_wmma_operand(v, loc=loc) for v in operands]
    return _ods_wmma_i32_16x16x16_iu4(result_type, ops, loc=loc, ip=ip).result


def wmma_i32_16x16x32_iu4(result_type, operands, *, loc=None, ip=None):
    """WMMA int4->i32, 16x16x32 (gfx12). Operands: [A_sign, A, B_sign, B, C, clamp]. Returns Value."""
    if _ods_wmma_i32_16x16x32_iu4 is None:
        raise AttributeError("ROCDL op not found: wmma_i32_16x16x32_iu4")
    ops = [_unwrap_wmma_operand(v, loc=loc) for v in operands]
    return _ods_wmma_i32_16x16x32_iu4(result_type, ops, loc=loc, ip=ip).result


# --- WMMA Scale variants (gfx1250 mxfp4) ---

def wmma_scale_f32_16x16x128_f8f6f4(result_type, a, b, c, scaleA, scaleB,
                                      *, fmtA=4, fmtB=4, modC=0,
                                      scaleAType=0, fmtScaleA=0,
                                      scaleBType=0, fmtScaleB=0,
                                      reuseA=False, reuseB=False,
                                      loc=None, ip=None):
    """V_WMMA_SCALE_F32_16X16X128_F8F6F4 for gfx1250 (wave32).

    Operand types (wave32):
        a: vector<8xi32> (16x128 FP4 data)
        b: vector<8xi32> (128x16 FP4 data)
        c: vector<8xf32> (16x16 FP32 accumulator)
        scaleA: i32 (A scale VGPR)
        scaleB: i32 (B scale VGPR)

    fmtA/fmtB: data type encoding (0=FP8/E4M3, 1=FP8/E5M2, 2=FP6/E2M3, 3=FP6/E3M2, 4=FP4/E2M1)
    scaleAType/scaleBType: opsel – selects lo/hi 16-bit half of scale VGPR (0=lo, 1=hi)
    fmtScaleA/fmtScaleB: scale format (0=E8M0, 1=E5M3, 2=E4M3)
    """
    if _ods_wmma_scale_f32_16x16x128_f8f6f4 is None:
        raise AttributeError("ROCDL op not found: wmma_scale_f32_16x16x128_f8f6f4")
    a_v = _unwrap_wmma_operand(a, loc=loc)
    b_v = _unwrap_wmma_operand(b, loc=loc)
    c_v = _unwrap_wmma_operand(c, loc=loc)
    sA = _unwrap_wmma_operand(scaleA, loc=loc)
    sB = _unwrap_wmma_operand(scaleB, loc=loc)
    return _ods_wmma_scale_f32_16x16x128_f8f6f4(
        result_type, a_v, b_v, c_v, sA, sB,
        fmtA=fmtA, fmtB=fmtB, modC=modC,
        scaleAType=scaleAType, fmtScaleA=fmtScaleA,
        scaleBType=scaleBType, fmtScaleB=fmtScaleB,
        reuseA=reuseA, reuseB=reuseB,
        loc=loc, ip=ip,
    ).result


def wmma_scale_f32_32x16x128_f4(result_type, a, b, c, scaleA, scaleB,
                                  *, modC=0,
                                  scaleAType=0, fmtScaleA=0,
                                  scaleBType=0, fmtScaleB=0,
                                  reuseA=False, reuseB=False,
                                  loc=None, ip=None):
    """V_WMMA_SCALE_F32_32X16X128_F4 for gfx1250 (wave32).

    Operand types (wave32):
        a: vector<16xi32> (32x128 FP4 data)
        b: vector<8xi32>  (128x16 FP4 data)
        c: vector<16xf32> (32x16 FP32 accumulator)
        scaleA: i32 (A scale VGPR)
        scaleB: i32 (B scale VGPR)

    scaleAType/scaleBType: lane half-select (0=lanes 0-15, 1=lanes 16-31)
        — maps to VOP3PX2 scale_op_sel bits (OPSEL)
    fmtScaleA/fmtScaleB: scale data format (0=E8M0, 2=E4M3)
        — maps to VOP3PX2 neg_lo/neg_hi bits (repurposed)
    """
    if _ods_wmma_scale_f32_32x16x128_f4 is None:
        raise AttributeError("ROCDL op not found: wmma_scale_f32_32x16x128_f4")
    a_v = _unwrap_wmma_operand(a, loc=loc)
    b_v = _unwrap_wmma_operand(b, loc=loc)
    c_v = _unwrap_wmma_operand(c, loc=loc)
    sA = _unwrap_wmma_operand(scaleA, loc=loc)
    sB = _unwrap_wmma_operand(scaleB, loc=loc)
    return _ods_wmma_scale_f32_32x16x128_f4(
        result_type, a_v, b_v, c_v, sA, sB,
        modC=modC,
        scaleAType=scaleAType, fmtScaleA=fmtScaleA,
        scaleBType=scaleBType, fmtScaleB=fmtScaleB,
        reuseA=reuseA, reuseB=reuseB,
        loc=loc, ip=ip,
    ).result


def wave_id():
    """Get wave-id-in-workgroup as SGPR (via TTMP8[29:25]).

    On gfx1250 this reads an architected SGPR, so the result stays in
    the SGPR pipeline and all derived computations are automatically
    scalarized by LLVM uniformity analysis.

    Returns:
        i32 value (SGPR) with the wave ID within the workgroup.
    """
    from .._mlir import ir
    i32 = ir.IntegerType.get_signless(32)
    return _ods_wave_id(i32)


def cluster_workgroup_id_x():
    """Get workgroup position within cluster along X (SGPR, gfx1250). """
    from .._mlir import ir
    i32 = ir.IntegerType.get_signless(32)
    return _ods_cluster_workgroup_id_x(i32)


def cluster_workgroup_id_y():
    """Get workgroup position within cluster along Y (SGPR, gfx1250). """
    from .._mlir import ir
    i32 = ir.IntegerType.get_signless(32)
    return _ods_cluster_workgroup_id_y(i32)


def cluster_workgroup_id_z():
    """Get workgroup position within cluster along Z (SGPR, gfx1250). """
    from .._mlir import ir
    i32 = ir.IntegerType.get_signless(32)
    return _ods_cluster_workgroup_id_z(i32)


def cluster_load_async_to_lds(global_ptr, lds_ptr, size_bytes, offset=0, cpol=0, mask=None):
    """Per-lane cluster broadcast load: Global -> LDS with MCAST (gfx1250).

    Args:
        global_ptr: ``!llvm.ptr<1>`` — global address space pointer.
        lds_ptr:    ``!llvm.ptr<3>`` — LDS address space pointer.
        size_bytes: Load width: 1, 4, 8, or 16 bytes (selects b8/b32/b64/b128).
        offset:     Byte offset (int, default 0).
        cpol:       Cache policy (int, default 0).
        mask:       i32 workgroup_mask for MCAST broadcast. None means no mask
                    (falls back to non-cluster global_load_async_to_lds).

    Raises:
        ValueError: If ``size_bytes`` is not 1, 4, 8, or 16.
    """
    _dispatch = {
        1: _ods_cluster_load_async_to_lds_b8,
        4: _ods_cluster_load_async_to_lds_b32,
        8: _ods_cluster_load_async_to_lds_b64,
        16: _ods_cluster_load_async_to_lds_b128,
    }
    fn = _dispatch.get(size_bytes)
    if fn is None:
        raise ValueError(
            f"cluster_load_async_to_lds: size_bytes must be 1, 4, 8, or 16, "
            f"got {size_bytes}")
    if mask is None:
        from .._mlir import ir
        from . import arith as _arith
        mask = _arith.unwrap(_arith.constant(0, type=ir.IntegerType.get_signless(32)))
    fn(global_ptr, lds_ptr, offset, cpol, mask)


def disable_xdl_arb_stall():
    """Disable WMMA multicycle arbitration stall by setting SCHED_MODE bit 4."""
    from .._mlir.dialects import llvm as _llvm
    from . import arith as _arith
    from .typing import T

    # hwreg encoding: ID=26(SCHED_MODE), Offset=4, Size=1 -> 282
    imm_val = _arith.unwrap(_arith.constant(282, type=T.i32))
    val_val = _arith.unwrap(_arith.constant(1, type=T.i32))

    _llvm.call_intrinsic(None, "llvm.amdgcn.s.setreg", [imm_val, val_val], [], [])


def s_wait_asynccnt(count=0):
    """Wait for outstanding async load/store operations (ASYNCcnt counter).

    Args:
        count: Maximum number of outstanding operations to allow.
               0 = wait for all.
    """
    _ods_s_wait_asynccnt(count)


def lds_transpose_load(result_type, lds_memref, elem_offset, elem_bytes):
    """Transpose-load from LDS memref via ds_load_tr16_b128 (gfx1250).

    Args:
        result_type: Vector result type, e.g. ``VectorType.get([8], f16)``.
        lds_memref:  LDS memref value (address-space 3), typically from
                     ``SmemPtr.get()`` or ``get_op_result_or_value(...)``.
        elem_offset: Per-lane linearized element offset into the memref
                     (ArithValue / ir.Value of index type / Python int).
        elem_bytes:  Element size in bytes (Python int, e.g. 2 for f16).

    Returns:
        Loaded and transposed vector ``ir.Value``.
    """
    from .._mlir import ir as _ir
    from .._mlir.dialects import (
        llvm as _llvm,
        memref as _memref,
        rocdl as _rocdl,
    )
    from . import arith as _arith
    from .arith import _to_raw
    from .typing import T
    from .utils.arith import ArithValue as _AV

    lds_ptr_ty = _ir.Type.parse("!llvm.ptr<3>")
    raw_memref = _arith.unwrap(lds_memref)
    lds_base = _memref.extract_aligned_pointer_as_index(raw_memref)

    byte_off = _AV(_arith.unwrap(elem_offset, index=True)) * _arith.index(elem_bytes)
    total_byte_idx = _AV(lds_base) + byte_off
    addr_i32 = _to_raw(_arith.index_cast(T.i32, total_byte_idx))
    ptr_val = _llvm.inttoptr(lds_ptr_ty, addr_i32)

    return _rocdl.ds_load_tr16_b128(result_type, ptr_val)


__all__ = [
    # Thread/Block/Grid IDs and dimensions
    "workitem_id_x",
    "workitem_id_y",
    "workitem_id_z",
    "workgroup_id_x",
    "workgroup_id_y",
    "workgroup_id_z",
    "workgroup_dim_x",
    "workgroup_dim_y",
    "workgroup_dim_z",
    "grid_dim_x",
    "grid_dim_y",
    "grid_dim_z",
    "wavefrontsize",
    "wave_id",
    # Synchronization
    "barrier",
    "s_barrier",
    "s_barrier_signal",
    "s_barrier_wait",
    "s_waitcnt",
    "s_wait_loadcnt",
    "s_wait_storecnt",
    "s_wait_dscnt",
    "s_wait_expcnt",
    "s_wait_asynccnt",
    "disable_xdl_arb_stall",
    # Matrix operations - MFMA (Matrix Fused Multiply-Add)
    "mfma_f32_32x32x8f16",
    "mfma_f32_16x16x16f16",
    "mfma_f32_16x16x16bf16_1k",
    "mfma_f32_32x32x4bf16",
    "mfma_f32_16x16x8bf16",
    "mfma_i32_32x32x8i8",
    "mfma_i32_16x16x16i8",
    "mfma_i32_16x16x32_i8",
    "mfma_scale_f32_16x16x128_f8f6f4",
    # Raw-op constructors (return op view) for the above
    "mfma_f32_16x16x16f16_op",
    "mfma_f32_16x16x32_fp8_fp8_op",
    "mfma_f32_16x16x16bf16_1k_op",
    "mfma_i32_16x16x32_i8_op",
    "mfma_scale_f32_16x16x128_f8f6f4_op",
    # Matrix operations - WMMA (Wave Matrix Multiply-Accumulate)
    "wmma_f32_16x16x16_f16",
    "wmma_f32_16x16x16_bf16",
    "wmma_f16_16x16x16_f16",
    "wmma_bf16_16x16x16_bf16",
    "wmma_i32_16x16x16_iu8",
    "wmma_i32_16x16x16_iu4",
    "wmma_f32_16x16x16_fp8_fp8",
    "wmma_f32_16x16x16_fp8_bf8",
    "wmma_f32_16x16x16_bf8_fp8",
    "wmma_f32_16x16x16_bf8_bf8",
    "wmma_i32_16x16x32_iu4",
    "wmma_scale_f32_16x16x128_f8f6f4",   # gfx1250 WMMA_SCALE 16x16x128 (FP4/FP6/FP8)
    "wmma_scale_f32_32x16x128_f4",        # gfx1250 WMMA_SCALE 32x16x128 (FP4 only)
    # Matrix operations - SMFMAC (Sparse Matrix FMA)
    "smfmac_f32_32x32x16_f16",
    "smfmac_f32_32x32x16_bf16",
    "smfmac_i32_32x32x32_i8",
    # Shuffle and permutation
    "ds_swizzle",
    "ds_bpermute",
    "permlanex16",
    "permlane16_swap",
    "permlane32_swap",
    "readlane",
    "readfirstlane",
    "update_dpp",
    "ballot",
    # Data movement
    "raw_buffer_load",
    "raw_buffer_store",
    "raw_ptr_buffer_load",
    "raw_ptr_buffer_store",
    "load_to_lds",
    "global_load_lds",
    "make_buffer_rsrc",
    # Atomic operations
    "raw_buffer_atomic_fadd",
    "raw_buffer_atomic_fmax",
    "raw_buffer_atomic_smax",
    "raw_buffer_atomic_umin",
    "raw_ptr_buffer_atomic_fadd",
    "raw_ptr_buffer_atomic_fmax",
    # Bit manipulation
    "mbcnt_lo",
    "mbcnt_hi",
    # Scheduling and optimization
    "s_setprio",
    "s_sleep",
    "sched_barrier",
    "sched_group_barrier",
    "iglp_opt",
    # Type conversions
    "cvt_f32_bf8",
    "cvt_f32_fp8",
    "cvt_pk_f32_bf8",
    "cvt_pk_f32_fp8",
    # Atom state enums
    "AtomStateField",
    # Copy atom types
    "CopyOpCDNA3BufferCopyType",
    "BufferCopy",
    "BufferCopy32b",
    "BufferCopy64b",
    "BufferCopy128b",
    # MMA atom types
    "MmaOpCDNA3_MFMAType",
    "MFMA",
    "MmaOpGFX1250_WMMAType",
    "WMMA",
    # Convenience wrappers
    "make_buffer_tensor",
    "lds_transpose_load",       # memref-level wrapper for gfx1250 ds_load_tr16_b128
    # gfx1250 TDM - descriptor-driven tile copy (preferred over per-lane)
    "tensor_load_to_lds",       # 4-group, up to 5D tensor
    "tensor_load_to_lds_d2",    # 2-group, up to 2D tensor
    "tensor_store_from_lds",    # 4-group store
    "tensor_store_from_lds_d2", # 2-group store
    "s_wait_tensorcnt",
    # gfx1250 L2 prefetch
    "global_prefetch",          # per-lane 1-byte prefetch hint
    # Cluster (gfx1250 workgroup clustering)
    "cluster_workgroup_id_x",
    "cluster_workgroup_id_y",
    "cluster_workgroup_id_z",
    "cluster_load_async_to_lds",   # per-lane MCAST load (Global → LDS)
]


# ── Wrappers that accept DSL Numeric args (fx.Int32, fx.Float32, etc.) ───────
# The ODS-generated ops require raw ir.Value. These wrappers call ir_value()
# on any DSL Numeric argument before forwarding to the underlying MLIR op.


def _to_ir(v):
    """Coerce DSL Numeric to ir.Value if needed."""
    if not isinstance(v, __import__("flydsl._mlir.ir", fromlist=["Value"]).Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def raw_ptr_buffer_atomic_fadd(vdata, rsrc, offset, soffset, aux, **kw):
    from .._mlir.dialects.rocdl import raw_ptr_buffer_atomic_fadd as _op

    return _op(_to_ir(vdata), _to_ir(rsrc), _to_ir(offset), _to_ir(soffset), _to_ir(aux), **kw)


def raw_ptr_buffer_atomic_fmax(vdata, rsrc, offset, soffset, aux, **kw):
    from .._mlir.dialects.rocdl import raw_ptr_buffer_atomic_fmax as _op

    return _op(_to_ir(vdata), _to_ir(rsrc), _to_ir(offset), _to_ir(soffset), _to_ir(aux), **kw)


def cvt_pk_fp8_f32(res, src_a, src_b, old, word_sel, **kw):
    from .._mlir.dialects.rocdl import cvt_pk_fp8_f32 as _op

    return _op(res=res, src_a=_to_ir(src_a), src_b=_to_ir(src_b), old=_to_ir(old), word_sel=word_sel, **kw)


def raw_ptr_buffer_load_lds(rsrc, lds_ptr, size, voffset, soffset, offset, aux, **kw):
    from .._mlir.dialects.rocdl import raw_ptr_buffer_load_lds as _op

    return _op(
        _to_ir(rsrc), _to_ir(lds_ptr), _to_ir(size), _to_ir(voffset), _to_ir(soffset), _to_ir(offset), _to_ir(aux), **kw
    )
