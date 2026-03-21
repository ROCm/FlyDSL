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

from .._mlir.dialects.rocdl import *  # noqa: F401,F403
from .._mlir._mlir_libs._fly_rocdl import CopyOpCDNA3BufferLDSTType

from .._mlir._mlir_libs._fly_rocdl import MmaAtomCDNA3_MFMAType

BufferLDST = lambda bit_size: CopyOpCDNA3BufferLDSTType.get(bit_size)  # noqa: E731
BufferLDST32b = lambda: CopyOpCDNA3BufferLDSTType.get(32)  # noqa: E731
BufferLDST64b = lambda: CopyOpCDNA3BufferLDSTType.get(64)  # noqa: E731
BufferLDST128b = lambda: CopyOpCDNA3BufferLDSTType.get(128)  # noqa: E731


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

    if isinstance(elem_type, type) and hasattr(elem_type, 'ir_type'):
        ty = elem_type.ir_type
    elif isinstance(elem_type, ir.Type):
        ty = elem_type
    else:
        raise TypeError(f"MFMA: unsupported elem_type {elem_type}")

    ty_b = ty if elem_type_b is None else (elem_type_b.ir_type if hasattr(elem_type_b, 'ir_type') else elem_type_b)
    ty_acc = ty if elem_type_acc is None else (elem_type_acc.ir_type if hasattr(elem_type_acc, 'ir_type') else elem_type_acc)
    return MmaAtomCDNA3_MFMAType.get(m, n, k, ty, ty_b, ty_acc)


def make_buffer_tensor(memref, alignment=4, loc=None, ip=None):
    """Convert a global-address-space fly memref to a buffer_desc memref.

    Extracts the raw pointer from the input memref, builds an AMD buffer
    resource descriptor (base, stride, numRecords, flags), and wraps it
    back into a fly.memref with BufferDesc address space.
    """
    from . import primitive as _prim
    from .meta import _to_raw_value
    from .._mlir import ir
    from .._mlir.dialects import fly, arith as _arith

    raw_memref = _to_raw_value(memref)
    layout = _prim.get_layout(memref, loc=loc, ip=ip)
    elem_type = fly.MemRefType(raw_memref.type).element_type

    llvm_ptr_ty = ir.Type.parse("!llvm.ptr")
    base = fly.extract_aligned_pointer_as_index(llvm_ptr_ty, raw_memref, loc=loc, ip=ip)
    i16 = ir.IntegerType.get_signless(16)
    i32 = ir.IntegerType.get_signless(32)
    i64 = ir.IntegerType.get_signless(64)
    stride = _arith.ConstantOp(i16, ir.IntegerAttr.get(i16, 0)).result
    num_records = _arith.ConstantOp(i64, ir.IntegerAttr.get(i64, 0xFFFFFFFF)).result
    from .buffer_ops import _get_buffer_flags
    flags = _arith.ConstantOp(i32, ir.IntegerAttr.get(i32, _get_buffer_flags())).result

    bd_ptr_type = fly.PointerType.get(
        elem_type,
        address_space=int(fly.AddressSpace.BufferDesc),
        alignment=alignment,
    )
    bd_ptr = _prim.make_ptr(bd_ptr_type, [base, stride, num_records, flags], loc=loc, ip=ip)
    return _prim.make_view(bd_ptr, layout, loc=loc, ip=ip)

# Keep references to ODS-generated builders so we can wrap them without losing access.
_ods_mfma_f32_16x16x16f16 = mfma_f32_16x16x16f16
_ods_mfma_f32_16x16x16bf16_1k = globals().get("mfma_f32_16x16x16bf16_1k", None)
_ods_mfma_f32_16x16x32_fp8_fp8 = mfma_f32_16x16x32_fp8_fp8
_ods_mfma_i32_16x16x32_i8 = mfma_i32_16x16x32_i8
_ods_mfma_scale_f32_16x16x128_f8f6f4 = (
    globals().get("mfma_scale_f32_16x16x128_f8f6f4", None)
    or globals().get("mfma_scale_f32_16x16x128_f8f6f4_", None)
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
        result_type, a, b, c, cbsz, blgp, opselA, scaleA, opselB, scaleB,
        loc=loc, ip=ip,
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


__all__ = [
    # Thread/Block/Grid IDs and dimensions
    'workitem_id_x', 'workitem_id_y', 'workitem_id_z',
    'workgroup_id_x', 'workgroup_id_y', 'workgroup_id_z', 
    'workgroup_dim_x', 'workgroup_dim_y', 'workgroup_dim_z',
    'grid_dim_x', 'grid_dim_y', 'grid_dim_z',
    'wavefrontsize',
    
    # Synchronization
    'barrier', 's_barrier', 's_barrier_signal', 's_barrier_wait',
    's_waitcnt', 's_wait_loadcnt', 's_wait_storecnt',
    's_wait_dscnt', 's_wait_expcnt',
    
    # Matrix operations - MFMA (Matrix Fused Multiply-Add)
    'mfma_f32_32x32x8f16', 'mfma_f32_16x16x16f16',
    'mfma_f32_16x16x16bf16_1k',
    'mfma_f32_32x32x4bf16', 'mfma_f32_16x16x8bf16',
    'mfma_i32_32x32x8i8', 'mfma_i32_16x16x16i8',
    'mfma_i32_16x16x32_i8',
    'mfma_scale_f32_16x16x128_f8f6f4',
    # Raw-op constructors (return op view) for the above
    'mfma_f32_16x16x16f16_op', 'mfma_f32_16x16x32_fp8_fp8_op',
    'mfma_f32_16x16x16bf16_1k_op',
    'mfma_i32_16x16x32_i8_op',
    'mfma_scale_f32_16x16x128_f8f6f4_op',
    
    # Matrix operations - WMMA (Wave Matrix Multiply-Accumulate)
    'wmma_f32_16x16x16_f16', 'wmma_f32_16x16x16_bf16',
    'wmma_f16_16x16x16_f16', 'wmma_bf16_16x16x16_bf16',
    'wmma_i32_16x16x16_iu8', 'wmma_i32_16x16x16_iu4',
    'wmma_f32_16x16x16_fp8_fp8', 'wmma_f32_16x16x16_fp8_bf8',
    'wmma_f32_16x16x16_bf8_fp8', 'wmma_f32_16x16x16_bf8_bf8',
    'wmma_i32_16x16x32_iu4',
    
    # Matrix operations - SMFMAC (Sparse Matrix FMA)
    'smfmac_f32_32x32x16_f16', 'smfmac_f32_32x32x16_bf16',
    'smfmac_i32_32x32x32_i8',
    
    # Shuffle and permutation
    'ds_swizzle', 'ds_bpermute',
    'permlanex16', 'permlane16_swap', 'permlane32_swap',
    'readlane', 'readfirstlane',
    'update_dpp',
    'ballot',
    
    # Data movement
    'raw_buffer_load', 'raw_buffer_store',
    'raw_ptr_buffer_load', 'raw_ptr_buffer_store',
    'load_to_lds', 'global_load_lds',
    'make_buffer_rsrc',
    
    # Atomic operations
    'raw_buffer_atomic_fadd', 'raw_buffer_atomic_fmax',
    'raw_buffer_atomic_smax', 'raw_buffer_atomic_umin',
    'raw_ptr_buffer_atomic_fadd', 'raw_ptr_buffer_atomic_fmax',
    
    # Bit manipulation
    'mbcnt_lo', 'mbcnt_hi',
    
    # Scheduling and optimization
    's_setprio', 's_sleep',
    'sched_barrier', 'sched_group_barrier',
    'iglp_opt',
    
    # Type conversions
    'cvt_f32_bf8', 'cvt_f32_fp8',
    'cvt_pk_f32_bf8', 'cvt_pk_f32_fp8',

    # Copy atom types
    'CopyOpCDNA3BufferLDSTType',
    'BufferLDST', 'BufferLDST32b', 'BufferLDST64b', 'BufferLDST128b',

    # MMA atom types
    'MmaAtomCDNA3_MFMAType', 'MFMA',

    # Convenience wrappers
    'make_buffer_tensor',
]


# ── Wrappers that accept DSL Numeric args (fx.Int32, fx.Float32, etc.) ───────
# The ODS-generated ops require raw ir.Value. These wrappers call ir_value()
# on any DSL Numeric argument before forwarding to the underlying MLIR op.

def _to_ir(v):
    """Coerce DSL Numeric to ir.Value if needed."""
    if not isinstance(v, __import__('flydsl._mlir.ir', fromlist=['Value']).Value) and hasattr(v, 'ir_value'):
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
    return _op(_to_ir(rsrc), _to_ir(lds_ptr), _to_ir(size), _to_ir(voffset),
               _to_ir(soffset), _to_ir(offset), _to_ir(aux), **kw)
