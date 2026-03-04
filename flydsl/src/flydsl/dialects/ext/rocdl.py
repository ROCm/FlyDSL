"""ROCDL dialect extension for ROCm/AMD GPU programming.

This module provides access to ROCm-specific GPU operations including:
- Thread/block/grid identifiers and dimensions
- Synchronization primitives (barriers, wait operations)
- Matrix multiplication acceleration (MFMA, WMMA, SMFMAC)
- Data movement and shuffle operations
- Atomic operations
- Type conversion operations

Example:
    >>> from flydsl.dialects.ext import rocdl
    >>> tid_x = rocdl.workitem_id_x()
    >>> rocdl.barrier()
"""

from _mlir.dialects.rocdl import *  # noqa: F401,F403

# Keep references to ODS-generated builders so we can wrap them without losing access.
_ods_mfma_f32_16x16x16f16 = mfma_f32_16x16x16f16
_ods_mfma_f32_16x16x16bf16_1k = globals().get("mfma_f32_16x16x16bf16_1k", None)
_ods_mfma_f32_16x16x32_fp8_fp8 = mfma_f32_16x16x32_fp8_fp8
_ods_mfma_i32_16x16x32_i8 = mfma_i32_16x16x32_i8
_ods_mfma_scale_f32_16x16x128_f8f6f4 = globals().get(
    "mfma_scale_f32_16x16x128_f8f6f4", None
) or globals().get("mfma_scale_f32_16x16x128_f8f6f4_", None)
_ods_readlane = readlane
_ods_readfirstlane = readfirstlane
_ods_ds_swizzle = ds_swizzle
_ods_raw_ptr_buffer_atomic_fadd = raw_ptr_buffer_atomic_fadd

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
    from _mlir.ir import IntegerType
    from . import arith as _arith_ext

    if isinstance(v, int):
        return _arith_ext.constant(v, type=IntegerType.get_signless(32), loc=loc)._value
    return _arith_ext.unwrap(v, loc=loc)


def mfma_f32_16x16x16f16_op(result_type, operands, *, loc=None, ip=None):
    """Return the op view (original behavior)."""
    ops = [_unwrap_mfma_operand(v, loc=loc) for v in operands]
    return _ods_mfma_f32_16x16x16f16(result_type, ops, loc=loc, ip=ip)


def mfma_f32_16x16x16f16(result_type, operands, *, loc=None, ip=None):
    """Return the op result directly (no `.result` needed at call sites)."""
    return mfma_f32_16x16x16f16_op(result_type, operands, loc=loc, ip=ip).result


# for bf16 version mfma
def mfma_f32_16x16x16bf16_1k_op(result_type, operands, *, loc=None, ip=None):
    """Return the op view (original behavior)."""
    if _ods_mfma_f32_16x16x16bf16_1k is None:
        raise AttributeError("ROCDL op not found: mfma_f32_16x16x16bf16_1k")
    ops = [_unwrap_mfma_operand(v, loc=loc) for v in operands]
    return _ods_mfma_f32_16x16x16bf16_1k(result_type, ops, loc=loc, ip=ip)


def mfma_f32_16x16x16bf16_1k(result_type, operands, *, loc=None, ip=None):
    """Return the op result directly (no `.result` needed at call sites)."""
    return mfma_f32_16x16x16bf16_1k_op(result_type, operands, loc=loc, ip=ip).result


def mfma_f32_16x16x32_fp8_fp8_op(result_type, operands, *, loc=None, ip=None):
    """Return the op view (original behavior)."""
    ops = [_unwrap_mfma_operand(v, loc=loc) for v in operands]
    return _ods_mfma_f32_16x16x32_fp8_fp8(result_type, ops, loc=loc, ip=ip)


def mfma_f32_16x16x32_fp8_fp8(result_type, operands, *, loc=None, ip=None):
    """Return the op result directly (no `.result` needed at call sites)."""
    return mfma_f32_16x16x32_fp8_fp8_op(result_type, operands, loc=loc, ip=ip).result


def mfma_i32_16x16x32_i8_op(result_type, operands, *, loc=None, ip=None):
    """Return the op view (original behavior)."""
    ops = [_unwrap_mfma_operand(v, loc=loc) for v in operands]
    return _ods_mfma_i32_16x16x32_i8(result_type, ops, loc=loc, ip=ip)


def mfma_i32_16x16x32_i8(result_type, operands, *, loc=None, ip=None):
    """Return the op result directly (no `.result` needed at call sites)."""
    return mfma_i32_16x16x32_i8_op(result_type, operands, loc=loc, ip=ip).result


def mfma_scale_f32_16x16x128_f8f6f4_op(result_type, operands, *, loc=None, ip=None):
    """Return the op view (original behavior)."""
    if _ods_mfma_scale_f32_16x16x128_f8f6f4 is None:
        raise AttributeError("ROCDL op not found: mfma_scale_f32_16x16x128_f8f6f4(_)")
    ops = [_unwrap_mfma_operand(v, loc=loc) for v in operands]
    return _ods_mfma_scale_f32_16x16x128_f8f6f4(result_type, ops, loc=loc, ip=ip)


def mfma_scale_f32_16x16x128_f8f6f4(result_type, operands, *, loc=None, ip=None):
    """Return the op result directly (no `.result` needed at call sites)."""
    return mfma_scale_f32_16x16x128_f8f6f4_op(
        result_type, operands, loc=loc, ip=ip
    ).result


# ---------------------------------------------------------------------------
# WMMA wrappers  (Wave Matrix Multiply-Accumulate – RDNA3/RDNA4)
#
# WMMA operands are [A, B, C] – all MLIR Values, no integer flags.
# For IU variants the operand list is [A_sign, A, B_sign, B, C, clamp].
# For OPSEL variants (f16->f16, bf16->bf16) the list is [A, B, C, op_sel].
# ---------------------------------------------------------------------------


def _unwrap_wmma_operand(v, *, loc=None):
    """Accept Python ints (for flags like op_sel/clamp/signed) and ArithValue wrappers."""
    from _mlir.ir import IntegerType
    from . import arith as _arith_ext

    if isinstance(v, bool):
        return _arith_ext.constant(
            int(v), type=IntegerType.get_signless(1), loc=loc
        )._value
    if isinstance(v, int):
        return _arith_ext.constant(v, type=IntegerType.get_signless(32), loc=loc)._value
    return _arith_ext.unwrap(v, loc=loc)


# --- f32 output variants ---


def wmma_f32_16x16x16_f16_op(result_type, operands, *, loc=None, ip=None):
    ops = [_unwrap_wmma_operand(v, loc=loc) for v in operands]
    return _ods_wmma_f32_16x16x16_f16(result_type, ops, loc=loc, ip=ip)


def wmma_f32_16x16x16_f16(result_type, operands, *, loc=None, ip=None):
    """WMMA f16->f32, 16x16x16. Operands: [A, B, C]. Returns Value."""
    return wmma_f32_16x16x16_f16_op(result_type, operands, loc=loc, ip=ip).result


def wmma_f32_16x16x16_bf16_op(result_type, operands, *, loc=None, ip=None):
    ops = [_unwrap_wmma_operand(v, loc=loc) for v in operands]
    return _ods_wmma_f32_16x16x16_bf16(result_type, ops, loc=loc, ip=ip)


def wmma_f32_16x16x16_bf16(result_type, operands, *, loc=None, ip=None):
    """WMMA bf16->f32, 16x16x16. Operands: [A, B, C]. Returns Value."""
    return wmma_f32_16x16x16_bf16_op(result_type, operands, loc=loc, ip=ip).result


# --- fp8 variants (gfx12 / RDNA4 only) ---


def wmma_f32_16x16x16_fp8_fp8_op(result_type, operands, *, loc=None, ip=None):
    if _ods_wmma_f32_16x16x16_fp8_fp8 is None:
        raise AttributeError("ROCDL op not found: wmma_f32_16x16x16_fp8_fp8")
    ops = [_unwrap_wmma_operand(v, loc=loc) for v in operands]
    return _ods_wmma_f32_16x16x16_fp8_fp8(result_type, ops, loc=loc, ip=ip)


def wmma_f32_16x16x16_fp8_fp8(result_type, operands, *, loc=None, ip=None):
    """WMMA fp8->f32, 16x16x16 (gfx12). Operands: [A, B, C]. Returns Value."""
    return wmma_f32_16x16x16_fp8_fp8_op(result_type, operands, loc=loc, ip=ip).result


def wmma_f32_16x16x16_fp8_bf8_op(result_type, operands, *, loc=None, ip=None):
    if _ods_wmma_f32_16x16x16_fp8_bf8 is None:
        raise AttributeError("ROCDL op not found: wmma_f32_16x16x16_fp8_bf8")
    ops = [_unwrap_wmma_operand(v, loc=loc) for v in operands]
    return _ods_wmma_f32_16x16x16_fp8_bf8(result_type, ops, loc=loc, ip=ip)


def wmma_f32_16x16x16_fp8_bf8(result_type, operands, *, loc=None, ip=None):
    """WMMA fp8+bf8->f32, 16x16x16 (gfx12). Operands: [A, B, C]. Returns Value."""
    return wmma_f32_16x16x16_fp8_bf8_op(result_type, operands, loc=loc, ip=ip).result


def wmma_f32_16x16x16_bf8_fp8_op(result_type, operands, *, loc=None, ip=None):
    if _ods_wmma_f32_16x16x16_bf8_fp8 is None:
        raise AttributeError("ROCDL op not found: wmma_f32_16x16x16_bf8_fp8")
    ops = [_unwrap_wmma_operand(v, loc=loc) for v in operands]
    return _ods_wmma_f32_16x16x16_bf8_fp8(result_type, ops, loc=loc, ip=ip)


def wmma_f32_16x16x16_bf8_fp8(result_type, operands, *, loc=None, ip=None):
    """WMMA bf8+fp8->f32, 16x16x16 (gfx12). Operands: [A, B, C]. Returns Value."""
    return wmma_f32_16x16x16_bf8_fp8_op(result_type, operands, loc=loc, ip=ip).result


def wmma_f32_16x16x16_bf8_bf8_op(result_type, operands, *, loc=None, ip=None):
    if _ods_wmma_f32_16x16x16_bf8_bf8 is None:
        raise AttributeError("ROCDL op not found: wmma_f32_16x16x16_bf8_bf8")
    ops = [_unwrap_wmma_operand(v, loc=loc) for v in operands]
    return _ods_wmma_f32_16x16x16_bf8_bf8(result_type, ops, loc=loc, ip=ip)


def wmma_f32_16x16x16_bf8_bf8(result_type, operands, *, loc=None, ip=None):
    """WMMA bf8->f32, 16x16x16 (gfx12). Operands: [A, B, C]. Returns Value."""
    return wmma_f32_16x16x16_bf8_bf8_op(result_type, operands, loc=loc, ip=ip).result


# --- f16/bf16 output variants (OPSEL: operands include op_sel flag) ---


def wmma_f16_16x16x16_f16_op(result_type, operands, *, loc=None, ip=None):
    ops = [_unwrap_wmma_operand(v, loc=loc) for v in operands]
    return _ods_wmma_f16_16x16x16_f16(result_type, ops, loc=loc, ip=ip)


def wmma_f16_16x16x16_f16(result_type, operands, *, loc=None, ip=None):
    """WMMA f16->f16, 16x16x16. Operands: [A, B, C, op_sel]. Returns Value."""
    return wmma_f16_16x16x16_f16_op(result_type, operands, loc=loc, ip=ip).result


def wmma_bf16_16x16x16_bf16_op(result_type, operands, *, loc=None, ip=None):
    ops = [_unwrap_wmma_operand(v, loc=loc) for v in operands]
    return _ods_wmma_bf16_16x16x16_bf16(result_type, ops, loc=loc, ip=ip)


def wmma_bf16_16x16x16_bf16(result_type, operands, *, loc=None, ip=None):
    """WMMA bf16->bf16, 16x16x16. Operands: [A, B, C, op_sel]. Returns Value."""
    return wmma_bf16_16x16x16_bf16_op(result_type, operands, loc=loc, ip=ip).result


# --- Integer variants (IU: operands include sign flags and clamp) ---


def wmma_i32_16x16x16_iu8_op(result_type, operands, *, loc=None, ip=None):
    ops = [_unwrap_wmma_operand(v, loc=loc) for v in operands]
    return _ods_wmma_i32_16x16x16_iu8(result_type, ops, loc=loc, ip=ip)


def wmma_i32_16x16x16_iu8(result_type, operands, *, loc=None, ip=None):
    """WMMA int8->i32, 16x16x16. Operands: [A_sign, A, B_sign, B, C, clamp]. Returns Value."""
    return wmma_i32_16x16x16_iu8_op(result_type, operands, loc=loc, ip=ip).result


def wmma_i32_16x16x16_iu4_op(result_type, operands, *, loc=None, ip=None):
    ops = [_unwrap_wmma_operand(v, loc=loc) for v in operands]
    return _ods_wmma_i32_16x16x16_iu4(result_type, ops, loc=loc, ip=ip)


def wmma_i32_16x16x16_iu4(result_type, operands, *, loc=None, ip=None):
    """WMMA int4->i32, 16x16x16. Operands: [A_sign, A, B_sign, B, C, clamp]. Returns Value."""
    return wmma_i32_16x16x16_iu4_op(result_type, operands, loc=loc, ip=ip).result


def wmma_i32_16x16x32_iu4_op(result_type, operands, *, loc=None, ip=None):
    if _ods_wmma_i32_16x16x32_iu4 is None:
        raise AttributeError("ROCDL op not found: wmma_i32_16x16x32_iu4")
    ops = [_unwrap_wmma_operand(v, loc=loc) for v in operands]
    return _ods_wmma_i32_16x16x32_iu4(result_type, ops, loc=loc, ip=ip)


def wmma_i32_16x16x32_iu4(result_type, operands, *, loc=None, ip=None):
    """WMMA int4->i32, 16x16x32 (gfx12). Operands: [A_sign, A, B_sign, B, C, clamp]. Returns Value."""
    return wmma_i32_16x16x32_iu4_op(result_type, operands, loc=loc, ip=ip).result


def readlane(result_type, src, lane_id, *, loc=None, ip=None):
    """Lane read that accepts ArithValue / wrappers."""
    from . import arith as _arith_ext

    return _ods_readlane(
        result_type, _arith_ext.unwrap(src), _arith_ext.unwrap(lane_id), loc=loc, ip=ip
    )


def readfirstlane(result_type, src, *, loc=None, ip=None):
    """Read-firstlane that accepts ArithValue / wrappers."""
    from . import arith as _arith_ext

    return _ods_readfirstlane(result_type, _arith_ext.unwrap(src), loc=loc, ip=ip)


def ds_swizzle(result_type, src, offset, *, loc=None, ip=None):
    """DS swizzle that accepts ArithValue / wrappers."""
    from . import arith as _arith_ext

    return _ods_ds_swizzle(
        result_type, _arith_ext.unwrap(src), _arith_ext.unwrap(offset), loc=loc, ip=ip
    )


def raw_ptr_buffer_atomic_fadd(
    val, rsrc, voffset, soffset, cache, *, loc=None, ip=None
):
    """Atomic fadd that accepts `ArithValue` / wrappers (no explicit `arith.unwrap(...)` needed).

    Signature intentionally matches the underlying ODS builder:
      (val, rsrc, voffset, soffset, cache)
    """
    from . import arith as _arith_ext

    return _ods_raw_ptr_buffer_atomic_fadd(
        _arith_ext.unwrap(val),
        _arith_ext.unwrap(rsrc),
        _arith_ext.unwrap(voffset),
        _arith_ext.unwrap(soffset),
        _arith_ext.unwrap(cache),
        loc=loc,
        ip=ip,
    )


# Keep raw ODS builders available (rare: for tests that want the op object).
_mfma_f32_16x16x16f16_ods = _ods_mfma_f32_16x16x16f16
_mfma_f32_16x16x32_fp8_fp8_ods = _ods_mfma_f32_16x16x32_fp8_fp8

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
    # Raw-op constructors (return op view) for WMMA
    "wmma_f32_16x16x16_f16_op",
    "wmma_f32_16x16x16_bf16_op",
    "wmma_f16_16x16x16_f16_op",
    "wmma_bf16_16x16x16_bf16_op",
    "wmma_i32_16x16x16_iu8_op",
    "wmma_i32_16x16x16_iu4_op",
    "wmma_f32_16x16x16_fp8_fp8_op",
    "wmma_f32_16x16x16_fp8_bf8_op",
    "wmma_f32_16x16x16_bf8_fp8_op",
    "wmma_f32_16x16x16_bf8_bf8_op",
    "wmma_i32_16x16x32_iu4_op",
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
]
