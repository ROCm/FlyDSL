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
from ..meta import dsl_loc_tracing
from . import cdna3 as cdna3
from . import cdna4 as cdna4
from . import cdna5 as cdna5
from . import rdna3 as rdna3
from . import rdna4 as rdna4
from .enum import SyncScope as SyncScope
from .universal import *

__all__ = [
    # Targets
    "cdna3",
    "cdna4",
    # "cdna5", unstable for now
    "rdna3",
    "rdna4",
    # Enums
    "SyncScope",
    # Re-exported from .universal
    "s_waitcnt",
    "asyncmark",
    "wait_asyncmark",
    "BufferCopy",
    "BufferCopy8b",
    "BufferCopy16b",
    "BufferCopy32b",
    "BufferCopy64b",
    "BufferCopy128b",
    "BufferCopyLDS",
    "BufferCopyLDS32b",
    "BufferCopyLDS64b",  # Deprecated: no 8-byte LDS DMA exists; use 32b / 128b
    "BufferCopyLDS128b",
    "BufferAtomic",
    "BufferAtomicAdd",
    "BufferAtomicMax",
    "BufferAtomicMin",
    "BufferAtomicPkAdd",
    "MFMA",
    "WMMA",
    "make_buffer_ptr",
    "make_buffer_tensor",
    "get_buffer_rsrc",
    # Operations
    "sched_mfma",
    "sched_vmem",
    "sched_dsrd",
    "sched_dswr",
]

# Keep references to ODS-generated builders so we can wrap them without losing access.
_ods_wmma_scale_f32_16x16x128_f8f6f4 = globals().get("wmma_scale_f32_16x16x128_f8f6f4", None)
_ods_wmma_scale_f32_32x16x128_f4 = globals().get("wmma_scale_f32_32x16x128_f4", None)
_ods_wmma_f32_16x16x128_fp8_fp8 = globals().get("wmma_f32_16x16x128_fp8_fp8", None)
_ods_wave_id = wave_id  # ODS: wave_id(res, ...) -> i32
_ods_cluster_workgroup_id_x = cluster_workgroup_id_x
_ods_cluster_workgroup_id_y = cluster_workgroup_id_y
_ods_cluster_workgroup_id_z = cluster_workgroup_id_z
_ods_cluster_load_async_to_lds_b8 = cluster_load_async_to_lds_b8
_ods_cluster_load_async_to_lds_b32 = cluster_load_async_to_lds_b32
_ods_cluster_load_async_to_lds_b64 = cluster_load_async_to_lds_b64
_ods_cluster_load_async_to_lds_b128 = cluster_load_async_to_lds_b128
_ods_s_wait_asynccnt = s_wait_asynccnt
_ods_readfirstlane = readfirstlane
_ods_ballot = ballot
_ods_readlane = readlane
_ods_mfma_f32_32x32x8f16 = globals().get("mfma_f32_32x32x8f16", None)
_ods_mfma_f32_32x32x8bf16_1k = globals().get("mfma_f32_32x32x8bf16_1k", None)
_ods_mfma_f32_32x32x16_f16 = globals().get("mfma_f32_32x32x16_f16", None)
_ods_mfma_f32_32x32x16_bf16 = globals().get("mfma_f32_32x32x16_bf16", None)
_ods_mfma_f32_16x16x16f16 = mfma_f32_16x16x16f16
_ods_mfma_f32_16x16x16bf16_1k = globals().get("mfma_f32_16x16x16bf16_1k", None)
_ods_mfma_f32_16x16x32_fp8_fp8 = mfma_f32_16x16x32_fp8_fp8
_ods_mfma_i32_16x16x32_i8 = mfma_i32_16x16x32_i8
_ods_mfma_f32_16x16x32_f16 = globals().get("mfma_f32_16x16x32_f16", None)
_ods_mfma_f32_16x16x32_bf16 = globals().get("mfma_f32_16x16x32_bf16", None)
_ods_mfma_scale_f32_16x16x128_f8f6f4 = globals().get("mfma_scale_f32_16x16x128_f8f6f4", None) or globals().get(
    "mfma_scale_f32_16x16x128_f8f6f4_", None
)
_ods_mfma_scale_f32_32x32x64_f8f6f4 = globals().get("mfma_scale_f32_32x32x64_f8f6f4", None) or globals().get(
    "mfma_scale_f32_32x32x64_f8f6f4_", None
)
mask_mfma = 0x008
mask_vmem_rd = 0x020
mask_dsrd = 0x100
mask_dswr = 0x200

_ods_sched_barrier = globals().get("sched_barrier")
_ods_sched_group_barrier = globals().get("sched_group_barrier")

_SCHED_MASK_INT_TO_KW = {
    0x000: "none",
    0x001: "non_mem_non_sideeffect",
    0x002: "valu",
    0x004: "salu",
    0x008: "mfma_wmma",
    0x010: "all_vmem",
    0x020: "vmem_read",
    0x040: "vmem_write",
    0x080: "all_ds",
    0x100: "ds_read",
    0x200: "ds_write",
    0x400: "transcendental",
    0x800: "ldsdma",
}


def _mask_to_attr(mask):
    """Convert an int or keyword mask to a SchedGroupMask attribute."""
    from ..._mlir import ir as _ir

    if isinstance(mask, _ir.Attribute):
        return mask
    if isinstance(mask, str):
        return _ir.Attribute.parse(f"#rocdl<sched_group_mask {mask}>")
    val = int(mask)
    if val == 0:
        return _ir.Attribute.parse("#rocdl<sched_group_mask none>")
    parts = [kw for bit, kw in _SCHED_MASK_INT_TO_KW.items() if bit and val & bit]
    if not parts:
        return _ir.Attribute.parse("#rocdl<sched_group_mask none>")
    return _ir.Attribute.parse(f"#rocdl<sched_group_mask {'|'.join(parts)}>")


@dsl_loc_tracing
def sched_barrier(mask, **kw):
    return _ods_sched_barrier(_mask_to_attr(mask), **kw)


@dsl_loc_tracing
def sched_group_barrier(mask, size, group_id, **kw):
    return _ods_sched_group_barrier(_mask_to_attr(mask), size, group_id, **kw)


@dsl_loc_tracing
def sched_mfma(cnt):
    sched_group_barrier(mask_mfma, cnt, 0)


@dsl_loc_tracing
def sched_vmem(cnt):
    sched_group_barrier(mask_vmem_rd, cnt, 0)


@dsl_loc_tracing
def sched_dsrd(cnt):
    sched_group_barrier(mask_dsrd, cnt, 0)


@dsl_loc_tracing
def sched_dswr(cnt):
    sched_group_barrier(mask_dswr, cnt, 0)


@dsl_loc_tracing
def fence_acquire(syncscope):
    """Emit an acquire fence for the selected AMDGPU memory scope."""
    from ..._mlir.dialects import llvm as _llvm

    _llvm.FenceOp(_llvm.AtomicOrdering.acquire, syncscope=syncscope)


@dsl_loc_tracing
def fence_release(syncscope):
    """Emit a release fence for the selected AMDGPU memory scope."""
    from ..._mlir.dialects import llvm as _llvm

    _llvm.FenceOp(_llvm.AtomicOrdering.release, syncscope=syncscope)


def _unwrap_mfma_operand(v):
    """MFMA operands are MLIR Values; some trailing operands are i32 flags.

    Accept Python ints and materialize them as i32 signless constants.
    """
    from flydsl._mlir.ir import IntegerType

    from .. import arith as _arith_ext

    if isinstance(v, int):
        return _arith_ext.unwrap(_arith_ext.constant(v, type=IntegerType.get_signless(32)))
    return _arith_ext.unwrap(v)


_BLGP_INT_TO_KW = {
    0: "none",
    1: "bcast_first_32",
    2: "bcast_second_32",
    4: "bcast_first_16",
    5: "bcast_second_16",
}


def _blgp_attr(val):
    """Convert an int blgp value to a ROCDL MFMAPermB attribute."""
    from ..._mlir import ir as _ir

    if isinstance(val, _ir.Attribute):
        return val
    kw = _BLGP_INT_TO_KW.get(int(val), "none")
    return _ir.Attribute.parse(f"#rocdl<mfma_perm_b {kw}>")


def _split_mfma_operands(operands):
    """Split [a, b, c, cbsz, abid, blgp] into (a, b, c) Values + (cbsz, abid, blgp) attrs."""
    a = _unwrap_mfma_operand(operands[0])
    b = _unwrap_mfma_operand(operands[1])
    c = _unwrap_mfma_operand(operands[2])
    cbsz = int(operands[3]) if len(operands) > 3 else 0
    abid = int(operands[4]) if len(operands) > 4 else 0
    blgp = _blgp_attr(operands[5]) if len(operands) > 5 else _blgp_attr(0)
    return a, b, c, cbsz, abid, blgp


@dsl_loc_tracing
def mfma_f32_32x32x8f16(result_type, operands):
    if _ods_mfma_f32_32x32x8f16 is None:
        raise AttributeError("ROCDL op not found: mfma_f32_32x32x8f16")
    a, b, c, cbsz, abid, blgp = _split_mfma_operands(operands)
    return _ods_mfma_f32_32x32x8f16(result_type, a, b, c, cbsz, abid, blgp).result


@dsl_loc_tracing
def mfma_f32_32x32x8bf16_1k(result_type, operands):
    if _ods_mfma_f32_32x32x8bf16_1k is None:
        raise AttributeError("ROCDL op not found: mfma_f32_32x32x8bf16_1k")
    a, b, c, cbsz, abid, blgp = _split_mfma_operands(operands)
    return _ods_mfma_f32_32x32x8bf16_1k(result_type, a, b, c, cbsz, abid, blgp).result


@dsl_loc_tracing
def mfma_f32_32x32x16_f16(result_type, operands):
    if _ods_mfma_f32_32x32x16_f16 is None:
        raise AttributeError("ROCDL op not found: mfma_f32_32x32x16_f16")
    a, b, c, cbsz, abid, blgp = _split_mfma_operands(operands)
    return _ods_mfma_f32_32x32x16_f16(result_type, a, b, c, cbsz, abid, blgp).result


@dsl_loc_tracing
def mfma_f32_32x32x16_bf16(result_type, operands):
    if _ods_mfma_f32_32x32x16_bf16 is None:
        raise AttributeError("ROCDL op not found: mfma_f32_32x32x16_bf16")
    a, b, c, cbsz, abid, blgp = _split_mfma_operands(operands)
    return _ods_mfma_f32_32x32x16_bf16(result_type, a, b, c, cbsz, abid, blgp).result


@dsl_loc_tracing
def mfma_f32_16x16x16f16(result_type, operands):
    a, b, c, cbsz, abid, blgp = _split_mfma_operands(operands)
    return _ods_mfma_f32_16x16x16f16(result_type, a, b, c, cbsz, abid, blgp).result


@dsl_loc_tracing
def mfma_f32_16x16x16bf16_1k(result_type, operands):
    if _ods_mfma_f32_16x16x16bf16_1k is None:
        raise AttributeError("ROCDL op not found: mfma_f32_16x16x16bf16_1k")
    a, b, c, cbsz, abid, blgp = _split_mfma_operands(operands)
    return _ods_mfma_f32_16x16x16bf16_1k(result_type, a, b, c, cbsz, abid, blgp).result


@dsl_loc_tracing
def mfma_f32_16x16x32_fp8_fp8(result_type, operands):
    a, b, c, cbsz, abid, blgp = _split_mfma_operands(operands)
    return _ods_mfma_f32_16x16x32_fp8_fp8(result_type, a, b, c, cbsz, abid, blgp).result


@dsl_loc_tracing
def mfma_i32_16x16x32_i8(result_type, operands):
    a, b, c, cbsz, abid, blgp = _split_mfma_operands(operands)
    return _ods_mfma_i32_16x16x32_i8(result_type, a, b, c, cbsz, abid, blgp).result


@dsl_loc_tracing
def mfma_f32_16x16x32_f16(result_type, operands):
    if _ods_mfma_f32_16x16x32_f16 is None:
        raise AttributeError("ROCDL op not found: mfma_f32_16x16x32_f16 (gfx950+)")
    a, b, c, cbsz, abid, blgp = _split_mfma_operands(operands)
    return _ods_mfma_f32_16x16x32_f16(result_type, a, b, c, cbsz, abid, blgp).result


@dsl_loc_tracing
def mfma_f32_16x16x32_bf16(result_type, operands):
    if _ods_mfma_f32_16x16x32_bf16 is None:
        raise AttributeError("ROCDL op not found: mfma_f32_16x16x32_bf16 (gfx950+)")
    a, b, c, cbsz, abid, blgp = _split_mfma_operands(operands)
    return _ods_mfma_f32_16x16x32_bf16(result_type, a, b, c, cbsz, abid, blgp).result


def _mfma_scale_call(ods_op, name, result_type, operands):
    """Shared body for the scaled-MFMA wrappers.

    ``cbsz``/``blgp`` are ROCDL MatrixFormat enum attributes, so every
    scale variant must route them through ``_wmma_fmt``.
    """
    if ods_op is None:
        raise AttributeError(f"ROCDL op not found: {name}(_)")
    a = _unwrap_mfma_operand(operands[0])
    b = _unwrap_mfma_operand(operands[1])
    c = _unwrap_mfma_operand(operands[2])
    cbsz = operands[3] if len(operands) > 3 else 0
    blgp = operands[4] if len(operands) > 4 else 0
    opselA = int(operands[5]) if len(operands) > 5 else 0
    scaleA = _unwrap_mfma_operand(operands[6]) if len(operands) > 6 else a
    opselB = int(operands[7]) if len(operands) > 7 else 0
    scaleB = _unwrap_mfma_operand(operands[8]) if len(operands) > 8 else b
    return ods_op(
        result_type,
        a,
        b,
        c,
        _wmma_fmt(cbsz),
        _wmma_fmt(blgp),
        opselA,
        scaleA,
        opselB,
        scaleB,
    ).result


@dsl_loc_tracing
def mfma_scale_f32_16x16x128_f8f6f4(result_type, operands):
    return _mfma_scale_call(
        _ods_mfma_scale_f32_16x16x128_f8f6f4,
        "mfma_scale_f32_16x16x128_f8f6f4",
        result_type,
        operands,
    )


@dsl_loc_tracing
def mfma_scale_f32_32x32x64_f8f6f4(result_type, operands):
    return _mfma_scale_call(
        _ods_mfma_scale_f32_32x32x64_f8f6f4,
        "mfma_scale_f32_32x32x64_f8f6f4",
        result_type,
        operands,
    )


_WMMA_FMT_INT_TO_KW = {
    0: "fp8_e4m3",
    1: "fp8_e5m2",
    2: "fp6_e2m3",
    3: "fp6_e3m2",
    4: "fp4_e2m1",
}
_WMMA_MODC_INT_TO_KW = {0: "none", 1: "neg", 2: "abs", 3: "neg_abs"}
_WMMA_SCALE_TYPE_INT_TO_KW = {0: "row0", 1: "row1"}
_WMMA_SCALE_FMT_INT_TO_KW = {0: "e8", 1: "e5m3", 2: "e4m3"}


def _wmma_attr(val, mapping, attr_name):
    """Convert an int to a parsed ROCDL enum attribute for WMMA ops."""
    from ..._mlir import ir as _ir

    if val is None or isinstance(val, _ir.Attribute):
        return val
    if isinstance(val, bool):
        return val
    kw = mapping.get(int(val))
    if kw is None:
        return val
    return _ir.Attribute.parse(f"#rocdl<{attr_name} {kw}>")


def _wmma_fmt(val):
    return _wmma_attr(val, _WMMA_FMT_INT_TO_KW, "matrix_format")


def _wmma_modc(val):
    return _wmma_attr(val, _WMMA_MODC_INT_TO_KW, "wmma_c_modifier")


def _wmma_scale_type(val):
    return _wmma_attr(val, _WMMA_SCALE_TYPE_INT_TO_KW, "wmma_matrix_scale")


def _wmma_scale_fmt(val):
    return _wmma_attr(val, _WMMA_SCALE_FMT_INT_TO_KW, "wmma_matrix_scale_format")


@dsl_loc_tracing
def wmma_scale_f32_16x16x128_f8f6f4(
    result_type,
    a,
    b,
    c,
    scaleA,
    scaleB,
    *,
    fmtA=4,
    fmtB=4,
    modC=0,
    scaleAType=0,
    fmtScaleA=0,
    scaleBType=0,
    fmtScaleB=0,
    reuseA=False,
    reuseB=False,
):
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
    a_v = _unwrap_mfma_operand(a)
    b_v = _unwrap_mfma_operand(b)
    c_v = _unwrap_mfma_operand(c)
    sA = _unwrap_mfma_operand(scaleA)
    sB = _unwrap_mfma_operand(scaleB)
    return _ods_wmma_scale_f32_16x16x128_f8f6f4(
        result_type,
        a_v,
        b_v,
        c_v,
        sA,
        sB,
        fmtA=_wmma_fmt(fmtA),
        fmtB=_wmma_fmt(fmtB),
        modC=_wmma_modc(modC),
        scaleAType=_wmma_scale_type(scaleAType),
        fmtScaleA=_wmma_scale_fmt(fmtScaleA),
        scaleBType=_wmma_scale_type(scaleBType),
        fmtScaleB=_wmma_scale_fmt(fmtScaleB),
        reuseA=reuseA,
        reuseB=reuseB,
    ).result


@dsl_loc_tracing
def wmma_scale_f32_32x16x128_f4(
    result_type,
    a,
    b,
    c,
    scaleA,
    scaleB,
    *,
    modC=0,
    scaleAType=0,
    fmtScaleA=0,
    scaleBType=0,
    fmtScaleB=0,
    reuseA=False,
    reuseB=False,
):
    """V_WMMA_SCALE_F32_32X16X128_F4 for gfx1250 (wave32).

    Operand types (wave32):
        a: vector<16xi32> (32x128 FP4 data)
        b: vector<8xi32>  (128x16 FP4 data)
        c: vector<16xf32> (32x16 FP32 accumulator)
        scaleA: i32 (A scale VGPR)
        scaleB: i32 (B scale VGPR)
    """
    if _ods_wmma_scale_f32_32x16x128_f4 is None:
        raise AttributeError("ROCDL op not found: wmma_scale_f32_32x16x128_f4")
    a_v = _unwrap_mfma_operand(a)
    b_v = _unwrap_mfma_operand(b)
    c_v = _unwrap_mfma_operand(c)
    sA = _unwrap_mfma_operand(scaleA)
    sB = _unwrap_mfma_operand(scaleB)
    return _ods_wmma_scale_f32_32x16x128_f4(
        result_type,
        a_v,
        b_v,
        c_v,
        sA,
        sB,
        modC=_wmma_modc(modC),
        scaleAType=_wmma_scale_type(scaleAType),
        fmtScaleA=_wmma_scale_fmt(fmtScaleA),
        scaleBType=_wmma_scale_type(scaleBType),
        fmtScaleB=_wmma_scale_fmt(fmtScaleB),
        reuseA=reuseA,
        reuseB=reuseB,
    ).result


@dsl_loc_tracing
def wmma_f32_16x16x128_fp8_fp8(result_type, a, b, c, *, modC=0, reuseA=False, reuseB=False):
    """Non-scale V_WMMA_F32_16X16X128 (E4M3) for gfx1250 (wave32).

    Operand types (wave32):
        a: vector<16xi32> (16x128 FP8/E4M3 data)
        b: vector<16xi32> (128x16 FP8/E4M3 data)
        c: vector<8xf32>  (16x16 FP32 accumulator)
    """
    if _ods_wmma_f32_16x16x128_fp8_fp8 is None:
        raise AttributeError("ROCDL op not found: wmma_f32_16x16x128_fp8_fp8")
    a_v = _unwrap_mfma_operand(a)
    b_v = _unwrap_mfma_operand(b)
    c_v = _unwrap_mfma_operand(c)
    return _ods_wmma_f32_16x16x128_fp8_fp8(
        result_type, a_v, b_v, c_v, modC=_wmma_modc(modC), reuseA=reuseA, reuseB=reuseB
    ).result


@dsl_loc_tracing
def wave_id():
    """Get wave-id-in-workgroup as SGPR (via TTMP8[29:25]).

    Returns:
        i32 value (SGPR) with the wave ID within the workgroup.
    """
    from ..._mlir import ir

    i32 = ir.IntegerType.get_signless(32)
    return _ods_wave_id(i32)


@dsl_loc_tracing
def cluster_workgroup_id_x():
    """Get workgroup position within cluster along X (SGPR, gfx1250)."""
    from ..._mlir import ir

    i32 = ir.IntegerType.get_signless(32)
    return _ods_cluster_workgroup_id_x(i32)


@dsl_loc_tracing
def cluster_workgroup_id_y():
    """Get workgroup position within cluster along Y (SGPR, gfx1250)."""
    from ..._mlir import ir

    i32 = ir.IntegerType.get_signless(32)
    return _ods_cluster_workgroup_id_y(i32)


@dsl_loc_tracing
def cluster_workgroup_id_z():
    """Get workgroup position within cluster along Z (SGPR, gfx1250)."""
    from ..._mlir import ir

    i32 = ir.IntegerType.get_signless(32)
    return _ods_cluster_workgroup_id_z(i32)


@dsl_loc_tracing
def cluster_load_async_to_lds(global_ptr, lds_ptr, size_bytes, offset=0, cpol=0, mask=None):
    """Per-lane cluster broadcast load: Global -> LDS with MCAST (gfx1250).

    Args:
        global_ptr: ``!llvm.ptr<1>`` -- global address space pointer.
        lds_ptr:    ``!llvm.ptr<3>`` -- LDS address space pointer.
        size_bytes: Load width: 1, 4, 8, or 16 bytes (selects b8/b32/b64/b128).
        offset:     Byte offset (int, default 0).
        cpol:       Cache policy (int, default 0).
        mask:       i32 workgroup_mask for MCAST broadcast. None means no mask.
    """
    _dispatch = {
        1: _ods_cluster_load_async_to_lds_b8,
        4: _ods_cluster_load_async_to_lds_b32,
        8: _ods_cluster_load_async_to_lds_b64,
        16: _ods_cluster_load_async_to_lds_b128,
    }
    fn = _dispatch.get(size_bytes)
    if fn is None:
        raise ValueError(f"cluster_load_async_to_lds: size_bytes must be 1, 4, 8, or 16, got {size_bytes}")
    if mask is None:
        from ..._mlir import ir
        from .. import arith as _arith

        mask = _arith.unwrap(_arith.constant(0, type=ir.IntegerType.get_signless(32)))
    fn(global_ptr, lds_ptr, offset, cpol, mask)


@dsl_loc_tracing
def disable_xdl_arb_stall():
    """Disable WMMA multicycle arbitration stall by setting SCHED_MODE bit 4."""
    from ..._mlir.dialects import llvm as _llvm
    from .. import arith as _arith
    from ..typing import T

    # hwreg encoding: ID=26(SCHED_MODE), Offset=4, Size=1 -> 282
    imm_val = _arith.unwrap(_arith.constant(282, type=T.i32))
    val_val = _arith.unwrap(_arith.constant(1, type=T.i32))

    _llvm.call_intrinsic(None, "llvm.amdgcn.s.setreg", [imm_val, val_val], [], [])


@dsl_loc_tracing
def s_wait_asynccnt(count=0):
    """Wait for outstanding async load/store operations (ASYNCcnt counter)."""
    _ods_s_wait_asynccnt(count)


@dsl_loc_tracing
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
    from ..._mlir import ir as _ir
    from ..._mlir.dialects import (
        llvm as _llvm,
    )
    from ..._mlir.dialects import (
        memref as _memref,
    )
    from ..._mlir.dialects import (
        rocdl as _rocdl,
    )
    from .. import arith as _arith
    from ..arith import _to_raw
    from ..typing import T
    from ..utils.arith import ArithValue as _AV

    lds_ptr_ty = _ir.Type.parse("!llvm.ptr<3>")
    raw_memref = _arith.unwrap(lds_memref)
    lds_base = _memref.extract_aligned_pointer_as_index(raw_memref)

    byte_off = _AV(_arith.unwrap(elem_offset, index=True)) * _arith.index(elem_bytes)
    total_byte_idx = _AV(lds_base) + byte_off
    addr_i32 = _to_raw(_arith.index_cast(T.i32, total_byte_idx))
    ptr_val = _llvm.inttoptr(lds_ptr_ty, addr_i32)

    return _rocdl.ds_load_tr16_b128(result_type, ptr_val)


# ── New high-level helpers from universal.py ──────────────────────────
from .cdna5 import *  # noqa: E402,F401,F403,I001
from .inline_asm import *  # noqa: E402,F401,F403,I001

# ── Wrappers: accept DSL Numeric args (fx.Int32, fx.Float32, etc.) ─────────
# ODS-generated ops require raw ir.Value. These wrappers auto-convert.


def _to_ir(v):
    """Coerce DSL Numeric to ir.Value if needed."""
    from ..._mlir import ir as _ir
    from .. import arith as _arith_ext

    if isinstance(v, int):
        return _arith_ext.unwrap(_arith_ext.constant(v, type=_ir.IntegerType.get_signless(32)))
    if isinstance(v, float):
        return _arith_ext.unwrap(_arith_ext.constant(v, type=_ir.F32Type.get()))
    if not isinstance(v, _ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


@dsl_loc_tracing
def raw_ptr_buffer_atomic_fadd(vdata, rsrc, offset, soffset, aux=None, **kw):
    from ..._mlir import ir as _ir
    from ..._mlir.dialects.rocdl import raw_ptr_buffer_atomic_fadd as _op

    vdata_ir = _to_ir(vdata)
    if aux is not None and not isinstance(aux, _ir.Attribute):
        if isinstance(aux, int):
            aux = _ir.IntegerAttr.get(_ir.IntegerType.get_signless(32), aux)
        else:
            aux = None
    return _op(vdata_ir.type, vdata_ir, _to_ir(rsrc), _to_ir(offset), _to_ir(soffset), aux=aux, **kw)


@dsl_loc_tracing
def raw_ptr_buffer_atomic_fmax(vdata, rsrc, offset, soffset, aux=None, **kw):
    from ..._mlir import ir as _ir
    from ..._mlir.dialects.rocdl import raw_ptr_buffer_atomic_fmax as _op

    vdata_ir = _to_ir(vdata)
    if aux is not None and not isinstance(aux, _ir.Attribute):
        if isinstance(aux, int):
            aux = _ir.IntegerAttr.get(_ir.IntegerType.get_signless(32), aux)
        else:
            aux = None
    return _op(vdata_ir.type, vdata_ir, _to_ir(rsrc), _to_ir(offset), _to_ir(soffset), aux=aux, **kw)


@dsl_loc_tracing
def cvt_pk_fp8_f32(res, src_a, src_b, old, word_sel, **kw):
    from ..._mlir.dialects.rocdl import cvt_pk_fp8_f32 as _op

    return _op(
        res=res,
        src_a=_to_ir(src_a),
        src_b=_to_ir(src_b),
        old=_to_ir(old),
        word_sel=word_sel,
        **kw,
    )


@dsl_loc_tracing
def cvt_pk_f32_fp8(res, src, word_sel, **kw):
    """ROCDL ``cvt_pk_f32_fp8``: unpack one i32 (4 packed fp8) into ``vector<2xf32>``.

    ``word_sel=False`` decodes the low half (fp8 elems 0,1); ``word_sel=True`` the
    high half (fp8 elems 2,3). A full v4f32 unpack requires both halves stitched
    via a shuffle.
    """
    from ..._mlir.dialects.rocdl import cvt_pk_f32_fp8 as _op

    return _op(res=res, src=_to_ir(src), word_sel=word_sel, **kw)


@dsl_loc_tracing
def cvt_scalef32_pk_f32_fp4(res, src, scale, src_sel_index, **kw):
    """ROCDL ``cvt_scalef32_pk_f32_fp4``: unpack 2 fp4 (from one i32 holding 8 packed
    fp4 elems) into ``vector<2xf32>``, multiplied by ``scale``.

    ``src_sel_index`` (Python int in ``[0,3]``) selects which fp4 pair within the
    i32 lane is decoded. A full v8f32 unpack requires 4 calls (sel=0..3) plus
    two-stage shuffle to stitch.
    """
    from ..._mlir.dialects.rocdl import cvt_scalef32_pk_f32_fp4 as _op

    return _op(res=res, src=_to_ir(src), scale=_to_ir(scale), src_sel_index=src_sel_index, **kw)


@dsl_loc_tracing
def cvt_scalef32_pk_bf16_fp4(res, src, scale, src_sel_index, **kw):
    """ROCDL ``cvt_scalef32_pk_bf16_fp4``: unpack 2 fp4 (from one i32 holding 8 packed
    fp4 elems) into ``vector<2xbf16>``, multiplied by ``scale``.

    Same operand shape as :func:`cvt_scalef32_pk_f32_fp4` but the destination is bf16
    (the mxfp4->bf16 upconvert used by the a16w4 MoE path). ``src_sel_index`` (Python
    int in ``[0,3]``) selects which fp4 pair within the i32 lane is decoded; a full
    v8bf16 unpack requires 4 calls (sel=0..3).
    """
    from ..._mlir.dialects.rocdl import cvt_scalef32_pk_bf16_fp4 as _op

    return _op(res=res, src=_to_ir(src), scale=_to_ir(scale), src_sel_index=src_sel_index, **kw)


@dsl_loc_tracing
def cvt_scalef32_pk_fp4_f32(res, old_vdst, src0, src1, scale, dst_sel_index, **kw):
    """ROCDL ``cvt_scalef32_pk_fp4_f32``: pack 2 fp32 into 2 fp4 and write them into
    slot ``dst_sel_index`` of the i32 lane ``old_vdst`` (other slots preserved).

    A full v8f32→i32 repack requires 4 calls (dst_sel=0..3) chaining ``old_vdst``
    so each call accumulates a different pair into the running i32 value.
    """
    from ..._mlir.dialects.rocdl import cvt_scalef32_pk_fp4_f32 as _op

    return _op(
        res=res,
        old_vdst=_to_ir(old_vdst),
        src0=_to_ir(src0),
        src1=_to_ir(src1),
        scale=_to_ir(scale),
        dst_sel_index=dst_sel_index,
        **kw,
    )


@dsl_loc_tracing
def rcp(res, arg, **kw):
    from ..._mlir.dialects.rocdl import rcp as _op

    return _op(res=res, arg=_to_ir(arg), **kw)


@dsl_loc_tracing
def perm_b32(src_hi, src_lo, sel, **kw):
    """Wrapper for ``llvm.amdgcn.perm`` returning one i32 lane value."""
    from ..._mlir.dialects import llvm as _llvm
    from ..typing import T

    return _llvm.call_intrinsic(
        T.i32,
        "llvm.amdgcn.perm",
        [_to_ir(src_hi), _to_ir(src_lo), _to_ir(sel)],
        [],
        [],
        **kw,
    )


@dsl_loc_tracing
def raw_ptr_buffer_load(res, rsrc, offset, soffset, aux=None, **kw):
    from ..._mlir import ir as _ir
    from ..._mlir.dialects.rocdl import raw_ptr_buffer_load as _op

    if aux is not None and not isinstance(aux, _ir.Attribute):
        if isinstance(aux, int):
            aux = _ir.IntegerAttr.get(_ir.IntegerType.get_signless(32), aux)
        else:
            aux = None
    return _op(
        res=res,
        rsrc=_to_ir(rsrc),
        offset=_to_ir(offset),
        soffset=_to_ir(soffset),
        aux=aux,
        **kw,
    )


@dsl_loc_tracing
def raw_ptr_buffer_load_lds(rsrc, lds_ptr, size, voffset, soffset, offset, aux=None, **kw):
    from ..._mlir import ir as _ir
    from ..._mlir.dialects.rocdl import raw_ptr_buffer_load_lds as _op

    if aux is not None and not isinstance(aux, _ir.Attribute):
        if isinstance(aux, int):
            aux = _ir.IntegerAttr.get(_ir.IntegerType.get_signless(32), aux)
        else:
            aux = None
    return _op(
        _to_ir(rsrc),
        _to_ir(lds_ptr),
        _to_ir(size),
        _to_ir(voffset),
        _to_ir(soffset),
        _to_ir(offset),
        aux=aux,
        **kw,
    )


@dsl_loc_tracing
def buffer_load_to_lds(rsrc, lds_ptr, voffset, size_bytes=4, soffset=0, offset=0, **kw):
    """Load ``size_bytes`` from a buffer resource into LDS.

    Simplified wrapper around :func:`raw_ptr_buffer_load_lds` with
    sensible defaults (``soffset=0``, ``offset=0``, ``aux=0``).
    Python int arguments are auto-materialised as i32 constants.
    """
    return raw_ptr_buffer_load_lds(rsrc, lds_ptr, size_bytes, voffset, soffset, offset, 0, **kw)


@dsl_loc_tracing
def tensor_load_to_lds(dgroup0, dgroup1, dgroup2, dgroup3, dgroup4, cache_policy=None, **kw):
    from ..._mlir import ir as _ir
    from ..._mlir.dialects.rocdl import tensor_load_to_lds as _op

    if cache_policy is not None and not isinstance(cache_policy, _ir.Attribute):
        if isinstance(cache_policy, int):
            cache_policy = _ir.IntegerAttr.get(_ir.IntegerType.get_signless(32), cache_policy)
        else:
            cache_policy = None
    return _op(dgroup0, dgroup1, dgroup2, dgroup3, dgroup4, cache_policy=cache_policy, **kw)


@dsl_loc_tracing
def tensor_store_from_lds(dgroup0, dgroup1, dgroup2, dgroup3, dgroup4, cache_policy=None, **kw):
    from ..._mlir import ir as _ir
    from ..._mlir.dialects.rocdl import tensor_store_from_lds as _op

    if cache_policy is not None and not isinstance(cache_policy, _ir.Attribute):
        if isinstance(cache_policy, int):
            cache_policy = _ir.IntegerAttr.get(_ir.IntegerType.get_signless(32), cache_policy)
        else:
            cache_policy = None
    return _op(dgroup0, dgroup1, dgroup2, dgroup3, dgroup4, cache_policy=cache_policy, **kw)


@dsl_loc_tracing
def ds_bpermute(res, index, src, **kw):
    from ..._mlir.dialects.rocdl import ds_bpermute as _op

    return _op(res=res, index=_to_ir(index), src=_to_ir(src), **kw)


@dsl_loc_tracing
def readfirstlane(res, src, **kw):
    return _ods_readfirstlane(res=res, src=_to_ir(src), **kw)


@dsl_loc_tracing
def ballot(res, pred, **kw):
    """Wrap ROCDL ``ballot``: coerce ``pred`` to ``i1`` if needed.

    ``res`` selects the lane-mask width (``i32`` on wave32, ``i64`` on wave64).
    """
    from ..._mlir.dialects import llvm as _llvm
    from ..._mlir.ir import IntegerType

    pred_v = _to_ir(pred)
    i1 = IntegerType.get_signless(1)
    if pred_v.type != i1:
        pred_v = _llvm.TruncOp(i1, pred_v).result
    return _ods_ballot(res=res, pred=pred_v, **kw)


@dsl_loc_tracing
def readlane(res, src, lane, **kw):
    """Wrap ROCDL ``readlane`` with ``_to_ir`` coercion (Python ``int`` ok for ``lane``)."""
    return _ods_readlane(res=res, src0=_to_ir(src), src1=_to_ir(lane), **kw)
