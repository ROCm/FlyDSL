# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""GCN/CDNA inline assembly wrappers for ROCm GPU instructions.

These emit LLVM inline asm ops for instructions that have no corresponding
MLIR ROCDL dialect op yet.  The underlying ISA instructions are defined in
LLVM's AMDGPU backend (VOP1Instructions.td / VOP3Instructions.td) but the
MLIR ROCDLOps.td tablegen does not surface them.

TODO: Remove these inline asm wrappers once upstream MLIR adds proper ROCDL
dialect ops for v_cvt_off_f32_i4, v_cvt_pk_bf16_f32, s_memrealtime and s_getreg_b32.
"""

from ..meta import dsl_loc_tracing


def _to_ir(v):
    """Coerce DSL Numeric to ir.Value if needed."""
    from ..._mlir import ir as _ir

    if not isinstance(v, _ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


@dsl_loc_tracing
def cvt_off_f32_i4(src_i32, byte_sel=None):
    """gfx9xx: v_cvt_off_f32_i4 — convert low nibble (bits[3:0]) to f32.

    With byte_sel=0..3, uses SDWA to select the byte before conversion,
    avoiding an explicit shift.  byte_sel=None uses the plain VOP1 form.
    """
    from ..._mlir import ir
    from ..._mlir.dialects import llvm as _llvm

    if byte_sel is not None:
        sel = ["BYTE_0", "BYTE_1", "BYTE_2", "BYTE_3"][int(byte_sel)]
        return _llvm.inline_asm(
            ir.F32Type.get(),
            [_to_ir(src_i32)],
            f"v_cvt_off_f32_i4_sdwa $0, $1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:{sel}",
            "=v,v",
            has_side_effects=False,
        )
    return _llvm.inline_asm(
        ir.F32Type.get(),
        [_to_ir(src_i32)],
        "v_cvt_off_f32_i4 $0, $1",
        "=v,v",
        has_side_effects=False,
    )


@dsl_loc_tracing
def cvt_pk_bf16_f32(src_a_f32, src_b_f32):
    """gfx950: v_cvt_pk_bf16_f32 vdst, vsrc0, vsrc1.

    Pack two f32 values into 2xbf16 in i32.
    dst[15:0] = bf16(src_a), dst[31:16] = bf16(src_b).
    """
    from ..._mlir import ir
    from ..._mlir.dialects import llvm as _llvm

    return _llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [_to_ir(src_a_f32), _to_ir(src_b_f32)],
        "v_cvt_pk_bf16_f32 $0, $1, $2",
        "=v,v,v",
        has_side_effects=False,
    )


# HW register ids and field layout, from AMD's shipped HIP header
# (/opt/rocm/include/hip/amd_detail/amd_device_functions.h).  Only the ids are
# needed here; callers decode the fields host-side.
_HW_REG_HW_ID = 4  # WAVE_ID[3:0] SIMD_ID[5:4] PIPE_ID[7:6] CU_ID[11:8] SH_ID[12] SE_ID[14:13]
_HW_REG_XCC_ID = 20  # XCC_ID[3:0]; gfx942/gfx950 only, absent on gfx90a


def _require_cdna_timestamp_arch():
    """s_memrealtime / HW_REG_XCC_ID exist on CDNA3 and CDNA4 only.

    Gates on the COMPILE TARGET, not the host GPU.  ``ARCH`` overrides the target
    (``compiler/backends/rocm.py``: ``env.compile.arch or get_rocm_arch()``) while
    ``get_rocm_arch()`` only consults ``FLYDSL_GPU_ARCH``/hardware, so gating on the
    latter would let ``ARCH=<rdna>`` cross-compiles through.  That matters because
    ``s_getreg`` does not error on RDNA -- ``hwreg(4)`` and ``hwreg(20, 0, 4)``
    silently retarget to WAVE_STATE_PRIV / WAVE_SCRATCH_BASE_LO, which is worse than
    a kernel that fails to assemble.
    """
    from ...runtime.device import get_rocm_arch
    from ...utils import env

    arch = env.compile.arch or get_rocm_arch()
    if not arch.startswith(("gfx942", "gfx950")):
        raise ValueError(
            f"in-kernel timestamps are not supported on target arch {arch!r}; "
            "supported: gfx942 (CDNA3) and gfx950 (CDNA4). "
            "HW_REG_SHADER_CYCLES and s_sendmsg_rtn are not available on these targets, "
            "and s_memrealtime is absent on RDNA."
        )
    return arch


@dsl_loc_tracing
def s_memrealtime():
    """gfx942/gfx950: s_memrealtime — 64-bit constant-rate wall clock.

    Reads REF_CLK (100 MHz on MI300/MI350, i.e. 10 ns per tick), which is
    invariant under DVFS.  Prefer this over s_memtime, whose shader-clock
    domain varies with clock throttling.

    s_memrealtime is an SMEM operation, so the result is not readable until
    ``s_waitcnt lgkmcnt(0)``; the wait is emitted inside the asm string so the
    value is usable immediately.

    Note the effective granularity is coarser than one tick: the counter
    advances in steps of 4 (40 ns) on gfx950.

    :return: i64 timestamp
    """
    from ..._mlir import ir
    from ..._mlir.dialects import llvm as _llvm

    _require_cdna_timestamp_arch()
    return _llvm.inline_asm(
        ir.IntegerType.get_signless(64),
        [],
        "s_memrealtime $0\ns_waitcnt lgkmcnt(0)",
        "=s",
        has_side_effects=True,
    )


@dsl_loc_tracing
def s_getreg_hw_id():
    """gfx942/gfx950: s_getreg_b32 hwreg(HW_REG_HW_ID) — raw wave placement.

    Returns the whole register so every field stays available to the host:
    WAVE_ID[3:0], SIMD_ID[5:4], PIPE_ID[7:6], CU_ID[11:8], SH_ID[12],
    SE_ID[14:13] (2 bits on gfx942/gfx950, unlike the 3 bits used on
    gfx908/gfx90a).

    The value is wave-invariant for the lifetime of a wave -- waves are
    CU-resident on CDNA -- so callers may read it once at kernel entry.

    :return: i32 raw HW_ID
    """
    from ..._mlir import ir
    from ..._mlir.dialects import llvm as _llvm

    _require_cdna_timestamp_arch()
    return _llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [],
        f"s_getreg_b32 $0, hwreg({_HW_REG_HW_ID})",
        "=s",
        has_side_effects=True,
    )


@dsl_loc_tracing
def s_getreg_xcc_id():
    """gfx942/gfx950: s_getreg_b32 hwreg(HW_REG_XCC_ID, 0, 4) — XCD index.

    Identifies which accelerator complex die the wave is running on; MI355X
    has 8.  The hardware extracts the field, so no shift/mask is needed.

    Like HW_ID this is wave-invariant and may be read once at kernel entry.

    :return: i32 XCC_ID in [0, 15]
    """
    from ..._mlir import ir
    from ..._mlir.dialects import llvm as _llvm

    _require_cdna_timestamp_arch()
    return _llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [],
        f"s_getreg_b32 $0, hwreg({_HW_REG_XCC_ID}, 0, 4)",
        "=s",
        has_side_effects=True,
    )
