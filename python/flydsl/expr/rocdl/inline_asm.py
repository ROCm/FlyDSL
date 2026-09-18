# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""GCN/CDNA wrappers for ROCm GPU instructions the MLIR ROCDL dialect does not surface.

Two escape hatches, in order of preference:

* ``llvm.call_intrinsic`` when LLVM defines an ``llvm.amdgcn.*`` intrinsic for the
  instruction -- ``s_memrealtime`` and ``s_getreg_b32`` both do.  The backend keeps the
  semantics rather than treating the body as opaque text: for ``s_memrealtime``, an SMEM
  read, it places the required ``s_waitcnt lgkmcnt(0)`` at the first use of the value, so
  independent work fills the latency.  (``s_getreg_b32`` is SOPK and needs no wait at
  all; it gains the symbolic ``hwreg(...)`` operand and argument checking instead.)
* ``llvm.inline_asm`` when no intrinsic exists, or when one exists but cannot express the
  form actually used.  The block is opaque to the scheduler and the constraint string is
  hand-written, so this is the fallback, not the default.  ``v_cvt_pk_bf16_f32`` has no
  intrinsic at all.  ``int_amdgcn_cvt_off_f32_i4`` does exist, but takes only the i32
  source and cannot carry the SDWA ``src0_sel`` modifier that every in-tree call site
  passes (``byte_sel=``) to avoid an explicit shift -- swapping it in would be a codegen
  regression, not a like-for-like move.

TODO: v_cvt_pk_bf16_f32 can move once LLVM defines an intrinsic for it. v_cvt_off_f32_i4
needs its SDWA form expressible first -- an MLIR ROCDL op, or an intrinsic taking the
byte selector.
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


def _hwreg_immarg(reg_id: int, size: int):
    """Materialise ``s.getreg``'s immarg: ``id | (size - 1) << 11``, offset 0.

    The intrinsic declares the operand ``immarg``, so it must be a real constant rather
    than a runtime value.  Both ktrace registers start at bit 0, so the ``offset << 6``
    term is fixed at zero rather than exposed as an untested parameter; the encoding is
    the same one ``rocdl.sched_mode_expert`` spells out for ``s.setreg``.

    Out-of-range arguments are rejected instead of masked: a silently truncated id is
    still a well-formed selector for a *different* register, so it assembles and returns
    plausible garbage.
    """
    from .. import arith as _arith
    from ..typing import T

    if not 0 <= reg_id <= 0x3F:
        raise ValueError(f"hwreg id must fit 6 bits, got {reg_id}")
    if not 1 <= size <= 32:
        raise ValueError(f"hwreg field size must be in [1, 32], got {size}")
    return _arith.unwrap(_arith.constant(reg_id | ((size - 1) << 11), type=T.i32))


def _require_cdna_timestamp_arch():
    """s_memrealtime / HW_REG_XCC_ID exist on CDNA3 and CDNA4 only.

    Gates the three timestamp wrappers below. They no longer back ktrace -- the
    conversion pass emits the same intrinsics itself and enforces the same arch
    restriction (`lib/Conversion/FlyKtraceToROCDL`) -- so they are now reachable only
    from a kernel that calls them directly.

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
    ``s_waitcnt lgkmcnt(0)``.  Emitted as an intrinsic rather than inline asm, the backend
    places that wait itself -- at the first use of the value rather than immediately after
    the read -- so independent work can fill the SMEM latency.

    Note the effective granularity is coarser than one tick: the counter
    advances in steps of 4 (40 ns) on gfx950.

    :return: i64 timestamp
    """
    from ..._mlir import ir
    from ..._mlir.dialects import llvm as _llvm

    _require_cdna_timestamp_arch()
    return _llvm.call_intrinsic(ir.IntegerType.get_signless(64), "llvm.amdgcn.s.memrealtime", [], [], [])


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
    return _llvm.call_intrinsic(
        ir.IntegerType.get_signless(32),
        "llvm.amdgcn.s.getreg",
        [_hwreg_immarg(_HW_REG_HW_ID, 32)],
        [],
        [],
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
    return _llvm.call_intrinsic(
        ir.IntegerType.get_signless(32),
        "llvm.amdgcn.s.getreg",
        [_hwreg_immarg(_HW_REG_XCC_ID, 4)],
        [],
        [],
    )
