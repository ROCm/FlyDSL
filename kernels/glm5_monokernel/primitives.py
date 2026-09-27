# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""GLM-5-specific wrappers around low-level AMDGPU operations."""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import rocdl
from flydsl.expr.typing import T, as_ir_value

_UMAX_ASM = "\n".join(
    [
        "s_nop 1\nv_max_u32_dpp $0, $0, $0 " + control
        for control in (
            "row_shr:1 bound_ctrl:0",
            "row_shr:2 bound_ctrl:0",
            "row_shr:4 bound_ctrl:0",
            "row_shr:8 bound_ctrl:0",
            "row_bcast:15 row_mask:0xa",
            "row_bcast:31 row_mask:0xc",
        )
    ]
)


def spin_pause() -> None:
    """Keep tagged-mailbox polling loads inside the retry loop."""

    llvm.InlineAsmOp(None, [], "s_nop 0", "", has_side_effects=True)


def read_lane_i32(value, lane):
    """Read one i32 from ``lane`` while keeping IR conversion local."""

    return fx.Int32(rocdl.readlane(T.i32, fx.Int32(value), fx.Int32(lane)))


def write_lane_i32(value, lane, vector):
    """Write one i32 into ``lane`` of a wave-distributed value."""

    return fx.Int32(
        llvm.call_intrinsic(
            T.i32,
            "llvm.amdgcn.writelane.i32",
            [as_ir_value(fx.Int32(item)) for item in (value, lane, vector)],
            [],
            [],
        )
    )


def bpermute_i32(byte_offset, value):
    """Read an i32 VGPR value from the lane selected by a byte offset."""

    return fx.Int32(rocdl.ds_bpermute(T.i32, fx.Int32(byte_offset), fx.Int32(value)))


def mem_realtime():
    """Read the device-wide 64-bit realtime counter."""

    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))


def wave_umax(value):
    """Return an unsigned wave maximum through the tuned fused-DPP sequence."""

    result = llvm.InlineAsmOp(T.i32, [as_ir_value(fx.Int32(value))], _UMAX_ASM, "=v,0").result
    return read_lane_i32(result, 63)
