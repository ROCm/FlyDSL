# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FP8 flex-attention forward on the FlyDSL layout API (gfx950).

FP8 (e4m3fn) variant of flex_attention_layout_gfx950.py. Uses the CDNA4
scaled MFMA (32x32x64) for QK and PV GEMMs, ds_read_tr8_b64 for V transpose
reads, and per-tensor Q/K/V descales. Output is bf16.

Key differences from bf16 path:
- QK GEMM: mfma_scale_f32_32x32x64_f8f6f4 (2 MFMAs for D=128 vs 8 bf16 MFMAs)
- PV GEMM: same scaled MFMA with fp8 P (packed from f32 via cvt_pk_fp8_f32)
- V reads: LDSReadTrans8_64b (8 fp8/lane vs 4 bf16/lane)
- Q scale: deferred (fused into c_logit_scale after QK GEMM, cannot pre-scale fp8)
- in_data_bytes=1 (half the LDS and DMA of bf16)
- Output always bf16 with V descale folded into O normalization
"""

from typing import Optional
import warnings
import math

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl, arith
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch

from kernels.attention.flex_attention_layout_gfx950 import (
    FlexAttnParam,
    FlexMod,
    CausalMask,
    SlidingWindowMask,
    AlibiScore,
    CompositeMod,
    _build_mod,
    _elem_dtype,
    _hw_exp2,
    _permlane32_reduce,
    _mfma_acc,
    _FM,
    _LOG2E,
    _size_scalar,
    make_flex_attn_param,
    make_flex_attn_kernel_name,
    GFX950_WAVE_SIZE,
    GFX950_DMA_BYTES,
    FLEX_DTYPE_FP8,
    MASK_NONE,
    MASK_CAUSAL,
    MASK_SLIDING_WINDOW,
    SCORE_NONE,
    SCORE_ALIBI,
)

try:
    from flydsl.expr.rocdl.universal import make_buffer_ptr as _make_buffer_ptr
except ImportError:
    from flydsl.expr import buffer_ops
    from flydsl._mlir.dialects.fly_rocdl import TargetAddressSpace

    def _make_buffer_ptr(ptr, num_records_bytes=None):
        if num_records_bytes is None:
            num_records_bytes = fx.Int64(0xFFFFFFFF)
        elif not isinstance(num_records_bytes, fx.Int64):
            num_records_bytes = fx.Int64(num_records_bytes)
        buf_ptr_ty = fx.PointerType.get(
            elem_ty=ptr.element_type.ir_type,
            address_space=TargetAddressSpace.BufferDesc,
            alignment=ptr.alignment,
        )
        return fx.make_ptr(
            buf_ptr_ty,
            [
                ptr,
                fx.Int16(0).ir_value(),
                num_records_bytes.ir_value(),
                fx.Int32(buffer_ops._get_buffer_flags()).ir_value(),
            ],
        )


def _to_elem(val, dtype):
    """Convert a DSL value to *dtype* (bf16/f16/fp8), returning the DSL wrapper."""
    return dtype(val)


# ── FP8 MFMA helper ──────────────────────────────────────────────────────────
# Unit E8M0 scales: all exponent bytes = 127 = scale 1.0.
_FP8_UNIT_SCALE = 0x7F7F7F7F


def _mfma_scale_fp8(a_i32x8, b_i32x8, c_v16f32):
    """Single 32x32x64 scaled MFMA: C += A × B with unit E8M0 scales."""
    from flydsl.expr.typing import Vector as Vec
    v16f32_type = c_v16f32.type
    return rocdl.mfma_scale_f32_32x32x64_f8f6f4(
        v16f32_type,
        [
            fx.as_mlir_value(a_i32x8) if hasattr(a_i32x8, 'ir_value') else a_i32x8,
            fx.as_mlir_value(b_i32x8) if hasattr(b_i32x8, 'ir_value') else b_i32x8,
            fx.as_mlir_value(c_v16f32) if hasattr(c_v16f32, 'ir_value') else c_v16f32,
            0,  # cbsz (A block scale format)
            0,  # blgp (B block scale format)
            0,  # neg (negate flags)
            fx.Int32(_FP8_UNIT_SCALE).ir_value(),  # A scale
            0,  # A scale_hi
            fx.Int32(_FP8_UNIT_SCALE).ir_value(),  # B scale
        ],
    )


# ── FP8 P packing ────────────────────────────────────────────────────────────

def _pack_fp8_i32x8(f32_vals):
    """Pack 32 f32 values into i32x8 (32 fp8 e4m3fn values).

    Uses cvt_pk_fp8_f32 which packs 2 f32 → 2 fp8 bytes at a time:
    - Call with word_selector=0 to fill low 2 bytes of i32
    - Call with word_selector=1 to fill high 2 bytes
    Result: 4 fp8 per i32 word, 8 words = 32 fp8.
    """
    from flydsl._mlir.dialects import llvm
    from flydsl.expr.typing import Vector as Vec
    words = []
    for g in range(8):
        base = g * 4
        c0 = llvm.mlir_poison(T.i32)
        w = rocdl.cvt_pk_fp8_f32(T.i32, fx.Float32(f32_vals[base]).ir_value(),
                                  fx.Float32(f32_vals[base + 1]).ir_value(), c0, 0)
        w = rocdl.cvt_pk_fp8_f32(T.i32, fx.Float32(f32_vals[base + 2]).ir_value(),
                                  fx.Float32(f32_vals[base + 3]).ir_value(), w, 1)
        words.append(fx.Int32(w))
    return Vec.from_elements(words, fx.Int32).ir_value()


# TODO: FP8 kernel implementation
# This file is a work-in-progress. The kernel body will include:
# - FP8 QK GEMM via _mfma_scale_fp8
# - FP8 V reads via LDSReadTrans8_64b
# - FP8 P packing via _pack_p_fp8
# - Deferred Q/K scaling (c_logit_scale)
# - bf16 O output with V descale
