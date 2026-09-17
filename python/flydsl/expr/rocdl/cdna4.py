# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from ..._mlir.dialects import llvm
from ..._mlir.dialects.fly_rocdl import (
    CopyOpCDNA4BufferLoadAsyncLDSType,
    CopyOpCDNA4GlobalLoadAsyncLDSType,
    CopyOpCDNA4LdsReadTransposeType,
    MmaOpCDNA4_MFMAScaleType,
)
from ..._mlir.extras import types as T
from ..numeric import Boolean, Float32, Uint32
from ..typing import BFloat16x2

__all__ = [
    "BufferLoadAsyncLDS",
    "BufferLoadAsyncLDS32b",
    "BufferLoadAsyncLDS96b",
    "BufferLoadAsyncLDS128b",
    "GlobalLoadAsyncLDS",
    "GlobalLoadAsyncLDS32b",
    "GlobalLoadAsyncLDS96b",
    "GlobalLoadAsyncLDS128b",
    "LDSReadTrans",
    "MFMAScale",
    "MFMA_Scale",
    "cvt_sr_f32_to_bf16",
]


def LDSReadTrans(trans_granularity, bit_size):
    """Create a GFX950 LDS read-transpose copy atom (ds_read_tr series)."""
    return CopyOpCDNA4LdsReadTransposeType.get(trans_granularity, bit_size)


LDSReadTrans4_64b = lambda: CopyOpCDNA4LdsReadTransposeType.get(4, 64)
LDSReadTrans8_64b = lambda: CopyOpCDNA4LdsReadTransposeType.get(8, 64)
LDSReadTrans6_96b = lambda: CopyOpCDNA4LdsReadTransposeType.get(6, 96)
LDSReadTrans16_64b = lambda: CopyOpCDNA4LdsReadTransposeType.get(16, 64)


def BufferLoadAsyncLDS(bit_size):
    """Create a CDNA4 async buffer-to-LDS copy atom.

    Same BufferDesc -> Shared direction and state as the synchronous
    :func:`flydsl.expr.rocdl.BufferCopyLDS`, but emits the async LDS DMA: the
    backend does not insert the ``vmcnt`` wait before the staged LDS data is
    read. Bracket the copies with ``fx.rocdl.asyncmark()`` and wait with
    ``fx.rocdl.wait_asyncmark(n)``.

    ``bit_size`` must be 32, 96 or 128.

    Current atom state:
    - `soffset` (`i32`), default zero
    - `imm_offset` (const `i32`), default zero
    """
    return CopyOpCDNA4BufferLoadAsyncLDSType.get(bit_size)


BufferLoadAsyncLDS32b = lambda: CopyOpCDNA4BufferLoadAsyncLDSType.get(32)
BufferLoadAsyncLDS96b = lambda: CopyOpCDNA4BufferLoadAsyncLDSType.get(96)
BufferLoadAsyncLDS128b = lambda: CopyOpCDNA4BufferLoadAsyncLDSType.get(128)


def GlobalLoadAsyncLDS(bit_size):
    """Create a CDNA4 async global-to-LDS copy atom.

    Completion tracking is the same: ``fx.rocdl.asyncmark()`` / ``fx.rocdl.wait_asyncmark(n)``.

    ``bit_size`` must be 32, 96 or 128, same as :func:`BufferLoadAsyncLDS`.
    """
    return CopyOpCDNA4GlobalLoadAsyncLDSType.get(bit_size)


GlobalLoadAsyncLDS32b = lambda: CopyOpCDNA4GlobalLoadAsyncLDSType.get(32)
GlobalLoadAsyncLDS96b = lambda: CopyOpCDNA4GlobalLoadAsyncLDSType.get(96)
GlobalLoadAsyncLDS128b = lambda: CopyOpCDNA4GlobalLoadAsyncLDSType.get(128)


def MFMA_Scale(m, n, k, elem_ty_a, elem_ty_b=None, elem_ty_acc=None, *, opsel_a=0, opsel_b=0):
    """Create a CDNA4 scaled MFMA atom (mfma.scale.f32.*.f8f6f4).

    Current atom state:
    - `scale_a` (`i32`), default zero
    - `scale_b` (`i32`), default zero
    """
    ty_a = elem_ty_a.ir_type if hasattr(elem_ty_a, "ir_type") else elem_ty_a
    if elem_ty_b is None:
        ty_b = ty_a
    else:
        ty_b = elem_ty_b.ir_type if hasattr(elem_ty_b, "ir_type") else elem_ty_b
    if elem_ty_acc is None:
        ty_acc = T.f32()
    else:
        ty_acc = elem_ty_acc.ir_type if hasattr(elem_ty_acc, "ir_type") else elem_ty_acc
    return MmaOpCDNA4_MFMAScaleType.get(m, n, k, ty_a, ty_b, ty_acc, opsel_a=opsel_a, opsel_b=opsel_b)


MFMAScale = MFMA_Scale


def cvt_sr_f32_to_bf16(x, rand):
    """Convert f32 to bf16 with the gfx950 stochastic-rounding instruction.

    ``rand`` supplies 32 random bits. The conversion preserves the native AMD
    non-saturating behavior.
    """
    # The result is a tied 2xbf16: word_sel picks the half that gets written and
    # the other half keeps the old value.
    return BFloat16x2(
        llvm.call_intrinsic(
            BFloat16x2.ir_type,
            "llvm.amdgcn.cvt.sr.bf16.f32",
            [
                llvm.mlir_undef(BFloat16x2.ir_type),
                Float32(x).ir_value(),
                Uint32(rand).ir_value(),
                Boolean(0).ir_value(),  # word_sel
            ],
            [],
            [],
        )
    )[0]
