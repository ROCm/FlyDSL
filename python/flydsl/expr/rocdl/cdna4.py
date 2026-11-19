# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from ..._mlir import ir
from ..._mlir.dialects import llvm
from ..._mlir.dialects.fly_rocdl import (
    CopyOpCDNA4BufferLoadAsyncLDSType,
    CopyOpCDNA4GlobalLoadAsyncLDSType,
    CopyOpCDNA4LdsReadTransposeType,
    MmaOpCDNA4_MFMAScaleType,
)
from ..._mlir.extras import types as T
from ..numeric import (
    BFloat16,
    Boolean,
    Float4E2M1FN,
    Float6E2M3FN,
    Float6E3M2FN,
    Float8E4M3FN,
    Float8E5M2,
    Float16,
    Float32,
    Int8,
    Uint32,
)
from ..typing import BFloat16x2
from . import universal

# gfx950 has the gfx942 buffer copies and atomics, so their checked constructors are shared.
from .cdna3 import (
    BufferAtomic,
    BufferAtomicAdd,
    BufferAtomicMax,
    BufferAtomicMin,
    BufferCopy,
    BufferCopy8b,
    BufferCopy16b,
    BufferCopy32b,
    BufferCopy64b,
    BufferCopy128b,
    make_buffer_rsrc,
    s_waitcnt,
)
from .universal import (
    asyncmark,
    get_buffer_rsrc,
    make_buffer_ptr,
    make_buffer_tensor,
    wait_asyncmark,
)
from .utils import _check_mfma_shape, _check_numeric_type

__all__ = [
    # Atoms
    "MFMA",
    "MFMAScale",
    "MFMA_Scale",
    "BufferCopy",
    "BufferCopy8b",
    "BufferCopy16b",
    "BufferCopy32b",
    "BufferCopy64b",
    "BufferCopy128b",
    "BufferLoadAsyncLDS",
    "BufferLoadAsyncLDS32b",
    "BufferLoadAsyncLDS96b",
    "BufferLoadAsyncLDS128b",
    "GlobalLoadAsyncLDS",
    "GlobalLoadAsyncLDS32b",
    "GlobalLoadAsyncLDS96b",
    "GlobalLoadAsyncLDS128b",
    "LDSReadTrans",
    "LDSReadTrans4_64b",
    "LDSReadTrans8_64b",
    "LDSReadTrans6_96b",
    "LDSReadTrans16_64b",
    "BufferAtomic",
    "BufferAtomicAdd",
    "BufferAtomicMax",
    "BufferAtomicMin",
    "BufferAtomicPkAdd",
    # Functions
    "make_buffer_ptr",
    "make_buffer_rsrc",
    "make_buffer_tensor",
    "get_buffer_rsrc",
    "cvt_sr_f32_to_bf16",
    "s_waitcnt",
    "asyncmark",
    "wait_asyncmark",
]

# The A/B types and shapes the MFMA atom lowers to a gfx950 instruction.
_FP8 = (Float8E4M3FN, Float8E5M2)  # gfx950 reads FP8 / BF8 data as the OCP formats
_MFMA_SHAPES = {
    (Float32, Float32): ((32, 32, 2), (16, 16, 4)),
    (Float16, Float16): ((32, 32, 8), (16, 16, 16), (16, 16, 32), (32, 32, 16)),
    (BFloat16, BFloat16): ((32, 32, 8), (16, 16, 16), (16, 16, 32), (32, 32, 16)),
    (Int8, Int8): ((16, 16, 32), (32, 32, 16), (16, 16, 64), (32, 32, 32)),
    **{(a, b): ((16, 16, 32), (32, 32, 16)) for a in _FP8 for b in _FP8},
}
_F8F6F4 = (*_FP8, Float6E2M3FN, Float6E3M2FN, Float4E2M1FN)
_MFMA_SCALE_SHAPES = {(a, b): ((16, 16, 128), (32, 32, 64)) for a in _F8F6F4 for b in _F8F6F4}


def BufferAtomicPkAdd(val_type):
    """Create a CDNA3 packed buffer atomic add copy atom.

    Current atom state:
    - `soffset` (uniform `i32`), default zero
    """
    _check_numeric_type("BufferAtomicPkAdd on gfx950", val_type, (Float16, BFloat16))
    return universal.BufferAtomicPkAdd(val_type)


def MFMA(m, n, k, elem_ty_ab, elem_ty_b_or_acc=None, elem_ty_acc=None):
    elem_ty_b = elem_ty_ab if elem_ty_acc is None or elem_ty_b_or_acc is None else elem_ty_b_or_acc
    _check_mfma_shape("MFMA on gfx950", _MFMA_SHAPES, m, n, k, elem_ty_ab, elem_ty_b)
    return universal.MFMA(m, n, k, elem_ty_ab, elem_ty_b_or_acc, elem_ty_acc)


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
    - `soffset` (uniform `i32`), default zero
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

    An extra scale operand in an atom-call group (``[data, scale]``) overrides
    the corresponding ``scale_a`` or ``scale_b`` state for that call. If a
    group omits the scale operand, its scale is read from atom state.
    """
    # skip check ir.Type for backward compatibility.
    if not isinstance(elem_ty_a, ir.Type) and not isinstance(elem_ty_b, ir.Type):
        b = elem_ty_a if elem_ty_b is None else elem_ty_b
        _check_mfma_shape("MFMA_Scale on gfx950", _MFMA_SCALE_SHAPES, m, n, k, elem_ty_a, b)
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
