# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

from ..._mlir.dialects import rocdl as mlir_rocdl
from ..._mlir.dialects.fly import AtomicOp
from ..meta import dsl_loc_tracing
from ..numeric import (
    BFloat16,
    Float8E4M3FNUZ,
    Float8E5M2FNUZ,
    Float16,
    Float32,
    Float64,
    Int8,
    Int32,
    Int64,
    Uint32,
    Uint64,
)
from ..primitive import get_iter
from ..typing import Pointer, Tensor
from . import universal
from .universal import (
    asyncmark,
    get_buffer_rsrc,
    make_buffer_ptr,
    make_buffer_tensor,
    wait_asyncmark,
)
from .utils import _check_mfma_shape, _check_numeric_type, normalize_s_waitcnt_field

__all__ = [
    # Atoms
    "MFMA",
    "BufferCopy",
    "BufferCopy8b",
    "BufferCopy16b",
    "BufferCopy32b",
    "BufferCopy64b",
    "BufferCopy128b",
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
    "s_waitcnt",
    "asyncmark",
    "wait_asyncmark",
]

_BUFFER_COPY_BITS = (8, 16, 32, 64, 128)
_BUFFER_ATOMIC_TYPES = {
    AtomicOp.Add: (Float32, Float64),
    AtomicOp.Max: (Float64, Int32, Int64),
    AtomicOp.Min: (Uint32, Uint64),
}
_FP8 = (Float8E4M3FNUZ, Float8E5M2FNUZ)  # gfx942 reads FP8 / BF8 data as the FNUZ formats
_MFMA_SHAPES = {
    (Float32, Float32): ((32, 32, 2), (16, 16, 4)),
    (Float16, Float16): ((32, 32, 8), (16, 16, 16)),
    (BFloat16, BFloat16): ((32, 32, 8), (16, 16, 16)),
    (Int8, Int8): ((16, 16, 32), (32, 32, 16)),
    **{(a, b): ((16, 16, 32), (32, 32, 16)) for a in _FP8 for b in _FP8},
}


def BufferCopy(bit_size, cache_modifier=0):
    """Create a CDNA3 buffer copy atom (cache_modifier: 0=cached, 2=nt).

    Current atom state:
    - `soffset` (uniform `i32`), default zero
    """
    if bit_size not in _BUFFER_COPY_BITS:
        raise ValueError(f"BufferCopy has a buffer load / store for {_BUFFER_COPY_BITS} bits, got {bit_size}")
    return universal.BufferCopy(bit_size, cache_modifier)


BufferCopy8b = lambda cache_modifier=0: BufferCopy(8, cache_modifier)
BufferCopy16b = lambda cache_modifier=0: BufferCopy(16, cache_modifier)
BufferCopy32b = lambda cache_modifier=0: BufferCopy(32, cache_modifier)
BufferCopy64b = lambda cache_modifier=0: BufferCopy(64, cache_modifier)
BufferCopy128b = lambda cache_modifier=0: BufferCopy(128, cache_modifier)


def BufferAtomic(atomic_op, val_type):
    """Create a CDNA3 buffer atomic copy atom.

    Current atom state:
    - `soffset` (uniform `i32`), default zero
    """
    op = AtomicOp(int(atomic_op))
    _check_numeric_type(f"BufferAtomic({op.name})", val_type, _BUFFER_ATOMIC_TYPES.get(op, ()))
    return universal.BufferAtomic(atomic_op, val_type)


BufferAtomicAdd = lambda val_type: BufferAtomic(AtomicOp.Add, val_type)
BufferAtomicMax = lambda val_type: BufferAtomic(AtomicOp.Max, val_type)
BufferAtomicMin = lambda val_type: BufferAtomic(AtomicOp.Min, val_type)


def BufferAtomicPkAdd(val_type):
    """Create a CDNA3 packed buffer atomic add copy atom.

    Current atom state:
    - `soffset` (uniform `i32`), default zero
    """
    _check_numeric_type("BufferAtomicPkAdd on gfx942", val_type, (Float16,))
    return universal.BufferAtomicPkAdd(val_type)


def MFMA(m, n, k, elem_ty_ab, elem_ty_b_or_acc=None, elem_ty_acc=None):
    elem_ty_b = elem_ty_ab if elem_ty_acc is None or elem_ty_b_or_acc is None else elem_ty_b_or_acc
    _check_mfma_shape("MFMA on gfx942", _MFMA_SHAPES, m, n, k, elem_ty_ab, elem_ty_b)
    return universal.MFMA(m, n, k, elem_ty_ab, elem_ty_b_or_acc, elem_ty_acc)


def s_waitcnt(vmcnt=None, lgkmcnt=None, expcnt=None):
    """
    Emit a CDNA3/GFX942-encoded s_waitcnt operation.

    vmcnt: split across [3:0] and [15:14]
    expcnt: [6:4]
    lgkmcnt: [11:8]
    """
    vmcnt = normalize_s_waitcnt_field("vmcnt", vmcnt, 63)
    expcnt = normalize_s_waitcnt_field("expcnt", expcnt, 7)
    lgkmcnt = normalize_s_waitcnt_field("lgkmcnt", lgkmcnt, 15)
    encoded_vmcnt = (vmcnt & 0xF) | ((vmcnt & 0x30) << 10)
    return mlir_rocdl.s_waitcnt(encoded_vmcnt | (lgkmcnt << 8) | (expcnt << 4))


@dsl_loc_tracing
def make_buffer_rsrc(
    src: Pointer | Tensor,
    *,
    max_size: bool = True,
    num_records_bytes=None,
):
    """Construct a raw ROCDL buffer resource (``!llvm.ptr<8>``) from a global
    pointer or tensor.

    ``max_size`` and ``num_records_bytes`` mean what they mean for
    :func:`make_buffer_tensor`; a pointer has no layout to derive the byte
    count from, so ``max_size=False`` then requires ``num_records_bytes``.
    """
    if isinstance(src, Tensor):
        return get_buffer_rsrc(get_iter(make_buffer_tensor(src, max_size, num_records_bytes=num_records_bytes)))
    if not max_size and num_records_bytes is None:
        raise ValueError("make_buffer_rsrc: max_size=False with a pointer requires num_records_bytes")
    return get_buffer_rsrc(make_buffer_ptr(src, num_records_bytes=num_records_bytes))
