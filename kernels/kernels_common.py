# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Common helpers shared by kernel modules.

Keep helper naming consistent with other kernel helpers (e.g. `mfma_preshuffle_pipeline.py`),
but this module is intentionally small and MLIR-dialect facing.
"""

from contextlib import contextmanager

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as _std_arith
from flydsl._mlir.dialects import builtin
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import gpu as _gpu
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import scf as _scf
from flydsl.expr import arith as _expr_arith
from flydsl.expr import buffer_ops, const_expr
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch, is_rdna_arch


def get_llvm_ptr(ptr, offset, dtype_bytes, ptr_type=None):
    """Build a global (address-space 1) ``!llvm.ptr`` at ``ptr + offset*dtype_bytes``.

    Shared home for the LLVM-ptr arithmetic used by atomic/global accesses
    (previously duplicated in hgemm_splitk.py and rmsnorm_kernel.py).
    """
    if ptr_type is None:
        ptr_type = ir.Type.parse("!llvm.ptr<1>")
    base_ptr = _fly.extract_aligned_pointer_as_index(ptr_type, ptr)
    base_ptr = _llvm.PtrToIntOp(T.i64, base_ptr).result
    byte_offset = _expr_arith.index_cast(T.i64, fx.Index(offset) * fx.Index(dtype_bytes))
    llvm_ptr = _llvm.AddOp(base_ptr, byte_offset, _llvm.IntegerOverflowFlags(0)).result
    llvm_ptr = _llvm.IntToPtrOp(ptr_type, llvm_ptr).result
    return llvm_ptr._value if const_expr(hasattr(llvm_ptr, "_value")) else llvm_ptr


@contextmanager
def _if_then(if_op, scf=None):
    """Context manager for SCF IfOp then-region across old/new Python APIs.

    Ensures the then block always ends with a YieldOp.
    The optional *scf* parameter is accepted for backward compatibility
    but ignored — the module-level import is used.
    """
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], _scf.YieldOp):
                _scf.YieldOp([])


_VALID_A_DTYPES = frozenset(("fp8", "fp16", "int8", "fp4"))
_VALID_B_DTYPES = frozenset(("fp8", "fp16", "int8", "int4", "fp4"))


def validate_moe_dtypes(a_dtype: str, b_dtype: str) -> None:
    """Validate a_dtype/b_dtype strings for mixed MoE kernels."""
    if a_dtype not in _VALID_A_DTYPES:
        raise ValueError(f"a_dtype must be one of {tuple(sorted(_VALID_A_DTYPES))}, got {a_dtype!r}")
    if b_dtype not in _VALID_B_DTYPES:
        raise ValueError(f"b_dtype must be one of {tuple(sorted(_VALID_B_DTYPES))}, got {b_dtype!r}")


def dtype_to_elem_type(dtype_str: str):
    """Map a dtype string to its FlyDSL numeric type.

    Supported: 'f32', 'f16', 'bf16', 'fp8' (OCP e4m3fn, not the fnuz variant).
    """
    if dtype_str == "f32":
        return fx.Float32
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    if dtype_str == "fp8":
        return fx.Float8E4M3FN
    raise ValueError(f"unsupported dtype: {dtype_str!r} (expected 'f32', 'f16', 'bf16', or 'fp8')")


def get_warp_size(arch=None):
    """Return the wavefront/warp size for the given GPU architecture.

    CDNA (gfx9xx) uses wave64, RDNA (gfx10xx/gfx11xx/gfx12xx) uses wave32.
    """
    if arch is None:
        arch = get_rocm_arch()
    return 32 if is_rdna_arch(arch) else 64


def _create_llvm_ptr(value, address_space: int = 1):
    value = buffer_ops._unwrap_value(value)
    if isinstance(value.type, ir.IndexType):
        i64_type = T.i64
        value = buffer_ops._unwrap_value(_std_arith.IndexCastOp(i64_type, value).result)
    ptr_type = ir.Type.parse(f"!llvm.ptr<{address_space}>")
    return _llvm.IntToPtrOp(ptr_type, value).result


def stream_ptr_to_async_token(stream_ptr_value, loc=None, ip=None):
    stream_llvm_ptr = _create_llvm_ptr(stream_ptr_value)

    async_token_type = _gpu.AsyncTokenType.get()
    cast_op = builtin.UnrealizedConversionCastOp([async_token_type], [stream_llvm_ptr], loc=loc, ip=ip)
    return cast_op.results[0]
