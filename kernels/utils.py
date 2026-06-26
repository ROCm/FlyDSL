# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Common low-level helpers shared across kernels.

Hosts integer-arithmetic helpers (pow2 div/rem, cdiv) and the generic
global-memory load API used by the paged-attention decode kernels.  Keep
additions here small and broadly reusable.
"""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, buffer_ops, const_expr, rocdl
from flydsl.expr.typing import T


def _rcp_f32(value):
    return rocdl.rcp(T.f32, value)


def _exp2_amdgcn_scalar(scalar_value):
    """Direct ``llvm.amdgcn.exp2.f32`` intrinsic call on one f32 scalar.

    The default ``fly_math.exp2`` lowering routes through ``__ocml_exp2_f32``,
    which (for full IEEE range/subnormal correctness) expands at codegen time
    into ``v_exp_f32 + v_ldexp_f32`` per element.  The amdgcn intrinsic compiles
    to a single ``v_exp_f32``, matching what Gluon emits for its softmax.
    Skipping ldexp is safe only when the operand is pre-clamped to the
    fast-range (as PA softmax does via `safe_qk_max`/`safe_partition_max`).
    """
    raw = (
        arith.unwrap(scalar_value)
        if hasattr(scalar_value, "ir_value") or hasattr(scalar_value, "type")
        else scalar_value
    )
    f32_ty = ir.F32Type.get()
    return llvm.call_intrinsic(f32_ty, "llvm.amdgcn.exp2.f32", [raw], [], [])


def _exp2_f32_fast(value):
    """Compute 2^value (elementwise for vectors) via the amdgcn intrinsic,
    avoiding the ``v_exp_f32 + v_ldexp_f32`` pair OCML lowering produces.
    Requires pre-clamped inputs (see :func:`_exp2_amdgcn_scalar`)."""
    from flydsl._mlir.dialects import vector as _vector_dialect

    raw = arith.unwrap(value) if hasattr(value, "ir_value") or hasattr(value, "type") else value
    ty = raw.type
    if isinstance(ty, ir.VectorType):
        n = ty.shape[0]
        elems = []
        for i in range(n):
            scalar = _vector_dialect.extract(raw, static_position=[i], dynamic_position=[])
            elems.append(_exp2_amdgcn_scalar(scalar))
        return _vector_dialect.from_elements(ty, elems)
    return _exp2_amdgcn_scalar(raw)


def _cdiv(numer: int, denom: int) -> int:
    return (numer + denom - 1) // denom


def _pow2_shift(value: int) -> int:
    assert value > 0 and (value & (value - 1)) == 0
    return value.bit_length() - 1


def _is_pow2(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _udiv_pow2(value, divisor: int):
    return value >> fx.Int32(_pow2_shift(divisor))


def _urem_pow2(value, divisor: int):
    return value & fx.Int32(divisor - 1)


def _udiv_const(value, divisor: int):
    if const_expr(_is_pow2(divisor)):
        return _udiv_pow2(value, divisor)
    return value // fx.Int32(divisor)


def _urem_const(value, divisor: int):
    if const_expr(_is_pow2(divisor)):
        return _urem_pow2(value, divisor)
    return value % fx.Int32(divisor)


def _maxnumf(a, b):
    """Non-NaN-propagating max, equivalent to ``a.maximumf(b)`` for non-NaN
    inputs but lowers to a single ``v_max_f32`` instead of the
    ``v_max_f32 + v_cmp_o_f32 + s_nop 1 + v_cndmask`` chain that
    ``arith.maximumf`` emits for IEEE 754 NaN-propagation semantics.

    Safe for PA softmax: inputs are either finite or -inf (from masking),
    never NaN.  Each call site saves ~3 instructions + a 1-cycle VCC->VALU
    s_nop hazard in the cross-warp max chain.
    """
    return type(a)(arith.maxnumf(arith.unwrap(a), arith.unwrap(b)))


def _unflatten_k(k_flat, qkhe_loop: int = 2):
    # k_flat carries ``tloop * qkhe_loop * 2`` i64 scalars; recover tloop from
    # its length so this helper needs no geometry constant.
    n = qkhe_loop * 2
    return [[k_flat[td * n + j] for j in range(n)] for td in range(len(k_flat) // n)]


def extract_global_ptr(tensor):
    """Extract the raw ``!llvm.ptr<1>`` global pointer from a tensor argument."""
    from flydsl._mlir.dialects import fly as _fly

    raw = tensor.ir_value() if hasattr(tensor, "ir_value") and not isinstance(tensor, ir.Value) else tensor
    ptr_type = ir.Type.parse("!llvm.ptr<1>")
    return _fly.extract_aligned_pointer_as_index(ptr_type, raw)


def global_ptr_from_addr(addr_i64):
    """Make a global (addrspace-1) ``!llvm.ptr<1>`` from a raw i64 device address.

    The raw-pointer-kernargs counterpart of :func:`extract_global_ptr`: use it
    when the kernel argument is a bare ``fx.Int64`` ``data_ptr()`` instead of an
    ``fx.Tensor`` memref.  Result is interchangeable with ``extract_global_ptr``
    (e.g. feeds :func:`global_load` / ``buffer_ops.get_element_ptr``).
    """
    raw = addr_i64.ir_value() if hasattr(addr_i64, "ir_value") and not isinstance(addr_i64, ir.Value) else addr_i64
    return llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), raw).result


def global_load(global_ptr, byte_offset, result_type, *, alignment):
    """Load ``result_type`` from global memory at ``byte_offset`` bytes from ``global_ptr``.

    This is the generic primitive; ``global_load_i64x2`` / ``global_load_i32``
    are thin convenience wrappers for the common widths.

    Args:
        global_ptr: raw global pointer (see :func:`extract_global_ptr`).
        byte_offset: offset in bytes (any int-like; wrapped to i64).
        result_type: MLIR type of the loaded value (e.g. ``T.i64x2``, ``T.i32``).
        alignment: load alignment in bytes.
    """
    ptr = buffer_ops.get_element_ptr(global_ptr, byte_offset=fx.Int64(byte_offset), elem_type=T.i8)
    return llvm.LoadOp(result_type, ptr, alignment=alignment).result


def global_load_i64x2(global_ptr, byte_offset_i64):
    """Aligned 128-bit (``i64x2``) global load at a byte offset."""
    return global_load(global_ptr, byte_offset_i64, T.i64x2, alignment=16)


def global_load_i32(global_ptr, elem_offset_i32):
    """Aligned 32-bit (``i32``) global load at an *element* (4-byte) offset."""
    return global_load(global_ptr, fx.Int64(elem_offset_i32) * fx.Int64(4), T.i32, alignment=4)


def _ptr8_to_v4i32(ptr8_val):
    """Convert a ``!llvm.ptr addrspace(8)`` buffer descriptor to ``<4 x i32>``
    via ``ptrtoint(i128)`` + ``bitcast``.  Both forms are 128-bit type-puns of
    the same SGPR-resident descriptor — the LLVM backend emits zero
    instructions for this conversion (the descriptor stays in SGPRs)."""
    from flydsl.expr.rocdl import _to_ir

    i128_ty = ir.IntegerType.get_signless(128)
    v4i32_ty = ir.VectorType.get([4], ir.IntegerType.get_signless(32))
    i128_val = llvm.ptrtoint(i128_ty, _to_ir(ptr8_val))
    return llvm.bitcast(v4i32_ty, i128_val)


def buffer_load(rsrc, soffset_i32, vec_width=4, *, is_scalar=False, dtype=None, cache_modifier=0):
    """AMD buffer load, scalar (``s_buffer_load``) or vector (``buffer_load``).

    ``is_scalar`` selects the hardware path:

    * ``is_scalar=False`` (default) — vector-memory load (``buffer_load_dword[xN]``),
      one result per lane.  Delegates to :func:`flydsl.expr.buffer_ops.buffer_load`;
      ``soffset_i32`` is interpreted as the per-lane *element* offset.
      ``dtype`` selects the element type.

    * ``is_scalar=True`` — scalar load (``s_buffer_load_dword[x4]``) via the
      ``llvm.amdgcn.s.buffer.load`` intrinsic, landing the result directly in an
      SGPR shared across all lanes.  Requires ``soffset_i32`` to be
      wave-uniform and expressed in *bytes*.  Saves the ``vmcnt(0)`` drain +
      ``readfirstlane`` the VMEM path imposes (frees VMEM queue slots and keeps
      the result off the VGPRs).  Only ``vec_width`` 1 (``i32``) and 4 (``<4 x i32>``)
      are supported; ``dtype`` is ignored (the load operates on dwords).

    Args:
        rsrc: buffer resource descriptor (``!llvm.ptr<8>``).
        soffset_i32: per-lane element offset (vector path) or wave-uniform byte
            offset (scalar path, ``is_scalar=True``).
        vec_width: number of 32-bit words / elements to load.
        is_scalar: route through ``s_buffer_load`` instead of ``buffer_load``.
        dtype: element type for the vector path (defaults to f32).
        cache_modifier: cache-policy / aux flags (0 for default).
    """
    if not is_scalar:
        return buffer_ops.buffer_load(
            rsrc,
            soffset_i32,
            vec_width=vec_width,
            dtype=dtype,
            cache_modifier=cache_modifier,
        )

    from flydsl.expr.rocdl import _to_ir

    i32_ty = ir.IntegerType.get_signless(32)
    if vec_width == 1:
        result_type = i32_ty
        suffix = "i32"
    elif vec_width == 4:
        result_type = ir.VectorType.get([4], i32_ty)
        suffix = "v4i32"
    else:
        raise ValueError(f"buffer_load(is_scalar=True): unsupported vec_width={vec_width}")

    rsrc_v4 = _ptr8_to_v4i32(rsrc)
    cache_policy = arith.constant(cache_modifier, type=T.i32)
    return llvm.call_intrinsic(
        result_type,
        f"llvm.amdgcn.s.buffer.load.{suffix}",
        [
            _to_ir(rsrc_v4),
            _to_ir(soffset_i32),
            _to_ir(cache_policy),
        ],
        [],
        [],
    )
