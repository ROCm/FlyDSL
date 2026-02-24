"""Compatibility layer for data-movement helpers in the flyc environment.

Layout algebra ops (make_layout, crd2idx, ...) are forwarded to flydsl.expr.
``swizzle_xor16`` is implemented with pure arithmetic.
``copy`` generates vector load/store MLIR ops for efficient data movement.
"""

from typing import List, Optional, Union, Tuple

from .._mlir.ir import (
    Type,
    Value,
    Location,
    InsertionPoint,
    IndexType,
    IntegerAttr,
    IntegerType,
    MemRefType,
)
from .._mlir.dialects import memref, arith, scf, gpu, vector, math, llvm
from .._mlir.extras import types as T


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_location(loc: Optional[Location] = None) -> Location:
    """Get location, using current location if none provided."""
    try:
        if loc is not None and str(loc) == "loc(unknown)":
            loc = None
    except Exception:
        pass
    if loc is None:
        try:
            from flydsl._mlir.ir import Location as _Loc
            loc = _Loc.current
        except Exception:
            loc = None
        try:
            if loc is not None and str(loc) == "loc(unknown)":
                loc = None
        except Exception:
            pass
        if loc is None:
            try:
                loc = Location.current
            except Exception:
                loc = None
        if loc is None:
            loc = Location.unknown()
    return loc


def _unwrap_value(v):
    """Unwrap ArithValue or other value wrappers to get underlying MLIR Value."""
    if isinstance(v, int):
        from .._mlir.dialects import arith
        from .._mlir.ir import IndexType, IntegerAttr
        loc = _get_location(None)
        op = arith.ConstantOp(IndexType.get(), IntegerAttr.get(IndexType.get(), v), loc=loc)
        return _unwrap_value(op.result)
    try:
        internal = object.__getattribute__(v, "_value")
        return _unwrap_value(internal)
    except AttributeError:
        pass
    if hasattr(v, 'value') and callable(getattr(type(v).value, 'fget', None)):
        return v.value
    elif hasattr(v, '_value'):
        return v._value
    else:
        return v


def _get_insertion_point(ip: Optional[InsertionPoint] = None) -> InsertionPoint:
    """Get insertion point, using current if none provided."""
    if ip is None:
        return InsertionPoint.current
    return ip


def _try_get_constant_index(v: Value) -> Optional[int]:
    """Best-effort extract an int from an index-typed Value.

    Supports arith.constant and simple arithmetic chains (addi, subi, muli).
    """

    def _get_owner_op(val: Value):
        try:
            owner = val.owner
        except Exception:
            return None, None
        op = getattr(owner, "operation", owner)
        return owner, op

    def _const_from_op(owner, op) -> Optional[int]:
        try:
            if getattr(op, "name", None) == "arith.constant":
                attrs = getattr(op, "attributes", None)
                if attrs is None:
                    return None
                try:
                    attr = attrs["value"]
                except Exception:
                    attr = None
                if isinstance(attr, IntegerAttr):
                    return int(attr.value)
        except Exception:
            pass
        try:
            if isinstance(owner, arith.ConstantOp):
                attr = owner.value
                if isinstance(attr, IntegerAttr):
                    return int(attr.value)
        except Exception:
            pass
        return None

    def _operands(op) -> Optional[List[Value]]:
        try:
            return list(op.operands)
        except Exception:
            return None

    def _eval(val: Value, depth: int = 0) -> Optional[int]:
        if depth > 8:
            return None
        owner, op = _get_owner_op(val)
        if owner is None or op is None:
            return None
        c = _const_from_op(owner, op)
        if c is not None:
            return c
        opname = getattr(op, "name", None)
        if opname in ("arith.addi", "arith.subi", "arith.muli"):
            ops = _operands(op)
            if not ops or len(ops) != 2:
                return None
            a = _eval(ops[0], depth + 1)
            b = _eval(ops[1], depth + 1)
            if a is None or b is None:
                return None
            if opname == "arith.addi":
                return a + b
            if opname == "arith.subi":
                return a - b
            if opname == "arith.muli":
                return a * b
        return None

    return _eval(v, 0)


def _to_index_value(val, loc: Optional[Location] = None):
    """Convert python int or MLIR value to an index-typed MLIR value."""
    loc = _get_location(loc)
    val = _unwrap_value(val)
    if isinstance(val, Value):
        return val
    if isinstance(val, int):
        const = arith.ConstantOp(IndexType.get(), IntegerAttr.get(IndexType.get(), int(val)), loc=loc)
        return const.result
    return val


def _linear_idx_to_coords(index_value, dims):
    """Convert linear index into per-dimension coordinates."""
    coords = []
    remaining = _unwrap_value(_to_index_value(index_value))
    for size in reversed(dims):
        size_val = _unwrap_value(_to_index_value(size))
        rem = _unwrap_value(remaining)
        sz = _unwrap_value(size_val)
        coord = arith.RemUIOp(rem, sz).result
        coords.append(coord)
        remaining = arith.DivUIOp(rem, sz).result
    coords.reverse()
    return coords


def _add_index(base, offset):
    """Add an offset to a base coordinate."""
    if offset is None:
        return _to_index_value(base)
    base_val = _to_index_value(base)
    offset_val = _to_index_value(offset)
    return arith.AddIOp(base_val, offset_val).result


def _scale_index(value, factor):
    """Scale an index value by an integer factor."""
    value_val = _unwrap_value(_to_index_value(value))
    if isinstance(factor, int):
        if factor == 0:
            return _to_index_value(0)
        if factor == 1:
            return value_val
        factor_val = _unwrap_value(_to_index_value(int(factor)))
        return arith.MulIOp(value_val, factor_val).result
    factor_val = _unwrap_value(_to_index_value(factor))
    return arith.MulIOp(value_val, factor_val).result


# ---------------------------------------------------------------------------
# TensorView
# ---------------------------------------------------------------------------

class TensorView:
    """Lightweight view object representing a tensor slice."""

    def __init__(
        self,
        memref_value,
        shape,
        strides=None,
        base_indices=None,
        element_type=None,
        *,
        wrap_arith: bool = False,
    ):
        self.memref = _unwrap_value(memref_value) if memref_value is not None else None
        if shape is None:
            self.shape = ()
        else:
            _shape = []
            for s in shape:
                if isinstance(s, int):
                    _shape.append(int(s))
                    continue
                s_u = _unwrap_value(s)
                if isinstance(s_u, Value):
                    _shape.append(s_u)
                else:
                    _shape.append(int(s_u))
            self.shape = tuple(_shape)
        self.rank = len(self.shape)
        self.wrap_arith = bool(wrap_arith)
        if strides is None:
            if any(isinstance(d, Value) for d in self.shape):
                raise ValueError(
                    "TensorView(strides=None) requires fully-static `shape` (Python ints). "
                    "For dynamic shapes, pass explicit `strides=`."
                )
            strides = []
            stride = 1
            for size in reversed(self.shape):
                strides.insert(0, stride)
                stride *= int(size)
        _strides = []
        for s in strides:
            if isinstance(s, int):
                _strides.append(int(s))
                continue
            s_u = _unwrap_value(s)
            if isinstance(s_u, Value):
                _strides.append(s_u)
            else:
                _strides.append(int(s_u))
        self.strides = tuple(_strides)
        if base_indices is None:
            base_indices = [0] * self.rank
        self.base_indices = [_to_index_value(b) for b in base_indices]
        mem_type = getattr(self.memref, "type", None)
        if element_type is None and mem_type is not None and hasattr(mem_type, "element_type"):
            element_type = mem_type.element_type
        self.element_type = element_type

    def numel(self) -> int:
        """Return total number of elements in the view."""
        if any(isinstance(d, Value) for d in self.shape):
            raise ValueError("TensorView.numel() requires fully-static shape (Python ints).")
        total = 1
        for size in self.shape:
            total *= int(size)
        return total

    def offsets_from_linear(self, linear_idx):
        """Return per-dimension offsets for a given linear index."""
        idx_val = _to_index_value(linear_idx)
        return [_unwrap_value(v) for v in _linear_idx_to_coords(idx_val, self.shape)]

    def coords_from_linear(self, linear_idx):
        """Return absolute coordinates for a given linear index."""
        offsets = self.offsets_from_linear(linear_idx)
        coords = []
        for dim, offset in enumerate(offsets):
            base = self.base_indices[dim] if dim < len(self.base_indices) else _to_index_value(0)
            coords.append(_unwrap_value(_add_index(base, offset)))
        return coords

    def _normalize_coords(self, coords):
        if not isinstance(coords, (list, tuple)):
            coords = [coords]
        norm = []
        for idx in coords:
            if isinstance(idx, int):
                norm.append(_unwrap_value(_to_index_value(idx)))
            else:
                norm.append(_unwrap_value(idx))
        return norm

    def load(self, coords, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None):
        if self.memref is None:
            raise ValueError("TensorView has no backing memref for load")
        loc = _get_location(loc)
        coords = self._normalize_coords(coords)
        with ip or InsertionPoint.current:
            op = memref.load(self.memref, coords, loc=loc)
        val = _unwrap_value(op.result if hasattr(op, "result") else op)
        if self.wrap_arith:
            try:
                from . import arith as _arith_ext
            except Exception:  # pragma: no cover
                from . import arith as _arith_ext
            return _arith_ext.ArithValue(val)
        return val

    def store(self, value, coords, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None):
        if self.memref is None:
            raise ValueError("TensorView has no backing memref for store")
        loc = _get_location(loc)
        coords = self._normalize_coords(coords)
        with ip or InsertionPoint.current:
            memref.store(_unwrap_value(value), self.memref, coords, loc=loc)

    def __getitem__(self, coords):
        return self.load(coords)

    def __setitem__(self, coords, value):
        self.store(value, coords)


# ---------------------------------------------------------------------------
# swizzle_xor16
# ---------------------------------------------------------------------------

def swizzle_xor16(row: Value, col: Value, k_blocks16: Value,
                  loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """XOR-with-row swizzle on the K dimension at 16B granularity.

    Computes: col xor ((row % k_blocks16) * 16)
    """
    loc = _get_location(loc)
    with ip or InsertionPoint.current:
        r = _unwrap_value(row)
        c = _unwrap_value(col)
        kb = _unwrap_value(k_blocks16)
        rem = arith.RemUIOp(r, kb, loc=loc).result
        c16 = arith.ConstantOp(IndexType.get(), IntegerAttr.get(IndexType.get(), 16), loc=loc).result
        mul = arith.MulIOp(rem, c16, loc=loc).result
        return arith.XOrIOp(c, mul, loc=loc).result


# ---------------------------------------------------------------------------
# CopyAtom
# ---------------------------------------------------------------------------

class CopyAtom:
    """Copy atom descriptor for data movement operations."""

    def __init__(self, element_type: Type, vector_size: int, is_coalesced: bool = True):
        self.element_type = element_type
        self.vector_size = vector_size
        self.is_coalesced = is_coalesced

    def __repr__(self):
        return f"CopyAtom({self.element_type}, vec={self.vector_size}, coalesced={self.is_coalesced})"


def make_copy_atom(element_type: Type, vector_size: int = 8, is_coalesced: bool = True,
                   loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> CopyAtom:
    """Create a copy atom for data movement operations."""
    return CopyAtom(element_type, vector_size, is_coalesced)


# ---------------------------------------------------------------------------
# _normalize_indices_to_memref
# ---------------------------------------------------------------------------

def _normalize_indices_to_memref(memref_val: Value, indices: List[Value], strides: Optional[tuple], loc: Location) -> List[Value]:
    """Normalize indices to match the backing memref's rank.

    If the memref rank is less than the number of indices, linearize the
    multi-dimensional indices into a flat index using the strides.
    """
    from .._mlir.ir import MemRefType

    memref_type = memref_val.type
    if not isinstance(memref_type, MemRefType):
        return indices

    memref_rank = memref_type.rank
    num_indices = len(indices)

    if memref_rank == num_indices:
        return indices

    if memref_rank == 1 and num_indices > 1:
        if strides is None or len(strides) != num_indices:
            pass  # fall through to use whatever strides we have

        linear_idx = None
        for i, (idx, stride) in enumerate(zip(indices, strides if strides else [1]*num_indices)):
            idx = _unwrap_value(idx)
            if stride == 1:
                term = idx
            else:
                stride_val = _to_index_value(stride, loc)
                term = arith.muli(idx, stride_val, loc=loc)
                term = _unwrap_value(term)
            if linear_idx is None:
                linear_idx = term
            else:
                linear_idx = _unwrap_value(linear_idx)
                term = _unwrap_value(term)
                linear_idx = arith.addi(linear_idx, term, loc=loc)
        return [_unwrap_value(linear_idx)]

    return indices


# ---------------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------------

def copy(copy_desc, src, dst,
         src_indices: Optional[List[Value]] = None,
         dst_indices: Optional[List[Value]] = None,
         pred: Optional[Value] = None,
         dst_swizzle_xor16_kblocks: Optional[Value] = None,
         dst_swizzle_xor16_dims: Optional[Tuple[int, int]] = (0, 1),
         *,
         src_buffer_resource: Optional[Value] = None,
         src_buffer_offset_in_bytes: bool = True,
         nontemporal: Optional[bool] = None,
         alignment: Optional[int] = None,
         return_vector: bool = False,
         loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None):
    """Execute a copy/load/store using a CopyAtom descriptor.

    Supported call patterns:
      copy(atom, src_TensorView, None, return_vector=True, ...)   -- vector load
      copy(atom, vector_value,   dst_TensorView, ...)             -- vector store
      copy(atom, src_TensorView, dst_TensorView, ...)             -- memref copy
      copy(atom, src_val, dst_val, src_indices=..., dst_indices=...)
    """
    from .._mlir.dialects import memref as memref_dialect
    loc = _get_location(loc)

    captured_vec = {"val": None}

    def _maybe_swizzle_dst_indices(idx_list: List[Value]) -> List[Value]:
        if dst_swizzle_xor16_kblocks is None or dst_swizzle_xor16_dims is None:
            return idx_list
        if len(idx_list) < 2:
            return idx_list
        row_dim, col_dim = dst_swizzle_xor16_dims
        if row_dim < 0 or col_dim < 0:
            return idx_list
        if row_dim >= len(idx_list) or col_dim >= len(idx_list):
            return idx_list
        row = _unwrap_value(idx_list[row_dim])
        col = _unwrap_value(idx_list[col_dim])
        kblocks = _unwrap_value(dst_swizzle_xor16_kblocks)
        swz = swizzle_xor16(row, col, kblocks, loc=loc, ip=ip)
        out = list(idx_list)
        out[col_dim] = _unwrap_value(swz)
        return out

    def emit_tensor_copy(copy_shape, src_view: TensorView, dst_view: TensorView,
                         pred_view: Optional[Union[TensorView, Value]]):
        from .._mlir.dialects import vector as _vec
        from .._mlir.ir import VectorType

        vector_size = getattr(copy_desc, "vector_size", 1)

        def recurse(dim, src_idx, dst_idx, pred_idx):
            if dim == len(copy_shape):
                load_idx = _normalize_indices_to_memref(src_view.memref, [_unwrap_value(i) for i in src_idx], src_view.strides, loc)
                dst_idx2 = _maybe_swizzle_dst_indices([_unwrap_value(i) for i in dst_idx])
                store_idx = _normalize_indices_to_memref(dst_view.memref, dst_idx2, dst_view.strides, loc)
                load_op = memref_dialect.load(src_view.memref, load_idx)
                val = _unwrap_value(load_op.result if hasattr(load_op, "result") else load_op)

                cond = None
                if pred_view is not None:
                    if isinstance(pred_view, TensorView):
                        pred_idx_vals = [_unwrap_value(i) for i in pred_idx]
                        pred_op = memref_dialect.load(pred_view.memref, pred_idx_vals)
                        flag = _unwrap_value(pred_op.result if hasattr(pred_op, "result") else pred_op)
                        zero = _unwrap_value(arith.ConstantOp(flag.type, IntegerAttr.get(flag.type, 0), loc=loc).result)
                        cond = arith.CmpIOp(arith.CmpIPredicate.ne, flag, zero, loc=loc).result
                    else:
                        cond = _unwrap_value(pred_view)

                if cond is not None:
                    cond = _unwrap_value(cond)
                    if_op = scf.IfOp(cond, [], loc=loc)
                    with InsertionPoint(if_op.then_block):
                        memref_dialect.store(val, dst_view.memref, store_idx)
                        scf.YieldOp([])
                else:
                    memref_dialect.store(val, dst_view.memref, store_idx)
                return

            extent_dim = _unwrap_value(copy_shape[dim])
            if isinstance(extent_dim, Value):
                c = _try_get_constant_index(extent_dim)
                if c is None:
                    raise ValueError("flir.copy requires statically-known copy extents")
                extent = int(c)
            else:
                extent = int(extent_dim)

            if dim == len(copy_shape) - 1 and vector_size > 1 and extent % vector_size == 0:
                base_src = src_view.base_indices[dim] if dim < len(src_view.base_indices) else _to_index_value(0, loc)
                base_dst = dst_view.base_indices[dim] if dim < len(dst_view.base_indices) else _to_index_value(0, loc)
                base_pred = pred_view.base_indices[dim] if isinstance(pred_view, TensorView) else None

                hoisted_cond = None
                if pred_view is not None and not isinstance(pred_view, TensorView):
                    hoisted_cond = _unwrap_value(pred_view)

                def emit_vector_loop_body():
                    for i in range(0, extent, vector_size):
                        off = _to_index_value(i, loc)
                        vec_src_idx = src_idx + [_add_index(base_src, off)]
                        vec_dst_idx = dst_idx + [_add_index(base_dst, off)]
                        elem_type = src_view.element_type
                        vec_type = VectorType.get((vector_size,), elem_type)
                        load_indices = _normalize_indices_to_memref(src_view.memref, [_unwrap_value(idx) for idx in vec_src_idx], src_view.strides, loc)
                        vec_dst_idx2 = _maybe_swizzle_dst_indices([_unwrap_value(idx) for idx in vec_dst_idx])
                        store_indices = _normalize_indices_to_memref(dst_view.memref, vec_dst_idx2, dst_view.strides, loc)

                        vec_load_op = _vec.load(vec_type, src_view.memref, load_indices,
                                                nontemporal=nontemporal, alignment=alignment)
                        vec_val = _unwrap_value(vec_load_op.result if hasattr(vec_load_op, "result") else vec_load_op)
                        if return_vector and captured_vec["val"] is None:
                            captured_vec["val"] = vec_val

                        if pred_view is not None and isinstance(pred_view, TensorView):
                            curr_pred_base = _add_index(base_pred, off)
                            p_idx = pred_idx + [curr_pred_base]
                            p_idx_vals = [_unwrap_value(p) for p in p_idx]
                            pred_val_op = memref_dialect.load(pred_view.memref, p_idx_vals)
                            flag = _unwrap_value(pred_val_op.result if hasattr(pred_val_op, "result") else pred_val_op)
                            zero = _unwrap_value(arith.ConstantOp(flag.type, IntegerAttr.get(flag.type, 0), loc=loc).result)
                            cond = _unwrap_value(arith.CmpIOp(arith.CmpIPredicate.ne, flag, zero, loc=loc).result)
                            if_op = scf.IfOp(cond, [], loc=loc)
                            with InsertionPoint(if_op.then_block):
                                _vec.store(vec_val, dst_view.memref, store_indices,
                                           nontemporal=nontemporal, alignment=alignment)
                                scf.YieldOp([])
                        else:
                            _vec.store(vec_val, dst_view.memref, store_indices,
                                       nontemporal=nontemporal, alignment=alignment)

                if hoisted_cond is not None:
                    if_op = scf.IfOp(hoisted_cond, [], loc=loc)
                    with InsertionPoint(if_op.then_block):
                        emit_vector_loop_body()
                        scf.YieldOp([])
                else:
                    emit_vector_loop_body()
                return

            base_src = src_view.base_indices[dim] if dim < len(src_view.base_indices) else _to_index_value(0, loc)
            base_dst = dst_view.base_indices[dim] if dim < len(dst_view.base_indices) else _to_index_value(0, loc)
            base_pred = pred_view.base_indices[dim] if isinstance(pred_view, TensorView) else None
            for i in range(extent):
                off = _to_index_value(i, loc)
                next_src = _add_index(base_src, off)
                next_dst = _add_index(base_dst, off)
                next_pred_idx = pred_idx
                if isinstance(pred_view, TensorView):
                    next_pred_idx = pred_idx + [_add_index(base_pred, off)]
                recurse(dim + 1, src_idx + [next_src], dst_idx + [next_dst], next_pred_idx)

        with ip or InsertionPoint.current:
            recurse(0, [], [], [])

    def emit_tensor_load(copy_shape, src_view: TensorView, pred_val: Optional[Value] = None):
        """Load-only path (no dst), for gmem->register vector loads."""
        from .._mlir.dialects import vector as _vec
        from .._mlir.ir import VectorType

        if not return_vector:
            raise ValueError("copy(load-only) requires return_vector=True when dst is None")
        if len(copy_shape) != 1:
            raise ValueError("copy(load-only) currently supports only 1D shapes")

        extent0 = _unwrap_value(copy_shape[0])
        if isinstance(extent0, Value):
            c = _try_get_constant_index(extent0)
            if c is None:
                raise ValueError("copy(load-only) requires a statically-known 1D extent")
            extent = int(c)
        else:
            extent = int(extent0)

        vector_size = getattr(copy_desc, "vector_size", 1)
        if extent != int(vector_size):
            raise ValueError(f"copy(load-only) expects extent==vector_size (got {extent} vs {vector_size})")

        if src_buffer_resource is not None:
            try:
                from . import buffer_ops as _buffer_ops
            except Exception:
                from . import buffer_ops as _buffer_ops  # type: ignore

            elem_ty = src_view.element_type
            elem_ty_str = str(elem_ty)
            is_f8 = ("f8" in elem_ty_str) or ("Float8" in elem_ty_str)
            is_i8 = False
            try:
                is_i8 = IntegerType.isinstance(elem_ty) and (IntegerType(elem_ty).width == 8)
            except Exception:
                is_i8 = (elem_ty_str == "i8")
            is_f16 = ("f16" in elem_ty_str) or ("Float16" in elem_ty_str)
            is_bf16 = ("bf16" in elem_ty_str) or ("BFloat16" in elem_ty_str)
            is_1byte = is_f8 or is_i8
            is_2byte = is_f16 or is_bf16
            use_buffer_fast = (is_1byte or is_2byte) and extent in (8, 16) and (extent % 4 == 0)

            if use_buffer_fast:
                i32_ty = IntegerType.get_signless(32)
                elem_size = 2 if is_2byte else 1
                load_bytes = extent * elem_size
                vec_width = load_bytes // 4
                base0 = src_view.base_indices[0] if len(src_view.base_indices) else _to_index_value(0, loc)
                if src_buffer_offset_in_bytes:
                    c4 = arith.ConstantOp(IndexType.get(), IntegerAttr.get(IndexType.get(), 4), loc=loc).result
                    idx_i32 = arith.DivSIOp(_unwrap_value(base0), _unwrap_value(c4), loc=loc).result
                elif is_2byte:
                    c2 = arith.ConstantOp(IndexType.get(), IntegerAttr.get(IndexType.get(), 2), loc=loc).result
                    byte_idx = arith.MulIOp(_unwrap_value(base0), _unwrap_value(c2), loc=loc).result
                    c4 = arith.ConstantOp(IndexType.get(), IntegerAttr.get(IndexType.get(), 4), loc=loc).result
                    idx_i32 = arith.DivSIOp(_unwrap_value(byte_idx), _unwrap_value(c4), loc=loc).result
                else:
                    idx_i32 = _unwrap_value(base0)
                mask = _unwrap_value(pred_val) if pred_val is not None else None
                i32_vec = _buffer_ops.buffer_load(
                    _unwrap_value(src_buffer_resource), idx_i32,
                    vec_width=vec_width, dtype=i32_ty, mask=mask,
                )
                vec_elem_ty = VectorType.get((extent,), elem_ty)
                return _vec.BitCastOp(vec_elem_ty, _unwrap_value(i32_vec)).result

        base = src_view.base_indices[0] if len(src_view.base_indices) else _to_index_value(0, loc)
        idxs = _normalize_indices_to_memref(src_view.memref, [_unwrap_value(base)], src_view.strides, loc)
        vec_type = VectorType.get((extent,), src_view.element_type)
        with ip or InsertionPoint.current:
            return _vec.load(vec_type, src_view.memref, idxs,
                             nontemporal=nontemporal, alignment=alignment)

    # -- Dispatch --------------------------------------------------------

    # Vector store: src is a vector Value, dst is a TensorView.
    try:
        from .._mlir.ir import VectorType as _VectorType
    except Exception:
        _VectorType = None  # type: ignore

    if isinstance(dst, TensorView):
        v = _unwrap_value(src)
        if _VectorType is not None and isinstance(getattr(v, "type", None), _VectorType):
            from .._mlir.dialects import vector as vector_dialect
            d_idx = [_unwrap_value(i) for i in dst.base_indices]
            d_idx2 = _maybe_swizzle_dst_indices(d_idx)
            store_indices = _normalize_indices_to_memref(dst.memref, d_idx2, dst.strides, loc)
            with ip or InsertionPoint.current:
                vector_dialect.store(v, dst.memref, store_indices,
                                     nontemporal=nontemporal, alignment=alignment)
            return v if return_vector else None

    # Load-only: dst=None and src is a TensorView.
    if isinstance(src, TensorView) and dst is None:
        pred_val = None
        if pred is not None and not isinstance(pred, TensorView):
            pred_val = _unwrap_value(pred)
        return emit_tensor_load(src.shape, src, pred_val=pred_val)

    # TensorView-to-TensorView copy.
    if isinstance(src, TensorView) and isinstance(dst, TensorView):
        emit_tensor_copy(src.shape, src, dst, pred)
        return captured_vec["val"] if return_vector else None

    # Raw value fallback.
    src_val = _unwrap_value(src)
    dst_val = _unwrap_value(dst)
    with ip or InsertionPoint.current:
        if src_indices is not None and dst_indices is not None:
            s_idx = [_unwrap_value(i) for i in src_indices]
            d_idx = [_unwrap_value(i) for i in dst_indices]
            val = memref_dialect.load(src_val, s_idx)
            memref_dialect.store(val, dst_val, d_idx)
        else:
            raise ValueError("copy requires explicit indices for raw values")
    return captured_vec["val"] if return_vector else None


# ---------------------------------------------------------------------------
# Module self-reference + print re-export
# ---------------------------------------------------------------------------

print = print

import sys as _sys
flir_compat = _sys.modules[__name__]


# ---------------------------------------------------------------------------
# Layout algebra: forwarded to flydsl.expr (fly dialect)
# ---------------------------------------------------------------------------

def _get_fx():
    import flydsl.expr as _fx
    return _fx

def make_shape(*dims, loc=None, ip=None):
    return _get_fx().make_shape(*dims, loc=loc, ip=ip)

def make_stride(*strides, loc=None, ip=None):
    return _get_fx().make_stride(*strides, loc=loc, ip=ip)

def make_layout(shape, stride=None, loc=None, ip=None):
    if stride is None:
        stride = 1
    return _get_fx().make_layout(shape, stride, loc=loc, ip=ip)

def make_coord(*coords, loc=None, ip=None):
    return _get_fx().make_coord(*coords, loc=loc, ip=ip)

def crd2idx(coord, layout, loc=None, ip=None):
    return _get_fx().crd2idx(coord, layout, loc=loc, ip=ip)

def idx2crd(idx, layout, loc=None, ip=None):
    return _get_fx().idx2crd(idx, layout, loc=loc, ip=ip)

def get(input_val, mode, loc=None, ip=None):
    return _get_fx().get(input_val, mode, loc=loc, ip=ip)
