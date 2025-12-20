"""Python bindings for Rocir dialect operations.

This module provides Python wrappers for Rocir layout algebra operations,
making it easier to construct layouts and perform layout transformations
from Python code.
"""

from typing import List, Optional, Sequence, Union

from _mlir.ir import (
    Type,
    Value,
    Location,
    InsertionPoint,
    IndexType,
    IntegerAttr,
    IntegerType,
    MemRefType,
)
from _mlir.dialects import memref, arith, scf, gpu, vector, math, llvm
from _mlir.extras import types as T
from rocdsl.lang.ir.module import MlirModule, kernel, jit
from _mlir.dialects import rocir as rocir_ops

# Also expose RocDSL "extended" wrappers (like test_eltwise_add.py uses) so
# tests can route everything through `rocir.*` without importing `mlir.*`.
from . import arith as arith_ext  # noqa: E402
from . import scf as scf_ext      # noqa: E402
from . import gpu as gpu_ext      # noqa: E402



def _get_location(loc: Optional[Location] = None) -> Location:
    """Get location, using current location if none provided."""
    if loc is None:
        loc = Location.unknown()
    return loc



def _unwrap_value(v):
    """Unwrap ArithValue or other value wrappers to get underlying MLIR Value."""
    if isinstance(v, int):
        from _mlir.dialects import arith
        from _mlir.ir import IndexType, IntegerAttr
        op = arith.ConstantOp(IndexType.get(), IntegerAttr.get(IndexType.get(), v))
        return _unwrap_value(op.result)
    try:
        internal = object.__getattribute__(v, "_value")
        return _unwrap_value(internal)
    except AttributeError:
        pass
    if hasattr(v, 'value') and callable(getattr(type(v).value, 'fget', None)):
        # It's a property, call it
        return v.value
    elif hasattr(v, '_value'):
        # Direct attribute access
        return v._value
    else:
        # Already a Value or compatible
        return v

def _get_insertion_point(ip: Optional[InsertionPoint] = None) -> InsertionPoint:
    """Get insertion point, using current if none provided."""
    if ip is None:
        return InsertionPoint.current
    return ip


def _try_get_constant_index(v: Value) -> Optional[int]:
    """Best-effort extract an int from an index-typed Value.

    Supports:
    - arith.constant
    - a small set of arithmetic ops when all operands are constant (addi, subi, muli)

    This is intentionally conservative: if we can't prove it's constant, return None.
    """

    def _get_owner_op(val: Value):
        try:
            owner = val.owner
        except Exception:
            return None, None

        # Normalize to an Operation handle (some APIs return OpView, others Operation).
        op = getattr(owner, "operation", owner)
        return owner, op

    def _const_from_op(owner, op) -> Optional[int]:
        # Operation path (newer bindings)
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

        # OpView path (older bindings)
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


def _count_leaves_in_tuple_spec(spec: str) -> int:
    """Count leaf entries in a canonical tuple spec like '(9,(4,8))' or '(?,(?,?))'."""
    s = "".join(ch for ch in spec if not ch.isspace())
    i = 0
    leaves = 0
    while i < len(s):
        c = s[i]
        if c == "?":
            leaves += 1
            i += 1
            continue
        if c.isdigit() or (c == "-" and i + 1 < len(s) and s[i + 1].isdigit()):
            # integer literal
            i += 1 if c == "-" else 0
            while i < len(s) and s[i].isdigit():
                i += 1
            leaves += 1
            continue
        i += 1
    return leaves


def _extract_rank_from_rocir_type_str(type_str: str) -> int:
    """Extract rank from a rocir type string like '!rocir.layout<2>' or '!rocir.layout<(?,?)>'."""
    if "<" not in type_str or ">" not in type_str:
        raise ValueError(f"Cannot extract rank from type string: {type_str}")
    inner = type_str.split("<", 1)[1].rsplit(">", 1)[0].strip()
    if inner.startswith("("):
        return _count_leaves_in_tuple_spec(inner)
    if len(inner) >= 2 and inner[0] == '"' and inner[-1] == '"':
        return _count_leaves_in_tuple_spec(inner[1:-1])
    # numeric
    return int(inner)


def _extract_tuple_spec_from_rocir_type(type_str: str) -> Optional[str]:
    """If type_str is like '!rocir.shape<\"(...)\">' or '!rocir.stride<\"(...)\">', return '(...)'."""
    if "<" not in type_str or ">" not in type_str:
        return None
    inner = type_str.split("<", 1)[1].rsplit(">", 1)[0]
    inner = inner.strip()
    if len(inner) >= 2 and inner[0] == '"' and inner[-1] == '"':
        return inner[1:-1]
    return None


def _is_rocir_type(v: Value, prefix: str) -> bool:
    try:
        return str(v.type).startswith(prefix)
    except Exception:
        return False


def _flatten_nested_shape_or_stride_value(v: Value, kind: str) -> Optional[tuple]:
    """If v is produced by rocir.make_shape/make_stride, flatten it into (spec, leaf_values)."""
    try:
        op = v.owner
    except Exception:
        return None
    op_name = ""
    try:
        op_name = op.operation.name
    except Exception:
        try:
            op_name = op.name
        except Exception:
            op_name = ""
    expected = "rocir.make_shape" if kind == "shape" else "rocir.make_stride"
    if op_name != expected:
        return None

    try:
        operands = list(op.values)
    except Exception:
        # Generated opviews may use accessors like getValues().
        try:
            operands = list(op.getValues())
        except Exception:
            return None

    child_specs = []
    leaf_vals: List[Value] = []
    for ov in operands:
        ov = _unwrap_value(ov)
        if isinstance(ov, Value) and _is_rocir_type(ov, f"!rocir.{kind}<"):
            rec = _flatten_nested_shape_or_stride_value(ov, kind)
            if rec is None:
                return None
            sub_spec, sub_leaves = rec
            child_specs.append(sub_spec)
            leaf_vals.extend(sub_leaves)
        else:
            # Leaf index value
            const = _try_get_constant_index(ov) if isinstance(ov, Value) else None
            child_specs.append(str(const) if const is not None else "?")
            leaf_vals.append(ov)
    spec = "(" + ",".join(child_specs) + ")"
    return spec, leaf_vals


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
        self.shape = tuple(int(s) for s in shape) if shape is not None else ()
        self.rank = len(self.shape)
        self.wrap_arith = bool(wrap_arith)
        if strides is None:
            strides = []
            stride = 1
            for size in reversed(self.shape):
                strides.insert(0, stride)
                stride *= int(size)
        self.strides = tuple(int(s) for s in strides)
        if base_indices is None:
            base_indices = [0] * self.rank
        self.base_indices = [_to_index_value(b) for b in base_indices]
        mem_type = getattr(self.memref, "type", None)
        if element_type is None and mem_type is not None and hasattr(mem_type, "element_type"):
            element_type = mem_type.element_type
        self.element_type = element_type

    def numel(self) -> int:
        """Return total number of elements in the view."""
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
                norm.append(const_index(idx))
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
            # Local import to avoid circular import issues.
            try:
                from rocdsl.dialects.ext import arith as _arith_ext
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


def _to_index_value(val, loc: Optional[Location] = None):
    """Convert python int or MLIR value to an index-typed MLIR value."""
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


def const_index(value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None):
    """Create an index-typed constant Value.

    Compatibility note: with Python range-loop lowering enabled, loop induction
    variables are MLIR Values. In that case, `const_index(v)` will behave like
    `to_index(v)`.
    """
    loc = _get_location(loc)
    with ip or InsertionPoint.current:
        # If `value` is already an MLIR value (or a wrapper around one), just cast.
        try:
            from _mlir.ir import Value as _MlirValue  # local import to avoid import cycles
            if isinstance(value, _MlirValue) or (hasattr(value, "value") and isinstance(value.value, _MlirValue)):
                return _unwrap_value(_to_index_value(value, loc))
        except Exception:
            pass
        op = arith.ConstantOp(IndexType.get(), IntegerAttr.get(IndexType.get(), int(value)), loc=loc)
        return _unwrap_value(op.result if hasattr(op, "result") else op)


def to_index(val, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None):
    """Convert python int or MLIR Value to an index-typed MLIR Value.

    Useful when writing code that may run either as a Python-unrolled loop
    (where loop indices are Python ints) or as an IR loop (scf.for) where
    induction variables are MLIR Values.
    """
    loc = _get_location(loc)
    with ip or InsertionPoint.current:
        return _unwrap_value(_to_index_value(val, loc))


def thread_idx(axis: str = "x"):
    """Return the current thread index along the given axis."""
    return gpu.thread_id(axis)


def block_idx(axis: str = "x"):
    """Return the current block index along the given axis."""
    return gpu.block_id(axis)


def block_dim(axis: str = "x"):
    """Return the block dimension along the given axis."""
    return gpu.block_dim(axis)


class ShapeType(Type):
    """Rocir shape type."""
    
    @staticmethod
    def get(rank: int, context=None):
        """Create a shape type with given rank."""
        # This would need to be implemented in C++ bindings
        # For now, return a generic type
        from _mlir.ir import Context
        if context is None:
            context = Context.current
        # Placeholder - would use actual ODS-generated type
        return Type.parse(f"!rocir.shape<{rank}>", context=context)


class StrideType(Type):
    """Rocir stride type."""
    
    @staticmethod
    def get(rank: int, context=None):
        """Create a stride type with given rank."""
        from _mlir.ir import Context
        if context is None:
            context = Context.current
        return Type.parse(f"!rocir.stride<{rank}>", context=context)


class LayoutType(Type):
    """Rocir layout type."""
    
    @staticmethod
    def get(rank: int, context=None):
        """Create a layout type with given rank."""
        from _mlir.ir import Context
        if context is None:
            context = Context.current
        return Type.parse(f"!rocir.layout<{rank}>", context=context)


class CoordType(Type):
    """Rocir coordinate type."""
    
    @staticmethod
    def get(rank: int, context=None):
        """Create a coordinate type with given rank."""
        from _mlir.ir import Context
        if context is None:
            context = Context.current
        return Type.parse(f"!rocir.coord<{rank}>", context=context)



def _flatten_nested(values, result=None):
    """Flatten nested tuples/lists into a flat list of values."""
    if result is None:
        result = []
    
    for v in values:
        if isinstance(v, (tuple, list)):
            _flatten_nested(v, result)
        else:
            result.append(v)
    
    return result


def _count_total_dims(dims):
    """Count total dimensions in potentially nested structure."""
    count = 0
    for d in dims:
        if isinstance(d, (tuple, list)):
            count += _count_total_dims(d)
        else:
            count += 1
    return count


def make_shape(*dims, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Create a shape from dimension values (supports nested shapes).
    
    Args:
        *dims: Index values or tuples of index values for nested dimensions
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        A Rocir shape value
        
    Example:
        >>> # Flat shape
        >>> c8 = arith.constant(8, index=True)
        >>> c16 = arith.constant(16, index=True)
        >>> shape = rocir.make_shape(c8, c16)  # Creates shape<2>
        >>> 
        >>> # Nested shape example: (9, (4, 8))
        >>> c9 = arith.constant(9, index=True)
        >>> c4 = arith.constant(4, index=True)
        >>> shape = rocir.make_shape(c9, (c4, c8))  # Creates nested shape
    """
    
    loc = _get_location(loc)
    
    # If a single tuple/list is passed, unpack it
    if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
        dims = dims[0]
    
    # Nested form: allow tuple/list and nested rocir.make_shape results.
    # If we can fully flatten to leaf index values, emit a tuple-spec type: !rocir.shape<(...)>.
    nested_spec = None
    nested_leaf_vals = None

    # Normalize to a list for processing
    dims_list = list(dims)
    if len(dims_list) == 1 and isinstance(dims_list[0], (tuple, list)):
        dims_list = list(dims_list[0])

    # Detect nested structure (tuple/list or shape values)
    has_nested = any(isinstance(d, (tuple, list)) for d in dims_list) or any(
        isinstance(_unwrap_value(d), Value) and _is_rocir_type(_unwrap_value(d), "!rocir.shape<") for d in dims_list
    )

    if has_nested:
        # Build a spec and leaf list by recursively expanding nested make_shape ops and tuples.
        child_specs = []
        leaf_vals: List[Value] = []

        def consume_node(node):
            nonlocal child_specs, leaf_vals
            if isinstance(node, (tuple, list)):
                sub_child_specs = []
                for x in node:
                    consume_node(x)
                    # move last spec into sub list
                    sub_child_specs.append(child_specs.pop())
                    # leaf_vals already appended
                child_specs.append("(" + ",".join(sub_child_specs) + ")")
                return

            v = _unwrap_value(node)
            if isinstance(v, Value) and _is_rocir_type(v, "!rocir.shape<"):
                rec = _flatten_nested_shape_or_stride_value(v, "shape")
                if rec is None:
                    raise ValueError("Cannot flatten nested rocir.shape value without defining rocir.make_shape")
                sub_spec, sub_leaves = rec
                child_specs.append(sub_spec)
                leaf_vals.extend(sub_leaves)
                return

            # Leaf index
            const = _try_get_constant_index(v) if isinstance(v, Value) else None
            child_specs.append(str(const) if const is not None else "?")
            leaf_vals.append(v)

        try:
            for d in dims_list:
                consume_node(d)
            nested_spec = "(" + ",".join(child_specs) + ")"
            nested_leaf_vals = leaf_vals
        except Exception:
            nested_spec = None
            nested_leaf_vals = None

    if nested_spec is not None and nested_leaf_vals is not None:
        # Type carries structure; rank is implied by leaf count.
        result_type = Type.parse(f'!rocir.shape<{nested_spec}>')
        flat_dims = nested_leaf_vals
    else:
        # Fallback: flat shape only
        flat_dims = _flatten_nested(dims)
        rank = len(flat_dims)
        # If any dimension is a compile-time constant, encode it in the type spec
        # so downstream passes can "see" static information early:
        #   !rocir.shape<(4,32)> or partial !rocir.shape<(4,?)>
        spec_elems = []
        has_any_const = False
        for d in flat_dims:
            v = _unwrap_value(d)
            const = None
            if isinstance(v, int):
                const = v
            elif isinstance(v, Value):
                const = _try_get_constant_index(v)
            if const is not None:
                has_any_const = True
                spec_elems.append(str(const))
            else:
                spec_elems.append("?")
        if has_any_const:
            flat_spec = "(" + ",".join(spec_elems) + ")"
            result_type = Type.parse(f"!rocir.shape<{flat_spec}>")
        else:
            result_type = ShapeType.get(rank)
    
    # Keep all operands (dense) for compatibility with existing lowering passes
    operands = [_unwrap_value(d) for d in flat_dims]

    with ip or InsertionPoint.current:
        return rocir_ops.MakeShapeOp(result_type, operands, loc=loc).result


def make_stride(*strides, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Create a stride from stride values (supports nested strides).
    
    Args:
        *strides: Index values or tuples of index values for nested strides
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        A Rocir stride value
        
    Example:
        >>> # Flat stride
        >>> c1 = arith.constant(1, index=True)
        >>> c8 = arith.constant(8, index=True)
        >>> stride = rocir.make_stride(c1, c8)  # Creates stride<2>
        >>> 
        >>> # Nested stride example: (59, (13, 1))
        >>> c59 = arith.constant(59, index=True)
        >>> c13 = arith.constant(13, index=True)
        >>> stride = rocir.make_stride(c59, (c13, c1))  # Creates nested stride
    """
    
    loc = _get_location(loc)
    
    # If a single tuple/list is passed, unpack it
    if len(strides) == 1 and isinstance(strides[0], (tuple, list)):
        strides = strides[0]
    
    nested_spec = None
    nested_leaf_vals = None

    strides_list = list(strides)
    if len(strides_list) == 1 and isinstance(strides_list[0], (tuple, list)):
        strides_list = list(strides_list[0])

    has_nested = any(isinstance(s, (tuple, list)) for s in strides_list) or any(
        isinstance(_unwrap_value(s), Value) and _is_rocir_type(_unwrap_value(s), "!rocir.stride<") for s in strides_list
    )

    if has_nested:
        child_specs = []
        leaf_vals: List[Value] = []

        def consume_node(node):
            nonlocal child_specs, leaf_vals
            if isinstance(node, (tuple, list)):
                sub_child_specs = []
                for x in node:
                    consume_node(x)
                    sub_child_specs.append(child_specs.pop())
                child_specs.append("(" + ",".join(sub_child_specs) + ")")
                return

            v = _unwrap_value(node)
            if isinstance(v, Value) and _is_rocir_type(v, "!rocir.stride<"):
                rec = _flatten_nested_shape_or_stride_value(v, "stride")
                if rec is None:
                    raise ValueError("Cannot flatten nested rocir.stride value without defining rocir.make_stride")
                sub_spec, sub_leaves = rec
                child_specs.append(sub_spec)
                leaf_vals.extend(sub_leaves)
                return

            const = _try_get_constant_index(v) if isinstance(v, Value) else None
            child_specs.append(str(const) if const is not None else "?")
            leaf_vals.append(v)

        try:
            for s in strides_list:
                consume_node(s)
            nested_spec = "(" + ",".join(child_specs) + ")"
            nested_leaf_vals = leaf_vals
        except Exception:
            nested_spec = None
            nested_leaf_vals = None

    if nested_spec is not None and nested_leaf_vals is not None:
        result_type = Type.parse(f'!rocir.stride<{nested_spec}>')
        flat_strides = nested_leaf_vals
    else:
        flat_strides = _flatten_nested(strides)
        rank = len(flat_strides)
        # Same idea as make_shape: encode constant strides into the type spec
        # e.g. !rocir.stride<(32,1)> or partial !rocir.stride<(?,1)>
        spec_elems = []
        has_any_const = False
        for s in flat_strides:
            v = _unwrap_value(s)
            const = None
            if isinstance(v, int):
                const = v
            elif isinstance(v, Value):
                const = _try_get_constant_index(v)
            if const is not None:
                has_any_const = True
                spec_elems.append(str(const))
            else:
                spec_elems.append("?")
        if has_any_const:
            flat_spec = "(" + ",".join(spec_elems) + ")"
            result_type = Type.parse(f"!rocir.stride<{flat_spec}>")
        else:
            result_type = StrideType.get(rank)
    
    # Keep all operands (dense) for compatibility with existing lowering passes
    operands = [_unwrap_value(s) for s in flat_strides]

    with ip or InsertionPoint.current:
        return rocir_ops.MakeStrideOp(result_type, operands, loc=loc).result


def make_layout(shape, stride=None, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Create a layout from shape and stride (supports nested layouts).
    
    Args:
        shape: A Rocir shape value, or a tuple/int for creating shape
        stride: A Rocir stride value, or a tuple/int for creating stride, 
                or None to create default column-major stride
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        A Rocir layout value
        
    Example:
        >>> # Using shape and stride values
        >>> shape = rocir.make_shape(c8, c16)
        >>> stride = rocir.make_stride(c1, c8)
        >>> layout = rocir.make_layout(shape, stride)
        >>> 
        >>> # Using tuples directly in nested form
        >>> layout = rocir.make_layout((c9, (c4, c8)), stride=(c59, (c13, c1)))
        >>> 
        >>> # Using single value
        >>> layout = rocir.make_layout(c6, stride=c1)  # 1D layout 6:1
    """
    
    loc = _get_location(loc)
    
    # If shape is not already a Value, create shape from it
    if not isinstance(shape, Value):
        if isinstance(shape, (tuple, list)):
            shape = make_shape(*shape, loc=loc, ip=ip)
        else:
            # Single value
            shape = make_shape(shape, loc=loc, ip=ip)
    
    # If stride is not already a Value, create stride from it
    if stride is not None:
        if not isinstance(stride, Value):
            if isinstance(stride, (tuple, list)):
                stride = make_stride(*stride, loc=loc, ip=ip)
            else:
                # Single value
                stride = make_stride(stride, loc=loc, ip=ip)
    else:
        # Create default column-major stride (1, prev_dim, prev_dim*prev_stride, ...)
        # For now, just use unit stride
        # TODO: Implement proper default stride computation
        raise ValueError("Default stride not yet implemented, please provide explicit stride")
    
    def _extract_spec_and_rank_from_rocir_shape_or_stride_type(type_str: str):
        """Return (spec, rank) where spec is a tuple-spec like '(9,(4,8))' or None."""
        type_content = type_str.split("<")[1].split(">")[0].strip()
        if type_content.startswith("("):
            spec = type_content
            rank = _count_leaves_in_tuple_spec(spec)
            return spec, rank
        if len(type_content) >= 2 and type_content[0] == '"' and type_content[-1] == '"':
            spec = type_content[1:-1]
            rank = _count_leaves_in_tuple_spec(spec)
            return spec, rank
        if "," in type_content:
            # Legacy format: !rocir.shape<rank, ...>
            return None, int(type_content.split(",")[0].strip())
        return None, int(type_content)

    shape_spec, rank = _extract_spec_and_rank_from_rocir_shape_or_stride_type(str(shape.type))
    stride_spec, _ = _extract_spec_and_rank_from_rocir_shape_or_stride_type(str(stride.type))

    if shape_spec is not None and stride_spec is not None:
        result_type = Type.parse(f"!rocir.layout<{shape_spec}:{stride_spec}>")
    else:
        result_type = LayoutType.get(rank)
    
    with ip or InsertionPoint.current:
        return rocir_ops.MakeLayoutOp(result_type, _unwrap_value(shape), _unwrap_value(stride), loc=loc).result


def make_coord(*coords: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Create a coordinate from index values.
    
    Args:
        *coords: Index values representing each coordinate dimension
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        A Rocir coordinate value
        
    Example:
        >>> i = arith.constant(4, index=True)
        >>> j = arith.constant(7, index=True)
        >>> coord = rocir.make_coord(i, j)  # Creates coord<2>
    """
    
    loc = _get_location(loc)
    rank = len(coords)
    result_type = CoordType.get(rank)
    
    with ip or InsertionPoint.current:
        return rocir_ops.MakeCoordOp(result_type, [_unwrap_value(c) for c in coords], loc=loc).result


def crd2idx(coord: Value, layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Convert a coordinate to a linear index using a layout.
    
    Computes: sum(coord[i] * stride[i]) for all dimensions i.
    
    Args:
        coord: A Rocir coordinate value
        layout: A Rocir layout value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        An index value representing the linear offset
        
    Example:
        >>> coord = rocir.make_coord(i, j)
        >>> layout = rocir.make_layout(shape, stride)
        >>> idx = rocir.crd2idx(coord, layout)  # Returns i*stride[0] + j*stride[1]
    """
    
    loc = _get_location(loc)
    result_type = IndexType.get()
    
    with ip or InsertionPoint.current:
        op = rocir_ops.Crd2IdxOp(result_type, _unwrap_value(coord), _unwrap_value(layout), loc=loc, ip=ip)
        return op.result


def idx2crd(idx: Value, layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Convert a linear index to a coordinate using a layout.
    
    This is the inverse operation of crd2idx.
    
    Args:
        idx: An index value representing the linear offset
        layout: A Rocir layout value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        A Rocir coordinate value
        
    Example:
        >>> idx = arith.constant(42, index=True)
        >>> layout = rocir.make_layout(shape, stride)
        >>> coord = rocir.idx2crd(idx, layout)  # Inverse of crd2idx
    """
    
    loc = _get_location(loc)
    # Extract rank from layout type
    layout_type_str = str(layout.type)
    rank = _extract_rank_from_rocir_type_str(layout_type_str)
    result_type = CoordType.get(rank)
    
    with ip or InsertionPoint.current:
        op = rocir_ops.Idx2CrdOp(result_type, _unwrap_value(idx), _unwrap_value(layout), loc=loc, ip=ip)
        return op.results[0]



def size(layout_or_tensor, mode: Optional[List[int]] = None, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Get the size of a layout or tensor.
    
    Args:
        layout_or_tensor: Layout, tensor, or shape to query
        mode: Optional list of mode indices to query specific dimensions
              If None, returns total size
              If [0], returns size of mode 0 (e.g., thread count)
              If [1], returns size of mode 1 (e.g., tile count)
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        Size as an index value
        
    Example:
        >>> shape = rocir.make_shape(c8, c16)
        >>> total = rocir.size(shape)  # Returns 128
        >>> thread_count = rocir.size(tv_layout, mode=[0])  # Thread dimension
        >>> tile_count = rocir.size(gC, mode=[1])  # Tile dimension
    """
    
    loc = _get_location(loc)
    result_type = IndexType.get()
    
    # Handle tensor/layout with shape attribute and mode query
    if hasattr(layout_or_tensor, 'shape') and mode is not None:
        # Extract specific dimension
        if isinstance(layout_or_tensor.shape, (tuple, list)):
            idx = mode[0]
            if idx < len(layout_or_tensor.shape):
                return _unwrap_value(layout_or_tensor.shape[idx])
    
    with ip or InsertionPoint.current:
        op = rocir_ops.SizeOp(result_type, _unwrap_value(layout_or_tensor), loc=loc, ip=ip)
        return op.result


def cosize(layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the cosize (stride extent) of a layout.
    
    Args:
        layout: A Rocir layout value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        An index value representing the cosize
    """
    
    loc = _get_location(loc)
    result_type = IndexType.get()
    
    with ip or InsertionPoint.current:
        op = rocir_ops.CosizeOp(result_type, _unwrap_value(layout), loc=loc, ip=ip)
        return op.result


def rank(shape_or_layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Get the rank (number of dimensions) of a shape or layout.
    
    Args:
        shape_or_layout: A Rocir shape or layout value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        An index value representing the rank
    """
    
    loc = _get_location(loc)
    result_type = IndexType.get()
    
    with ip or InsertionPoint.current:
        op = rocir_ops.RankOp(result_type, _unwrap_value(shape_or_layout), loc=loc, ip=ip)
        return op.result


def get_shape(layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Extract the shape from a layout.
    
    Args:
        layout: A Rocir layout value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        The shape component of the layout
    """
    
    loc = _get_location(loc)
    # Extract rank from layout type
    layout_type_str = str(layout.type)
    rank = _extract_rank_from_rocir_type_str(layout_type_str)
    result_type = ShapeType.get(rank)
    
    with ip or InsertionPoint.current:
        return rocir_ops.GetShapeOp(result_type, _unwrap_value(layout), loc=loc).result


def get_stride(layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Extract the stride from a layout.
    
    Args:
        layout: A Rocir layout value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        The stride component of the layout
    """
    
    loc = _get_location(loc)
    # Extract rank from layout type
    layout_type_str = str(layout.type)
    rank = _extract_rank_from_rocir_type_str(layout_type_str)
    result_type = StrideType.get(rank)
    
    with ip or InsertionPoint.current:
        return rocir_ops.GetStrideOp(result_type, _unwrap_value(layout), loc=loc).result


def get(input: Value, index: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Extract element from shape/stride/coord at given index.
    
    Args:
        input: A Rocir shape, stride, or coord value
        index: Index of element to extract
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        The element at the given index (as an index value)
        
    Example:
        >>> shape = rocir.make_shape(c2, c3, c4)
        >>> dim0 = rocir.get(shape, Const.index(0))  # Returns 2
        >>> dim1 = rocir.get(shape, Const.index(1))  # Returns 3
    """
    
    loc = _get_location(loc)
    result_type = IndexType.get()
    
    with ip or InsertionPoint.current:
        return rocir_ops.GetOp(result_type, _unwrap_value(input), _unwrap_value(index), loc=loc, ip=ip).result


def composition(layout_a: Value, layout_b: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compose two layouts.
    
    Args:
        layout_a: First layout
        layout_b: Second layout
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        The composed layout
        
    Example:
        >>> # Compose a column-major layout with a tiler
        >>> composed = rocir.composition(col_major, tiler)
    """
    
    loc = _get_location(loc)
    result_type = LayoutType.get(-1)
    
    with ip or InsertionPoint.current:
        return rocir_ops.CompositionOp(result_type, _unwrap_value(layout_a), _unwrap_value(layout_b), loc=loc).result


def complement(tiler: Value, target_size: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the complement of a tiler layout.
    
    The complement finds the "rest" modes not covered by the tiler.
    This is used internally by logical_divide.
    
    Algorithm:
    1. Filters out stride-0 and size-1 modes from the tiler
    2. Sorts modes by stride (ascending)
    3. Folds to compute rest modes
    4. Returns coalesced layout of rest modes
    
    Args:
        tiler: The tiler layout
        target_size: The target size to complement against
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        The complement layout
        
    Example:
        >>> # For a layout of size 12 with tiler of size 3
        >>> # complement returns a layout covering the remaining 4 elements
        >>> tiler = rocir.make_layout(c3, stride=c1)
        >>> target = Const.index(12)
        >>> comp = rocir.complement(tiler, target)  # Returns 4:3
    """
    
    loc = _get_location(loc)
    result_type = LayoutType.get(-1)
    
    with ip or InsertionPoint.current:
        return rocir_ops.ComplementOp(result_type, _unwrap_value(tiler), _unwrap_value(target_size), loc=loc).result


def coalesce(layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Coalesce/simplify a layout by flattening and combining modes.
    
    Ensures post-conditions:
    - Preserves size: size(layout) == size(result)
    - Flattened: depth(result) <= 1
    - Preserves function: For all i, layout(i) == result(i)
    
    Args:
        layout: Layout to coalesce
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        The coalesced layout
        
    Example:
        >>> layout = rocir.make_layout((c2, (c1, c6)), stride=(c1, (c6, c2)))
        >>> coalesced = rocir.coalesce(layout)  # Simplifies to 12:1
    """
    from _mlir import ir as _ir
    
    loc = _get_location(loc)
    result_type = LayoutType.get(-1)
    
    with ip or InsertionPoint.current:
        # Create the operation directly using generic OpView
        unwrapped = _unwrap_value(layout)
        op = _ir.Operation.create(
            "rocir.coalesce",
            results=[result_type],
            operands=[unwrapped],
            loc=loc
        )
        return op.results[0]


# Product operations

def logical_product(block: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the logical product of two layouts (basic tiling).
    
    Args:
        block: Block layout
        tiler: Tiler layout
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        The tiled layout
    """
    
    loc = _get_location(loc)
    result_type = LayoutType.get(-1)
    
    with ip or InsertionPoint.current:
        return rocir_ops.LogicalProductOp(result_type, _unwrap_value(block), _unwrap_value(tiler), loc=loc).result


def zipped_product(block: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the zipped product of two layouts."""
    
    loc = _get_location(loc)
    result_type = LayoutType.get(-1)
    
    with ip or InsertionPoint.current:
        return rocir_ops.ZippedProductOp(result_type, _unwrap_value(block), _unwrap_value(tiler), loc=loc).result


def tiled_product(block: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the tiled product of two layouts."""
    
    loc = _get_location(loc)
    result_type = LayoutType.get(-1)
    
    with ip or InsertionPoint.current:
        return rocir_ops.TiledProductOp(result_type, _unwrap_value(block), _unwrap_value(tiler), loc=loc).result


def flat_product(block: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the flat product of two layouts."""
    
    loc = _get_location(loc)
    result_type = LayoutType.get(-1)
    
    with ip or InsertionPoint.current:
        return rocir_ops.FlatProductOp(result_type, _unwrap_value(block), _unwrap_value(tiler), loc=loc).result


def raked_product(block: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the raked product of two layouts."""
    
    loc = _get_location(loc)
    result_type = LayoutType.get(-1)
    
    with ip or InsertionPoint.current:
        return rocir_ops.RakedProductOp(result_type, _unwrap_value(block), _unwrap_value(tiler), loc=loc).result


def blocked_product(block: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the blocked product of two layouts."""
    
    loc = _get_location(loc)
    result_type = LayoutType.get(-1)
    
    with ip or InsertionPoint.current:
        return rocir_ops.BlockedProductOp(result_type, _unwrap_value(block), _unwrap_value(tiler), loc=loc).result


# Divide operations

def logical_divide(layout: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Divide a layout by a tiler (basic partitioning).
    
    Args:
        layout: Layout to partition
        tiler: Tiler layout
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        The partitioned layout
    """
    
    loc = _get_location(loc)
    result_type = LayoutType.get(-1)
    
    with ip or InsertionPoint.current:
        return rocir_ops.LogicalDivideOp(result_type, _unwrap_value(layout), _unwrap_value(tiler), loc=loc).result


def zipped_divide(layout_or_tensor, tiler_or_shape, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None):
    """Compute the zipped divide of a layout or partition a tensor view.
    
    This function handles two use cases:
    1. Layout division: zipped_divide(layout, tiler) -> divided layout
    2. Tensor partitioning: zipped_divide(tensor_view, tile_shape) -> ZippedTensor
    
    Args:
        layout_or_tensor: Either a layout Value or a TensorView
        tiler_or_shape: Either a tiler layout Value or a tile shape tuple
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        Either a divided layout Value or a ZippedTensor
    """
    # Check if this is a TensorView operation
    if isinstance(layout_or_tensor, TensorView):
        # TensorView partitioning case
        return ZippedTensor(layout_or_tensor, tiler_or_shape)
    
    # Layout division case
    loc = _get_location(loc)
    result_type = LayoutType.get(-1)
    
    with ip or InsertionPoint.current:
        return rocir_ops.ZippedDivideOp(result_type, _unwrap_value(layout_or_tensor), _unwrap_value(tiler_or_shape), loc=loc).result


def tiled_divide(layout: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the tiled divide of a layout."""
    
    loc = _get_location(loc)
    result_type = LayoutType.get(-1)
    
    with ip or InsertionPoint.current:
        return rocir_ops.TiledDivideOp(result_type, _unwrap_value(layout), _unwrap_value(tiler), loc=loc).result


def flat_divide(layout: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the flat divide of a layout."""
    
    loc = _get_location(loc)
    result_type = LayoutType.get(-1)
    
    with ip or InsertionPoint.current:
        return rocir_ops.FlatDivideOp(result_type, _unwrap_value(layout), _unwrap_value(tiler), loc=loc).result


# Local operations

def local_partition(layout: Value, tile: Value, index: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Partition a layout for a specific thread or block index.
    
    Args:
        layout: Layout to partition
        tile: Tile layout
        index: Thread/block index
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        The partitioned layout for the given index
        
    Example:
        >>> # Partition data among threads
        >>> thread_data = rocir.local_partition(global_layout, tile, thread_idx)
    """
    
    loc = _get_location(loc)
    result_type = LayoutType.get(-1)
    
    with ip or InsertionPoint.current:
        return rocir_ops.LocalPartitionOp(result_type, _unwrap_value(layout), _unwrap_value(tile), _unwrap_value(index), loc=loc).result


def local_tile(layout: Value, tiler: Value, coord: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Extract a tile from a layout at specific coordinates.
    
    Args:
        layout: Layout to tile
        tiler: Tile shape
        coord: Coordinate to extract
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        The tile at the given coordinate
        
    Example:
        >>> # Extract CTA tile at block coordinates
        >>> cta_data = rocir.local_tile(global_layout, cta_shape, block_coord)
    """
    
    loc = _get_location(loc)
    result_type = layout.type
    
    with ip or InsertionPoint.current:
        return rocir_ops.LocalTileOp(result_type, _unwrap_value(layout), _unwrap_value(tiler), _unwrap_value(coord), loc=loc).result


#===----------------------------------------------------------------------===//
# Copy Atom and Tiled Copy Classes
#===----------------------------------------------------------------------===//

class CopyAtom:
    """Copy atom descriptor for data movement operations.
    
    Encapsulates a copy operation with metadata about vector size and coalescing.
    Used to construct tiled copy operations.
    """
    
    def __init__(self, element_type: Type, vector_size: int, is_coalesced: bool = True):
        """Initialize a copy atom.
        
        Args:
            element_type: Element type being copied (e.g., f16, f32)
            vector_size: Number of elements per copy instruction
            is_coalesced: Whether memory accesses are coalesced
        """
        self.element_type = element_type
        self.vector_size = vector_size
        self.is_coalesced = is_coalesced
        self._value = None  # Will be set when MLIR op is created
    
    def __repr__(self):
        return f"CopyAtom({self.element_type}, vec={self.vector_size}, coalesced={self.is_coalesced})"


class TiledCopy:
    """Tiled copy operation descriptor.
    
    Represents a data movement operation distributed across threads in a block.
    Created by combining a CopyAtom with thread-value layouts.
    """
    
    def __init__(self, copy_atom: CopyAtom, tv_layout=None, tiler=None):
        """Initialize a tiled copy.
        
        Args:
            copy_atom: Base copy atom
            tv_layout: Thread-value layout (optional)
            tiler: Tiler shape (optional)
        """
        self.copy_atom = copy_atom
        self.tv_layout = tv_layout
        self.tiler = tiler
        self._value = None  # MLIR value
    
    def get_slice(self, thread_idx: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None):
        """Get per-thread slice of the tiled copy.
        
        Args:
            thread_idx: Thread index within the block
            loc: Optional source location
            ip: Optional insertion point
            
        Returns:
            ThrCopy instance for the specific thread
        """
        return ThrCopy(self, thread_idx)
    
    def partition_S(self, tensor: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
        """Partition source tensor according to this tiled copy's layout.
        
        Args:
            tensor: Tensor to partition
            loc: Optional source location
            ip: Optional insertion point
            
        Returns:
            Partitioned tensor
        """
        # This would use a partition operation in MLIR
        # For now, return the tensor (placeholder)
        return tensor
    
    def partition_D(self, tensor: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
        """Partition destination tensor according to this tiled copy's layout.
        
        Args:
            tensor: Tensor to partition
            loc: Optional source location
            ip: Optional insertion point
            
        Returns:
            Partitioned tensor
        """
        return tensor
    
    def __repr__(self):
        return f"TiledCopy({self.copy_atom})"


class ThrCopy:
    """Per-thread copy descriptor.
    
    Represents the portion of a tiled copy assigned to a specific thread.
    """
    
    def __init__(self, tiled_copy: TiledCopy, thread_idx: Value):
        """Initialize per-thread copy.
        
        Args:
            tiled_copy: Parent tiled copy
            thread_idx: Thread index
        """
        self.tiled_copy = tiled_copy
        self.thread_idx = thread_idx
    
    def partition_S(self, tensor: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
        """Partition source tensor for this thread.
        
        Args:
            tensor: Tensor to partition
            loc: Optional source location
            ip: Optional insertion point
            
        Returns:
            Thread's portion of the tensor
        """
        return partition_src(self.tiled_copy, tensor, self.thread_idx, loc=loc, ip=ip)
    
    def partition_D(self, tensor: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
        """Partition destination tensor for this thread."""
        return partition_dst(self.tiled_copy, tensor, self.thread_idx, loc=loc, ip=ip)
    
    def __repr__(self):
        return f"ThrCopy({self.tiled_copy}, tid={self.thread_idx})"


#===----------------------------------------------------------------------===//
# Tensor and Fragment Operations
#===----------------------------------------------------------------------===//

class Fragment:
    """Fragment (register memory tensor) wrapper.
    
    Represents a tensor allocated in register memory for per-thread computation.
    """
    
    def __init__(self, value: Value, element_type: Type = None):
        """Initialize fragment.
        
        Args:
            value: Underlying MLIR value
            element_type: Element type of the fragment
        """
        self._value = value
        self.element_type = element_type
        self.shape = None
    
    def load(self):
        """Load fragment data for computation.
        
        Returns:
            The loaded value (identity for now, will be optimized by compiler)
        """
        return self._value
    
    def store(self, value):
        """Store computed value into fragment.
        
        Args:
            value: Value to store
        """
        # In a full implementation, this would generate appropriate store operations
        # For now, this is handled by the compiler optimization
        pass
    
    def __getitem__(self, index):
        """Access fragment element at index."""
        # Would use appropriate indexing operation
        return self._value
    
    def __setitem__(self, index, value):
        """Set fragment element at index."""
        pass


def make_fragment_like(template, element_type: Type = None, loc: Optional[Location] = None,
                       ip: Optional[InsertionPoint] = None):
    """Create a fragment buffer mirroring the provided template."""
    loc = _get_location(loc)
    if isinstance(template, TensorView):
        elem_ty = element_type or template.element_type
        memref_type = MemRefType.get(template.shape, elem_ty)
        with ip or InsertionPoint.current:
            buffer = memref.AllocaOp(memref_type, [], [], loc=loc).result
        zeros = [0] * template.rank
        return TensorView(
            buffer,
            template.shape,
            strides=template.strides,
            base_indices=zeros,
            element_type=elem_ty,
            wrap_arith=True,
        )
    if isinstance(template, TiledCopy):
        if element_type is None:
            raise ValueError("make_fragment_like(tiled_copy, element_type=...) requires element_type")
        val_shape = template.val_shape
        memref_type = MemRefType.get(val_shape, element_type)
        with ip or InsertionPoint.current:
            buffer = memref.AllocaOp(memref_type, [], [], loc=loc).result
        zeros = [0] * len(val_shape)
        return TensorView(
            buffer,
            val_shape,
            strides=None,
            base_indices=zeros,
            element_type=element_type,
            wrap_arith=True,
        )
    raise ValueError("Unsupported template type for make_fragment_like")


def make_rmem_tensor(shape, element_type: Type, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> TensorView:
    """Create a tensor in register memory with given shape and type.
    
    Args:
        shape: Shape tuple
        element_type: Element type
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        Fragment allocated in register memory (memref)
        
    Example:
        >>> frgPred = rocir.make_rmem_tensor((4, 4), Boolean)
    """
    loc = _get_location(loc)
    if not isinstance(shape, (tuple, list)):
        raise NotImplementedError("Dynamic shape not supported in make_rmem_tensor yet")
    memref_type = MemRefType.get(shape, element_type)
    with ip or InsertionPoint.current:
        buffer = memref.AllocaOp(memref_type, [], [], loc=loc).result
    zeros = [0] * len(shape)
    return TensorView(buffer, tuple(int(s) for s in shape), strides=None, base_indices=zeros, element_type=element_type)


def make_identity_tensor(shape, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None):
    """Create an identity tensor (coordinate tensor) with given shape.
    
    An identity tensor maps each coordinate to itself, useful for tracking
    coordinates during partitioning.
    
    Args:
        shape: Shape tuple or Value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        Identity tensor
        
    Example:
        >>> idC = rocir.make_identity_tensor(mC.shape)
        >>> cC = rocir.zipped_divide(idC, tiler=tiler_mn)
    """
    loc = _get_location(loc)
    if isinstance(shape, TensorView):
        dims = shape.shape
    elif isinstance(shape, (tuple, list)):
        dims = tuple(int(s) for s in shape)
    else:
        raise ValueError("make_identity_tensor expects a TensorView or shape tuple")
    strides = []
    stride = 1
    for size in reversed(dims):
        strides.insert(0, stride)
        stride *= int(size)
    base = [_to_index_value(0, loc) for _ in dims]
    return TensorView(None, dims, strides=strides, base_indices=base, element_type=IndexType.get())


def make_ordered_layout(shape: tuple, order: tuple = None, stride: tuple = None,
                       loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Create a layout with specified dimension ordering.
    
    Args:
        shape: Shape tuple
        order: Dimension order tuple (e.g., (1, 0) for column-major)
        stride: Optional explicit stride (computed from order if not provided)
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        Layout value
        
    Example:
        >>> # Row-major: order (1, 0) means dim 1 varies fastest
        >>> thr_layout = rocir.make_ordered_layout((4, 32), order=(1, 0))
        >>> # Equivalent to make_layout((4, 32), stride=(32, 1))
    """
    if stride is not None:
        return make_layout(shape, stride=stride, loc=loc, ip=ip)
    
    if order is None:
        order = tuple(range(len(shape) - 1, -1, -1))  # Default: row-major
    
    # Compute strides from order
    # For order (1, 0): fastest dimension is 1, so stride[1] = 1, stride[0] = shape[1]
    computed_stride = [1] * len(shape)
    sorted_dims = sorted(range(len(shape)), key=lambda i: order[i])
    
    stride_val = 1
    for dim in sorted_dims:
        computed_stride[dim] = stride_val
        stride_val *= shape[dim]
    
    return make_layout(shape, stride=tuple(computed_stride), loc=loc, ip=ip)


def product_each(shape: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute element-wise product of shape dimensions.
    
    For nested shapes, computes the product at each hierarchical level.
    Example: ((2,3), (4,5)) -> (6, 20)
    
    Args:
        shape: Input shape
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        Shape with products computed at each level
        
    Example:
        >>> shape = make_shape((2, 3), (4, 5))  # Nested shape
        >>> result = product_each(shape)  # -> (6, 20)
    """
    loc = _get_location(loc)
    shape = _unwrap_value(shape)
    with _get_insertion_point(ip):
        # from _mlir.dialects import rocir as rocir_ops  # Not available
        op = rocir_ops.ProductEachOp(shape, loc=loc)
        return op.result


def make_layout_tv(thr_layout: Value, val_layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None):
    """Create tiler and TV layout from thread and value layouts.
    
    Combines thread layout (thread → tile coords) and value layout (value → tile coords)
    to produce a tiler and thread-value layout.
    
    Computes:
    1. layout_mn = raked_product(thr_layout, val_layout)
    2. tiler_mn = product_each(layout_mn.shape)
    3. layout_tv = composition(right_inverse(layout_mn), make_layout((thr_size, val_size)))
    
    Args:
        thr_layout: Thread layout
        val_layout: Value layout
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        Tuple of (tiler, tv_layout)
        
    Example:
        >>> thr_layout = rocir.make_layout((4, 32), stride=(32, 1))
        >>> val_layout = rocir.make_layout((4, 4), stride=(4, 1))
        >>> tiler_mn, tv_layout = rocir.make_layout_tv(thr_layout, val_layout)
    """
    loc = _get_location(loc)
    thr_layout = _unwrap_value(thr_layout)
    val_layout = _unwrap_value(val_layout)
    with _get_insertion_point(ip):
        # Note: MakeLayoutTVOp not implemented in MLIR, compute at Python level
        # This is a placeholder - full implementation would use product_each
        # and other layout algebra operations
        return (thr_layout, val_layout)


def elem_less(a, b, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None):
    """Element-wise less-than comparison.
    
    Args:
        a: First value
        b: Second value (can be Shape or Value)
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        Boolean result
        
    Example:
        >>> val = rocir.elem_less(thrCrd[i], shape)
        >>> frgPred[i] = val
    """
    from _mlir.dialects import arith
    loc = _get_location(loc)

    def _to_limits(b_val, rank):
        if isinstance(b_val, (tuple, list)):
            return [_to_index_value(dim, loc) for dim in b_val]
        return [_to_index_value(b_val, loc) for _ in range(rank)]

    with ip or InsertionPoint.current:
        if isinstance(a, (list, tuple)):
            limits = _to_limits(b, len(a))
            cond = None
            for coord, limit in zip(a, limits):
                cmp = arith.CmpIOp(arith.CmpIPredicate.ult, _unwrap_value(coord), _unwrap_value(limit), loc=loc).result
                if cond is None:
                    cond = cmp
                else:
                    cond = arith.AndIOp(_unwrap_value(cond), _unwrap_value(cmp), loc=loc).result
            return _unwrap_value(cond)
        a_val = _unwrap_value(a)
        limit = _to_limits(b, 1)[0]
        return arith.CmpIOp(arith.CmpIPredicate.ult, a_val, _unwrap_value(limit), loc=loc).result


#===----------------------------------------------------------------------===//
# Copy Atom Construction Functions
#===----------------------------------------------------------------------===//

def make_copy_atom(element_type: Type, vector_size: int = 8, is_coalesced: bool = True, 
                   loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> CopyAtom:
    """Create a copy atom for data movement operations.
    
    Args:
        element_type: Type of elements being copied (e.g., f16, f32)
        vector_size: Number of elements per copy instruction (default: 8)
        is_coalesced: Whether accesses should be coalesced (default: True)
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        CopyAtom descriptor
        
    Example:
        >>> from _mlir.ir import F16Type
        >>> atom = rocir.make_copy_atom(F16Type.get(), vector_size=8)
    """
    return CopyAtom(element_type, vector_size, is_coalesced)


def make_tiled_copy_tv(copy_atom: CopyAtom, thr_layout: Value, val_layout: Value,
                       thr_shape: Optional[tuple] = None,
                       val_shape: Optional[tuple] = None,
                       loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> TiledCopy:
    """Create a tiled copy from copy atom and separate thread/value layouts.
    
    Args:
        copy_atom: Copy atom descriptor
        thr_layout: Thread layout mapping threads to tile coordinates
        val_layout: Value layout mapping per-thread values to tile coordinates
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        TiledCopy descriptor
        
    Example:
        >>> atom = rocir.make_copy_atom(F16Type.get(), 8)
        >>> thr_layout = rocir.make_layout((4, 32), stride=(32, 1))
        >>> val_layout = rocir.make_layout((4, 4), stride=(4, 1))
        >>> tiled_copy = rocir.make_tiled_copy_tv(atom, thr_layout, val_layout)
    """
    tiled_copy = TiledCopy(copy_atom, tv_layout=None, tiler=None)
    tiled_copy.thr_layout = thr_layout
    tiled_copy.val_layout = val_layout
    tiled_copy.thr_shape = thr_shape
    tiled_copy.val_shape = val_shape
    return tiled_copy


def make_tensor(memref, layout=None, shape=None, strides=None, loc: Optional[Location] = None,
                ip: Optional[InsertionPoint] = None) -> TensorView:
    """Create a tensor view from a memref with a specific layout and shape."""
    if shape is None:
        raise ValueError("make_tensor requires explicit shape information")
    _get_location(loc)
    if layout is not None:
        _unwrap_value(layout)
    if strides is None:
        strides = []
        stride = 1
        for size in reversed(shape):
            strides.insert(0, stride)
            stride *= int(size)
    base = [0] * len(shape)
    return TensorView(memref, shape, strides=strides, base_indices=base)


class ZippedTensor:
    """Represents a tensor partitioned into tiles for each block."""

    def __init__(self, tensor_view: TensorView, tile_shape):
        self.tensor = tensor_view
        self.tile_shape = tuple(int(s) for s in tile_shape)
        self.rank = tensor_view.rank
        self.block_shape = tuple(
            max(1, (tensor_view.shape[i] + self.tile_shape[i] - 1) // self.tile_shape[i])
            for i in range(self.rank)
        )

    def _normalize_indices(self, key):
        if isinstance(key, tuple):
            if len(key) == 2 and isinstance(key[0], tuple):
                key = key[1]
            if len(key) == self.rank:
                return [_to_index_value(k) for k in key]
        # Fallback to linear index
        return _linear_idx_to_coords(_to_index_value(key), self.block_shape)

    def __getitem__(self, block_idx):
        coords = self._normalize_indices(block_idx)
        base = []
        for dim in range(self.rank):
            offset = _scale_index(coords[dim], self.tile_shape[dim])
            base.append(_add_index(self.tensor.base_indices[dim], offset))
        return TensorView(
            self.tensor.memref,
            self.tile_shape,
            strides=self.tensor.strides,
            base_indices=base,
            element_type=self.tensor.element_type,
        )


def partition_src(tiled_copy, tensor, thread_id, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None):
    """Partition source tensor for a specific thread."""
    loc = _get_location(loc)
    if not isinstance(tensor, TensorView):
        raise ValueError("partition_src expects a TensorView produced by make_tensor")
    thr_shape = getattr(tiled_copy, "thr_shape", None)
    val_shape = getattr(tiled_copy, "val_shape", None)
    if thr_shape is None or val_shape is None:
        raise ValueError("tiled_copy is missing thr_shape/val_shape metadata")
    tid_val = _to_index_value(thread_id)
    coords = _linear_idx_to_coords(tid_val, thr_shape)
    base_indices = []
    for dim in range(tensor.rank):
        coord = coords[dim] if dim < len(coords) else _to_index_value(0)
        step = val_shape[dim] if dim < len(val_shape) else 1
        offset = _scale_index(coord, step)
        base_indices.append(_add_index(tensor.base_indices[dim], offset))
    return TensorView(tensor.memref, val_shape, strides=tensor.strides, base_indices=base_indices, element_type=tensor.element_type)


def partition_dst(tiled_copy, tensor, thread_id, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None):
    """Partition destination tensor for a specific thread."""
    return partition_src(tiled_copy, tensor, thread_id, loc=loc, ip=ip)


def fragment_load(fragment, index, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None):
    """Load an element from a register fragment."""
    from _mlir.dialects import memref as memref_dialect
    loc = _get_location(loc)
    fragment = _unwrap_value(fragment)
    index = _unwrap_value(index)
    with _get_insertion_point(ip):
        return memref_dialect.load(fragment, [index])


def fragment_store(value, fragment, index, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None):
    """Store an element to a register fragment."""
    from _mlir.dialects import memref as memref_dialect
    loc = _get_location(loc)
    value = _unwrap_value(value)
    fragment = _unwrap_value(fragment)
    index = _unwrap_value(index)
    with _get_insertion_point(ip):
        memref_dialect.store(value, fragment, [index])


def _normalize_indices_to_memref(memref_val: Value, indices: List[Value], strides: Optional[tuple], loc: Location) -> List[Value]:
    """Normalize indices to match the backing memref's rank.
    
    If the memref rank is less than the number of indices, linearize the multi-dimensional 
    indices into a flat index using the strides.
    
    Args:
        memref_val: The memref Value to access
        indices: List of index Values (possibly multi-dimensional)
        strides: Strides tuple from TensorView (or None for default row-major)
        loc: Source location for generated operations
        
    Returns:
        List of index Values matching the memref's rank
    """
    from _mlir.ir import MemRefType
    
    memref_type = memref_val.type
    if not isinstance(memref_type, MemRefType):
        # Not a memref, return indices as-is
        return indices
    
    memref_rank = memref_type.rank
    num_indices = len(indices)
    
    if memref_rank == num_indices:
        # Ranks match, no conversion needed
        return indices
    
    if memref_rank == 1 and num_indices > 1:
        # Linearize multi-dimensional indices into 1D
        # linear_idx = idx[0] * stride[0] + idx[1] * stride[1] + ...
        
        # If no strides provided, assume row-major (C-style) layout
        if strides is None or len(strides) != num_indices:
            # Default row-major strides
            # For a 2D array [M, N]: strides = [N, 1]
            # We can infer from memref shape if available
            shape = memref_type.shape
            if len(shape) == 1 and len(shape) > 0:
                # 1D memref with known size
                # Assume the original layout was row-major with last stride = 1
                # But we don't have the original shape, so we need to compute from strides
                # For now, assume the last stride is 1 and work backwards
                # This is a simplified heuristic
                computed_strides = [1] * num_indices
                # Example: for 2D with total size N, strides = [N, 1]
                # But we don't know N from just the 1D memref
                # FALLBACK: We'll use the provided strides from TensorView
                pass
        
        # Use the strides from TensorView
        linear_idx = None
        for i, (idx, stride) in enumerate(zip(indices, strides if strides else [1]*num_indices)):
            # Ensure idx is a proper Value by unwrapping
            idx = _unwrap_value(idx)
            
            if stride == 1:
                term = idx
            else:
                stride_val = _to_index_value(stride, loc)
                term = arith.muli(idx, stride_val, loc=loc)
                # Unwrap in case arith.muli returns an ArithValue wrapper
                term = _unwrap_value(term)
            
            if linear_idx is None:
                linear_idx = term
            else:
                # Unwrap both operands to ensure they're raw MLIR Values
                linear_idx = _unwrap_value(linear_idx)
                term = _unwrap_value(term)
                linear_idx = arith.addi(linear_idx, term, loc=loc)
        
        # Unwrap the final result before returning
        return [_unwrap_value(linear_idx)]
    
    # For other cases, return indices as-is and let MLIR validation catch mismatches
    return indices


def copy(copy_desc, src, dst, 
         src_indices: Optional[List[Value]] = None,
         dst_indices: Optional[List[Value]] = None,
         pred: Optional[Value] = None,
         *,
         nontemporal: Optional[bool] = None,
         alignment: Optional[int] = None,
         return_vector: bool = False,
         loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> None:
    """Execute a copy operation using the given copy atom.
    
    Args:
        copy_desc: Copy atom or tiled copy descriptor
        src: Source tensor (memref)
        dst: Destination tensor (memref)
        src_indices: Indices for source access
        dst_indices: Indices for destination access
        pred: Optional predicate mask for conditional copying. 
              Can be a TensorView (element-wise mask) or a scalar Value (broadcast mask).
        loc: Optional source location
        ip: Optional insertion point
        
    Example:
        >>> rocir.copy(atom, src, dst, src_indices=[i,j], dst_indices=[k])
    """
    from _mlir.dialects import memref as memref_dialect
    loc = _get_location(loc)

    # If return_vector=True, we capture the last vector value loaded by the vectorized path.
    captured_vec = {"val": None}

    def emit_tensor_copy(copy_shape, src_view: TensorView, dst_view: TensorView, pred_view: Optional[Union[TensorView, Value]]):
        from _mlir.dialects import vector
        from _mlir.ir import VectorType

        # Attempt vectorization if copy_desc has vector_size
        vector_size = 1
        if isinstance(copy_desc, TiledCopy) and copy_desc.copy_atom:
            vector_size = copy_desc.copy_atom.vector_size
        elif hasattr(copy_desc, "vector_size"):
            vector_size = copy_desc.vector_size

        def recurse(dim, src_idx, dst_idx, pred_idx):
            if dim == len(copy_shape):
                # Scalar fall-back (should be covered by vectorized path if possible)
                load_idx = _normalize_indices_to_memref(src_view.memref, [_unwrap_value(i) for i in src_idx], src_view.strides, loc)
                store_idx = _normalize_indices_to_memref(dst_view.memref, [_unwrap_value(i) for i in dst_idx], dst_view.strides, loc)
                load_op = memref_dialect.load(src_view.memref, load_idx)
                val = load_op.result if hasattr(load_op, "result") else load_op
                val = _unwrap_value(val)
                
                cond = None
                if pred_view is not None:
                    if isinstance(pred_view, TensorView):
                        # Scalar masked store logic from tensor view
                        pred_idx_vals = [_unwrap_value(i) for i in pred_idx]
                        pred_op = memref_dialect.load(pred_view.memref, pred_idx_vals)
                        flag = pred_op.result if hasattr(pred_op, "result") else pred_op
                        flag = _unwrap_value(flag)
                        zero_op = arith.ConstantOp(flag.type, IntegerAttr.get(flag.type, 0), loc=loc)
                        zero = _unwrap_value(zero_op.result if hasattr(zero_op, "result") else zero_op)
                        cond = arith.CmpIOp(arith.CmpIPredicate.ne, flag, zero, loc=loc).result
                    else:
                        # Scalar broadcast mask
                        cond = _unwrap_value(pred_view)

                if cond is not None:
                    cond = _unwrap_value(cond)
                    
                    # Optimization: If cond is a broadcast scalar mask (not dependent on indices inside recurse loop),
                    # we can hoist the if check outside. However, here we are inside the scalar/vector loop.
                    # Since we are inside the scalar recursion, 'cond' might depend on 'pred_idx' if it came from TensorView.
                    # If it came from scalar broadcast, it is invariant.
                    
                    # For scalar broadcast, 'pred_view' is a Value (not TensorView).
                    # 'cond' is that value.
                    
                    if_op = scf.IfOp(cond, [], loc=loc)
                    with InsertionPoint(if_op.then_block):
                        memref_dialect.store(val, dst_view.memref, store_idx)
                        scf.YieldOp([])
                else:
                    memref_dialect.store(val, dst_view.memref, store_idx)
                return

            extent = int(copy_shape[dim])
            
            # Check for vectorization opportunity on the last dimension
            if dim == len(copy_shape) - 1 and vector_size > 1 and extent % vector_size == 0:
                # 1. Verify contiguity (stride=1) for innermost dim
                # Simplified check: assume TensorView with default inner stride 1 if not specified
                # Ideally check src_view.strides[-1] == 1
                
                # Iterate in chunks of vector_size
                base_src = src_view.base_indices[dim] if dim < len(src_view.base_indices) else _to_index_value(0, loc)
                base_dst = dst_view.base_indices[dim] if dim < len(dst_view.base_indices) else _to_index_value(0, loc)
                base_pred = pred_view.base_indices[dim] if isinstance(pred_view, TensorView) else None

                # Optimization: Scalar Broadcast Mask Hoisting
                # If pred_view is a scalar Value (broadcast mask), we can hoist the check outside the vector loop.
                hoisted_cond = None
                if pred_view is not None and not isinstance(pred_view, TensorView):
                     hoisted_cond = _unwrap_value(pred_view)

                def emit_vector_loop_body():
                    for i in range(0, extent, vector_size):
                        off = _to_index_value(i, loc)
                        # For vector load, we use the start index of the vector
                        vec_src_idx = src_idx + [_add_index(base_src, off)]
                        vec_dst_idx = dst_idx + [_add_index(base_dst, off)]
                        
                        # Prepare vector type
                        elem_type = src_view.element_type
                        vec_type = VectorType.get((vector_size,), elem_type)
                        
                        load_indices = _normalize_indices_to_memref(src_view.memref, [_unwrap_value(idx) for idx in vec_src_idx], src_view.strides, loc)
                        store_indices = _normalize_indices_to_memref(dst_view.memref, [_unwrap_value(idx) for idx in vec_dst_idx], dst_view.strides, loc)

                        # Vector Load
                        vec_val = vector.load(
                            vec_type,
                            src_view.memref,
                            load_indices,
                            nontemporal=nontemporal,
                            alignment=alignment,
                        )
                        if return_vector and captured_vec["val"] is None:
                            captured_vec["val"] = vec_val

                        # Handle Predicate (TensorView case only inside loop)
                        if pred_view is not None and isinstance(pred_view, TensorView):
                            cond = None
                            # FALLBACK STRATEGY for TensorView mask: 
                            # Check first element of the vector range
                            curr_pred_base = _add_index(base_pred, off)
                            p_idx = pred_idx + [curr_pred_base]
                            p_idx_vals = [_unwrap_value(p) for p in p_idx]
                            
                            pred_val_op = memref_dialect.load(pred_view.memref, p_idx_vals)
                            flag = pred_val_op.result if hasattr(pred_val_op, "result") else pred_val_op
                            flag = _unwrap_value(flag)
                            
                            zero_op = arith.ConstantOp(flag.type, IntegerAttr.get(flag.type, 0), loc=loc)
                            zero = _unwrap_value(zero_op.result if hasattr(zero_op, "result") else zero_op)
                            cond = arith.CmpIOp(arith.CmpIPredicate.ne, flag, zero, loc=loc).result
                            
                            cond = _unwrap_value(cond)
                            if_op = scf.IfOp(cond, [], loc=loc)
                            with InsertionPoint(if_op.then_block):
                                vector.store(
                                    vec_val,
                                    dst_view.memref,
                                    store_indices,
                                    nontemporal=nontemporal,
                                    alignment=alignment,
                                )
                                scf.YieldOp([])
                        else:
                            # No predicate or handled by hoisted check
                            vector.store(
                                vec_val,
                                dst_view.memref,
                                store_indices,
                                nontemporal=nontemporal,
                                alignment=alignment,
                            )

                if hoisted_cond is not None:
                    # Hoist the scf.If outside the vector loop
                    if_op = scf.IfOp(hoisted_cond, [], loc=loc)
                    with InsertionPoint(if_op.then_block):
                        emit_vector_loop_body()
                        scf.YieldOp([])
                else:
                    emit_vector_loop_body()
                
                return

            # Scalar recursion
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

    if isinstance(copy_desc, TiledCopy) and isinstance(src, TensorView) and isinstance(dst, TensorView):
        # pred can be TensorView or scalar Value
        emit_tensor_copy(copy_desc.val_shape, src, dst, pred)
        return captured_vec["val"] if return_vector else None

    if isinstance(src, TensorView) and isinstance(dst, TensorView):
        # pred can be TensorView or scalar Value
        emit_tensor_copy(src.shape, src, dst, pred)
        return captured_vec["val"] if return_vector else None

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


#===----------------------------------------------------------------------===//
# Printing operations
#===----------------------------------------------------------------------===//

# Use Python's built-in print for static compile-time values
# This mirrors the behavior where print shows compile-time information
print = print  # Re-export Python's built-in print


def printf(format_str: str, *args, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None):
    """Print formatted output at runtime (dynamic values).
    
    This function prints dynamic values that are only known at runtime.
    It uses GPU printf to display values during kernel execution.
    
    Args:
        format_str: Format string (e.g., "value: {}")
        *args: Values to print (can be dynamic runtime values)
        loc: Optional source location
        ip: Optional insertion point
        
    Example:
        >>> # Print static value (compile time)
        >>> rocir.print(">>>", b)  # Shows static value
        >>> rocir.print(">>>", a)  # Shows "?" for dynamic value
        >>> 
        >>> # Print dynamic value (runtime)
        >>> rocir.printf(">?? {}", a)  # Shows actual runtime value
        >>> rocir.printf(">?? {}", b)  # Also works for static values
        >>> 
        >>> # Print layout
        >>> layout = rocir.make_layout(shape, stride)
        >>> rocir.print(">>>", layout)  # Shows layout with "?" for dynamic parts
        >>> rocir.printf(">?? {}", layout)  # Shows actual runtime values
    
    Note:
        - Use `rocir.print` (Python's print) for compile-time/static values
        - Use `rocir.printf` for runtime/dynamic values
        - Format strings use "{}" as placeholders (similar to Python f-strings)
    """
    from _mlir.dialects import gpu as _gpu
    
    loc = _get_location(loc)
    
    # Unwrap all argument values
    unwrapped_args = [_unwrap_value(arg) for arg in args]
    
    with ip or InsertionPoint.current:
        return _gpu.printf(format=format_str, args=unwrapped_args, loc=loc, ip=ip)


__all__ = [
    "T",
    "MlirModule",
    "kernel",
    "jit",
    # Types
    "ShapeType",
    "StrideType",
    "LayoutType",
    "CoordType",
    # Basic operations
    "make_shape",
    "make_stride",
    "make_layout",
    "make_coord",
    "crd2idx",
    "idx2crd",
    "size",
    "cosize",
    "rank",
    "get",
    "get_shape",
    "get_stride",
    "composition",
    "coalesce",
    "const_index",
    "to_index",
    "thread_idx",
    "block_idx",
    "block_dim",
    # Product operations
    "logical_product",
    "zipped_product",
    "tiled_product",
    "flat_product",
    "raked_product",
    "blocked_product",
    # Divide operations
    "logical_divide",
    "zipped_divide",
    "tiled_divide",
    "flat_divide",
    # Local operations
    "local_partition",
    "local_tile",
    # Copy atom classes and operations
    "CopyAtom",
    "TiledCopy",
    "ThrCopy",
    "make_copy_atom",
    "make_tiled_copy_tv",
    "copy",
    # Tensor and fragment operations
    "Fragment",
    "make_fragment_like",
    "make_rmem_tensor",
    "make_identity_tensor",
    "make_ordered_layout",
    "make_layout_tv",
    "elem_less",
    # Printing operations
    "print",
    "printf",
]
