"""Python bindings for Flir dialect operations.

This module provides Python wrappers for Flir layout algebra operations,
making it easier to construct layouts and perform layout transformations
from Python code.
"""

from typing import List, Optional, Sequence, Union, Tuple

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
from flydsl.lang.ir.module import MlirModule, kernel, jit
from _mlir.dialects import flir as flir_ops

# Also expose FLIR "extended" wrappers (like test_eltwise_add.py uses) so
# tests can route everything through `flir.*` without importing `mlir.*`.
from . import arith as arith_ext  # noqa: E402
from . import scf as scf_ext      # noqa: E402
from . import gpu as gpu_ext      # noqa: E402



def _get_location(loc: Optional[Location] = None) -> Location:
    """Get location, using current location if none provided."""
    # Some call sites pass `Location.unknown()` explicitly. Treat it as missing
    # so we can still attach a useful debug location.
    try:
        if loc is not None and str(loc) == "loc(unknown)":
            loc = None
    except Exception:
        pass
    if loc is None:
        # Prefer a file/line location for better IR dump debugging.
        try:
            from flydsl.dialects.ext.func import get_user_code_loc

            loc = get_user_code_loc()
        except Exception:
            loc = None
        # `get_user_code_loc()` can legitimately fail in some dynamic/rewritten codepaths
        # and return `loc(unknown)`. Treat that the same as missing so we still inherit
        # a meaningful `Location.current` from the surrounding context.
        try:
            if loc is not None and str(loc) == "loc(unknown)":
                loc = None
        except Exception:
            pass
        # If we couldn't infer a file/line location, prefer inheriting the current
        # location from the surrounding `with Location(...)` context.
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
        from _mlir.dialects import arith
        from _mlir.ir import IndexType, IntegerAttr
        # Prefer a non-unknown location for implicit constant materialization so IR dumps
        # remain actionable.
        loc = _get_location(None)
        op = arith.ConstantOp(IndexType.get(), IntegerAttr.get(IndexType.get(), v), loc=loc)
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


def _extract_rank_from_flir_type_str(type_str: str) -> int:
    """Extract rank from a flir type string like '!flir.layout<2>' or '!flir.layout<(?,?)>'."""
    if "<" not in type_str or ">" not in type_str:
        raise ValueError(f"Cannot extract rank from type string: {type_str}")
    inner = type_str.split("<", 1)[1].rsplit(">", 1)[0].strip()
    # Layout types may encode both shape and stride specs:
    #   !flir.layout<(9,(4,8)):(59,(13,1))>
    # Rank should be derived from the *shape* side only.
    def _split_top_level_colon(s: str) -> tuple[str, str] | None:
        depth = 0
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == ":" and depth == 0:
                return s[:i], s[i + 1 :]
        return None

    if inner.startswith("("):
        parts = _split_top_level_colon(inner)
        if parts is not None:
            inner = parts[0].strip()
        return _count_leaves_in_tuple_spec(inner)
    if len(inner) >= 2 and inner[0] == '"' and inner[-1] == '"':
        inner2 = inner[1:-1]
        parts = _split_top_level_colon(inner2)
        if parts is not None:
            inner2 = parts[0].strip()
        return _count_leaves_in_tuple_spec(inner2)
    # numeric
    return int(inner)


def _extract_tuple_spec_from_flir_type(type_str: str) -> Optional[str]:
    """If type_str is like '!flir.shape<\"(...)\">' or '!flir.stride<\"(...)\">', return '(...)'."""
    if "<" not in type_str or ">" not in type_str:
        return None
    inner = type_str.split("<", 1)[1].rsplit(">", 1)[0]
    inner = inner.strip()
    if len(inner) >= 2 and inner[0] == '"' and inner[-1] == '"':
        return inner[1:-1]
    return None


def _require_index_value(v: Value, what: str) -> Value:
    """Ensure v is an index-typed Value. Raise a clear error otherwise."""
    from _mlir.ir import IndexType
    if not isinstance(v, Value):
        raise TypeError(f"{what} must be an MLIR Value (got {type(v)})")
    if v.type != IndexType.get():
        raise ValueError(f"{what} must have type 'index' (got {v.type})")
    return v


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
                from flydsl.dialects.ext import arith as _arith_ext
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
    try:
        from flydsl.dialects.ext.func import get_user_code_loc

        loc = get_user_code_loc()
    except Exception:
        loc = None
    return gpu.thread_id(axis, loc=loc)


def block_idx(axis: str = "x"):
    """Return the current block index along the given axis."""
    try:
        from flydsl.dialects.ext.func import get_user_code_loc

        loc = get_user_code_loc()
    except Exception:
        loc = None
    return gpu.block_id(axis, loc=loc)


def block_dim(axis: str = "x"):
    """Return the block dimension along the given axis."""
    try:
        from flydsl.dialects.ext.func import get_user_code_loc

        loc = get_user_code_loc()
    except Exception:
        loc = None
    return gpu.block_dim(axis, loc=loc)


class ShapeType(Type):
    """Flir shape type."""
    
    @staticmethod
    def get(rank: int, context=None):
        """Create a shape type with given rank."""
        from _mlir.ir import Context
        if context is None:
            context = Context.current
        spec = "(" + ",".join(["?"] * int(rank)) + ")"
        # Rank-1 can be printed as "?" in C++, but keep tuple form here for clarity/stability.
        return Type.parse(f"!flir.shape<{spec}>", context=context)


class StrideType(Type):
    """Flir stride type."""
    
    @staticmethod
    def get(rank: int, context=None):
        """Create a stride type with given rank."""
        from _mlir.ir import Context
        if context is None:
            context = Context.current
        spec = "(" + ",".join(["?"] * int(rank)) + ")"
        return Type.parse(f"!flir.stride<{spec}>", context=context)


class LayoutType(Type):
    """Flir layout type."""
    
    @staticmethod
    def get(rank: int, context=None):
        """Create a layout type with given rank."""
        from _mlir.ir import Context
        if context is None:
            context = Context.current
        spec = "(" + ",".join(["?"] * int(rank)) + ")"
        return Type.parse(f"!flir.layout<{spec}:{spec}>", context=context)


class CoordType(Type):
    """Flir coordinate type."""
    
    @staticmethod
    def get(rank: int, context=None):
        """Create a coordinate type with given rank."""
        from _mlir.ir import Context
        if context is None:
            context = Context.current
        spec = "(" + ",".join(["?"] * int(rank)) + ")"
        return Type.parse(f"!flir.coord<{spec}>", context=context)


# -----------------------------------------------------------------------------
# Type-level layout algebra inference (Python-side, for wrappers)
# -----------------------------------------------------------------------------

def _split_top_level_colon(spec: str) -> Optional[tuple[str, str]]:
    depth = 0
    for i, ch in enumerate(spec):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == ":" and depth == 0:
            return spec[:i], spec[i + 1 :]
    return None


class _PatternNode:
    __slots__ = ("is_leaf", "value", "children")

    def __init__(self, is_leaf: bool, value: Optional[int] = None, children: Optional[List["_PatternNode"]] = None):
        self.is_leaf = is_leaf
        self.value = value
        self.children = children or []

    @staticmethod
    def leaf(v: Optional[int]) -> "_PatternNode":
        return _PatternNode(True, value=v, children=[])

    @staticmethod
    def tup(children: List["_PatternNode"]) -> "_PatternNode":
        return _PatternNode(False, children=children)

    def flatten_leaves(self, out: List[Optional[int]]) -> None:
        if self.is_leaf:
            out.append(self.value)
            return
        for c in self.children:
            c.flatten_leaves(out)

    def leaf_count(self) -> int:
        leaves: List[Optional[int]] = []
        self.flatten_leaves(leaves)
        return len(leaves)

    def to_tuple_root(self) -> "_PatternNode":
        # layout<shapeSpec:strideSpec> syntax requires tuple-root specs (leading '(').
        if self.is_leaf:
            return _PatternNode.tup([self])
        return self

    def to_spec(self) -> str:
        # Normalize away redundant 1-tuples that can arise from intermediate algebra steps
        # (e.g. returning `_PatternNode.tup([leaf])`). For our type specs, `(x)` and `x`
        # are equivalent except at the root where layout<...> requires a tuple-root.
        def normalize(n: "_PatternNode") -> "_PatternNode":
            if n.is_leaf:
                return n
            kids = [normalize(c) for c in n.children]
            if len(kids) == 1:
                return kids[0]
            return _PatternNode.tup(kids)

        def emit(n: "_PatternNode") -> str:
            if n.is_leaf:
                return "?" if n.value is None else str(n.value)
            return "(" + ",".join(emit(c) for c in n.children) + ")"

        return emit(normalize(self).to_tuple_root())


def _parse_tuple_spec(spec: str) -> _PatternNode:
    # Supports "(9,(4,8))" and "(?,(?,?))". Also accepts leaf specs like "?" or "4".
    s = "".join(ch for ch in spec if not ch.isspace())
    i = 0

    def peek() -> str:
        return s[i] if i < len(s) else "\0"

    def consume(expected: Optional[str] = None) -> str:
        nonlocal i
        if i >= len(s):
            raise ValueError("unexpected end of spec")
        ch = s[i]
        if expected is not None and ch != expected:
            raise ValueError(f"expected '{expected}' but got '{ch}'")
        i += 1
        return ch

    def parse_int() -> int:
        nonlocal i
        neg = False
        if peek() == "-":
            consume("-")
            neg = True
        if not peek().isdigit():
            raise ValueError("expected integer")
        v = 0
        while peek().isdigit():
            v = v * 10 + int(consume())
        return -v if neg else v

    def parse_elem() -> _PatternNode:
        if peek() == "(":
            return parse_tuple()
        if peek() == "?":
            consume("?")
            return _PatternNode.leaf(None)
        return _PatternNode.leaf(parse_int())

    def parse_tuple() -> _PatternNode:
        consume("(")
        if peek() == ")":
            consume(")")
            return _PatternNode.tup([])
        children: List[_PatternNode] = []
        while True:
            children.append(parse_elem())
            if peek() == ",":
                consume(",")
                continue
            break
        consume(")")
        return _PatternNode.tup(children)

    if peek() == "(":
        node = parse_tuple()
    elif peek() == "?":
        consume("?")
        node = _PatternNode.leaf(None)
    else:
        node = _PatternNode.leaf(parse_int())

    if i != len(s):
        raise ValueError(f"trailing characters in spec: {s[i:]}")
    return node


def _parse_layout_type(type_str: str) -> tuple[int, _PatternNode, _PatternNode]:
    """Return (rank, shapeNode, strideNode).

    - For layout<shapeSpec:strideSpec>, rank is leaf count in shape spec.
    - For layout<(...)> (rank-only tuple form), rank is leaf count of that tuple spec, and
      shape/stride are treated as flat tuples of '?' leaves.
    - For layout<rank>, shape/stride are treated as flat tuples of '?' leaves.
    """
    inner = type_str.split("<", 1)[1].rsplit(">", 1)[0].strip()
    if inner.startswith("("):
        split = _split_top_level_colon(inner)
        if split is None:
            # Backward compatible: layout<(...)> only encodes rank.
            rank = _count_leaves_in_tuple_spec(inner)
            flat = _PatternNode.tup([_PatternNode.leaf(None) for _ in range(rank)])
            return rank, flat, flat
        shape_spec, stride_spec = split
        shape = _parse_tuple_spec(shape_spec)
        stride = _parse_tuple_spec(stride_spec)
        return shape.leaf_count(), shape, stride
    try:
        rank = int(inner)
    except Exception:
        rank = -1
    if rank < 0:
        return -1, _PatternNode.leaf(None), _PatternNode.leaf(None)
    flat = _PatternNode.tup([_PatternNode.leaf(None) for _ in range(rank)])
    return rank, flat, flat


def _mul(a: Optional[int], b: Optional[int]) -> Optional[int]:
    return None if (a is None or b is None) else a * b


def _div_ui(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None or b is None or b == 0:
        return None
    return a // b


def _ceil_div_ui(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None or b is None or b == 0:
        return None
    return (a + b - 1) // b


def _min_ui(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None or b is None:
        return None
    return a if a < b else b


def _composition_impl(lhs_shape: _PatternNode, lhs_stride: _PatternNode, rhs_shape: _PatternNode, rhs_stride: _PatternNode):
    # Mirrors FlirToStandard's composition_impl structural behavior:
    # - distribute when RHS is tuple
    # - fold over flattened LHS when RHS is leaf
    if not rhs_shape.is_leaf:
        out_shapes: List[_PatternNode] = []
        out_strides: List[_PatternNode] = []
        for i, sub_shape in enumerate(rhs_shape.children):
            sub_stride = rhs_stride.children[i] if (not rhs_stride.is_leaf and i < len(rhs_stride.children)) else rhs_stride
            s, d = _composition_impl(lhs_shape, lhs_stride, sub_shape, sub_stride)
            out_shapes.append(s)
            out_strides.append(d)
        return _PatternNode.tup(out_shapes), _PatternNode.tup(out_strides)

    if (not rhs_shape.is_leaf) or (not rhs_stride.is_leaf):
        return rhs_shape, rhs_stride

    lhs_shapes: List[Optional[int]] = []
    lhs_strides: List[Optional[int]] = []
    lhs_shape.flatten_leaves(lhs_shapes)
    lhs_stride.flatten_leaves(lhs_strides)

    rest_shape = rhs_shape.value
    rest_stride = rhs_stride.value

    out_shape_leaves: List[Optional[int]] = []
    out_stride_leaves: List[Optional[int]] = []

    for i in range(len(lhs_shapes)):
        curr_shape = lhs_shapes[i]
        curr_stride = lhs_strides[i] if i < len(lhs_strides) else None

        next_shape = _ceil_div_ui(curr_shape, rest_stride)
        next_stride = _ceil_div_ui(rest_stride, curr_shape)

        # Only apply simplifications when statically provable.
        if rest_shape == 1:
            rest_stride = next_stride
            break
        if next_shape == 1:
            rest_stride = next_stride
            continue

        new_shape = _min_ui(next_shape, rest_shape)
        new_stride = _mul(curr_stride, rest_stride)
        out_shape_leaves.append(new_shape)
        out_stride_leaves.append(new_stride)

        rest_shape = _div_ui(rest_shape, new_shape)
        rest_stride = next_stride

    if len(out_shape_leaves) == 0:
        last_lhs_stride = lhs_strides[-1] if lhs_strides else 1
        tail_stride = _mul(rest_stride, last_lhs_stride)
        return _PatternNode.leaf(rest_shape), _PatternNode.leaf(tail_stride)

    if rest_shape == 1:
        return _PatternNode.tup([_PatternNode.leaf(v) for v in out_shape_leaves]), _PatternNode.tup(
            [_PatternNode.leaf(v) for v in out_stride_leaves]
        )

    out_shape_leaves.append(rest_shape)
    last_lhs_stride = lhs_strides[-1] if lhs_strides else 1
    tail_stride = _mul(rest_stride, last_lhs_stride)
    out_stride_leaves.append(tail_stride)
    return _PatternNode.tup([_PatternNode.leaf(v) for v in out_shape_leaves]), _PatternNode.tup(
        [_PatternNode.leaf(v) for v in out_stride_leaves]
    )


def _size_of_shape(shape: _PatternNode) -> Optional[int]:
    leaves: List[Optional[int]] = []
    shape.flatten_leaves(leaves)
    prod: Optional[int] = 1
    for v in leaves:
        prod = _mul(prod, v)
    return prod


def _complement_impl(shape: _PatternNode, stride: _PatternNode, cosize_hi: Optional[int]) -> tuple[_PatternNode, _PatternNode]:
    # Type-level complement used by logical_divide: produces flat tuple leaves.
    shapes: List[Optional[int]] = []
    strides: List[Optional[int]] = []
    shape.flatten_leaves(shapes)
    stride.flatten_leaves(strides)
    if cosize_hi is None:
        return _PatternNode.leaf(None), _PatternNode.leaf(None)

    modes = list(zip(shapes, strides))
    all_static = all(s is not None for _, s in modes)
    if all_static:
        modes.sort(key=lambda x: int(x[1]))  # type: ignore[arg-type]

    curr_stride: Optional[int] = 1
    out_shape_leaves: List[Optional[int]] = []
    out_stride_leaves: List[Optional[int]] = []

    for (m_shape, m_stride) in modes:
        gap = _div_ui(m_stride, curr_stride)
        out_shape_leaves.append(gap)
        out_stride_leaves.append(curr_stride)
        curr_stride = _mul(m_stride, m_shape)

    final_rest = _ceil_div_ui(cosize_hi, curr_stride)
    out_shape_leaves.append(final_rest)
    out_stride_leaves.append(curr_stride)

    return _PatternNode.tup([_PatternNode.leaf(v) for v in out_shape_leaves]), _PatternNode.tup(
        [_PatternNode.leaf(v) for v in out_stride_leaves]
    )


def _infer_layout_type_composition(a: Value, b: Value) -> Type:
    a_rank, a_shape, a_stride = _parse_layout_type(str(a.type))
    b_rank, b_shape, b_stride = _parse_layout_type(str(b.type))
    if a_rank < 0 or b_rank < 0:
        raise ValueError(f"Cannot infer composition type: operand ranks are unknown: {a.type} , {b.type}")

    # Runtime-capable policy: if we can't compute an exact static pattern,
    # return a rank-only dynamic layout type and let lowering compute it.
    if _size_of_shape(a_shape) is None or _size_of_shape(b_shape) is None:
        return LayoutType.get(max(a_rank, b_rank))
    a_stride_leaves: List[Optional[int]] = []
    b_stride_leaves: List[Optional[int]] = []
    a_stride.flatten_leaves(a_stride_leaves)
    b_stride.flatten_leaves(b_stride_leaves)
    if any(v is None for v in a_stride_leaves + b_stride_leaves):
        return LayoutType.get(max(a_rank, b_rank))

    out_shape, out_stride = _composition_impl(a_shape, a_stride, b_shape, b_stride)
    return Type.parse(f"!flir.layout<{out_shape.to_spec()}:{out_stride.to_spec()}>")


def _infer_layout_type_logical_product(block: Value, tiler: Value) -> Type:
    b_rank, b_shape, b_stride = _parse_layout_type(str(block.type))
    t_rank, t_shape, t_stride = _parse_layout_type(str(tiler.type))
    if b_rank < 0 or t_rank < 0:
        raise ValueError(f"Cannot infer logical_product type: operand ranks are unknown: {block.type} , {tiler.type}")

    # Flatten then concatenate. Shapes: [block..., tiler...]
    b_shape_leaves: List[Optional[int]] = []
    b_stride_leaves: List[Optional[int]] = []
    t_shape_leaves: List[Optional[int]] = []
    t_stride_leaves: List[Optional[int]] = []
    b_shape.flatten_leaves(b_shape_leaves)
    b_stride.flatten_leaves(b_stride_leaves)
    t_shape.flatten_leaves(t_shape_leaves)
    t_stride.flatten_leaves(t_stride_leaves)

    block_size = _size_of_shape(b_shape)
    if block_size is None or any(v is None for v in b_stride_leaves + t_stride_leaves):
        # Runtime-capable fallback: only rank is known.
        return LayoutType.get(b_rank + t_rank)
    out_shape_leaves = b_shape_leaves + t_shape_leaves
    out_stride_leaves: List[Optional[int]] = list(b_stride_leaves)
    for s in t_stride_leaves:
        out_stride_leaves.append(_mul(s, block_size))

    out_shape = _PatternNode.tup([_PatternNode.leaf(v) for v in out_shape_leaves])
    out_stride = _PatternNode.tup([_PatternNode.leaf(v) for v in out_stride_leaves])
    return Type.parse(f"!flir.layout<{out_shape.to_spec()}:{out_stride.to_spec()}>")


def _infer_layout_type_logical_divide(layout: Value, tiler: Value) -> Type:
    l_rank, l_shape, l_stride = _parse_layout_type(str(layout.type))
    t_rank, t_shape, t_stride = _parse_layout_type(str(tiler.type))
    if l_rank < 0 or t_rank < 0:
        raise ValueError(f"Cannot infer logical_divide type: operand ranks are unknown: {layout.type} , {tiler.type}")

    input_size = _size_of_shape(l_shape)
    if input_size is None:
        return LayoutType.get(max(l_rank, t_rank))

    # Require static strides (matches C++ inference: no fallback).
    l_stride_leaves: List[Optional[int]] = []
    t_stride_leaves: List[Optional[int]] = []
    l_stride.flatten_leaves(l_stride_leaves)
    t_stride.flatten_leaves(t_stride_leaves)
    if any(v is None for v in l_stride_leaves + t_stride_leaves):
        return LayoutType.get(max(l_rank, t_rank))
    comp_shape, comp_stride = _complement_impl(t_shape, t_stride, input_size)

    rhs_shape = _PatternNode.tup([t_shape, comp_shape])
    rhs_stride = _PatternNode.tup([t_stride, comp_stride])

    out_shape, out_stride = _composition_impl(l_shape, l_stride, rhs_shape, rhs_stride)
    return Type.parse(f"!flir.layout<{out_shape.to_spec()}:{out_stride.to_spec()}>")


def _infer_layout_type_complement(tiler: Value, target_size: Value) -> Type:
    """Infer result type for flir.complement in a strict/no-fallback way.

    Requires:
    - tiler layout type is fully static
    - target_size is provably constant
    """
    t_rank, t_shape, t_stride = _parse_layout_type(str(tiler.type))
    if t_rank < 0:
        raise ValueError(f"Cannot infer complement type: tiler rank is unknown: {tiler.type}")

    cosize_hi = _try_get_constant_index(target_size)
    if cosize_hi is None:
        raise ValueError("Cannot infer complement type: target_size must be a compile-time constant index")

    # If we can't compute a static complement pattern, return a rank-1 dynamic
    # layout type and let lowering handle/diagnose.
    if _size_of_shape(t_shape) is None:
        return LayoutType.get(1)
    t_stride_leaves: List[Optional[int]] = []
    t_stride.flatten_leaves(t_stride_leaves)
    if any(v is None for v in t_stride_leaves):
        return LayoutType.get(1)

    comp_shape, comp_stride = _complement_impl(t_shape, t_stride, cosize_hi)
    return Type.parse(f"!flir.layout<{comp_shape.to_spec()}:{comp_stride.to_spec()}>")



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
        A Flir shape value
        
    Example:
        >>> # Flat shape
        >>> c8 = arith.constant(8, index=True)
        >>> c16 = arith.constant(16, index=True)
        >>> shape = flir.make_shape(c8, c16)  # Creates shape<2>
        >>> 
        >>> # Nested shape example: (9, (4, 8))
        >>> c9 = arith.constant(9, index=True)
        >>> c4 = arith.constant(4, index=True)
        >>> shape = flir.make_shape(c9, (c4, c8))  # Creates nested shape
    """
    
    loc = _get_location(loc)
    
    # If a single tuple/list is passed, unpack it
    if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
        dims = dims[0]
    
    nested_spec = None
    nested_dyn_vals: Optional[List[Value]] = None

    # Normalize to a list for processing
    dims_list = list(dims)
    if len(dims_list) == 1 and isinstance(dims_list[0], (tuple, list)):
        dims_list = list(dims_list[0])

    # Detect nested structure (tuple/list).
    #
    # NOTE: We intentionally avoid probing MLIR Value types here (e.g. via `str(v.type)`),
    # because doing so has been observed to trigger interpreter-level crashes when a value
    # becomes invalid/dangling in some edge cases. The common/idiomatic way to build nested
    # shapes is via tuples/lists, which we continue to support.
    has_nested = any(isinstance(d, (tuple, list)) for d in dims_list)

    if has_nested:
        # Build a spec and leaf list from tuples/lists only (type-mode).
        child_specs: List[str] = []
        dyn_leaf_vals: List[Value] = []

        def consume_node(node):
            nonlocal child_specs, dyn_leaf_vals
            if isinstance(node, (tuple, list)):
                sub_child_specs = []
                for x in node:
                    consume_node(x)
                    # move last spec into sub list
                    sub_child_specs.append(child_specs.pop())
                child_specs.append("(" + ",".join(sub_child_specs) + ")")
                return

            # IMPORTANT: do not call `_unwrap_value` on python ints here, otherwise
            # it will materialize `arith.constant` ops (exactly what we're avoiding).
            if isinstance(node, int):
                child_specs.append(str(int(node)))
                return

            v = _unwrap_value(node)
            # Disallow passing `!flir.shape` values as operands.
            if isinstance(v, Value) and str(v.type).startswith("!flir.shape<"):
                raise ValueError("Passing flir.shape values into make_shape is not supported; use tuple/list nesting instead.")

            # Leaf index
            const = None
            if isinstance(v, Value):
                const = _try_get_constant_index(v)
            child_specs.append(str(const) if const is not None else "?")
            if const is None:
                # Only dynamic leaves become operands.
                dyn_leaf_vals.append(_require_index_value(v, "shape leaf"))

        try:
            for d in dims_list:
                consume_node(d)
            nested_spec = "(" + ",".join(child_specs) + ")"
            nested_dyn_vals = dyn_leaf_vals
        except Exception:
            nested_spec = None
            nested_dyn_vals = None

    if nested_spec is not None and nested_dyn_vals is not None:
        result_type = Type.parse(f'!flir.shape<{nested_spec}>')
        # Type-mode: operands are the dynamic leaves only.
        operands = [_unwrap_value(v) for v in nested_dyn_vals]
    else:
        # Fallback: flat shape only
        flat_dims = _flatten_nested(dims)
        rank = len(flat_dims)
        spec_elems: List[str] = []
        dyn_leaf_vals: List[Value] = []
        for d in flat_dims:
            if isinstance(d, int):
                spec_elems.append(str(int(d)))
                continue
            v = _unwrap_value(d)
            const = None
            if isinstance(v, Value):
                const = _try_get_constant_index(v)
            spec_elems.append(str(const) if const is not None else "?")
            if const is None:
                dyn_leaf_vals.append(_require_index_value(v, "shape leaf"))
        flat_spec = "(" + ",".join(spec_elems) + ")"
        result_type = Type.parse(f"!flir.shape<{flat_spec}>")
    
    if nested_spec is None or nested_dyn_vals is None:
        # Type-mode: operands are the dynamic leaves only.
        operands = [_unwrap_value(v) for v in dyn_leaf_vals]

    with ip or InsertionPoint.current:
        return flir_ops.MakeShapeOp(result_type, operands, loc=loc).result


def make_stride(*strides, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Create a stride from stride values (supports nested strides).
    
    Args:
        *strides: Index values or tuples of index values for nested strides
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        A Flir stride value
        
    Example:
        >>> # Flat stride
        >>> c1 = arith.constant(1, index=True)
        >>> c8 = arith.constant(8, index=True)
        >>> stride = flir.make_stride(c1, c8)  # Creates stride<2>
        >>> 
        >>> # Nested stride example: (59, (13, 1))
        >>> c59 = arith.constant(59, index=True)
        >>> c13 = arith.constant(13, index=True)
        >>> stride = flir.make_stride(c59, (c13, c1))  # Creates nested stride
    """
    
    loc = _get_location(loc)
    
    # If a single tuple/list is passed, unpack it
    if len(strides) == 1 and isinstance(strides[0], (tuple, list)):
        strides = strides[0]
    
    nested_spec = None
    nested_dyn_vals: Optional[List[Value]] = None

    strides_list = list(strides)
    if len(strides_list) == 1 and isinstance(strides_list[0], (tuple, list)):
        strides_list = list(strides_list[0])

    # Detect nested structure (tuple/list). See make_shape() for rationale.
    has_nested = any(isinstance(s, (tuple, list)) for s in strides_list)

    if has_nested:
        child_specs: List[str] = []
        dyn_leaf_vals: List[Value] = []

        def consume_node(node):
            nonlocal child_specs, dyn_leaf_vals
            if isinstance(node, (tuple, list)):
                sub_child_specs = []
                for x in node:
                    consume_node(x)
                    sub_child_specs.append(child_specs.pop())
                child_specs.append("(" + ",".join(sub_child_specs) + ")")
                return

            if isinstance(node, int):
                child_specs.append(str(int(node)))
                return

            v = _unwrap_value(node)
            # Disallow passing `!flir.stride` values as operands.
            if isinstance(v, Value) and str(v.type).startswith("!flir.stride<"):
                raise ValueError("Passing flir.stride values into make_stride is not supported; use tuple/list nesting instead.")

            const = None
            if isinstance(v, Value):
                const = _try_get_constant_index(v)
            child_specs.append(str(const) if const is not None else "?")
            if const is None:
                dyn_leaf_vals.append(_require_index_value(v, "stride leaf"))

        try:
            for s in strides_list:
                consume_node(s)
            nested_spec = "(" + ",".join(child_specs) + ")"
            nested_dyn_vals = dyn_leaf_vals
        except Exception:
            nested_spec = None
            nested_dyn_vals = None

    if nested_spec is not None and nested_dyn_vals is not None:
        result_type = Type.parse(f'!flir.stride<{nested_spec}>')
        operands = [_unwrap_value(v) for v in nested_dyn_vals]
    else:
        flat_strides = _flatten_nested(strides)
        rank = len(flat_strides)
        spec_elems: List[str] = []
        dyn_leaf_vals: List[Value] = []
        for s in flat_strides:
            if isinstance(s, int):
                spec_elems.append(str(int(s)))
                continue
            v = _unwrap_value(s)
            const = None
            if isinstance(v, Value):
                const = _try_get_constant_index(v)
            spec_elems.append(str(const) if const is not None else "?")
            if const is None:
                dyn_leaf_vals.append(_require_index_value(v, "stride leaf"))
        flat_spec = "(" + ",".join(spec_elems) + ")"
        result_type = Type.parse(f"!flir.stride<{flat_spec}>")
    
    if nested_spec is None or nested_dyn_vals is None:
        operands = [_unwrap_value(v) for v in dyn_leaf_vals]

    with ip or InsertionPoint.current:
        return flir_ops.MakeStrideOp(result_type, operands, loc=loc).result


def make_layout(shape, stride=None, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Create a layout from shape and stride (supports nested layouts).
    
    Args:
        shape: A Flir shape value, or a tuple/int for creating shape
        stride: A Flir stride value, or a tuple/int for creating stride, 
                or None to create default column-major stride
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        A Flir layout value
        
    Example:
        >>> # Using shape and stride values
        >>> shape = flir.make_shape(c8, c16)
        >>> stride = flir.make_stride(c1, c8)
        >>> layout = flir.make_layout(shape, stride)
        >>> 
        >>> # Using tuples directly in nested form
        >>> layout = flir.make_layout((c9, (c4, c8)), stride=(c59, (c13, c1)))
        >>> 
        >>> # Using single value
        >>> layout = flir.make_layout(c6, stride=c1)  # 1D layout 6:1
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
    
    def _extract_spec_and_rank_from_flir_shape_or_stride_type(type_str: str):
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
            # Legacy format: !flir.shape<rank, ...>
            return None, int(type_content.split(",")[0].strip())
        return None, int(type_content)

    shape_spec, rank = _extract_spec_and_rank_from_flir_shape_or_stride_type(str(shape.type))
    stride_spec, _ = _extract_spec_and_rank_from_flir_shape_or_stride_type(str(stride.type))

    if shape_spec is not None and stride_spec is not None:
        result_type = Type.parse(f"!flir.layout<{shape_spec}:{stride_spec}>")
    else:
        result_type = LayoutType.get(rank)
    
    with ip or InsertionPoint.current:
        return flir_ops.MakeLayoutOp(result_type, _unwrap_value(shape), _unwrap_value(stride), loc=loc).result


def make_coord(*coords: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Create a coordinate from index values.
    
    Args:
        *coords: Index values representing each coordinate dimension
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        A Flir coordinate value
        
    Example:
        >>> i = arith.constant(4, index=True)
        >>> j = arith.constant(7, index=True)
        >>> coord = flir.make_coord(i, j)  # Creates coord<2>
    """
    
    loc = _get_location(loc)

    # If a single tuple/list is passed, unpack it
    if len(coords) == 1 and isinstance(coords[0], (tuple, list)):
        coords = tuple(coords[0])

    # Coordinates are represented as a flat list of index operands.
    # Legacy nested coord SSA trees are not supported.
    flat = _flatten_nested(coords)
    operands: List[Value] = []
    for c in flat:
        if isinstance(c, int):
            operands.append(_require_index_value(_unwrap_value(c), "coord leaf"))
        else:
            operands.append(_require_index_value(_unwrap_value(c), "coord leaf"))

    with ip or InsertionPoint.current:
        # Use a structured coord type by default with unknown domain shape "(?,?,...)".
        # (`idx2crd` may use a more structured type derived from the layout domain.)
        rank = len(operands)
        spec = "(" + ",".join(["?"] * rank) + ")"
        result_type = Type.parse(f"!flir.coord<{spec}>")
        return flir_ops.MakeCoordOp(result_type, operands, loc=loc).result


def crd2idx(coord: Value, layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Convert a coordinate to a linear index using a layout.
    
    Computes: sum(coord[i] * stride[i]) for all dimensions i.
    
    Args:
        coord: A Flir coordinate value
        layout: A Flir layout value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        An index value representing the linear offset
        
    Example:
        >>> coord = flir.make_coord(i, j)
        >>> layout = flir.make_layout(shape, stride)
        >>> idx = flir.crd2idx(coord, layout)  # Returns i*stride[0] + j*stride[1]
    """
    
    loc = _get_location(loc)
    result_type = IndexType.get()
    
    with ip or InsertionPoint.current:
        op = flir_ops.Crd2IdxOp(result_type, _unwrap_value(coord), _unwrap_value(layout), loc=loc, ip=ip)
        return op.result


def idx2crd(idx: Value, layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Convert a linear index to a coordinate using a layout.
    
    This is the inverse operation of crd2idx.
    
    Args:
        idx: An index value representing the linear offset
        layout: A Flir layout value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        A Flir coordinate value
        
    Example:
        >>> idx = arith.constant(42, index=True)
        >>> layout = flir.make_layout(shape, stride)
        >>> coord = flir.idx2crd(idx, layout)  # Inverse of crd2idx
    """
    
    loc = _get_location(loc)
    # Prefer preserving nested coord structure (domain shape) when layout type carries a shape spec.
    layout_type_str = str(layout.type)
    if "<" not in layout_type_str or ">" not in layout_type_str:
        rank = _extract_rank_from_flir_type_str(layout_type_str)
        spec = "(" + ",".join(["?"] * rank) + ")"
        result_type = Type.parse(f"!flir.coord<{spec}>")
    else:
        inner = layout_type_str.split("<", 1)[1].rsplit(">", 1)[0].strip()

        def _split_top_level_colon(s: str) -> Optional[tuple[str, str]]:
            depth = 0
            for i, ch in enumerate(s):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth = max(0, depth - 1)
                elif ch == ":" and depth == 0:
                    return s[:i], s[i + 1 :]
            return None

        shape_spec = None
        if inner.startswith("("):
            parts = _split_top_level_colon(inner)
            if parts is not None:
                shape_spec = parts[0].strip()
        elif len(inner) >= 2 and inner[0] == '"' and inner[-1] == '"':
            inner2 = inner[1:-1]
            parts = _split_top_level_colon(inner2)
            if parts is not None and inner2.strip().startswith("("):
                shape_spec = parts[0].strip()

        if shape_spec is not None:
            result_type = Type.parse(f"!flir.coord<{shape_spec}>")
        else:
            rank = _extract_rank_from_flir_type_str(layout_type_str)
            spec = "(" + ",".join(["?"] * rank) + ")"
            result_type = Type.parse(f"!flir.coord<{spec}>")
    
    with ip or InsertionPoint.current:
        op = flir_ops.Idx2CrdOp(result_type, _unwrap_value(idx), _unwrap_value(layout), loc=loc, ip=ip)
        return op.results[0]


def swizzle_xor16(row: Value, col: Value, k_blocks16: Value,
                  loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """XOR-with-row swizzle on the K dimension at 16B granularity.

    Computes: col xor ((row % k_blocks16) * 16)
    Lowered by FlirToStandard to arith ops (matching ROCDSL-style swizzle).
    """
    loc = _get_location(loc)
    with ip or InsertionPoint.current:
        # ODS-generated builders for dialect ops require passing the result type.
        return flir_ops.swizzle_xor16(
            IndexType.get(),
            _unwrap_value(row),
            _unwrap_value(col),
            _unwrap_value(k_blocks16),
            loc=loc,
            ip=ip,
        )



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
        >>> shape = flir.make_shape(c8, c16)
        >>> total = flir.size(shape)  # Returns 128
        >>> thread_count = flir.size(tv_layout, mode=[0])  # Thread dimension
        >>> tile_count = flir.size(gC, mode=[1])  # Tile dimension
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
        op = flir_ops.SizeOp(result_type, _unwrap_value(layout_or_tensor), loc=loc, ip=ip)
        return op.result


def cosize(layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the cosize (stride extent) of a layout.
    
    Args:
        layout: A Flir layout value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        An index value representing the cosize
    """
    
    loc = _get_location(loc)
    result_type = IndexType.get()
    
    with ip or InsertionPoint.current:
        op = flir_ops.CosizeOp(result_type, _unwrap_value(layout), loc=loc, ip=ip)
        return op.result


def rank(shape_or_layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Get the rank (number of dimensions) of a shape or layout.
    
    Args:
        shape_or_layout: A Flir shape or layout value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        An index value representing the rank
    """
    
    loc = _get_location(loc)
    result_type = IndexType.get()
    
    with ip or InsertionPoint.current:
        op = flir_ops.RankOp(result_type, _unwrap_value(shape_or_layout), loc=loc, ip=ip)
        return op.result


def get_shape(layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Extract the shape from a layout.
    
    Args:
        layout: A Flir layout value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        The shape component of the layout
    """
    
    loc = _get_location(loc)
    # Extract rank from layout type
    layout_type_str = str(layout.type)
    rank = _extract_rank_from_flir_type_str(layout_type_str)
    result_type = ShapeType.get(rank)
    
    with ip or InsertionPoint.current:
        return flir_ops.GetShapeOp(result_type, _unwrap_value(layout), loc=loc).result


def get_stride(layout: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Extract the stride from a layout.
    
    Args:
        layout: A Flir layout value
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        The stride component of the layout
    """
    
    loc = _get_location(loc)
    # Extract rank from layout type
    layout_type_str = str(layout.type)
    rank = _extract_rank_from_flir_type_str(layout_type_str)
    result_type = StrideType.get(rank)
    
    with ip or InsertionPoint.current:
        return flir_ops.GetStrideOp(result_type, _unwrap_value(layout), loc=loc).result


def get(input: Value, index: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Extract element from shape/stride/coord at given index.
    
    Args:
        input: A Flir shape, stride, or coord value
        index: Index of element to extract
        loc: Optional source location
        ip: Optional insertion point
        
    Returns:
        The element at the given index (as an index value)
        
    Example:
        >>> shape = flir.make_shape(c2, c3, c4)
        >>> dim0 = flir.get(shape, Const.index(0))  # Returns 2
        >>> dim1 = flir.get(shape, Const.index(1))  # Returns 3
    """
    
    loc = _get_location(loc)
    result_type = IndexType.get()
    
    with ip or InsertionPoint.current:
        return flir_ops.GetOp(result_type, _unwrap_value(input), _unwrap_value(index), loc=loc, ip=ip).result


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
        >>> composed = flir.composition(col_major, tiler)
    """
    
    loc = _get_location(loc)
    result_type = _infer_layout_type_composition(layout_a, layout_b)

    with ip or InsertionPoint.current:
        return flir_ops.CompositionOp(result_type, _unwrap_value(layout_a), _unwrap_value(layout_b), loc=loc).result


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
        >>> tiler = flir.make_layout(c3, stride=c1)
        >>> target = Const.index(12)
        >>> comp = flir.complement(tiler, target)  # Returns 4:3
    """
    
    loc = _get_location(loc)
    result_type = _infer_layout_type_complement(tiler, target_size)
    
    with ip or InsertionPoint.current:
        return flir_ops.ComplementOp(result_type, _unwrap_value(tiler), _unwrap_value(target_size), loc=loc).result


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
        >>> layout = flir.make_layout((c2, (c1, c6)), stride=(c1, (c6, c2)))
        >>> coalesced = flir.coalesce(layout)  # Simplifies to 12:1
    """
    from _mlir import ir as _ir
    
    loc = _get_location(loc)
    result_type = LayoutType.get(-1)
    
    with ip or InsertionPoint.current:
        # Create the operation directly using generic OpView
        unwrapped = _unwrap_value(layout)
        op = _ir.Operation.create(
            "flir.coalesce",
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
    
    try:
        loc = _get_location(loc)
        result_type = _infer_layout_type_logical_product(block, tiler)
    except Exception:
        loc = _get_location(loc)
        result_type = LayoutType.get(-1)

    with ip or InsertionPoint.current:
        return flir_ops.LogicalProductOp(result_type, _unwrap_value(block), _unwrap_value(tiler), loc=loc).result


def zipped_product(block: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the zipped product of two layouts."""
    
    loc = _get_location(loc)
    # Lowering currently treats zipped_product as logical_product.
    try:
        result_type = _infer_layout_type_logical_product(block, tiler)
    except Exception:
        result_type = LayoutType.get(-1)
    with ip or InsertionPoint.current:
        return flir_ops.ZippedProductOp(result_type, _unwrap_value(block), _unwrap_value(tiler), loc=loc).result


def tiled_product(block: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the tiled product of two layouts."""
    
    loc = _get_location(loc)
    # Lowering currently treats tiled_product as logical_product.
    try:
        result_type = _infer_layout_type_logical_product(block, tiler)
    except Exception:
        result_type = LayoutType.get(-1)
    with ip or InsertionPoint.current:
        return flir_ops.TiledProductOp(result_type, _unwrap_value(block), _unwrap_value(tiler), loc=loc).result


def flat_product(block: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the flat product of two layouts."""
    
    loc = _get_location(loc)
    # Lowering currently treats flat_product as logical_product (flattening happens later).
    try:
        result_type = _infer_layout_type_logical_product(block, tiler)
    except Exception:
        result_type = LayoutType.get(-1)
    with ip or InsertionPoint.current:
        return flir_ops.FlatProductOp(result_type, _unwrap_value(block), _unwrap_value(tiler), loc=loc).result


def raked_product(block: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the raked product of two layouts."""
    
    loc = _get_location(loc)
    # Lowering currently treats raked_product as logical_product.
    try:
        result_type = _infer_layout_type_logical_product(block, tiler)
    except Exception:
        result_type = LayoutType.get(-1)
    with ip or InsertionPoint.current:
        return flir_ops.RakedProductOp(result_type, _unwrap_value(block), _unwrap_value(tiler), loc=loc).result


def blocked_product(block: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the blocked product of two layouts."""
    
    loc = _get_location(loc)
    # Lowering currently treats blocked_product as logical_product.
    try:
        result_type = _infer_layout_type_logical_product(block, tiler)
    except Exception:
        result_type = LayoutType.get(-1)
    with ip or InsertionPoint.current:
        return flir_ops.BlockedProductOp(result_type, _unwrap_value(block), _unwrap_value(tiler), loc=loc).result


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
    
    try:
        loc = _get_location(loc)
        result_type = _infer_layout_type_logical_divide(layout, tiler)
    except Exception:
        loc = _get_location(loc)
        result_type = LayoutType.get(-1)

    with ip or InsertionPoint.current:
        return flir_ops.LogicalDivideOp(result_type, _unwrap_value(layout), _unwrap_value(tiler), loc=loc).result


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
    
    # Layout division case (lowering treats zipped_divide as logical_divide).
    loc = _get_location(loc)
    try:
        result_type = _infer_layout_type_logical_divide(layout_or_tensor, tiler_or_shape)
    except Exception:
        result_type = LayoutType.get(-1)
    with ip or InsertionPoint.current:
        return flir_ops.ZippedDivideOp(result_type, _unwrap_value(layout_or_tensor), _unwrap_value(tiler_or_shape), loc=loc).result


def tiled_divide(layout: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the tiled divide of a layout."""
    
    loc = _get_location(loc)
    try:
        result_type = _infer_layout_type_logical_divide(layout, tiler)
    except Exception:
        result_type = LayoutType.get(-1)
    with ip or InsertionPoint.current:
        return flir_ops.TiledDivideOp(result_type, _unwrap_value(layout), _unwrap_value(tiler), loc=loc).result


def flat_divide(layout: Value, tiler: Value, loc: Optional[Location] = None, ip: Optional[InsertionPoint] = None) -> Value:
    """Compute the flat divide of a layout."""
    
    loc = _get_location(loc)
    try:
        result_type = _infer_layout_type_logical_divide(layout, tiler)
    except Exception:
        result_type = LayoutType.get(-1)
    with ip or InsertionPoint.current:
        return flir_ops.FlatDivideOp(result_type, _unwrap_value(layout), _unwrap_value(tiler), loc=loc).result


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
        >>> thread_data = flir.local_partition(global_layout, tile, thread_idx)
    """
    
    loc = _get_location(loc)
    result_type = LayoutType.get(-1)
    
    with ip or InsertionPoint.current:
        return flir_ops.LocalPartitionOp(result_type, _unwrap_value(layout), _unwrap_value(tile), _unwrap_value(index), loc=loc).result


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
        >>> cta_data = flir.local_tile(global_layout, cta_shape, block_coord)
    """
    
    loc = _get_location(loc)
    result_type = layout.type
    
    with ip or InsertionPoint.current:
        return flir_ops.LocalTileOp(result_type, _unwrap_value(layout), _unwrap_value(tiler), _unwrap_value(coord), loc=loc).result


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
        >>> frgPred = flir.make_rmem_tensor((4, 4), Boolean)
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
        >>> idC = flir.make_identity_tensor(mC.shape)
        >>> cC = flir.zipped_divide(idC, tiler=tiler_mn)
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
        >>> thr_layout = flir.make_ordered_layout((4, 32), order=(1, 0))
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
        # from _mlir.dialects import flir as flir_ops  # Not available
        op = flir_ops.ProductEachOp(shape, loc=loc)
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
        >>> thr_layout = flir.make_layout((4, 32), stride=(32, 1))
        >>> val_layout = flir.make_layout((4, 4), stride=(4, 1))
        >>> tiler_mn, tv_layout = flir.make_layout_tv(thr_layout, val_layout)
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
        >>> val = flir.elem_less(thrCrd[i], shape)
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
        >>> atom = flir.make_copy_atom(F16Type.get(), vector_size=8)
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
        >>> atom = flir.make_copy_atom(F16Type.get(), 8)
        >>> thr_layout = flir.make_layout((4, 32), stride=(32, 1))
        >>> val_layout = flir.make_layout((4, 4), stride=(4, 1))
        >>> tiled_copy = flir.make_tiled_copy_tv(atom, thr_layout, val_layout)
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
         dst_swizzle_xor16_kblocks: Optional[Value] = None,
         dst_swizzle_xor16_dims: Optional[Tuple[int, int]] = (0, 1),
         *,
         src_buffer_resource: Optional[Value] = None,
         src_buffer_offset_in_bytes: bool = True,
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
        >>> flir.copy(atom, src, dst, src_indices=[i,j], dst_indices=[k])
    """
    from _mlir.dialects import memref as memref_dialect
    loc = _get_location(loc)

    # If return_vector=True, we capture the last vector value loaded by the vectorized path.
    captured_vec = {"val": None}

    def _maybe_swizzle_dst_indices(idx_list: List[Value]) -> List[Value]:
        """Optionally apply XOR-with-row swizzle to dst indices (2D only).

        This is intended for LDS bank-conflict avoidance where we permute the
        K dimension at 16B granularity:
          col' = col xor ((row % kBlocks16) * 16)
        """
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
        swz = flir_ops.swizzle_xor16(IndexType.get(), row, col, kblocks, loc=loc, ip=ip)
        out = list(idx_list)
        out[col_dim] = _unwrap_value(swz)
        return out

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
                dst_idx2 = _maybe_swizzle_dst_indices([_unwrap_value(i) for i in dst_idx])
                store_idx = _normalize_indices_to_memref(dst_view.memref, dst_idx2, dst_view.strides, loc)
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
                        vec_dst_idx2 = _maybe_swizzle_dst_indices([_unwrap_value(idx) for idx in vec_dst_idx])
                        store_indices = _normalize_indices_to_memref(dst_view.memref, vec_dst_idx2, dst_view.strides, loc)

                        # Vector Load
                        vec_load_op = vector.load(
                            vec_type,
                            src_view.memref,
                            load_indices,
                            nontemporal=nontemporal,
                            alignment=alignment,
                        )
                        # MLIR Python bindings may return an Op wrapper; vector.store expects a Value.
                        vec_val = vec_load_op.result if hasattr(vec_load_op, "result") else vec_load_op
                        vec_val = _unwrap_value(vec_val)
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

    def emit_tensor_load(copy_shape, src_view: TensorView, pred_val: Optional[Value] = None):
        """Load-only path (no dst), primarily for gmem->register vector loads.

        Currently supported:
        - 1D TensorView
        - return_vector=True
        - extent == vector_size
        """
        from _mlir.dialects import vector
        from _mlir.ir import VectorType

        if not return_vector:
            raise ValueError("copy(load-only) requires return_vector=True when dst is None")
        if len(copy_shape) != 1:
            raise ValueError("copy(load-only) currently supports only 1D shapes")

        extent = int(copy_shape[0])

        vector_size = 1
        if isinstance(copy_desc, TiledCopy) and copy_desc.copy_atom:
            vector_size = copy_desc.copy_atom.vector_size
        elif hasattr(copy_desc, "vector_size"):
            vector_size = copy_desc.vector_size

        if extent != int(vector_size):
            raise ValueError(
                f"copy(load-only) expects extent==vector_size (got {extent} vs {vector_size})"
            )

        if src_buffer_resource is not None:
            try:
                from flydsl.dialects.ext import buffer_ops as _buffer_ops
            except Exception:
                from . import buffer_ops as _buffer_ops  # type: ignore

            # Specialize for 1-byte element types (fp8/int8) so we can lower to
            # `buffer_load_dwordx{2,4}` and avoid per-load scalarization.
            elem_ty = src_view.element_type
            elem_ty_str = str(elem_ty)
            is_f8 = ("f8" in elem_ty_str) or ("Float8" in elem_ty_str)
            is_i8 = False
            try:
                is_i8 = IntegerType.isinstance(elem_ty) and (IntegerType(elem_ty).width == 8)
            except Exception:
                # Best-effort fallback for older bindings.
                is_i8 = (elem_ty_str == "i8")

            if (is_f8 or is_i8) and extent in (8, 16) and (extent % 4 == 0):
                i32_ty = IntegerType.get_signless(32)
                vec_width = extent // 4  # 8B -> dwordx2, 16B -> dwordx4
                base0 = src_view.base_indices[0] if len(src_view.base_indices) else _to_index_value(0, loc)
                if src_buffer_offset_in_bytes:
                    # base index is in bytes (1-byte elements). Convert to i32 element offset.
                    c4 = arith.ConstantOp(IndexType.get(), IntegerAttr.get(IndexType.get(), 4), loc=loc).result
                    idx_i32 = arith.DivSIOp(_unwrap_value(base0), _unwrap_value(c4), loc=loc).result
                else:
                    # base index is already an i32-element offset for dtype=i32 loads.
                    idx_i32 = _unwrap_value(base0)
                mask = _unwrap_value(pred_val) if pred_val is not None else None
                i32_vec = _buffer_ops.buffer_load(
                    _unwrap_value(src_buffer_resource),
                    idx_i32,
                    vec_width=vec_width,
                    dtype=i32_ty,
                    mask=mask,
                )
                vec_elem_ty = VectorType.get((extent,), elem_ty)
                return vector.BitCastOp(vec_elem_ty, _unwrap_value(i32_vec)).result

        # Generic path: vector.load
        base = src_view.base_indices[0] if len(src_view.base_indices) else _to_index_value(0, loc)
        idxs = _normalize_indices_to_memref(src_view.memref, [_unwrap_value(base)], src_view.strides, loc)
        vec_type = VectorType.get((extent,), src_view.element_type)
        with ip or InsertionPoint.current:
            return vector.load(
                vec_type,
                src_view.memref,
                idxs,
                nontemporal=nontemporal,
                alignment=alignment,
            )

    if isinstance(copy_desc, TiledCopy) and isinstance(src, TensorView) and isinstance(dst, TensorView):
        # pred can be TensorView or scalar Value
        emit_tensor_copy(copy_desc.val_shape, src, dst, pred)
        return captured_vec["val"] if return_vector else None

    # Vector store path: src is a vector Value and dst is a TensorView.
    try:
        from _mlir.ir import VectorType as _VectorType
    except Exception:
        _VectorType = None  # type: ignore
    if isinstance(dst, TensorView):
        v = _unwrap_value(src)
        if _VectorType is not None and isinstance(getattr(v, "type", None), _VectorType):
            from _mlir.dialects import vector as vector_dialect
            # Use TensorView base indices as the store address.
            d_idx = [_unwrap_value(i) for i in dst.base_indices]
            d_idx2 = _maybe_swizzle_dst_indices(d_idx)
            store_indices = _normalize_indices_to_memref(dst.memref, d_idx2, dst.strides, loc)
            with ip or InsertionPoint.current:
                # Use `vector.store` so we can propagate alignment/nontemporal hints
                # into lowering (this is important for LDS 16B ops selection).
                vector_dialect.store(
                    v,
                    dst.memref,
                    store_indices,
                    nontemporal=nontemporal,
                    alignment=alignment,
                )
            return v if return_vector else None

    # Load-only path: dst=None and src is a TensorView.
    if isinstance(src, TensorView) and dst is None:
        # For load-only, only scalar broadcast predicates are supported (pred TensorView
        # implies element-wise masking, which would require vector predication support).
        pred_val = None
        if pred is not None and not isinstance(pred, TensorView):
            pred_val = _unwrap_value(pred)
        vec = emit_tensor_load(src.shape, src, pred_val=pred_val)
        return vec

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

# Backwards compatibility: allow `flir.flir` (historical API / older tests) to
# refer to this module object. This keeps `flir.flir.print` equivalent to
# `flir.print` without changing the public surface area of operations.
import sys as _sys
flir = _sys.modules[__name__]


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
        >>> flir.print(">>>", b)  # Shows static value
        >>> flir.print(">>>", a)  # Shows "?" for dynamic value
        >>> 
        >>> # Print dynamic value (runtime)
        >>> flir.printf(">?? {}", a)  # Shows actual runtime value
        >>> flir.printf(">?? {}", b)  # Also works for static values
        >>> 
        >>> # Print layout
        >>> layout = flir.make_layout(shape, stride)
        >>> flir.print(">>>", layout)  # Shows layout with "?" for dynamic parts
        >>> flir.printf(">?? {}", layout)  # Shows actual runtime values
    
    Note:
        - Use `flir.print` (Python's print) for compile-time/static values
        - Use `flir.printf` for runtime/dynamic values
        - Format strings use "{}" as placeholders (similar to Python f-strings)
    """
    from _mlir.dialects import gpu as _gpu
    
    loc = _get_location(loc)
    
    # Unwrap all argument values
    unwrapped_args = [_unwrap_value(arg) for arg in args]
    
    with ip or InsertionPoint.current:
        return _gpu.printf(format_str, unwrapped_args, loc=loc, ip=ip)


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
    "swizzle_xor16",
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
