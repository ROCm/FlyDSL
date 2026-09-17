# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Collective operations on native expr values; no additional value types.

Numeric tiles use Vector; structured tiles use tuples of plain Struct values. Struct values
are rebuilt through the compiler protocol, including their native field metadata.
Shared scratch uses expr.Array or a Struct of native arrays for sub-byte fields.
"""

from functools import lru_cache

from ...compiler.protocol import (
    construct_from_ir_values,
    dsl_size_of,
    extract_to_ir_values,
)
from ...expr.gpu import shuffle_idx
from ...expr.numeric import Int32, Integer, Numeric, Uint32, Uint64, Uint128
from ...expr.primitive import inttoptr, ptrtoint
from ...expr.struct import Struct, is_struct_type
from ...expr.typing import Array, Constexpr, Pointer, Vector


def _fields(dtype):
    return getattr(
        dtype,
        "__dsl_effective_field_defs__",
        tuple((f.name, f.type_spec) for f in dtype.__dsl_field_defs__),
    )


def _unalign(dtype):
    return dtype.dtype if getattr(dtype, "__dsl_align_wrapper__", False) else dtype


@lru_cache(None)
def _leaf_fields(dtype):
    """Return runtime expr leaves in declaration order, skipping Constexpr."""
    dtype = _unalign(dtype)
    if isinstance(dtype, type) and issubclass(dtype, Constexpr):
        return ()
    if isinstance(dtype, type) and issubclass(dtype, (Numeric, Vector, Pointer)):
        return (((), dtype),)
    if not is_struct_type(dtype):
        raise TypeError(f"collective items require Numeric, Pointer, Vector or fx.Struct values, got {dtype!r}")
    leaves = []
    for name, field_type in _fields(dtype):
        try:
            leaves.extend(((name, *path), leaf) for path, leaf in _leaf_fields(field_type))
        except TypeError as exc:
            raise TypeError(f"unsupported collective field {name}: {exc}") from exc
    return tuple(leaves)


def _item_dtype(value):
    if isinstance(value, Numeric):
        return value.dtype
    if isinstance(value, Vector):
        return Vector[value.dtype, value.shape]
    if isinstance(value, Pointer):
        return type(value)
    if is_struct_type(type(value)):
        _leaf_fields(type(value))
        return type(value)
    raise TypeError(f"expected a native expr item, got {type(value).__name__}")


def _flatten_record(value):
    """Collect native field values without converting pointers or vector bits."""
    out = []
    for path, _ in _leaf_fields(_item_dtype(value)):
        leaf = value
        for name in path:
            leaf = getattr(leaf, name)
        out.append(leaf)
    return tuple(out)


def _rebuild_record(dtype, leaves, exemplar=None):
    """Delegate reconstruction and field metadata to the native DSL protocol."""
    return construct_from_ir_values(
        dtype,
        dtype if exemplar is None else exemplar,
        extract_to_ir_values(tuple(leaves)),
    )


def _record_default(dtype, fill=0):
    """Construct an explicit default in the native item type."""
    dtype = _unalign(dtype)
    if isinstance(fill, dtype):
        return fill
    if is_struct_type(type(fill)):
        raise TypeError("default Struct must have the requested dtype")
    if issubclass(dtype, Constexpr):
        if not dtype.is_specialized:
            raise TypeError("a default needs a specialized Constexpr field")
        return dtype.value
    if issubclass(dtype, Pointer):
        if not hasattr(dtype, "ir_type"):
            raise TypeError("pointer defaults require a specialized fx.Pointer field or an explicit value")
        if isinstance(fill, Pointer):
            return dtype.__coerce__(fill)
        if fill != 0:
            raise TypeError("only a null default can be broadcast into a pointer field")
        return dtype(inttoptr(dtype.ir_type, Uint64(0)))
    if issubclass(dtype, Vector):
        if dtype is Vector:
            raise TypeError("vector defaults require a specialized fx.Vector field or an explicit value")
        return dtype(fill)
    if issubclass(dtype, Numeric):
        return dtype(fill)
    _leaf_fields(dtype)
    return dtype(**{name: _record_default(field_type, fill) for name, field_type in _fields(dtype)})


@lru_cache(None)
def _constexpr_fields(dtype):
    if not is_struct_type(dtype):
        return ()
    fields = []
    for name, field_type in _fields(dtype):
        if isinstance(field_type, type) and issubclass(field_type, Constexpr):
            fields.append(((name,), field_type))
        else:
            fields.extend(((name, *path), leaf) for path, leaf in _constexpr_fields(field_type))
    return tuple(fields)


def _record_cast(value, dtype):
    """Convert corresponding native leaves, validating pointer field types."""
    if _item_dtype(value) is dtype:
        return value
    if _constexpr_fields(_item_dtype(value)) != _constexpr_fields(dtype):
        raise TypeError("item conversion must preserve Constexpr fields and their specialization")
    source, target = _flatten_record(value), _leaf_fields(dtype)
    if len(source) != len(target):
        raise TypeError("item conversion requires the same number of runtime fields")
    leaves = []
    for leaf, (_, target_type) in zip(source, target):
        if issubclass(target_type, Numeric):
            if not isinstance(leaf, Numeric):
                raise TypeError("numeric conversion requires a numeric field")
            leaves.append(leaf.to(target_type))
        elif issubclass(target_type, Pointer):
            if not isinstance(leaf, Pointer):
                raise TypeError("pointer conversion requires a Pointer field")
            leaves.append(leaf if target_type is Pointer else target_type.__coerce__(leaf))
        else:
            leaves.append(target_type.__coerce__(leaf))
    return _rebuild_record(dtype, leaves)


def _is_items(value):
    return isinstance(value, (Vector, tuple, list))


def _as_items(value):
    # Vector indexing need not raise IndexError, so explicitly bound iteration.
    items = tuple(value[i] for i in range(len(value))) if _is_items(value) else (value,)
    if not items:
        raise ValueError("a cooperative tile must contain at least one item")
    dtype = _item_dtype(items[0])
    if any(_item_dtype(item) is not dtype for item in items[1:]):
        raise TypeError("all cooperative tile items must have the same dtype")
    return items


def _items_dtype(value):
    return _item_dtype(_as_items(value)[0])


def _from_items(items, dtype=None, *, like=None):
    """Pack numeric items into a Vector, or structured items into a tuple."""
    items = _as_items(items)
    dtype = _item_dtype(items[0]) if dtype is None else dtype
    if issubclass(dtype, Numeric):
        return Vector.from_elements(items, dtype=dtype)
    return tuple(_record_cast(item, dtype) for item in items)


def _record_select(condition, lhs, rhs):
    if isinstance(lhs, (Numeric, Vector)):
        return condition.select(lhs, rhs)
    if isinstance(lhs, Pointer):
        if not isinstance(rhs, Pointer) or lhs.type != rhs.type:
            raise TypeError("pointer selection requires identical pointer types")
        return type(lhs)(inttoptr(lhs.type, condition.select(ptrtoint(lhs), ptrtoint(rhs))))
    if isinstance(lhs, (tuple, list)):
        if type(lhs) is not type(rhs) or len(lhs) != len(rhs):
            raise ValueError("selection tiles must have the same shape")
        return type(lhs)(
            _record_select(condition[i] if isinstance(condition, Vector) else condition, a, b)
            for i, (a, b) in enumerate(zip(lhs, rhs))
        )
    dtype = _item_dtype(lhs)
    if _item_dtype(rhs) is not dtype:
        raise TypeError("selection operands must have the same dtype and Constexpr specialization")
    return _rebuild_record(
        dtype,
        [_record_select(condition, a, b) for a, b in zip(_flatten_record(lhs), _flatten_record(rhs))],
        lhs,
    )


def _record_shuffle(value, source_lane, width):
    if isinstance(value, Pointer):
        bits = _shuffle_leaf(ptrtoint(value).to(Uint64), source_lane, width)
        return type(value)(inttoptr(value.type, bits))
    if isinstance(value, Vector):
        moved = Vector.from_elements(
            [_shuffle_leaf(item, source_lane, width) for item in value],
            dtype=value.dtype,
        )
        return Vector(moved, shape=value.shape, dtype=value.dtype)
    if isinstance(value, Numeric):
        return _shuffle_leaf(value, source_lane, width)
    if isinstance(value, (tuple, list)):
        return type(value)(_record_shuffle(item, source_lane, width) for item in value)
    return _rebuild_record(
        _item_dtype(value),
        [_record_shuffle(leaf, source_lane, width) for leaf in _flatten_record(value)],
        value,
    )


def _shuffle_leaf(leaf, source_lane, width):
    if isinstance(leaf, Integer) and leaf.dtype.width < 8:
        return shuffle_idx(leaf.to(Int32), source_lane, width).to(leaf.dtype)
    if leaf.dtype.width == 128:
        bits, result = leaf.bitcast(Uint128), Uint128(0)
        for shift in range(0, 128, 32):
            word = (bits >> shift).to(Uint32)
            result = result | (shuffle_idx(word, source_lane, width).to(Uint128) << shift)
        return result.bitcast(leaf.dtype)
    return shuffle_idx(leaf, source_lane, width)


# Only non-Storable items need a fieldwise scratch layout (e.g. scalar i1).
# These are ordinary expr Struct/Array schemas, never custom protocol classes.
_SCRATCH_SCHEMAS = {}


@lru_cache(None)
def _shared_array(dtype, size):
    """Build native scratch storage, widening sub-byte scalar fields to i32."""
    leaves = _leaf_fields(dtype)
    try:
        dsl_size_of(dtype)
    except TypeError:
        fields = []
        for i, (_, leaf) in enumerate(leaves):
            if issubclass(leaf, Numeric):
                if leaf.width < 8 and leaf.is_float:
                    raise TypeError("pack sub-byte floating fields in a native Vector for storage")
                element = Int32 if leaf.width < 8 else leaf
            else:
                element = Struct["value":leaf]
            fields.append(slice(f"leaf{i}", Array[element, size]))
        layout = Struct[tuple(fields)]
        _SCRATCH_SCHEMAS[layout] = (dtype, size)
        return layout
    return Array[dtype, size]


def _shared_load(slots, index):
    spec = _SCRATCH_SCHEMAS.get(type(slots))
    if spec is None:
        return slots[index]
    dtype = spec[0]
    leaves = []
    for i, (_, leaf_type) in enumerate(_leaf_fields(dtype)):
        value = getattr(slots, f"leaf{i}")[index]
        leaves.append(value.to(leaf_type) if issubclass(leaf_type, Numeric) else value.value)
    return _rebuild_record(dtype, leaves)


def _shared_store(slots, index, value):
    spec = _SCRATCH_SCHEMAS.get(type(slots))
    if spec is None:
        slots[index] = value
        return
    if _item_dtype(value) is not spec[0]:
        raise TypeError("shared item must have its declared dtype")
    for i, leaf in enumerate(_flatten_record(value)):
        array = getattr(slots, f"leaf{i}")
        array[index] = leaf.to(array.dtype) if isinstance(leaf, Numeric) else array.dtype(leaf)


def _shared_dtype(slots):
    spec = _SCRATCH_SCHEMAS.get(type(slots))
    return spec[0] if spec is not None else slots.dtype


def _shared_size(slots):
    spec = _SCRATCH_SCHEMAS.get(type(slots))
    return spec[1] if spec is not None else getattr(slots, "size", None)
