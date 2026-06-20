# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
from enum import Enum
from itertools import chain
from typing import Any, List

from .._mlir import ir
from ..compiler.protocol import (
    c_abi_spec,
    cache_signature,
    dsl_align_of,
    dsl_size_of,
    extract_to_ir_values,
    get_ir_types,
    peek_from_ptr,
    poke_into_ptr,
)
from .meta import dsl_loc_tracing
from .primitive import add_offset
from .typing import Array, Constexpr, Pointer

__all__ = [
    "struct",
    "Struct",
    "union",
    "Union",
    "Array",
    "Align",
    "Storage",
    "Arena",
    "is_composite_type",
    "is_struct_type",
    "is_specializable_struct_type",
]


class CompositeKind(Enum):
    Product = 0
    Sum = 1


def is_composite_type(obj: Any) -> bool:
    return isinstance(obj, type) and hasattr(obj, "__dsl_composite_kind__")


def is_struct_type(obj: Any) -> bool:
    return is_composite_type(obj) and obj.__dsl_composite_kind__ == CompositeKind.Product


@dataclass(slots=True)
class FieldDef:
    name: str
    type_spec: Any


def _is_constexpr_type(type_spec: Any) -> bool:
    return isinstance(type_spec, type) and issubclass(type_spec, Constexpr)


def _type_name(dtype: Any) -> str:
    return getattr(dtype, "__name__", repr(dtype))


def _display_name(schema: type) -> str:
    return getattr(schema, "__dsl_display_name__", getattr(schema, "__name__", repr(schema)))


_RESERVED_FIELD_NAMES = frozenset(
    {
        "replace",  # used by Struct
        "peek",  # used by Storage
        "poke",  # used by Storage
    }
)


def _validate_field_name(name: str, context: str):
    if name.startswith("_"):
        raise ValueError(f"{context}: field name '{name}' must not start with underscore")
    if name in _RESERVED_FIELD_NAMES:
        raise ValueError(f"{context}: field name '{name}' is reserved")


def _normalize_decorator_fields(klass: type) -> tuple[FieldDef, ...]:
    annotations = getattr(klass, "__annotations__", {})
    context = klass.__name__
    for name in annotations:
        _validate_field_name(name, context)
    return tuple(FieldDef(name, annotation) for name, annotation in annotations.items())


def _align_up(value: int, align: int) -> int:
    if align <= 0:
        raise ValueError(f"alignment must be positive, got {align}")
    return (value + align - 1) // align * align


def _storage_layout(schema: type) -> tuple[int, int, dict[str, int]]:
    cached = getattr(schema, "__dsl_storage_layout_cache__", None)
    if cached is not None:
        return cached

    fields = [field for field in schema.__dsl_field_defs__ if not _is_constexpr_type(field.type_spec)]
    if not fields:
        result = (0, 1, {})
        schema.__dsl_storage_layout_cache__ = result
        return result

    def _field_layout(field: FieldDef) -> tuple[int, int]:
        try:
            return dsl_size_of(field.type_spec), dsl_align_of(field.type_spec)
        except TypeError as exc:
            raise TypeError(
                f"Cannot compute layout for schema {_display_name(schema)}: field '{field.name}' has type "
                f"{_type_name(field.type_spec)} which does not implement the Storable protocol."
            ) from exc

    if schema.__dsl_composite_kind__ == CompositeKind.Sum:
        sizes_aligns = [_field_layout(field) for field in fields]
        align = max(a for _, a in sizes_aligns)
        size = max(s for s, _ in sizes_aligns)
        result = (_align_up(size, align), align, {field.name: 0 for field in fields})
        schema.__dsl_storage_layout_cache__ = result
        return result
    else:
        offset = 0
        align = 1
        offsets: dict[str, int] = {}
        for field in fields:
            field_size, field_align = _field_layout(field)
            offset = _align_up(offset, field_align)
            offsets[field.name] = offset
            offset += field_size
            align = max(align, field_align)
        result = (_align_up(offset, align), align, offsets)
        schema.__dsl_storage_layout_cache__ = result
        return result


def _coerce_value_type(schema: type, field: FieldDef, value: Any) -> Any:
    type_spec = field.type_spec
    coerce_fn = getattr(type_spec, "__coerce__", None)
    if coerce_fn is not None:
        try:
            return coerce_fn(value)
        except TypeError as exc:
            raise TypeError(f"{_display_name(schema)}(...) field '{field.name}' {exc}") from exc
    if isinstance(type_spec, type) and not isinstance(value, type_spec):
        raise TypeError(
            f"{_display_name(schema)}(...) field '{field.name}' expects "
            f"{_type_name(type_spec)}, got {type(value).__name__}."
        )
    return value


def _resolve_field(schema: type, key: int | str) -> FieldDef:
    fields = schema.__dsl_field_defs__
    if isinstance(key, int):
        if key < 0 or key >= len(fields):
            raise IndexError(f"Index {key} out of range for schema {_display_name(schema)} with {len(fields)} fields.")
        return fields[key]
    if isinstance(key, str):
        for f in fields:
            if f.name == key:
                return f
        available = [f.name for f in fields]
        raise KeyError(f"Field '{key}' not found in schema {_display_name(schema)}. Available fields: {available}.")


def _type_cache_key(type_spec: Any):
    if is_composite_type(type_spec):
        return ("composite", type_spec.__dsl_type_identity__)
    sig = getattr(type_spec, "__cache_signature__", None)
    if callable(sig):
        try:
            return ("sig", sig())
        except TypeError:
            pass
    try:
        hash(type_spec)
    except TypeError:
        return ("repr", repr(type_spec))
    return type_spec


def _make_type_identity(policy: CompositeKind, fields: tuple[FieldDef, ...]):
    return (policy, tuple((field.name, _type_cache_key(field.type_spec)) for field in fields))


def _field_values_from_args(schema: type, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    fields = schema.__dsl_field_defs__
    if len(args) > len(fields):
        raise TypeError(
            f"{_display_name(schema)}(...) expected {len(fields)} field(s), got {len(args)} positional value(s)."
        )

    values: dict[str, Any] = {}
    for field, value in zip(fields, args, strict=False):
        values[field.name] = value

    field_names = {field.name for field in fields}
    unexpected = set(kwargs) - field_names
    for field in fields:
        if field.name in kwargs:
            if field.name in values:
                raise TypeError(f"{_display_name(schema)}(...) got multiple values for field '{field.name}'.")
            values[field.name] = kwargs[field.name]

    if unexpected:
        raise TypeError(f"{_display_name(schema)}(...) got unexpected field(s): {sorted(unexpected)}.")

    missing = [field.name for field in fields if field.name not in values]
    if missing:
        raise TypeError(f"{_display_name(schema)}(...) missing required field(s): {missing}.")
    return values


_specialization_cache: dict[tuple, type] = {}


def _specialize_type(base_cls: type, fields: tuple[FieldDef, ...], values: dict[str, Any]) -> type:
    effective: list[tuple[str, Any]] = []
    needs_specialization = False
    for field in fields:
        value = values[field.name]
        type_spec = field.type_spec
        specializer = getattr(type_spec, "__specialize_for_value__", None)
        if specializer is not None:
            spec_type = specializer(value)
            effective.append((field.name, spec_type))
            if spec_type is not type_spec:
                needs_specialization = True
        elif is_composite_type(type(value)):
            sub_type = type(value)
            effective.append((field.name, sub_type))
            if sub_type is not type_spec:
                needs_specialization = True
        else:
            effective.append((field.name, type_spec))
    if not needs_specialization:
        return base_cls

    cache_key = (base_cls, tuple(effective))
    cached = _specialization_cache.get(cache_key)
    if cached is not None:
        return cached

    suffix_parts = []
    for (name, eff_type), field in zip(effective, fields, strict=True):
        if eff_type is field.type_spec:
            continue
        if isinstance(eff_type, type) and issubclass(eff_type, Constexpr) and eff_type.is_specialized:
            suffix_parts.append(f"{name}={eff_type.value!r}")
    suffix = f"[{', '.join(suffix_parts)}]" if suffix_parts else ""
    namespace: dict[str, Any] = {
        "__dsl_effective_field_defs__": tuple(effective),
        "__dsl_base_type__": base_cls,
        "__dsl_display_name__": _display_name(base_cls) + suffix,
    }
    specialized = type(base_cls.__name__ + suffix, (base_cls,), namespace)
    _specialization_cache[cache_key] = specialized
    return specialized


def _effective_field_defs(schema: type) -> tuple[tuple[str, Any], ...]:
    effective = getattr(schema, "__dsl_effective_field_defs__", None)
    if effective is not None:
        return effective
    base_cls = getattr(schema, "__dsl_base_type__", schema)
    return tuple(getattr(base_cls, "__annotations__", {}).items())


def _carrier_for_field(eff_type: Any, value: Any) -> Any:
    if isinstance(eff_type, type) and issubclass(eff_type, Constexpr):
        return eff_type
    return value


def _construct_field_from_ir(type_spec: Any, values):
    ctor = getattr(type_spec, "__construct_from_ir_values__", None)
    if ctor is None:
        raise TypeError(f"struct field type {_type_name(type_spec)} does not implement __construct_from_ir_values__")
    return ctor(values)


def _ir_value_count_from_type(type_spec: Any) -> int:
    if is_struct_type(type_spec):
        return sum(_ir_value_count_from_type(eff) for _, eff in _effective_field_defs(type_spec))
    types_fn = getattr(type_spec, "__get_ir_types__", None)
    if types_fn is not None and isinstance(type_spec, type):
        try:
            return len(types_fn())
        except TypeError:
            pass
    return 1


def _normalize_inline_fields(params) -> tuple[FieldDef, ...]:
    if not isinstance(params, tuple):
        params = (params,)
    if len(params) == 0:
        raise ValueError("inline schema requires at least one field")
    fields = []
    seen: set[str] = set()
    for idx, item in enumerate(params):
        if isinstance(item, slice):
            if not isinstance(item.start, str) or item.stop is None:
                raise TypeError("named inline fields must use Schema['name': Type] syntax")
            name = item.start
            _validate_field_name(name, "inline schema")
            type_spec = item.stop
        else:
            name = f"_{idx}"
            type_spec = item
        if name in seen:
            raise ValueError(f"duplicate inline field name '{name}'")
        seen.add(name)
        fields.append(FieldDef(name, type_spec))
    return tuple(fields)


def _inline_display_name(display: str, params, fields: tuple[FieldDef, ...]) -> str:
    raw = params if isinstance(params, tuple) else (params,)
    all_anonymous = all(not isinstance(item, slice) for item in raw)
    if all_anonymous:
        body = ", ".join(_type_name(field.type_spec) for field in fields)
    else:
        body = ", ".join(f"{field.name!r}: {_type_name(field.type_spec)}" for field in fields)
    return f"{display}[{body}]"


def is_specializable_struct_type(tp: Any) -> bool:
    """True if *tp* is a struct type carrying a (possibly nested) Constexpr field."""
    if not is_struct_type(tp):
        return False
    for _name, eff in _effective_field_defs(tp):
        if isinstance(eff, type) and issubclass(eff, Constexpr):
            return True
        if is_specializable_struct_type(eff):
            return True
    return False


def _make_composite_class(
    *,
    name: str,
    module: str,
    fields: tuple[FieldDef, ...],
    policy: CompositeKind,
    display_name: str,
):
    identity = _make_type_identity(policy, fields)

    def __init__(self, *args, **kwargs):
        if policy == CompositeKind.Sum:
            raise TypeError(
                f"Union {_display_name(type(self))} has no value form; use Storage[...] or allocator.allocate()."
            )
        base_cls = getattr(type(self), "__dsl_base_type__", type(self))
        values = _field_values_from_args(base_cls, args, kwargs)
        coerced = {field.name: _coerce_value_type(base_cls, field, values[field.name]) for field in fields}
        specialized = _specialize_type(base_cls, fields, coerced)
        if specialized is not type(self):
            object.__setattr__(self, "__class__", specialized)
        for field in fields:
            object.__setattr__(self, field.name, coerced[field.name])
        object.__setattr__(self, "_schema_frozen", True)

    def __setattr__(self, key, value):
        if getattr(self, "_schema_frozen", False):
            raise FrozenInstanceError(f"cannot assign to field '{key}'")
        object.__setattr__(self, key, value)

    def __delattr__(self, key):
        if getattr(self, "_schema_frozen", False):
            raise FrozenInstanceError(f"cannot delete field '{key}'")
        object.__delattr__(self, key)

    def __repr__(self):
        body = ", ".join(f"{field.name}={getattr(self, field.name)!r}" for field in fields)
        return f"{_display_name(type(self))}({body})"

    def __eq__(self, other):
        self_base = getattr(type(self), "__dsl_base_type__", None)
        other_base = getattr(type(other), "__dsl_base_type__", None)
        if self_base is None or self_base is not other_base:
            return NotImplemented
        return all(getattr(self, f.name) == getattr(other, f.name) for f in fields)

    def __hash__(self):
        base = getattr(type(self), "__dsl_base_type__", type(self))
        return hash((base,) + tuple(getattr(self, f.name) for f in fields))

    def replace(self, **kwargs):
        values = {field.name: getattr(self, field.name) for field in fields}
        for key, value in kwargs.items():
            field_def = _resolve_field(type(self), key)
            values[field_def.name] = value
        return type(self)(**values)

    def __extract_to_ir_values__(self) -> List[ir.Value]:
        return list(
            chain.from_iterable(
                extract_to_ir_values(_carrier_for_field(eff_type, getattr(self, name)))
                for name, eff_type in _effective_field_defs(type(self))
            )
        )

    @classmethod
    def __construct_from_ir_values__(cls, values):
        rebuilt = {}
        cursor = 0
        for name, eff_type in _effective_field_defs(cls):
            nvalues = _ir_value_count_from_type(eff_type)
            rebuilt[name] = _construct_field_from_ir(eff_type, values[cursor : cursor + nvalues])
            cursor += nvalues
        if cursor != len(values):
            raise ValueError(f"struct {_display_name(cls)} expected {cursor} ir.Values, got {len(values)}")
        return cls(**rebuilt)

    def __get_ir_types__(self) -> List[ir.Type]:
        return list(
            chain.from_iterable(
                get_ir_types(_carrier_for_field(eff_type, getattr(self, name)))
                for name, eff_type in _effective_field_defs(type(self))
            )
        )

    def __c_abi_spec__(self):
        # Recurse each non-constexpr field through the shared ABI dispatcher and
        # wrap every sub-slot fill so it reads the field off the struct instance.
        slots = []
        for name, eff_type in _effective_field_defs(type(self)):
            if _is_constexpr_type(eff_type):
                continue
            for ctype, subfill in c_abi_spec(getattr(self, name)):

                def fill(struct_arg, s, _n=name, _f=subfill):
                    _f(getattr(struct_arg, _n), s)

                slots.append((ctype, fill))
        return slots

    @classmethod
    def __dsl_size_of__(cls) -> int:
        return _storage_layout(cls)[0]

    @classmethod
    def __dsl_align_of__(cls) -> int:
        return _storage_layout(cls)[1]

    @classmethod
    def __peek_from_ptr__(cls, ptr: Pointer):
        if policy != CompositeKind.Product:
            raise NotImplementedError(f"{_display_name(cls)} does not support __peek_from_ptr__")
        _, _, offsets = _storage_layout(cls)
        values = {}
        for name, eff_type in _effective_field_defs(cls):
            if _is_constexpr_type(eff_type):
                values[name] = _construct_field_from_ir(eff_type, [])
                continue
            if name not in offsets:
                raise TypeError(
                    f"Cannot peek field '{name}' in schema {_display_name(cls)} because it has no storage offset."
                )
            values[name] = peek_from_ptr(eff_type, add_offset(ptr, offsets[name]))
        return cls(**values)

    @classmethod
    def __poke_into_ptr__(cls, ptr: Pointer, value):
        if policy != CompositeKind.Product:
            raise NotImplementedError(f"{_display_name(cls)} does not support __poke_into_ptr__")
        if not isinstance(value, cls):
            raise TypeError(
                f"{_display_name(cls)}.__poke_into_ptr__ expects {_display_name(cls)} value, "
                f"got {type(value).__name__}."
            )

        _, _, offsets = _storage_layout(cls)
        value_field_types = dict(_effective_field_defs(type(value))) if is_struct_type(type(value)) else {}
        for field in fields:
            eff_type = value_field_types.get(field.name, field.type_spec)
            if _is_constexpr_type(eff_type):
                continue
            if field.name not in offsets:
                raise TypeError(
                    f"Cannot poke field '{field.name}' in schema {_display_name(cls)} because it has no storage offset."
                )
            poke_into_ptr(eff_type, add_offset(ptr, offsets[field.name]), getattr(value, field.name))

    def __cache_signature__(self):
        parts = [type(self)]
        for field in fields:
            if _is_constexpr_type(field.type_spec):
                # Constexpr fields are already folded into type(self) by _specialize_type,
                # so only the non-constexpr field values need to be encoded here.
                continue
            parts.append((field.name, cache_signature(getattr(self, field.name))))
        return tuple(parts)

    namespace = {
        "__module__": module,
        "__annotations__": {field.name: field.type_spec for field in fields},
        "__dsl_composite_kind__": policy,
        "__dsl_field_defs__": fields,
        "__dsl_type_identity__": identity,
        "__dsl_display_name__": display_name,
        "__init__": __init__,
        "__setattr__": __setattr__,
        "__delattr__": __delattr__,
        "__repr__": __repr__,
        "__eq__": __eq__,
        "__hash__": __hash__,
        "__extract_to_ir_values__": __extract_to_ir_values__,
        "__construct_from_ir_values__": __construct_from_ir_values__,
        "__cache_signature__": __cache_signature__,
        "__get_ir_types__": __get_ir_types__,
        "__c_abi_spec__": __c_abi_spec__,
        "__dsl_size_of__": __dsl_size_of__,
        "__dsl_align_of__": __dsl_align_of__,
        "__peek_from_ptr__": __peek_from_ptr__,
        "__poke_into_ptr__": __poke_into_ptr__,
        "replace": replace,
    }
    schema = type(name, (), namespace)
    schema.__dsl_base_type__ = schema
    return schema


class CompositeMeta(type):
    def __new__(mcs, name, bases, namespace, *, policy=CompositeKind.Product, display=None, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        cls._policy = policy
        cls._display = display or name
        return cls

    def __call__(cls, klass=None, /, **kwargs):
        policy = cls._policy

        def wrap(wrapped):
            fields = _normalize_decorator_fields(wrapped)
            return _make_composite_class(
                name=wrapped.__name__,
                module=wrapped.__module__,
                fields=fields,
                policy=policy,
                display_name=wrapped.__name__,
            )

        if klass is None:
            return wrap
        return wrap(klass)

    def __getitem__(cls, params):
        fields = _normalize_inline_fields(params)
        display_name = _inline_display_name(cls._display, params, fields)
        return _make_composite_class(
            name=f"_Dsl{cls._display}_{abs(hash(_make_type_identity(cls._policy, fields)))}",
            module=__name__,
            fields=fields,
            policy=cls._policy,
            display_name=display_name,
        )

    def __repr__(cls):
        return cls._display.lower()


class struct(metaclass=CompositeMeta, policy=CompositeKind.Product, display="Struct"): ...


class union(metaclass=CompositeMeta, policy=CompositeKind.Sum, display="Union"): ...


Struct = struct
Union = union


class Align:
    __dsl_align_wrapper__: bool = False
    dtype: Any = None
    align: int | None = None

    def __class_getitem__(cls, params):
        if cls is not Align:
            raise TypeError(f"{cls.__name__} cannot be re-parametrized")
        if not isinstance(params, tuple) or len(params) != 2:
            raise TypeError("struct.Align expects struct.Align[Type, N]")
        dtype, requested_align = params
        if isinstance(requested_align, bool) or not isinstance(requested_align, int):
            raise TypeError(f"struct.Align alignment must be an int, got {requested_align!r}")
        if requested_align <= 0:
            raise ValueError(f"struct.Align alignment must be positive, got {requested_align}")
        if not (requested_align > 0 and (requested_align & (requested_align - 1)) == 0):
            raise ValueError(f"struct.Align alignment must be a power of two, got {requested_align}")
        natural = dsl_align_of(dtype)
        if requested_align < natural:
            raise ValueError(
                f"struct.Align[{_type_name(dtype)}, {requested_align}]: requested alignment {requested_align} "
                f"is smaller than natural alignment {natural} of {_type_name(dtype)}; use a value >= {natural}."
            )

        def _aligned_size_of(inner=dtype):
            return dsl_size_of(inner)

        def _aligned_align_of(val=requested_align):
            return val

        def _aligned_peek_from_ptr(ptr, inner=dtype):
            return peek_from_ptr(inner, ptr)

        def _aligned_poke_into_ptr(ptr, value, inner=dtype):
            poke_into_ptr(inner, ptr, value)

        inner_key = _type_cache_key(dtype)

        def _cache_sig(key=inner_key, a=requested_align):
            return ("align", key, a)

        return type(
            f"struct.Align[{_type_name(dtype)}, {requested_align}]",
            (cls,),
            {
                "dtype": dtype,
                "align": requested_align,
                "__dsl_align_wrapper__": True,
                "__cache_signature__": classmethod(lambda cls, _f=_cache_sig: _f()),
                "__dsl_size_of__": classmethod(lambda cls, _f=_aligned_size_of: _f()),
                "__dsl_align_of__": classmethod(lambda cls, _f=_aligned_align_of: _f()),
                "__peek_from_ptr__": classmethod(lambda cls, ptr, _f=_aligned_peek_from_ptr: _f(ptr)),
                "__poke_into_ptr__": classmethod(lambda cls, ptr, value, _f=_aligned_poke_into_ptr: _f(ptr, value)),
            },
        )


class Storage:
    """Typed memory view: ``Storage[T]`` wraps an ``i8`` pointer for Storable ``T``.

    - ``._ptr`` — underlying i8* pointer
    - ``._target_type`` — the Storable type `T`
    - ``.peek()`` — calls ``T.__peek_from_ptr__(ptr)``
    - ``.poke(value)`` — calls ``T.__poke_into_ptr__(ptr, value)``
    - For composite T: attribute access (``storage.field_name``) returns ``Storage[FieldType]``
    """

    _target_type = None
    _cache: dict[Any, type] = {}

    def __class_getitem__(cls, target_type):
        cached = Storage._cache.get(target_type)
        if cached is not None:
            return cached

        target_name = _type_name(target_type)

        class _StorageImpl(Storage):
            _target_type = target_type

            def __init__(self, ptr, prebuilt=None):
                object.__setattr__(self, "_ptr", ptr)
                object.__setattr__(self, "_prebuilt", prebuilt or {})

            def peek(self):
                dsl_type = type(self)._target_type
                prebuilt = object.__getattribute__(self, "_prebuilt")
                if prebuilt and is_struct_type(dsl_type):
                    values = {}
                    for name, eff_type in _effective_field_defs(dsl_type):
                        if _is_constexpr_type(eff_type):
                            values[name] = _construct_field_from_ir(eff_type, [])
                            continue
                        if name in prebuilt:
                            values[name] = prebuilt[name].peek()
                    return dsl_type(**values)
                ptr = object.__getattribute__(self, "_ptr")
                if ptr is None:
                    raise RuntimeError(
                        f"Storage[{target_name}].peek() requires a backing pointer; this Storage "
                        "has neither a base pointer nor a prebuilt sub-allocation tree."
                    )
                return peek_from_ptr(dsl_type, ptr)

            def poke(self, value):
                dsl_type = type(self)._target_type
                prebuilt = object.__getattribute__(self, "_prebuilt")
                if prebuilt and is_struct_type(dsl_type):
                    for name, eff_type in _effective_field_defs(dsl_type):
                        if _is_constexpr_type(eff_type):
                            continue
                        if name in prebuilt:
                            prebuilt[name].poke(getattr(value, name))
                    return
                ptr = object.__getattribute__(self, "_ptr")
                if ptr is None:
                    raise RuntimeError(
                        f"Storage[{target_name}].poke() requires a backing pointer; this Storage "
                        "has neither a base pointer nor a prebuilt sub-allocation tree."
                    )
                return poke_into_ptr(dsl_type, ptr, value)

            def __getattr__(self, name):
                prebuilt = object.__getattribute__(self, "_prebuilt")
                if name in prebuilt:
                    return prebuilt[name]
                dsl_type = type(self)._target_type
                if is_composite_type(dsl_type):
                    try:
                        field_def = _resolve_field(dsl_type, name)
                    except KeyError:
                        raise AttributeError(
                            f"Storage[{target_name}] has no field '{name}'. "
                            f"Available: {[f.name for f in dsl_type.__dsl_field_defs__]}"
                        ) from None
                    _, _, offsets = _storage_layout(dsl_type)
                    if field_def.name not in offsets:
                        raise AttributeError(
                            f"Storage[{target_name}] field '{field_def.name}' is compile-time only and has no storage"
                        ) from None
                    ptr = object.__getattribute__(self, "_ptr")
                    if ptr is None:
                        raise AttributeError(
                            f"Storage[{target_name}] has no backing pointer and no prebuilt entry for '{name}'."
                        ) from None
                    offset = offsets[field_def.name]
                    sub_ptr = add_offset(ptr, offset)
                    return Storage[field_def.type_spec](sub_ptr)
                raise AttributeError(f"Storage[{target_name}] has no attribute '{name}'")

            def __repr__(self):
                return f"Storage[{target_name}]({object.__getattribute__(self, '_ptr')})"

        _StorageImpl.__name__ = f"Storage[{target_name}]"
        _StorageImpl.__qualname__ = f"Storage[{target_name}]"
        Storage._cache[target_type] = _StorageImpl
        return _StorageImpl

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)


class Arena:
    """Bump-pointer allocator over a single base pointer (address-space agnostic).

    The default ``allocate(T)`` path bumps one contiguous region per call and
    returns ``Storage[T]`` wrapping ``add_offset(base_ptr, off)``.
    Arena does not opt into any backend-specific symbol-splitting mechanism.
    """

    DEFAULT_BASE_ALIGNMENT = 16

    def __init__(self, base_alignment: int = DEFAULT_BASE_ALIGNMENT):
        self._offset = 0
        self._base_alignment = base_alignment

    @property
    def base_ptr(self):
        raise NotImplementedError

    @property
    def allocated_bytes(self) -> int:
        return self._offset

    def _bump(self, nbytes: int, align: int) -> int:
        offset = _align_up(self._offset, align)
        self._offset = offset + nbytes
        return offset

    @dsl_loc_tracing
    def allocate(self, storable_or_int, alignment=None):
        """Allocate a Storable type or raw bytes, returning ``Storage[T]``.

        - ``allocate(StorableType)`` — allocate by storable layout, return ``Storage[StorableType]``
        - ``allocate(N: int)`` — allocate N raw bytes, return ``Storage[Array[UInt8, N]]``
        """
        if isinstance(storable_or_int, int):
            from .numeric import Uint8

            nbytes = storable_or_int
            if nbytes <= 0:
                raise ValueError(f"allocate size must be > 0, got {nbytes}")
            align = alignment if alignment is not None else self._base_alignment
            offset = self._bump(nbytes, align)
            base = add_offset(self.base_ptr, offset)
            return Storage[Array[Uint8, nbytes]](base)
        else:
            storable = storable_or_int
            nbytes = dsl_size_of(storable)
            align = dsl_align_of(storable) if alignment is None else max(dsl_align_of(storable), alignment)
            offset = self._bump(nbytes, align)
            base = add_offset(self.base_ptr, offset)
            return Storage[storable](base)
