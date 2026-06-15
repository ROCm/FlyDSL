#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Unit tests for unified struct / union / Array / Storage types."""

import importlib

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler import jit_function
from flydsl.compiler.protocol import (
    c_abi_spec,
    construct_from_ir_values,
    dsl_align_of,
    dsl_size_of,
    extract_to_ir_values,
    get_ir_types,
)
from flydsl.expr.numeric import Float32, Int32, Uint8
from flydsl.expr.struct import Storage
from flydsl.expr.typing import Array

pytestmark = pytest.mark.l0_backend_agnostic


@pytest.fixture
def frontend_only_jit(monkeypatch):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "rocm")
    monkeypatch.setenv("ARCH", "gfx942")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setattr(jit_function, "_flydsl_key", lambda: "test-flydsl-key")

    def compile_noop(cls, module, **_kwargs):
        return module

    monkeypatch.setattr(jit_function.MlirCompiler, "compile", classmethod(compile_noop))


# ---------------------------------------------------------------------------
# struct basics
# ---------------------------------------------------------------------------


def test_struct_decorator_creates_frozen_value_schema():
    @fx.struct
    class Pair:
        a: Int32
        b: Float32

    p = Pair(1, 2.0)
    assert tuple(Pair.__annotations__) == ("a", "b")
    assert isinstance(p.a, Int32)
    assert isinstance(p.b, Float32)
    assert p.a == 1
    assert p.b == 2.0

    q = p.replace(a=3)
    assert q.a == 3
    assert p.a == 1

    with pytest.raises(Exception, match="cannot assign"):
        p.a = Int32(4)


def test_struct_constructor_requires_exact_fields():
    @fx.struct
    class Pair:
        a: Int32
        b: Float32

    with pytest.raises(TypeError, match="missing required field"):
        Pair(a=1)
    with pytest.raises(TypeError, match="unexpected field"):
        Pair(a=1, b=2.0, c=3)
    with pytest.raises(TypeError, match="expects Int32"):
        Pair(a=object(), b=2.0)


def test_inline_struct_named_and_positional_forms():
    Named = fx.Struct["a":Int32, "b":Float32]
    assert Named is not fx.Struct["a":Int32, "b":Float32]
    assert Named.__dsl_type_identity__ == fx.Struct["a":Int32, "b":Float32].__dsl_type_identity__

    n = Named(1, 2.0)
    assert n.a == 1
    assert n.b == 2.0

    Pos = fx.Struct["a":Int32, Float32]
    assert tuple(Pos.__annotations__) == ("a", "_1")

    p = Pos(3, 4.0)
    assert p.a == 3
    assert p._1 == 4.0

    Anonymous = fx.Struct[Int32, Float32]
    Positional = fx.Struct[Int32, Float32]
    assert Anonymous.__dsl_type_identity__ == Positional.__dsl_type_identity__
    assert tuple(Anonymous.__annotations__) == ("_0", "_1")
    anon = Anonymous(5, 6.0)
    assert anon._0 == 5
    assert anon._1 == 6.0
    assert anon == Anonymous(5, 6.0)
    with pytest.raises(AttributeError):
        anon.nonexistent


def test_union_schema_has_no_value_form():
    @fx.union
    class Variant:
        i: Int32
        f: Float32

    with pytest.raises(TypeError, match="no value form"):
        Variant(i=Int32(1))

    Inline = fx.Union["i":Int32, "f":Float32]
    with pytest.raises(TypeError, match="no value form"):
        Inline(Int32(1))


def test_struct_flattens_non_constexpr_fields_only(frontend_only_jit):
    @fx.struct
    class Params:
        a: Int32
        b: Float32
        n: fx.Constexpr[int]

    @flyc.jit
    def build(p: Params):
        values = p.__extract_to_ir_values__()
        assert len(values) == 2
        assert isinstance(values[0].type, ir.IntegerType)
        assert isinstance(values[1].type, ir.F32Type)
        assert [str(t) for t in get_ir_types(p)] == [str(v.type) for v in values]
        assert p.n == 32

        rebuilt = construct_from_ir_values(type(p), p, values)
        assert isinstance(rebuilt, Params)
        assert rebuilt.n == 32
        assert [v.get_name() for v in extract_to_ir_values(rebuilt)] == [v.get_name() for v in values]

    build(Params(a=Int32(7), b=Float32(2.0), n=32))


def test_constexpr_is_not_part_of_storage_layout():
    @fx.struct
    class Params:
        n: fx.Constexpr[int]
        a: Int32

    assert dsl_size_of(Params) == 4
    assert dsl_align_of(Params) == 4
    storage = Storage[Params](None)
    with pytest.raises(AttributeError, match="compile-time only"):
        storage.n
    with pytest.raises(TypeError, match="Storable"):
        dsl_size_of(fx.Constexpr[int])


def test_nested_struct_round_trip_via_exemplar(frontend_only_jit):
    @fx.struct
    class Inner:
        x: Int32
        y: Int32

    @fx.struct
    class Outer:
        head: Int32
        inner: Inner
        tail: Float32

    @flyc.jit
    def build(outer: Outer):
        flat = outer.__extract_to_ir_values__()
        rebuilt = construct_from_ir_values(type(outer), outer, flat)
        assert isinstance(rebuilt.inner, Inner)
        assert [v.get_name() for v in rebuilt.__extract_to_ir_values__()] == [v.get_name() for v in flat]

    build(
        Outer(
            head=Int32(1),
            inner=Inner(x=Int32(2), y=Int32(3)),
            tail=Float32(4.0),
        )
    )


def test_align_wrapper_overrides_natural_alignment():
    Aligned = fx.Align[Int32, 16]
    assert Aligned.dtype is Int32
    assert Aligned.align == 16

    @fx.struct
    class WithAligned:
        a: Int32
        b: fx.Align[Int32, 16]

    assert dsl_align_of(WithAligned) == 16


@pytest.mark.parametrize(
    "align,exc,match",
    [
        (3, ValueError, "power of two"),
        (6, ValueError, "power of two"),
        (5, ValueError, "power of two"),
        (0, ValueError, "positive"),
        (-1, ValueError, "positive"),
        (2, ValueError, "smaller than natural"),
        (1.0, TypeError, "must be an int"),
        (True, TypeError, "must be an int"),
    ],
)
def test_align_validation_rejects_invalid_values(align, exc, match):
    with pytest.raises(exc, match=match):
        fx.Align[Int32, align]


def test_align_requires_two_parameters():
    with pytest.raises(TypeError, match="Align\\[Type, N\\]"):
        fx.Align[Int32]


def test_struct_equality_and_hash():
    @fx.struct
    class Pair:
        a: Int32
        b: Float32

    p1 = Pair(1, 2.0)
    p2 = Pair(1, 2.0)
    p3 = Pair(1, 3.0)

    assert p1 == p2
    assert p1 != p3

    @fx.struct
    class OtherPair:
        a: Int32
        b: Float32

    assert Pair(1, 2.0) != OtherPair(1, 2.0)


def test_inline_schema_rejects_duplicate_field_names():
    with pytest.raises(ValueError, match="duplicate"):
        fx.Struct["a":Int32, "a":Float32]

    with pytest.raises(ValueError, match="must not start with underscore"):
        fx.Struct["_1":Int32, Float32]


def test_field_name_validation_rejects_underscore_prefix():
    with pytest.raises(ValueError, match="must not start with underscore"):

        @fx.struct
        class Bad:
            _hidden: Int32

    with pytest.raises(ValueError, match="must not start with underscore"):
        fx.Struct["_x":Int32]


def test_field_name_validation_rejects_reserved_names():
    with pytest.raises(ValueError, match="reserved"):

        @fx.struct
        class Bad:
            peek: Int32

    with pytest.raises(ValueError, match="reserved"):

        @fx.struct
        class Bad2:
            poke: Int32

    with pytest.raises(ValueError, match="reserved"):
        fx.Union["replace":Int32, "b":Float32]


def test_host_jit_argument_protocol_pointers():
    @fx.struct
    class HostPair:
        a: Int32
        b: Int32

    with ir.Context(), ir.Location.unknown():
        p = HostPair(a=Int32(7), b=Int32(11))
        slots = c_abi_spec(p)
        assert len(slots) == 2
        # Each slot fills its storage in place from the struct instance.
        values = []
        for ctype, fill in slots:
            s = ctype(0)
            fill(p, s)
            values.append(s.value)
        assert values == [7, 11]


# ---------------------------------------------------------------------------
# Array type
# ---------------------------------------------------------------------------


def test_array_type_creation():
    A = Array[Float32, 32]
    assert issubclass(A, Array._Base)
    assert A.dtype is Float32
    assert A.size == 32
    assert A.align == 4

    A16 = Array[Float32, 32, 16]
    assert A16.align == 16


def test_array_type_caching():
    A1 = Array[Int32, 64]
    A2 = Array[Int32, 64]
    assert A1 is A2

    A3 = Array[Int32, 64, 8]
    assert A3 is not A1


def test_array_storable_protocol():
    A = Array[Float32, 32]
    assert dsl_size_of(A) == 4 * 32
    assert dsl_align_of(A) == 4

    A_aligned = Array[Float32, 32, 16]
    assert dsl_align_of(A_aligned) == 16


def test_array_dsl_size_of_free_function():
    A = Array[Int32, 16]
    assert dsl_size_of(A) == 4 * 16
    assert dsl_align_of(A) == 4


def test_array_rejects_non_numeric_dtype():
    with pytest.raises(TypeError, match="Numeric subclass"):
        Array[object, 32]


def test_array_rejects_invalid_size():
    with pytest.raises(TypeError, match="positive integer"):
        Array[Float32, 0]
    with pytest.raises(TypeError, match="positive integer"):
        Array[Float32, -1]


def test_array_subbyte_dtype():
    A = Array[Uint8, 64]
    assert dsl_size_of(A) == 64
    assert dsl_align_of(A) == 1


# ---------------------------------------------------------------------------
# Storage type
# ---------------------------------------------------------------------------


def test_storage_type_creation():
    S = Storage[Int32]
    assert S._target_type is Int32
    assert S.__name__ == "Storage[Int32]"


def test_storage_schema_field_access_struct():
    @fx.struct
    class Pair:
        a: Int32
        b: Float32

    SPair = Storage[Pair]
    assert SPair._target_type is Pair


def test_storage_schema_field_access_with_array():
    @fx.struct
    class SharedStorage:
        sharedA: Array[Float32, 32]
        sharedB: Array[Float32, 32]

    S = Storage[SharedStorage]
    assert S._target_type is SharedStorage


def test_struct_peek_composes_field_peeks(monkeypatch):
    struct_module = importlib.import_module("flydsl.expr.struct")
    monkeypatch.setattr(struct_module, "add_offset", lambda ptr, offset: (ptr, offset))

    class Word:
        width = 4

        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return type(self) is type(other) and self.value == other.value

        @classmethod
        def __dsl_size_of__(cls):
            return cls.width

        @classmethod
        def __dsl_align_of__(cls):
            return cls.width

        @classmethod
        def __peek_from_ptr__(cls, ptr):
            return cls(("peek", ptr))

        @classmethod
        def __poke_into_ptr__(cls, ptr, value):
            raise AssertionError("poke should not run during peek")

    class Wide(Word):
        width = 8

    @fx.struct
    class Pair:
        a: Word
        b: Wide

    result = Storage[Pair]("base").peek()

    assert result.a == Word(("peek", ("base", 0)))
    assert result.b == Wide(("peek", ("base", 8)))


def test_struct_poke_composes_field_pokes_recursively(monkeypatch):
    struct_module = importlib.import_module("flydsl.expr.struct")
    monkeypatch.setattr(struct_module, "add_offset", lambda ptr, offset: (ptr, offset))

    class Word:
        poked = []

        def __init__(self, value):
            self.value = value

        @classmethod
        def __dsl_size_of__(cls):
            return 4

        @classmethod
        def __dsl_align_of__(cls):
            return 4

        @classmethod
        def __peek_from_ptr__(cls, ptr):
            return cls(ptr)

        @classmethod
        def __poke_into_ptr__(cls, ptr, value):
            cls.poked.append((ptr, value.value))

    @fx.struct
    class Inner:
        x: Word
        y: Word

    @fx.struct
    class Outer:
        head: Word
        inner: Inner
        tail: Word

    value = Outer(head=Word(1), inner=Inner(x=Word(2), y=Word(3)), tail=Word(4))
    Storage[Outer]("base").poke(value)

    assert Word.poked == [
        (("base", 0), 1),
        ((("base", 4), 0), 2),
        ((("base", 4), 4), 3),
        (("base", 12), 4),
    ]


def test_struct_peek_and_poke_handle_constexpr_fields(monkeypatch):
    struct_module = importlib.import_module("flydsl.expr.struct")
    monkeypatch.setattr(struct_module, "add_offset", lambda ptr, offset: (ptr, offset))

    class Word:
        poked = []

        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return type(self) is type(other) and self.value == other.value

        @classmethod
        def __dsl_size_of__(cls):
            return 4

        @classmethod
        def __dsl_align_of__(cls):
            return 4

        @classmethod
        def __peek_from_ptr__(cls, ptr):
            return cls(("peek", ptr))

        @classmethod
        def __poke_into_ptr__(cls, ptr, value):
            cls.poked.append((ptr, value.value))

    @fx.struct
    class Params:
        n: fx.Constexpr[int]
        value: Word

    value = Params(n=32, value=Word(7))
    peeked = Storage[type(value)]("base").peek()
    Storage[Params]("base").poke(value)

    assert peeked.n == 32
    assert peeked.value == Word(("peek", ("base", 0)))
    assert Word.poked == [(("base", 0), 7)]


# ---------------------------------------------------------------------------
# Numeric Storable via NumericMeta
# ---------------------------------------------------------------------------


def test_numeric_storable_protocol():
    assert dsl_size_of(Int32) == 4
    assert dsl_align_of(Int32) == 4
    assert dsl_size_of(Float32) == 4
    assert dsl_align_of(Float32) == 4

    from flydsl.expr.numeric import Float64, Int64

    assert dsl_size_of(Int64) == 8
    assert dsl_align_of(Int64) == 8
    assert dsl_size_of(Float64) == 8
    assert dsl_align_of(Float64) == 8


def test_numeric_storable_via_free_functions():
    assert dsl_size_of(Int32) == 4
    assert dsl_align_of(Int32) == 4
    assert dsl_size_of(Float32) == 4
    assert dsl_align_of(Float32) == 4


def test_subbyte_numeric_not_storable():
    from flydsl.expr.numeric import Int4

    with pytest.raises(TypeError, match="sub-byte|Storable"):
        dsl_size_of(Int4)


# ---------------------------------------------------------------------------
# Struct layout with new types
# ---------------------------------------------------------------------------


def test_struct_with_array_fields_layout():
    @fx.struct
    class SharedStorage:
        sharedA: Array[Float32, 32]
        sharedB: Array[Float32, 32]

    assert dsl_size_of(SharedStorage) == 128 + 128
    assert dsl_align_of(SharedStorage) == 4


def test_struct_with_aligned_array_fields():
    @fx.struct
    class AlignedStorage:
        sharedA: Array[Float32, 32, 16]
        sharedB: Array[Float32, 32, 16]

    assert dsl_align_of(AlignedStorage) == 16


def test_union_storable_layout():
    @fx.union
    class Variant:
        i: Int32
        f: Float32

    assert dsl_size_of(Variant) == 4
    assert dsl_align_of(Variant) == 4


def test_struct_containing_union_layout():
    @fx.union
    class Variant:
        i: Int32
        f: Float32

    @fx.struct
    class Tagged:
        tag: Int32
        data: Variant

    assert dsl_size_of(Tagged) == 8
    assert dsl_align_of(Tagged) == 4

    from flydsl.expr.struct import _storage_layout

    _, _, offsets = _storage_layout(Tagged)
    assert offsets == {"tag": 0, "data": 4}


def test_union_containing_struct_layout():
    @fx.struct
    class Pair:
        a: Int32
        b: Int32

    @fx.union
    class UnionWithStruct:
        pair: Pair
        f: Float32

    assert dsl_size_of(UnionWithStruct) == 8
    assert dsl_align_of(UnionWithStruct) == 4

    from flydsl.expr.struct import _storage_layout

    _, _, offsets = _storage_layout(UnionWithStruct)
    assert offsets == {"pair": 0, "f": 0}


def test_nested_struct_union_with_alignment():
    @fx.union
    class DataVariant:
        arr: Array[Float32, 16]
        single: Int32

    @fx.struct
    class TaggedVariant:
        tag: Int32
        data: DataVariant

    assert dsl_size_of(DataVariant) == 64
    assert dsl_align_of(DataVariant) == 4
    assert dsl_size_of(TaggedVariant) == 68
    assert dsl_align_of(TaggedVariant) == 4


def test_union_with_aligned_field_pads_size():
    @fx.struct
    class Inner:
        x: Int32

    @fx.union
    class Outer:
        inner: Inner
        aligned: fx.Align[Int32, 8]

    assert dsl_align_of(Outer) == 8
    assert dsl_size_of(Outer) == 8
