#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Conformance tests for ``docs/language/composite_types.md``.

Same scope as that spec: declaring a ``@fx.struct`` / ``@fx.union``, what may be
a field, how composites nest, and the one rule the whole type is built on — a
composite is closed under each protocol separately, satisfying it exactly when
all of its non-``Constexpr`` fields do. Keep the two in sync when either changes.

    Part 1  →  ## Declaring a composite
    Part 2  →  ## What can be a field
    Part 3  →  ## Nesting
    Part 4  →  ## Closure over the protocols
    Part 5  →  ## Compile-time fields
    Part 6  →  ## JIT and kernel boundaries

Byte layout, ``Storage`` and the allocators are the ``Storable`` side of the
story and live in ``test_storage_and_allocator.py``.
"""

from dataclasses import FrozenInstanceError

import pytest
from lang_utils import source_ir

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler.protocol import (
    c_abi_spec,
    cache_signature,
    construct_from_ir_values,
    dsl_size_of,
    extract_to_ir_values,
    get_ir_types,
)
from flydsl.expr.struct import Storage

pytestmark = pytest.mark.l1a_compile_no_target_dialect


# ###########################################################################
# Shared fixtures & helpers
#   (docs/language/composite_types.md → types reused across the parts)
# ###########################################################################


@fx.struct
class Pair:
    left: fx.Int32
    right: fx.Float32


@fx.struct
class Inner:
    x: fx.Int32
    y: fx.Int32


@fx.struct
class Outer:
    head: fx.Int32
    inner: Inner
    tail: fx.Float32


@fx.struct
class Params:
    tile: fx.Constexpr[int]
    scale: fx.Float32


@fx.union
class Scratch:
    fp16: fx.Array[fx.Float16, 128]
    fp32: fx.Array[fx.Float32, 64]


@fx.struct
class WithVector:
    scalar: fx.Int32
    vector: fx.Vector


# ###########################################################################
# Part 1 — Declaring a composite
#   (docs/language/composite_types.md → ## Declaring a composite)
# ###########################################################################


# ── Product form ────────────────────────────────────────────────────────────


class TestProductForm:
    """`@fx.struct` is an ordered product with an immutable value form."""

    def test_positional_and_named_construction(self):
        assert Pair(1, 2.0) == Pair(left=1, right=2.0)

    def test_field_types_coerce_python_literals(self):
        pair = Pair(1, 2.0)
        assert isinstance(pair.left, fx.Int32)
        assert isinstance(pair.right, fx.Float32)
        assert (pair.left, pair.right) == (1, 2.0)

    def test_fields_follow_annotation_order(self):
        assert tuple(Outer.__annotations__) == ("head", "inner", "tail")

    def test_value_is_frozen(self):
        pair = Pair(1, 2.0)
        with pytest.raises(FrozenInstanceError):
            pair.left = fx.Int32(4)
        with pytest.raises(FrozenInstanceError):
            del pair.left

    def test_replace_returns_a_new_value(self):
        pair = Pair(1, 2.0)
        assert pair.replace(left=3).left == 3
        assert pair.left == 1

    def test_equality_and_hash_are_structural(self):
        assert Pair(1, 2.0) == Pair(1, 2.0)
        assert hash(Pair(1, 2.0)) == hash(Pair(1, 2.0))
        assert Pair(1, 2.0) != Pair(3, 2.0)

    def test_equality_is_per_type(self):
        @fx.struct
        class OtherPair:
            left: fx.Int32
            right: fx.Float32

        assert Pair(1, 2.0) != OtherPair(1, 2.0)

    @pytest.mark.parametrize(
        "make, match",
        [
            (lambda: Pair(left=1), "missing required field"),
            (lambda: Pair(1, 2.0, extra=3), "unexpected field"),
            (lambda: Pair(1, left=2), "multiple values"),
            (lambda: Pair(object(), 2.0), "expects Int32"),
            (lambda: Pair(1, 2.0, 3.0), "expected 2 field"),
        ],
    )
    def test_constructor_errors(self, make, match):
        with pytest.raises(TypeError, match=match):
            make()


# ── Overlay form ────────────────────────────────────────────────────────────


class TestOverlayForm:
    """`@fx.union` is a storage overlay: no value form, no tag."""

    def test_union_has_no_value_form(self):
        with pytest.raises(TypeError, match="no value form"):
            Scratch(fp16=None)

    def test_inline_union_has_no_value_form(self):
        Inline = fx.Union["i" : fx.Int32, "f" : fx.Float32]
        with pytest.raises(TypeError, match="no value form"):
            Inline(1)


# ── Inline forms and field names ────────────────────────────────────────────


class TestInlineForms:

    def test_named_fields(self):
        Named = fx.Struct["left" : fx.Int32, "right" : fx.Float32]
        assert Named(left=1, right=2.0).right == 2.0

    def test_anonymous_fields_are_positionally_named(self):
        Anonymous = fx.Struct[fx.Int32, fx.Float32]
        assert tuple(Anonymous.__annotations__) == ("_0", "_1")
        assert Anonymous(1, 2.0)._0 == 1

    def test_mixed_named_and_anonymous_fields(self):
        Mixed = fx.Struct["left" : fx.Int32, fx.Float32]
        assert tuple(Mixed.__annotations__) == ("left", "_1")
        assert Mixed(1, 2.0)._1 == 2.0

    def test_each_spelling_is_a_distinct_class_with_the_same_identity(self):
        first = fx.Struct["left" : fx.Int32]
        second = fx.Struct["left" : fx.Int32]
        assert first is not second
        assert first.__dsl_type_identity__ == second.__dsl_type_identity__
        assert first(1) == first(1)
        assert first(1) != second(1)

    def test_cross_type_comparison_falls_back_to_identity(self):
        """`__eq__` declines across declarations, so Python answers `False`."""
        first = fx.Struct["left" : fx.Int32]
        second = fx.Struct["left" : fx.Int32]
        assert first(1).__eq__(second(1)) is NotImplemented

    def test_unknown_attribute_is_rejected(self):
        with pytest.raises(AttributeError):
            fx.Struct[fx.Int32](1).missing

    @pytest.mark.parametrize(
        "make, match",
        [
            (lambda: fx.Struct["a" : fx.Int32, "a" : fx.Float32], "duplicate"),
            (lambda: fx.Struct[()], "at least one field"),
        ],
    )
    def test_inline_declaration_errors(self, make, match):
        with pytest.raises(ValueError, match=match):
            make()


# ── Reserved field names ────────────────────────────────────────────────────


class TestReservedFieldNames:
    """A field may not collide with a real member of the value or its `Storage` view."""

    def test_the_reserved_names_really_are_members(self):
        assert callable(Pair(1, 2.0).replace)
        assert callable(Storage[Pair](None).peek)
        assert callable(Storage[Pair](None).poke)

    def test_underscore_names_are_the_implementations_own(self):
        assert object.__getattribute__(Pair(1, 2.0), "_schema_frozen") is True
        assert set(object.__getattribute__(Storage[Pair](None), "__dict__")) == {"_ptr", "_prebuilt"}
        assert Storage[Pair]._target_type is Pair

    @pytest.mark.parametrize("name", ["replace", "peek", "poke"])
    def test_member_names_are_rejected(self, name):
        with pytest.raises(ValueError, match="reserved"):
            fx.Struct[name : fx.Int32]
        with pytest.raises(ValueError, match="reserved"):
            fx.Union[name : fx.Int32, "other" : fx.Float32]

    @pytest.mark.parametrize("name", ["_x", "_ptr", "_schema_frozen", "__dsl_field_defs__"])
    def test_underscore_names_are_rejected(self, name):
        with pytest.raises(ValueError, match="must not start with underscore"):
            fx.Struct[name : fx.Int32]

    def test_the_decorator_form_validates_the_same_names(self):
        with pytest.raises(ValueError, match="reserved"):

            @fx.struct
            class Reserved:
                peek: fx.Int32

        with pytest.raises(ValueError, match="must not start with underscore"):

            @fx.struct
            class Hidden:
                _hidden: fx.Int32

    def test_generated_anonymous_names_are_exempt(self):
        assert tuple(fx.Struct[fx.Int32, fx.Float32].__annotations__) == ("_0", "_1")


# ###########################################################################
# Part 2 — What can be a field
#   (docs/language/composite_types.md → ## What can be a field)
# ###########################################################################


class TestFieldTypes:
    """Any `DslType` may be a field; `Constexpr` is the trace-time addition."""

    def test_numeric_field(self):
        assert isinstance(Pair(1, 2.0).left, fx.Int32)

    def test_vector_field(self):
        def body():
            vector = fx.Vector.filled(4, 1.0, fx.Float32)
            value = WithVector(scalar=fx.Int32(1), vector=vector)
            assert value.vector.dtype is fx.Float32
            assert len(extract_to_ir_values(value)) == 2

        source_ir(body)

    def test_vector_alias_field_pins_lanes_and_dtype(self):
        """An alias annotation rebuilds the value through the alias, re-checking it."""
        Aliased = fx.Struct["v" : fx.Float32x4]

        def body():
            value = Aliased(fx.Vector.filled(4, 1.0, fx.Float32))
            assert isinstance(value.v, fx.Float32x4)
            assert (value.v.dtype, value.v.shape) == (fx.Float32, (4,))

            assert isinstance(Aliased(1.0).v, fx.Float32x4)  # scalar broadcasts
            assert isinstance(Aliased([1.0, 2.0, 3.0, 4.0]).v, fx.Float32x4)

            with pytest.raises(TypeError, match="expects Float32x4"):
                Aliased(fx.Vector.filled(4, 1.0, fx.Float16))
            with pytest.raises(TypeError, match="expects Float32x4"):
                Aliased(fx.Vector.filled(8, 1.0, fx.Float32))

        source_ir(body)

    def test_pointer_and_array_fields(self):
        Buffers = fx.Struct["arr" : fx.Array[fx.Float32, 8], "ptr" : fx.Pointer]

        def body():
            arr = fx.Array[fx.Float32, 8].__peek_from_ptr__(fx.get_iter(fx.make_rmem_tensor(8, fx.Float32)))
            value = Buffers(arr, arr.ptr)
            assert isinstance(value.ptr, fx.Pointer)
            assert len(extract_to_ir_values(value)) == 2

        source_ir(body)

    def test_tensor_field(self):
        torch = pytest.importorskip("torch")
        WithTensor = fx.Struct["t" : fx.Tensor]

        def body(t: fx.Tensor):
            value = WithTensor(t)
            assert len(extract_to_ir_values(value)) == 1

        source_ir(body, torch.zeros(8))

    def test_struct_field(self):
        assert isinstance(Outer(head=1, inner=Inner(2, 3), tail=4.0).inner, Inner)

    def test_union_field_has_layout_but_no_value(self):
        @fx.struct
        class HasUnion:
            head: fx.Int32
            scratch: Scratch

        assert dsl_size_of(HasUnion) == 260
        with pytest.raises(TypeError, match="expects Scratch"):
            HasUnion(fx.Int32(1), None)

    def test_constexpr_field(self):
        assert Params(tile=32, scale=1.0).tile == 32


# ###########################################################################
# Part 3 — Nesting
#   (docs/language/composite_types.md → ## Nesting)
# ###########################################################################


class TestNesting:

    def test_struct_in_struct(self):
        outer = Outer(head=1, inner=Inner(2, 3), tail=4.0)
        assert outer.inner.y == 3

    def test_nesting_is_recursive(self):
        @fx.struct
        class Deep:
            outer: Outer
            extra: fx.Int32

        deep = Deep(outer=Outer(head=1, inner=Inner(2, 3), tail=4.0), extra=5)
        assert deep.outer.inner.x == 2

        def body(a: fx.Int32, b: fx.Float32):
            value = Deep(outer=Outer(head=a, inner=Inner(a, a), tail=b), extra=a)
            assert len(extract_to_ir_values(value)) == 5

        source_ir(body, 1, 2.0)

    def test_replace_keeps_nested_values(self):
        outer = Outer(head=1, inner=Inner(2, 3), tail=4.0)
        assert outer.replace(head=9).inner is outer.inner

    def test_nested_constexpr_travels_with_the_type(self):
        @fx.struct
        class WithParams:
            head: fx.Int32
            params: Params

        value = WithParams(head=1, params=Params(tile=32, scale=1.0))
        assert value.params.tile == 32
        assert dsl_size_of(WithParams) == 8


# ###########################################################################
# Part 4 — Closure over the protocols
#   (docs/language/composite_types.md → ## Closure over the protocols)
# ###########################################################################


class TestDslTypeClosure:
    """Fields all `DslType` ⇒ the composite is a `DslType`."""

    def test_flattens_in_declaration_order(self):
        def body(a: fx.Int32, b: fx.Float32):
            outer = Outer(head=a, inner=Inner(a, a), tail=b)
            flat = extract_to_ir_values(outer)
            assert len(flat) == 4
            assert isinstance(flat[0].type, ir.IntegerType)
            assert isinstance(flat[3].type, ir.F32Type)
            assert [str(t) for t in get_ir_types(outer)] == [str(v.type) for v in flat]

        source_ir(body, 1, 2.0)

    def test_round_trip_is_exact(self):
        def body(a: fx.Int32, b: fx.Float32):
            outer = Outer(head=a, inner=Inner(a, a), tail=b)
            flat = extract_to_ir_values(outer)
            rebuilt = construct_from_ir_values(type(outer), outer, flat)
            assert isinstance(rebuilt.inner, Inner)
            assert [v.get_name() for v in extract_to_ir_values(rebuilt)] == [v.get_name() for v in flat]

        source_ir(body, 1, 2.0)

    def test_round_trip_preserves_field_metadata(self):
        """A `Vector` field keeps its shape/dtype through the exemplar."""

        def body(a: fx.Int32):
            value = WithVector(scalar=a, vector=fx.Vector.filled(4, 1.0, fx.Float32))
            rebuilt = construct_from_ir_values(type(value), value, extract_to_ir_values(value))
            assert isinstance(rebuilt.vector, fx.Vector)
            assert rebuilt.vector.dtype is fx.Float32

        source_ir(body, 1)

    def test_constexpr_field_contributes_no_values(self):
        def body(b: fx.Float32):
            params = Params(tile=32, scale=b)
            flat = extract_to_ir_values(params)
            assert len(flat) == 1
            assert construct_from_ir_values(type(params), params, flat).tile == 32

        source_ir(body, 1.0)

    def test_surplus_values_are_rejected(self):
        def body(a: fx.Int32, b: fx.Float32):
            flat = extract_to_ir_values(Pair(a, b))
            with pytest.raises(ValueError, match="expected 2 ir.Values"):
                Pair.__construct_from_ir_values__(flat + flat)

        source_ir(body, 1, 2.0)


class TestJitArgumentClosure:
    """Fields all `JitArgument` ⇒ the composite is a `JitArgument`."""

    def test_abi_slots_are_one_per_run_time_field(self):
        with ir.Context(), ir.Location.unknown():
            value = Inner(x=fx.Int32(7), y=fx.Int32(11))
            slots = c_abi_spec(value)
            assert len(slots) == 2

            filled = []
            for ctype, fill in slots:
                storage = ctype(0)
                fill(value, storage)
                filled.append(storage.value)
            assert filled == [7, 11]

    def test_cache_signature_combines_the_fields(self):
        with ir.Context(), ir.Location.unknown():
            assert cache_signature(Pair(1, 2.0)) == cache_signature(Pair(3, 4.0))
            assert cache_signature(Pair(1, 2.0)) != cache_signature(Inner(1, 2))

    def test_one_non_qualifying_field_disqualifies_the_composite(self):
        def body(a: fx.Int32):
            value = WithVector(scalar=a, vector=fx.Vector.filled(4, 1.0, fx.Float32))
            with pytest.raises(TypeError, match="cache signature"):
                cache_signature(value)
            with pytest.raises(TypeError, match="C-ABI"):
                c_abi_spec(value)

        source_ir(body, 1)


class TestStorableClosure:
    """Fields all `Storable` ⇒ the composite is `Storable` (layout: see the storage spec)."""

    def test_all_storable_fields(self):
        assert dsl_size_of(Outer) == 16

    def test_one_non_storable_field_disqualifies_the_composite(self):
        with pytest.raises(TypeError, match="Storable"):
            dsl_size_of(WithVector)
        with pytest.raises(TypeError, match="Storable"):
            dsl_size_of(fx.Struct["t" : fx.Tensor])

    def test_constexpr_field_is_skipped(self):
        assert dsl_size_of(Params) == 4


# ###########################################################################
# Part 5 — Compile-time fields
#   (docs/language/composite_types.md → ## Compile-time fields)
# ###########################################################################


class TestConstexprField:

    def test_value_is_a_python_value(self):
        params = Params(tile=32, scale=1.0)
        assert params.tile == 32
        assert type(params.tile) is int

    def test_construction_specializes_the_type(self):
        assert type(Params(tile=32, scale=1.0)).__name__ == "Params[tile=32]"
        assert type(Params(tile=32, scale=1.0)) is not type(Params(tile=64, scale=1.0))

    def test_specialization_reaches_the_cache_signature(self):
        lhs = Params(tile=32, scale=1.0).__cache_signature__()
        rhs = Params(tile=64, scale=1.0).__cache_signature__()
        assert lhs != rhs

    def test_wrong_value_type_is_rejected(self):
        with pytest.raises(TypeError, match="expects int"):
            Params(tile=1.5, scale=1.0)

    def test_a_run_time_value_is_never_a_constexpr_value(self):
        with pytest.raises(TypeError, match="expects int"):
            Params(tile=fx.Int32(4), scale=1.0)

    def test_contributes_nothing_at_run_time(self):
        assert fx.Constexpr[int].__extract_to_ir_values__() == []
        assert fx.Constexpr[int].__get_ir_types__() == []

    def test_the_field_cannot_be_assigned(self):
        params = Params(tile=32, scale=1.0)
        with pytest.raises(FrozenInstanceError):
            params.tile = 64

    def test_replace_respecializes_the_type(self):
        """Changing the value is a change of type; changing a run-time field is not."""
        params = Params(tile=32, scale=1.0)
        retiled = params.replace(tile=64)
        assert type(retiled).__name__ == "Params[tile=64]"
        assert retiled.tile == 64
        assert type(params).__name__ == "Params[tile=32]"
        assert params != retiled
        assert type(params.replace(scale=2.0)) is type(params)

    def test_only_a_constexpr_change_moves_the_cache_key(self):
        params = Params(tile=32, scale=1.0)
        assert cache_signature(params) != cache_signature(params.replace(tile=64))
        assert cache_signature(params) == cache_signature(params.replace(scale=2.0))


# ###########################################################################
# Part 6 — JIT and kernel boundaries
#   (docs/language/composite_types.md → ## JIT and kernel boundaries)
# ###########################################################################


class TestJitKernelBoundary:

    def test_struct_built_in_a_jit_body_arrives_flattened(self):
        @flyc.kernel
        def pair_kernel(pair: Pair):
            _ = pair.left + fx.Int32(1)
            _ = pair.right * fx.Float32(2.0)

        def body(a: fx.Int32, b: fx.Float32):
            pair_kernel(Pair(a, b)).launch(grid=(1, 1, 1), block=(64, 1, 1))

        ir_text = source_ir(body, 1, 2.0)
        signature = next(line for line in ir_text.splitlines() if "gpu.func @pair_kernel" in line)
        assert signature.count("%arg") == 2
        assert "i32" in signature and "f32" in signature

    def test_constexpr_field_specializes_the_traced_kernel(self):
        @flyc.kernel
        def unrolled_kernel(params: Params):
            for _i in fx.range_constexpr(params.tile):
                _ = params.scale + fx.Float32(1.0)

        def body(b: fx.Float32):
            unrolled_kernel(Params(tile=3, scale=b)).launch(grid=(1, 1, 1), block=(64, 1, 1))

        ir_text = source_ir(body, 1.0)
        assert ir_text.count("arith.addf") == 3
        signature = next(line for line in ir_text.splitlines() if "gpu.func @unrolled_kernel" in line)
        assert signature.count("%arg") == 1

    def test_host_struct_argument_with_scalar_leaves(self):
        def body(pair: Pair):
            _ = pair.left + fx.Int32(1)

        assert "arith.addi" in source_ir(body, Pair(3, 4.0))

    def test_raw_framework_tensor_is_not_an_fx_tensor_field(self):
        torch = pytest.importorskip("torch")

        IOPair = fx.Struct["x" : fx.Tensor, "y" : fx.Tensor]
        host_tensor = torch.zeros(8)

        with pytest.raises(TypeError, match="expects Tensor"):
            IOPair(host_tensor, host_tensor)
        with pytest.raises(TypeError, match="expects Tensor"):
            IOPair(flyc.from_torch_tensor(host_tensor), flyc.from_torch_tensor(host_tensor))

    def test_tensor_struct_is_built_inside_the_jit_body(self):
        torch = pytest.importorskip("torch")

        IOPair = fx.Struct["x" : fx.Tensor, "y" : fx.Tensor]

        @flyc.kernel
        def tensor_kernel(io: IOPair):
            _ = io.x.shape

        def body(a: fx.Tensor, b: fx.Tensor):
            tensor_kernel(IOPair(a, b)).launch(grid=(1, 1, 1), block=(64, 1, 1))

        host_tensor = torch.zeros(8)
        ir_text = source_ir(body, host_tensor, host_tensor)
        signature = next(line for line in ir_text.splitlines() if "gpu.func @tensor_kernel" in line)
        assert signature.count("%arg") == 2
