#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Conformance tests for ``docs/language/storage_and_allocator.md``.

Same scope as that spec: ``Storage[T]`` as a typed address (the DSL's ``T*``),
what a `Storage` may point at, the byte layout it navigates, and the allocators
that produce one. Keep the two in sync when either changes.

    Part 1  →  ## fx.Storage[T]: a typed address
    Part 2  →  ## What a Storage can point at
    Part 3  →  ## Byte layout
    Part 4  →  ## Allocators

Declaring the composites themselves is ``test_composite_types.py``.

Checks run through the real DSL frontend (frontend-only; see ``conftest.py``),
so tracing needs no GPU. Only ``SharedAllocator`` requires a ``@flyc.kernel``
trace (``launch_ir``); everything else runs in a plain ``@flyc.jit`` body, over a
register-memory pointer where an address is needed.
"""

import importlib

import pytest
from lang_utils import launch_ir, source_ir

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.protocol import dsl_align_of, dsl_size_of
from flydsl.expr.struct import _storage_layout

pytestmark = pytest.mark.l1a_compile_no_target_dialect


# ###########################################################################
# Shared fixtures & helpers
#   (docs/language/storage_and_allocator.md → types reused across the parts)
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
class Padded:
    head: fx.Int32
    payload: fx.Align[fx.Int32, 16]


@fx.struct
class Shared:
    a: fx.Array[fx.Float32, 128, 16]
    b: fx.Array[fx.Float32, 128, 16]


@fx.struct
class Params:
    tile: fx.Constexpr[int]
    scale: fx.Float32


@fx.union
class Scratch:
    fp16: fx.Array[fx.Float16, 128]
    fp32: fx.Array[fx.Float32, 64]


# ── A Storable leaf that records the pointer it is read from / written to ──


class Word:
    width = 4
    poked: list = []

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
        cls.poked.append((ptr, value.value))


class Wide(Word):
    width = 8


@pytest.fixture
def symbolic_offsets(monkeypatch):
    """Make ``add_offset`` symbolic, so a peeked/poked address reads as ``(base, offset)``."""
    struct_module = importlib.import_module("flydsl.expr.struct")
    monkeypatch.setattr(struct_module, "add_offset", lambda ptr, offset: (ptr, offset))
    Word.poked.clear()
    yield
    Word.poked.clear()


def _offsets(dsl_type):
    """Field byte offsets of a storable type."""
    return _storage_layout(dsl_type)[2]


def _make_ptr_lines(ir_text):
    return [line.strip() for line in ir_text.splitlines() if "fly.make_ptr" in line]


# ###########################################################################
# Part 1 — fx.Storage[T]: a typed address
#   (docs/language/storage_and_allocator.md → ## fx.Storage[T]: a typed address)
# ###########################################################################


class TestTypedAddress:
    """`Storage[T]` is the DSL's `T*`: one MLIR pointer plus a trace-time `T`."""

    @pytest.mark.parametrize("target", [Pair, Scratch, fx.Array[fx.Int32, 4]])
    def test_an_mlir_pointer_cannot_name_a_composite(self, target):
        """Why the wrapper exists: a pointer's element type must be an MLIR type."""
        with pytest.raises(TypeError):
            fx.PointerType.get(elem_ty=target)

    def test_the_target_type_lives_in_python_not_in_the_pointer(self):
        def body():
            ptr = fx.get_iter(fx.make_rmem_tensor(2, fx.Int32))
            storage = fx.Storage[Pair](ptr)
            field_view = storage.right

            assert type(storage)._target_type is Pair
            assert type(field_view)._target_type is fx.Float32
            # Only the Python-side type differs; the MLIR pointer type is unchanged.
            assert str(object.__getattribute__(field_view, "_ptr").type) == str(ptr.type)

        source_ir(body)


class TestStorageView:

    def test_storage_targets_its_type_and_is_cached(self):
        assert fx.Storage[Pair]._target_type is Pair
        assert fx.Storage[Pair] is fx.Storage[Pair]
        assert fx.Storage[fx.Int32].__name__ == "Storage[Int32]"

    def test_unknown_field_is_rejected(self):
        with pytest.raises(AttributeError, match="has no field"):
            fx.Storage[Pair](None).missing

    def test_constexpr_field_has_no_view(self):
        with pytest.raises(AttributeError, match="compile-time only"):
            fx.Storage[Params](None).tile

    def test_peek_reads_each_field_at_its_offset(self, symbolic_offsets):
        @fx.struct
        class Mixed:
            head: Word
            tail: Wide

        value = fx.Storage[Mixed]("base").peek()
        assert value.head == Word(("peek", ("base", 0)))
        assert value.tail == Wide(("peek", ("base", 8)))

    def test_poke_descends_into_a_nested_struct(self, symbolic_offsets):
        @fx.struct
        class Leaves:
            x: Word
            y: Word

        @fx.struct
        class Tree:
            head: Word
            leaves: Leaves
            tail: Word

        fx.Storage[Tree]("base").poke(Tree(head=Word(1), leaves=Leaves(x=Word(2), y=Word(3)), tail=Word(4)))
        assert Word.poked == [
            (("base", 0), 1),
            ((("base", 4), 0), 2),
            ((("base", 4), 4), 3),
            (("base", 12), 4),
        ]

    def test_constexpr_fields_are_skipped_by_both(self, symbolic_offsets):
        @fx.struct
        class Config:
            n: fx.Constexpr[int]
            value: Word

        value = Config(n=32, value=Word(7))
        peeked = fx.Storage[type(value)]("base").peek()
        fx.Storage[Config]("base").poke(value)

        assert peeked.n == 32
        assert peeked.value == Word(("peek", ("base", 0)))
        assert Word.poked == [(("base", 0), 7)]

    def test_peek_and_poke_reach_the_backing_memory(self):
        def body():
            storage = fx.Storage[Pair](fx.get_iter(fx.make_rmem_tensor(2, fx.Int32)))
            storage.poke(Pair(fx.Int32(1), fx.Float32(2.0)))
            value = storage.peek()
            _ = value.left + fx.Int32(1)

        ir_text = source_ir(body)
        assert ir_text.count("fly.ptr.store") == 2
        assert ir_text.count("fly.ptr.load") == 2

    def test_union_has_no_value_form_to_peek(self):
        with pytest.raises(NotImplementedError):
            Scratch.__peek_from_ptr__(None)
        with pytest.raises(NotImplementedError):
            Scratch.__poke_into_ptr__(None, None)

    def test_union_variants_view_the_same_address(self, symbolic_offsets):
        storage = fx.Storage[Scratch]("base")
        assert object.__getattribute__(storage.fp16, "_ptr") == ("base", 0)
        assert object.__getattribute__(storage.fp32, "_ptr") == ("base", 0)


# ###########################################################################
# Part 2 — What a Storage can point at
#   (docs/language/storage_and_allocator.md → ## What a Storage can point at)
# ###########################################################################


class TestStorableLeaves:

    @pytest.mark.parametrize(
        "dtype, size",
        [(fx.Int8, 1), (fx.Int32, 4), (fx.Float32, 4), (fx.Int64, 8), (fx.Float64, 8)],
    )
    def test_numeric_leaves(self, dtype, size):
        assert (dsl_size_of(dtype), dsl_align_of(dtype)) == (size, size)

    @pytest.mark.parametrize("dtype", [fx.Boolean, fx.Int4])
    def test_sub_byte_numerics_are_not_storable(self, dtype):
        with pytest.raises(TypeError, match="sub-byte|Storable"):
            dsl_size_of(dtype)

    @pytest.mark.parametrize("dtype", [fx.Vector, fx.Pointer, fx.Tensor])
    def test_device_values_are_not_storable(self, dtype):
        with pytest.raises(TypeError, match="Storable"):
            dsl_size_of(dtype)


class TestArrayLeaf:

    def test_size_and_alignment(self):
        Tile = fx.Array[fx.Float32, 32, 16]
        assert (Tile.size, Tile.align) == (32, 16)
        assert (dsl_size_of(Tile), dsl_align_of(Tile)) == (128, 16)

    @pytest.mark.parametrize(
        "dtype, align",
        [(fx.Float32, 4), (fx.Float16, 2), (fx.Uint8, 1), (fx.Int64, 8)],
    )
    def test_alignment_defaults_to_the_element_byte_size(self, dtype, align):
        assert fx.Array[dtype, 32].align == align
        assert dsl_align_of(fx.Array[dtype, 32]) == align

    def test_array_types_are_cached(self):
        assert fx.Array[fx.Int32, 64] is fx.Array[fx.Int32, 64]
        assert fx.Array[fx.Int32, 64] is not fx.Array[fx.Int32, 64, 8]

    def test_indexing_and_view(self):
        def body():
            arr = fx.Array[fx.Float32, 8].__peek_from_ptr__(fx.get_iter(fx.make_rmem_tensor(8, fx.Float32)))
            arr[0] = fx.Float32(1.0)
            _ = arr[1]
            _ = arr.view(fx.make_layout(8, 1))

        ir_text = source_ir(body)
        assert "fly.ptr.store" in ir_text
        assert "fly.ptr.load" in ir_text
        assert "fly.make_view" in ir_text

    def test_whole_array_poke_is_not_implemented(self):
        with pytest.raises(NotImplementedError):
            fx.Array[fx.Int32, 64].__poke_into_ptr__(None, None)

    @pytest.mark.parametrize(
        "make, match",
        [
            (lambda: fx.Array[object, 4], "Numeric subclass"),
            (lambda: fx.Array[fx.Int32, 0], "positive integer"),
            (lambda: fx.Array[fx.Int32, -1], "positive integer"),
            (lambda: fx.Array[fx.Int32, 4, 0], "positive integer"),
            (lambda: fx.Array[fx.Int32], r"Array\[dtype, size\]"),
        ],
    )
    def test_parameter_errors(self, make, match):
        with pytest.raises(TypeError, match=match):
            make()


# ── fx.Align[T, A] ──────────────────────────────────────────────────────────


class TestAlignModifier:
    """`Align` only overrides alignment; it delegates size and access to `T`."""

    def test_size_is_unchanged_and_alignment_is_raised(self):
        Aligned = fx.Align[fx.Int32, 16]
        assert Aligned.dtype is fx.Int32
        assert (dsl_size_of(Aligned), dsl_align_of(Aligned)) == (4, 16)

    def test_access_is_delegated_to_the_inner_type(self, symbolic_offsets):
        Aligned = fx.Align[Word, 16]
        assert Aligned.__peek_from_ptr__("base") == Word(("peek", "base"))
        Aligned.__poke_into_ptr__("base", Word(7))
        assert Word.poked == [("base", 7)]

    @pytest.mark.parametrize(
        "align, match",
        [
            (0, "positive"),
            (-1, "positive"),
            (3, "power of two"),
            (24, "power of two"),
            (2, "smaller than natural alignment"),
        ],
    )
    def test_alignment_value_errors(self, align, match):
        with pytest.raises(ValueError, match=match):
            fx.Align[fx.Int32, align]

    @pytest.mark.parametrize(
        "make, match",
        [
            (lambda: fx.Align[fx.Int32, 1.0], "must be an int"),
            (lambda: fx.Align[fx.Int32, True], "must be an int"),
            (lambda: fx.Align[fx.Int32], r"Align\[Type, N\]"),
        ],
    )
    def test_alignment_type_errors(self, make, match):
        with pytest.raises(TypeError, match=match):
            make()


# ###########################################################################
# Part 3 — Byte layout
#   (docs/language/storage_and_allocator.md → ## Byte layout)
# ###########################################################################


class TestProductLayout:
    """Sequential placement, per-field alignment, trailing padding."""

    def test_sequential_fields(self):
        assert _offsets(Pair) == {"left": 0, "right": 4}
        assert (dsl_size_of(Pair), dsl_align_of(Pair)) == (8, 4)

    def test_alignment_gap_and_trailing_padding(self):
        assert _offsets(Padded) == {"head": 0, "payload": 16}
        assert (dsl_size_of(Padded), dsl_align_of(Padded)) == (32, 16)

    def test_nested_struct_layout_is_recursive(self):
        assert _offsets(Outer) == {"head": 0, "inner": 4, "tail": 12}
        assert (dsl_size_of(Outer), dsl_align_of(Outer)) == (16, 4)

    def test_array_fields_use_their_element_alignment_by_default(self):
        @fx.struct
        class Tiles:
            a: fx.Array[fx.Float32, 32]
            b: fx.Array[fx.Float32, 32]

        assert _offsets(Tiles) == {"a": 0, "b": 128}
        assert (dsl_size_of(Tiles), dsl_align_of(Tiles)) == (256, 4)

    def test_array_fields_carry_an_explicit_alignment(self):
        assert _offsets(Shared) == {"a": 0, "b": 512}
        assert (dsl_size_of(Shared), dsl_align_of(Shared)) == (1024, 16)

    def test_constexpr_fields_have_no_offset(self):
        assert dsl_size_of(Params) == 4
        assert "tile" not in _offsets(Params)


class TestUnionLayout:
    """Every field at offset zero; max size and max alignment."""

    def test_offsets_are_all_zero(self):
        assert _offsets(Scratch) == {"fp16": 0, "fp32": 0}

    def test_size_is_the_maximum_field_size(self):
        assert (dsl_size_of(Scratch), dsl_align_of(Scratch)) == (256, 4)

    def test_size_is_rounded_up_to_the_maximum_alignment(self):
        @fx.union
        class Mixed:
            small: fx.Array[fx.Uint8, 6]
            wide: fx.Align[fx.Int32, 16]

        assert (dsl_size_of(Mixed), dsl_align_of(Mixed)) == (16, 16)

    def test_union_inside_a_struct(self):
        @fx.struct
        class WithUnion:
            head: fx.Int32
            scratch: Scratch

        assert _offsets(WithUnion) == {"head": 0, "scratch": 4}
        assert dsl_size_of(WithUnion) == 260

    def test_struct_inside_a_union(self):
        @fx.union
        class WithStruct:
            pair: Inner
            single: fx.Float32

        assert _offsets(WithStruct) == {"pair": 0, "single": 0}
        assert (dsl_size_of(WithStruct), dsl_align_of(WithStruct)) == (8, 4)


# ###########################################################################
# Part 4 — Allocators
#   (docs/language/storage_and_allocator.md → ## Allocators)
# ###########################################################################


# ── fx.Arena — the target-neutral bump allocator ────────────────────────────


class TestArena:

    def test_starts_empty(self):
        assert fx.Arena().allocated_bytes == 0

    def test_base_pointer_is_supplied_by_a_subclass(self):
        with pytest.raises(NotImplementedError):
            fx.Arena().base_ptr


# ── fx.SharedAllocator — the LDS allocator ──────────────────────────────────


class TestAllocate:

    def test_allocate_returns_a_storage_tree(self):
        @flyc.kernel
        def tree_kernel():
            @fx.struct
            class Nested:
                inner: Inner
                scratch: Scratch

            storage = fx.SharedAllocator().allocate(Nested)
            assert storage._target_type is Nested
            assert storage.inner._target_type is Inner
            assert storage.inner.x._target_type is fx.Int32
            assert storage.scratch.fp32._target_type is fx.Array[fx.Float32, 64]

        launch_ir(tree_kernel)

    def test_allocated_bytes_follows_the_logical_layout(self):
        @flyc.kernel
        def bytes_kernel():
            allocator = fx.SharedAllocator()
            allocator.allocate(Shared)
            assert allocator.allocated_bytes == 1024
            allocator.allocate(Scratch)
            assert allocator.allocated_bytes == 1280

        launch_ir(bytes_kernel)

    def test_sequential_allocations_pad_to_alignment(self):
        @flyc.kernel
        def padding_kernel():
            allocator = fx.SharedAllocator()
            allocator.allocate(fx.Int32)  # 0..3
            allocator.allocate(Padded)  # aligned to 16 → 16..47
            assert allocator.allocated_bytes == 48

        launch_ir(padding_kernel)

    def test_explicit_alignment_raises_only_that_allocation(self):
        @flyc.kernel
        def alignment_kernel():
            allocator = fx.SharedAllocator()
            allocator.allocate(fx.Int32, alignment=32)
            allocator.allocate(fx.Int32)
            assert allocator.allocated_bytes == 8

        launch_ir(alignment_kernel)

    def test_constexpr_field_is_not_allocated(self):
        @flyc.kernel
        def constexpr_kernel():
            allocator = fx.SharedAllocator()
            storage = allocator.allocate(Params)
            assert allocator.allocated_bytes == 4
            with pytest.raises(AttributeError, match="compile-time only"):
                _ = storage.tile

        launch_ir(constexpr_kernel)

    def test_raw_byte_allocation(self):
        @flyc.kernel
        def raw_kernel():
            storage = fx.SharedAllocator().allocate(256)
            assert storage._target_type is fx.Array[fx.Uint8, 256]

        make_ptrs = _make_ptr_lines(launch_ir(raw_kernel))
        assert len(make_ptrs) == 1
        assert "allocBytes = 256" in make_ptrs[0]

    def test_raw_byte_allocation_rejects_non_positive_sizes(self):
        @flyc.kernel
        def bad_size_kernel():
            allocator = fx.SharedAllocator()
            with pytest.raises(ValueError, match="must be > 0"):
                allocator.allocate(0)
            with pytest.raises(ValueError, match="must be > 0"):
                allocator.allocate(-1)

        launch_ir(bad_size_kernel)

    def test_non_storable_type_is_rejected(self):
        @flyc.kernel
        def bad_type_kernel():
            allocator = fx.SharedAllocator()
            with pytest.raises(TypeError, match="Storable"):
                allocator.allocate(fx.Struct["t" : fx.Tensor])

        launch_ir(bad_type_kernel)

    def test_allocator_requires_a_kernel(self):
        def body():
            with pytest.raises(RuntimeError, match="@kernel"):
                fx.SharedAllocator()

        source_ir(body)

    def test_one_allocator_per_kernel(self):
        @flyc.kernel
        def two_allocators_kernel():
            fx.SharedAllocator()
            fx.SharedAllocator()

        with pytest.raises(RuntimeError, match="Only one SharedAllocator"):
            launch_ir(two_allocators_kernel)


class TestStaticPlacement:
    """`static=True` (default): one LDS allocation per struct leaf."""

    def test_one_allocation_per_leaf(self):
        @flyc.kernel
        def static_kernel():
            allocator = fx.SharedAllocator()
            assert allocator.is_static is True
            lds = allocator.allocate(Shared).peek()
            _ = lds.a.view(fx.make_layout(128, 1))
            _ = lds.b.view(fx.make_layout(128, 1))

        ir_text = launch_ir(static_kernel)
        make_ptrs = _make_ptr_lines(ir_text)
        assert len(make_ptrs) == 2
        assert all("allocBytes = 512" in line for line in make_ptrs)
        assert all("shared" in line for line in make_ptrs)
        assert "dynamic_shared_memory_size" not in ir_text

    def test_nested_struct_emits_one_allocation_per_leaf(self):
        @flyc.kernel
        def nested_kernel():
            @fx.struct
            class Nested:
                inner: Inner
                tail: fx.Float32

            fx.SharedAllocator().allocate(Nested)

        assert len(_make_ptr_lines(launch_ir(nested_kernel))) == 3

    def test_union_leaf_is_one_allocation_sized_to_the_widest_variant(self):
        @flyc.kernel
        def union_leaf_kernel():
            @fx.struct
            class WithUnion:
                head: fx.Int32
                scratch: Scratch

            storage = fx.SharedAllocator().allocate(WithUnion)
            assert object.__getattribute__(storage.scratch.fp16, "_ptr") is object.__getattribute__(
                storage.scratch.fp32, "_ptr"
            )

        make_ptrs = _make_ptr_lines(launch_ir(union_leaf_kernel))
        assert len(make_ptrs) == 2
        assert any("allocBytes = 256" in line for line in make_ptrs)

    def test_static_mode_has_no_base_pointer(self):
        @flyc.kernel
        def base_ptr_kernel():
            allocator = fx.SharedAllocator()
            with pytest.raises(RuntimeError, match="no shared base pointer"):
                _ = allocator.base_ptr

        launch_ir(base_ptr_kernel)


class TestDynamicPlacement:
    """`static=False`: one dynamic base pointer, sized at launch."""

    def test_single_base_pointer_and_inferred_smem(self):
        @flyc.kernel
        def dynamic_kernel():
            allocator = fx.SharedAllocator(static=False)
            assert allocator.is_static is False
            lds = allocator.allocate(Shared).peek()
            _ = lds.a.view(fx.make_layout(128, 1))

        ir_text = launch_ir(dynamic_kernel)
        assert ir_text.count("fly.get_dyn_shared") == 1
        assert "fly.make_ptr" not in ir_text
        assert "dynamic_shared_memory_size %c1024_i32" in ir_text

    def test_base_pointer_is_in_the_shared_address_space(self):
        @flyc.kernel
        def base_ptr_kernel():
            allocator = fx.SharedAllocator(static=False)
            assert allocator.base_ptr.address_space == fx.AddressSpace.Shared

        launch_ir(base_ptr_kernel)
