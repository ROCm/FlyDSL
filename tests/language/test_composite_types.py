#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Conformance tests for ``docs/language/composite_types.md``.

Same scope as that spec: declaring a ``@fx.struct`` / ``@fx.union``, what may be
a field, member behavior and caching, how composites nest, and the one rule the whole type is built on — a
composite is closed under each protocol separately, satisfying it exactly when
all of its non-``Constexpr`` fields do. Keep the two in sync when either changes.

    Part 1  →  ## Declaring a composite
    Part 2  →  ## Member methods and properties
    Part 3  →  ## What can be a field
    Part 4  →  ## Nesting
    Part 5  →  ## Closure over the protocols
    Part 6  →  ## Compile-time fields
    Part 7  →  ## JIT and kernel boundaries

Byte layout, ``Storage`` and the allocators are the ``Storable`` side of the
story and live in ``test_storage_and_allocator.py``.
"""

import importlib
import inspect
import json
import linecache
import subprocess
import sys
import textwrap
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
from lang_utils import source_ir

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl.compiler import jit_function
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


# ###########################################################################
# Part 2 — Member methods and properties
#   (docs/language/composite_types.md → ## Member methods and properties)
# ###########################################################################


_MEMBER_SCALE = 2
_NON_STRUCT_CACHE_TYPE = None


class _NonStructWithSignature:
    @classmethod
    def __cache_signature__(cls):
        raise AssertionError("the JIT must not query non-Struct type signatures")


@fx.struct
class Point:
    """A value with behavior; only the three annotations are fields."""

    x: fx.Int32
    y: fx.Int32
    scale: fx.Constexpr[int]

    def _sum(self):
        return self.x + self.y

    @property
    def total(self):
        return self._sum() * self.scale

    def shifted(self, offset):
        return self.replace(x=self.x + offset, y=self.y + offset)

    @classmethod
    def diagonal(cls, value, scale):
        return cls(value, value, scale)

    @staticmethod
    def dimensions():
        return 2

    def __add__(self, other):
        return self.replace(x=self.x + other.x, y=self.y + other.y)

    def __lt__(self, other):
        return self.total < other.total

    def __call__(self, offset):
        return self.total + offset

    @flyc.jit
    def bounded_x(self, bound):
        value = self.x
        if value < 0:
            value = -value
        if value > bound:
            value = bound
        return value


def test_members_factories_and_operators():
    p = Point.diagonal(3, scale=2)
    assert type(Point) is type
    assert type(type(p)) is type
    assert p.total == 12
    assert p.dimensions() == Point.dimensions() == 2
    assert p.shifted(1).total == 16
    assert p.total == 12
    assert (p + p).total == 24
    assert p < p.shifted(1)
    assert p(5) == 17
    assert Point.__doc__.startswith("A value with behavior")
    assert dsl_size_of(Point) == 8


def test_methods_survive_specialization_ir_roundtrip_and_control_flow():
    def body(x: fx.Int32):
        point = Point.diagonal(x, scale=3)
        rebuilt = construct_from_ir_values(type(point), point, extract_to_ir_values(point))
        assert type(rebuilt) is type(point)
        assert rebuilt.scale == 3
        assert isinstance(rebuilt.total, fx.Int32)
        assert isinstance(rebuilt.bounded_x(fx.Int32(7)), fx.Int32)
        if x > 0:
            rebuilt = rebuilt.shifted(1)
        assert isinstance(rebuilt(2), fx.Int32)

    text = source_ir(body, -2)
    assert text.count("scf.if") >= 3


def test_methods_survive_kernel_argument_reconstruction():
    @flyc.kernel
    def kernel(point: Point):
        assert point.scale == 2
        assert isinstance(point.total, fx.Int32)
        assert isinstance(point.bounded_x(fx.Int32(7)), fx.Int32)

    def body(x: fx.Int32):
        kernel(Point.diagonal(x, 2)).launch(grid=(1, 1, 1), block=(64, 1, 1))

    assert "scf.if" in source_ir(body, -2)


@pytest.mark.parametrize("mutation", ["assign", "delete", "augment", "new_attribute"])
def test_methods_cannot_mutate_self(mutation):
    @fx.struct
    class Frozen:
        x: fx.Int32

        def mutate(self, mutation):
            if mutation == "assign":
                self.x = fx.Int32(2)
            elif mutation == "delete":
                del self.x
            elif mutation == "augment":
                self.x += 1
            else:
                self.extra = 2

    value = Frozen(1)
    with pytest.raises(FrozenInstanceError):
        value.mutate(mutation)
    assert value.x == 1


def test_soa_indexing_and_stores_through_members():
    Item = fx.Struct["key" : fx.Int32, "weight" : fx.Float32]

    @fx.struct
    class Columns:
        keys: fx.Array[fx.Int32, 4]
        weights: fx.Array[fx.Float32, 4]

        @property
        def dtype(self):
            return Item

        def __getitem__(self, index):
            return Item(self.keys[index], self.weights[index])

        def __setitem__(self, index, item):
            self.keys[index] = item.key
            self.weights[index] = item.weight

    @flyc.kernel
    def kernel():
        columns = fx.SharedAllocator().allocate(Columns).peek()
        assert columns.dtype is Item
        columns[fx.thread_idx.x] = Item(fx.thread_idx.x, 2.0)
        item = columns[fx.thread_idx.x]
        assert type(item) is Item
        columns[fx.thread_idx.x] = item.replace(weight=item.weight * 3)

    def body():
        kernel().launch(grid=(1, 1, 1), block=(4, 1, 1))

    text = source_ir(body)
    assert text.count("fly.ptr.load") == 2
    assert text.count("fly.ptr.store") == 4


def test_custom_equality_respects_python_hash_contract():
    @fx.struct
    class Key:
        key: fx.Int32
        payload: fx.Int32

        def __eq__(self, other):
            return self.key == other.key

    assert Key(1, 2) == Key(1, 3)
    with pytest.raises(TypeError, match="unhashable"):
        hash(Key(1, 2))


def test_type_and_instance_queries_keep_field_metadata():
    class Field:
        def __init__(self, shape):
            self.shape = shape

        def __cache_signature__(self):
            return ("field", self.shape)

    Record = fx.Struct["field":Field]
    small, large = Record(Field(4)), Record(Field(8))
    assert type(Record) is type
    assert type(small).__cache_signature__() == type(large).__cache_signature__()
    assert cache_signature(small) != cache_signature(large)
    # The generic protocol retains its original instance-only behavior here.
    with pytest.raises(TypeError):
        cache_signature(fx.Int32)


def test_type_signature_protocol_supports_class_and_static_methods():
    class ClassBound:
        @classmethod
        def __cache_signature__(cls):
            return (cls.__name__, 7)

    class Static:
        @staticmethod
        def __cache_signature__():
            return ("static", 8)

    assert cache_signature(ClassBound) == ("ClassBound", 7)
    assert cache_signature(Static) == ("static", 8)


@pytest.mark.parametrize("dtype", [int, fx.Int32, _NonStructWithSignature, fx.Union["x" : fx.Int32]])
def test_non_struct_types_keep_existing_cache_paths(monkeypatch, dtype):
    # Even a non-Struct class offering a type signature must keep the old keys.
    monkeypatch.setitem(globals(), "_NON_STRUCT_CACHE_TYPE", dtype)

    def global_reference():
        return _NON_STRUCT_CACHE_TYPE

    def closure_reference():
        return dtype

    assert jit_function._snapshot_global_value(dtype, stable=True) == ("callable", dtype.__module__, dtype.__qualname__)
    assert jit_function._snapshot_global_value(dtype, stable=False) == ("callable", id(dtype), repr(dtype))
    for function in (global_reference, closure_reference):
        assert jit_function._collect_dependency_sources(function, inspect.getfile(function)) == []
    assert jit_function._collect_closure_scalar_vals(closure_reference) == [
        f"dtype={dtype.__module__}.{dtype.__qualname__}"
    ]

    @flyc.jit
    def build(schema: type):
        pass

    build._ensure_sig()
    assert ("schema", dtype) in build._resolve_and_make_cache_key({"schema": dtype})


def test_constexpr_callable_type_signatures_are_stable(monkeypatch):
    from typing import Callable

    @fx.struct
    class Config:
        operation: fx.Constexpr[Callable]

        def apply(self, x):
            return self.operation(x)

    def make():
        return Config(lambda x: x + 1)

    first = make()
    # Simulate rebuilding the specialization registry in another process.
    monkeypatch.setattr(fx.Constexpr, "_value_cache", {})
    second = make()
    assert type(first) is not type(second)
    assert type(first).__cache_signature__() == type(second).__cache_signature__()
    assert type(first).__cache_signature__() != type(Config(lambda x: x + 2)).__cache_signature__()


def test_definition_keys_are_stable_across_redeclaration():
    source = "class Same:\n x: fx.Int32\n def unused(self):\n  return self.x + 1\n"

    def make(text, filename):
        namespace = {"fx": fx, "__name__": "definition_stability"}
        linecache.cache[filename] = (len(text), None, text.splitlines(True), filename)
        exec(compile(text, filename, "exec"), namespace)
        return fx.struct(namespace["Same"])

    first = make(source, "/first/location.py")
    second = make("\n\n" + source, "/another/location.py")
    changed = make(source.replace("+ 1", "+ 2"), "/first/location.py")
    snapshot = lambda cls: jit_function._snapshot_global_value(cls, stable=True)
    assert snapshot(first) == snapshot(second)
    assert snapshot(first) != snapshot(changed)


def test_jit_entry_passes_runtime_fields_to_struct_member():
    @fx.struct
    class Program:
        x: fx.Int32

        @flyc.jit
        def trace(self):
            value = self.x + 1
            if value > 2:
                value = value + 3

    @flyc.jit
    def launch(program):
        program.trace()

    launch(Program(1))
    key, artifact = launch._last_compiled
    assert "scf.if" in artifact.source_ir
    assert "@launch(%arg0: i32" in artifact.source_ir
    launch(Program(8))
    assert launch._last_compiled[0] == key


def test_python_class_jit_receiver_and_runtime_arguments():
    class Program:
        x: fx.Int32

        @flyc.jit
        def trace(self, amount: fx.Int32, *, scale: fx.Constexpr[int]):
            value = (self.x + amount) * scale
            if value > 2:
                value = value + 3

    program = Program()
    program.x = 1
    jit = Program.trace
    program.trace(7, scale=2)
    key, artifact = jit._last_compiled
    assert "scf.if" in artifact.source_ir
    signature = next(line for line in artifact.source_ir.splitlines() if "func.func @trace(" in line)
    assert signature.count(": i32") == 1
    program.trace(9, scale=2)
    assert jit._last_compiled == (key, artifact)
    program.trace(9, scale=5)
    assert jit._last_compiled[0] != key


def test_annotations_and_jit_descriptor_combinations():
    @fx.struct
    class Annotated:
        x: fx.Int32

        def pair(self) -> tuple[fx.Int32, fx.Int32]:
            return self.x, self.x

        def offset(self, value: int | float):
            return self.x + value

        @property
        @flyc.jit
        def positive(self):
            result = self.x
            if result < 0:
                result = -result
            return result

        @classmethod
        @flyc.jit
        def make(cls, x):
            return cls(x)

        @staticmethod
        @flyc.jit
        def identity(x):
            return x

    assert cache_signature(Annotated(1))

    def body(x: fx.Int32):
        value = Annotated.make(x)
        assert isinstance(value.positive, fx.Int32)
        assert isinstance(value.identity(x), fx.Int32)

    assert "scf.if" in source_ir(body, -1)


def test_plain_struct_cache_never_rehashes_definitions(monkeypatch):
    struct_module = importlib.import_module("flydsl.expr.struct")

    def unexpected_hash(*args, **kwargs):
        raise AssertionError("warm cache computation must not fingerprint definitions")

    Pair = fx.Struct["x" : fx.Int32, "y" : fx.Float32]
    Outer = fx.Struct["child":Pair, "tile" : fx.Constexpr[int]]
    value = Outer(Pair(1, 2.0), 32)
    monkeypatch.setattr(struct_module, "_field_type_signature", unexpected_hash)
    assert cache_signature(value) == (type(value), ("child", (Pair, ("x", (fx.Int32,)), ("y", (fx.Float32,)))))
    assert jit_function._snapshot_global_value(Outer, stable=True) == ("callable", Outer.__module__, Outer.__qualname__)

    def captured():
        return Outer

    assert jit_function._collect_dependency_sources(captured, inspect.getfile(captured)) == []

    @flyc.jit
    def build(schema: type, value):
        pass

    build._ensure_sig()
    key = build._resolve_and_make_cache_key({"schema": Outer, "value": value})
    assert ("schema", Outer) in key


@pytest.mark.parametrize("entry", ["scalar", "plain_struct", "member_struct", "type"])
def test_warm_jit_cache_does_not_collect_dependencies(monkeypatch, tmp_path, entry):
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    if entry == "type":

        @flyc.jit
        def launch(schema: type, value: fx.Int32):
            _ = schema.diagonal(value, 2).total

        first, second = (Point, 1), (Point, 9)
    else:

        @flyc.jit
        def launch(value):
            pass

        first, second = {
            "scalar": ((fx.Int32(1),), (fx.Int32(9),)),
            "plain_struct": ((Pair(1, 2.0),), (Pair(9, 2.0),)),
            "member_struct": ((Point(1, 2, 3),), (Point(9, 2, 3),)),
        }[entry]

    launch(*first)
    compiled = launch._last_compiled

    def unexpected(*args, **kwargs):
        raise AssertionError("warm JIT calls must not collect dependencies")

    monkeypatch.setattr(jit_function, "_collect_dependency_sources", unexpected)
    monkeypatch.setattr(jit_function, "_collect_closure_scalar_vals", unexpected)
    launch(*second)
    assert launch._last_compiled == compiled
    assert len(launch._mem_cache) == 1


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.parametrize(
    "entry", ["value", "type", "global", "closure", "jit_helper", "global_kernel", "global_helper", "class_helper"]
)
def test_definition_cache_keys_are_stable_across_processes(tmp_path, monkeypatch, entry):
    monkeypatch.setenv("ARCH", "gfx942")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path / "cache"))
    entry_source = {
        "value": "@flyc.jit\ndef launch(value):\n    _ = value.read()\nlaunch(Wrapper(1))\n",
        "type": "@flyc.jit\ndef launch(schema: type, x: fx.Int32):\n    _ = schema(x).read()\nlaunch(Wrapper, 1)\n",
        "global": "@flyc.jit\ndef launch(x: fx.Int32):\n    _ = Wrapper(x).read()\nlaunch(1)\n",
        "closure": (
            "def make(schema):\n"
            "    @flyc.jit\n"
            "    def launch(x: fx.Int32):\n"
            "        _ = schema(x).read()\n"
            "    return launch\n"
            "launch = make(Wrapper)\nlaunch(1)\n"
        ),
        "jit_helper": (
            "@flyc.jit\ndef read(x: fx.Int32):\n    _ = Wrapper(x).read()\n"
            "@flyc.jit\ndef launch(x: fx.Int32):\n    read(x)\nlaunch(1)\n"
        ),
        "global_kernel": (
            "def make(schema):\n"
            "    @flyc.kernel\n"
            "    def kernel(x: fx.Int32):\n"
            "        _ = schema(x).read()\n"
            "    return kernel\n"
            "kernel = make(Wrapper)\n"
            "@flyc.jit\n"
            "def launch(x: fx.Int32):\n"
            "    kernel(x).launch(grid=(1, 1, 1), block=(64, 1, 1))\n"
            "launch(1)\n"
        ),
        "global_helper": (
            "def make(schema):\n"
            "    def read(x):\n"
            "        return schema(x).read()\n"
            "    return read\n"
            "read = make(Wrapper)\n"
            "@flyc.jit\ndef launch(x: fx.Int32):\n    _ = read(x)\nlaunch(1)\n"
        ),
        "class_helper": (
            "def make(schema):\n"
            "    def read(self, x):\n"
            "        return schema(x).read()\n"
            "    return read\n"
            "class Program:\n"
            "    read = make(Wrapper)\n"
            "    @flyc.jit\n"
            "    def launch(self, x: fx.Int32):\n"
            "        _ = self.read(x)\n"
            "Program().launch(1)\n"
            "launch = Program.launch\n"
        ),
    }[entry]
    script = tmp_path / "definition_cache.py"
    script.write_text(
        "import json, sys, linecache\n"
        "import flydsl.compiler as flyc\n"
        "import flydsl.expr as fx\n"
        "from flydsl.compiler import jit_function as j\n"
        "j._flydsl_key = lambda: 'fixed-toolchain'\n"
        "definition = 'class Item:\\n x: fx.Int32\\n def read(self):\\n  return self.x + ' + sys.argv[1] + '\\n'\n"
        "linecache.cache['<item>'] = (len(definition), None, definition.splitlines(True), '<item>')\n"
        "exec(compile(definition, '<item>', 'exec'))\n"
        "Item = fx.struct(Item)\n"
        "@fx.struct\n"
        "class Middle:\n"
        "    x: fx.Int32\n"
        "    def read(self):\n"
        "        return Item(self.x).read()\n"
        "@fx.struct\n"
        "class Wrapper:\n"
        "    x: fx.Int32\n"
        "    def read(self):\n"
        "        return Middle(self.x).read()\n" + entry_source + "artifact = next(iter(launch._mem_cache.values()))\n"
        "Inline = fx.Struct['x':fx.Int32]\n"
        "Fields = fx.Struct['rows':fx.Array[Inline, 4], 'aligned':fx.Align[Inline, 16], 'view':fx.Storage[Inline]]\n"
        "Config = fx.Struct['scale':fx.Constexpr[tuple]]\n"
        "field_key = (Fields.__cache_signature__(), type(Config((2, 3))).__cache_signature__())\n"
        "print(json.dumps({'compiled': launch._last_compiled is not None, 'hits': launch.cache_info().hits, 'ir': artifact.source_ir, 'field_key': field_key}))\n"
    )

    def run(offset):
        return json.loads(subprocess.check_output([sys.executable, str(script), str(offset)], text=True))

    first, reused, changed = run(1), run(1), run(9)
    assert first["field_key"] == reused["field_key"] == changed["field_key"]
    assert first["compiled"] and first["hits"] == 0
    assert not reused["compiled"] and reused["hits"] == 1
    assert changed["compiled"] and changed["hits"] == 0
    assert "arith.constant 1" in reused["ir"]
    assert "arith.constant 9" in changed["ir"]


def test_attribute_name_does_not_capture_unrelated_global(monkeypatch):
    def make(dependency):
        monkeypatch.setitem(globals(), "x", dependency)

        @fx.struct
        class Attribute:
            x: fx.Int32

            def read(self):
                return self.x

        return Attribute

    first, second = make(_declare_record(1)), make(_declare_record(9))
    assert first.__cache_signature__() == second.__cache_signature__()
    assert first(1).read() == second(1).read() == 1


@flyc.kernel(known_block_size=[64, 1, 1])
def _point_kernel(a: fx.Tensor, out: fx.Tensor):
    index = fx.thread_idx.x
    point = Point.diagonal(a[index], scale=2)
    out[index] = point.bounded_x(fx.Int32(7)) + (point + point).total + point.shifted(1)(3)


@flyc.jit
def _point_launch(a: fx.Tensor, out: fx.Tensor):
    _point_kernel(a, out).launch(grid=(1, 1, 1), block=(64, 1, 1))


@pytest.mark.l1b_target_dialect
@pytest.mark.rocm_lower
@pytest.mark.parametrize("arch", ["gfx942", "gfx1100"])
@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
def test_struct_member_target_compilation(monkeypatch, tmp_path, arch, default_device):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("ARCH", arch)
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))

    # A fresh launcher resolves each target independently.
    @flyc.jit
    def launch(a: fx.Tensor, out: fx.Tensor):
        _point_kernel(a, out).launch(grid=(1, 1, 1), block=(64, 1, 1))

    with torch.device(default_device):
        launch(torch.empty(64, dtype=torch.int32, device="cpu"), torch.empty(64, dtype=torch.int32, device="cpu"))
    assert "llvm.func" in launch._last_compiled[1]._ir_text


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
def test_struct_members_gpu(monkeypatch, tmp_path, default_device):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    monkeypatch.setenv("COMPILE_ONLY", "0")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    with torch.device(default_device):
        values = torch.arange(-32, 32, device="cuda", dtype=torch.int32)
        actual = torch.empty_like(values, device="cuda")
        _point_launch(values, actual)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, 12 * values + 7 + values.abs().clamp(max=7))


@pytest.mark.l2_device
@pytest.mark.rocm_lower
@pytest.mark.parametrize("default_device", ["cpu", "cuda"])
def test_python_class_jit_entry_calls_struct_member_gpu(monkeypatch, tmp_path, default_device):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    monkeypatch.setenv("COMPILE_ONLY", "0")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))

    @fx.struct
    class Config:
        bias: fx.Int32
        scale: fx.Constexpr[int]

        @flyc.jit
        def evaluate(self, value):
            return value * self.scale + self.bias

    class Program:
        @flyc.jit
        def run(self, config, values: fx.Tensor, output: fx.Tensor):
            @flyc.kernel
            def apply(config, values: fx.Tensor, output: fx.Tensor):
                index = fx.thread_idx.x
                output[index] = config.evaluate(values[index])

            apply(config, values, output).launch(grid=(1, 1, 1), block=(64, 1, 1))

    program = Program()
    with torch.device(default_device):
        values = torch.arange(64, device="cuda", dtype=torch.int32)
        output = torch.empty_like(values, device="cuda")
        expected_values = torch.arange(64, device="cpu", dtype=torch.int32)
        for bias, scale, compilations in ((3, 2, 1), (9, 2, 1), (9, 5, 2)):
            program.run(Config(bias, scale), values, output)
            torch.testing.assert_close(output.cpu(), expected_values * scale + bias)
            assert len(Program.run._mem_cache) == compilations
            assert len(Program.run._call_state_cache) == compilations


def _declare_record(offset, *, member_kind="method"):
    # Identical names and source locations: only the implementation changes.
    namespace = {"fx": fx, "__name__": "frozen_record"}
    source = f"class Record:\n x: fx.Int32\n def read(self):\n  return self.x + {offset}\n"
    filename = f"<record-{offset}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, "exec"), namespace)
    definition = namespace["Record"]
    if member_kind == "property":
        definition.read = property(definition.read)
    return fx.struct(definition)


def test_field_types_are_implicit_dependencies():
    Inner = _declare_record(1)

    @fx.struct
    class Outer:
        child: Inner

        def next(self):
            return Inner(self.child.x + 1)

    assert Outer(Inner(1)).next().read() == 3


@pytest.mark.parametrize(
    "wrapper", [lambda t: t, lambda t: fx.Array[t, 4], lambda t: fx.Align[t, 16], lambda t: fx.Storage[t]]
)
def test_nested_field_signatures_include_frozen_member_definitions(wrapper):
    first, second = _declare_record(1), _declare_record(9)
    before = fx.Struct["items" : wrapper(first)]
    after = fx.Struct["items" : wrapper(second)]
    assert jit_function._snapshot_global_value(before, stable=True) != jit_function._snapshot_global_value(
        after, stable=True
    )


@pytest.mark.parametrize("entry", ["value", "type", "capture"])
def test_redeclaration_recompiles_with_the_new_member_body(entry, monkeypatch, tmp_path):
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    First, Second = _declare_record(1), _declare_record(9)
    if entry == "value":

        @flyc.jit
        def launch(value):
            _ = value.read()

        first_args, second_args = (First(1),), (Second(1),)
    elif entry == "type":

        @flyc.jit
        def launch(schema: type, x: fx.Int32):
            _ = schema(x).read()

        first_args, second_args = (First, 1), (Second, 1)
    else:

        def make(schema):
            @flyc.jit
            def launch(x: fx.Int32):
                _ = schema(x).read()

            return launch

        launch = make(First)
        first_args = second_args = (1,)
    launch(*first_args)
    first_key, first_artifact = launch._last_compiled
    first_manager_key = launch.manager_key
    launch(*first_args)
    assert launch._last_compiled[1] is first_artifact
    if entry == "capture":
        launch = make(Second)
    launch(*second_args)
    key, artifact = launch._last_compiled
    assert (launch.manager_key, key) != (first_manager_key, first_key)
    assert "arith.constant 1" in first_artifact.source_ir
    assert "arith.constant 9" in artifact.source_ir


def test_literal_sets_nested_code_and_self_construction():
    @fx.struct
    class Record:
        x: fx.Int32

        def contains(self, value):
            return value in {1, 2}

        def next(self):
            return type(self)(self.x + 1)

        def nested(self, values):
            return [v + 1 for v in values]

    assert Record.__cache_signature__()
    assert Record(1).contains(2)
    assert Record(1).next().x == 2
    assert Record(1).nested([1, 2]) == [2, 3]


def test_signature_hot_path_never_rebuilds_definitions(monkeypatch):
    struct_module = importlib.import_module("flydsl.expr.struct")
    Item = _declare_record(1)

    @fx.struct
    class Record:
        x: fx.Int32

        def read(self):
            return Item(self.x).read()

    value = Record(1)
    before = cache_signature(value)

    def unexpected(*args):
        raise AssertionError("definition hashing on a warmed cache query")

    monkeypatch.setattr(struct_module, "_field_type_signature", unexpected)
    monkeypatch.setattr(struct_module, "_member_dependencies", unexpected)
    for _ in range(10):
        assert cache_signature(value) == before


@pytest.mark.parametrize("wrapper", [lambda t: fx.Array[t, 4], lambda t: fx.Align[t, 16], lambda t: fx.Storage[t]])
def test_field_wrappers_supply_implicit_dependencies(wrapper):
    Item = _declare_record(1)

    @fx.struct
    class Outer:
        data: wrapper(Item)

        @staticmethod
        def make(x):
            return Item(x)

    assert Outer.make(1).read() == 2


@pytest.mark.parametrize("kind", ["method", "property", "staticmethod", "classmethod", "jit", "operator"])
def test_every_declared_member_affects_the_signature(kind):
    def first(self):
        return 1

    def second(self):
        return 9

    wrappers = {
        "method": lambda f: f,
        "property": property,
        "staticmethod": staticmethod,
        "classmethod": classmethod,
        "jit": flyc.jit,
        "operator": lambda f: f,
    }
    name = "__call__" if kind == "operator" else "unused"

    def make(function):
        return fx.struct(type("Record", (), {"__annotations__": {"x": fx.Int32}, name: wrappers[kind](function)}))

    assert make(first).__cache_signature__() != make(second).__cache_signature__()


@pytest.mark.parametrize("jit", [False, True])
def test_members_keep_native_functions_and_descriptors(jit):
    def read(self):
        return self.x + 1

    function = flyc.jit(read) if jit else read
    getter = property(read)
    static = staticmethod(read)
    factory = classmethod(read)
    Record = fx.struct(
        type(
            "Record",
            (),
            {
                "__annotations__": {"x": fx.Int32},
                "read": function,
                "value": getter,
                "static": static,
                "factory": factory,
            },
        )
    )
    assert vars(Record)["read"] is function
    assert vars(Record)["value"] is getter
    assert vars(Record)["static"] is static
    assert vars(Record)["factory"] is factory


@pytest.mark.parametrize("jit", [False, True])
def test_closure_constants_are_not_part_of_struct_type_key(jit):
    def make(constant):
        def read(self):
            return self.x + constant

        return fx.struct(
            type(
                "Record",
                (),
                {
                    "__annotations__": {"x": fx.Int32},
                    "read": flyc.jit(read) if jit else read,
                },
            )
        )

    first, second = make(2), make(9)
    assert first.__cache_signature__() == second.__cache_signature__()
    if not jit:
        assert first(1).read() == 3
        assert second(1).read() == 10


def test_global_constants_are_not_part_of_struct_type_key():
    source = "def read(self):\n return self.x * SCALE\n"
    filename = "<struct-global-constant>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)

    def make(scale):
        namespace = {"SCALE": scale}
        exec(compile(source, filename, "exec"), namespace)
        return fx.struct(type("Record", (), {"__annotations__": {"x": fx.Int32}, "read": namespace["read"]}))

    first, second = make(2), make(9)
    assert first.__cache_signature__() == second.__cache_signature__()
    assert first(3).read() == 6
    assert second(3).read() == 27


def test_constexpr_field_is_the_explicit_constant_cache_key():
    @fx.struct
    class Record:
        x: fx.Int32
        scale: fx.Constexpr[int]

        def read(self):
            return self.x * self.scale

    first, second = Record(3, 2), Record(3, 9)
    assert type(first).__cache_signature__() != type(second).__cache_signature__()
    assert first.read() == 6
    assert second.read() == 27


def test_direct_closure_dependencies_are_automatic():
    def make(item):
        @fx.struct
        class Record:
            x: fx.Int32

            def read(self):
                return item(self.x).read()

        return Record

    first, second = make(_declare_record(1)), make(_declare_record(9))
    assert first.__cache_signature__() != second.__cache_signature__()
    assert first(1).read() == 2
    assert second(1).read() == 10


@pytest.mark.parametrize("scope", ["global", "closure"])
@pytest.mark.parametrize("call", ["Item(self.x).read()", "Item.static(self.x)", "Item.create(self.x).read()"])
@pytest.mark.parametrize("jit", [False, True])
def test_direct_type_calls_track_dependency_changes(scope, call, jit):
    def make(offset):
        source = (
            "@fx.struct\n"
            "class Item:\n"
            "    x: fx.Int32\n"
            "    def read(self):\n"
            f"        return self.x + {offset}\n"
            "    @staticmethod\n"
            "    def static(x):\n"
            f"        return x + {offset}\n"
            "    @classmethod\n"
            "    def create(cls, x):\n"
            "        return cls(x)\n"
        )
        wrapper = "@fx.struct\nclass Wrapper:\n    x: fx.Int32\n"
        if jit:
            wrapper += "    @flyc.jit\n"
        wrapper += f"    def read(self):\n        return {call}\n"
        if scope == "closure":
            wrapper = "def make(Item):\n" + textwrap.indent(wrapper + "return Wrapper\n", "    ")
            wrapper += "Wrapper = make(Item)\n"
        source += wrapper
        filename = f"<direct-type-{scope}-{jit}-{call}-{offset}>"
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        namespace = {"fx": fx, "flyc": flyc, "__name__": "direct_type_calls"}
        exec(compile(source, filename, "exec"), namespace)
        return namespace["Wrapper"]

    first, equivalent, changed = make(1), make(1), make(9)
    assert first.__cache_signature__() == equivalent.__cache_signature__()
    assert first.__cache_signature__() != changed.__cache_signature__()

    @flyc.jit
    def launch(value):
        _ = value.read()

    launch(first(3))
    first_key, first_artifact = launch._last_compiled
    launch(changed(3))
    changed_key, changed_artifact = launch._last_compiled
    assert first_key != changed_key
    assert "arith.constant 1" in first_artifact.source_ir
    assert "arith.constant 9" in changed_artifact.source_ir


def test_dependency_signatures_preserve_name_bindings():
    def make(left, right, repeated):
        @fx.struct
        class Record:
            x: fx.Int32

            def read(self):
                return left(self.x).read() - right(self.x).read() + repeated(self.x).read()

        return Record

    first, second = _declare_record(1), _declare_record(9)
    records = (make(first, second, first), make(second, first, first), make(first, second, second))
    assert len({record.__cache_signature__() for record in records}) == 3
    assert [record(1).read() for record in records] == [-6, 10, 2]


@pytest.mark.parametrize("scope", ["global", "closure"])
def test_direct_types_in_nested_member_code_are_dependencies(scope):
    def make(item):
        source = "@fx.struct\nclass Record:\n x: fx.Int32\n def read(self):\n  return (lambda: Item(self.x).read())()\n"
        if scope == "closure":
            source = "def make(Item):\n" + textwrap.indent(source + "return Record\n", "    ")
            source += "Record = make(Item)\n"
        filename = f"<nested-type-{scope}>"
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        namespace = {"fx": fx, "Item": item, "__name__": "nested_type_calls"}
        exec(compile(source, filename, "exec"), namespace)
        return namespace["Record"]

    first, second = make(_declare_record(1)), make(_declare_record(9))
    assert first.__cache_signature__() != second.__cache_signature__()
    assert first(1).read() == 2
    assert second(1).read() == 10


@pytest.mark.parametrize("expression", ["holder.Item", "holder['Item']", "holder()"])
def test_indirect_type_references_are_not_scanned(expression):
    def make(item):
        holder = {
            "holder.Item": SimpleNamespace(Item=item),
            "holder['Item']": {"Item": item},
            "holder()": lambda: item,
        }[expression]
        source = f"class Record:\n x: fx.Int32\n def read(self):\n  return {expression}(self.x).read()\n"
        filename = f"<indirect-type-{expression}>"
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        namespace = {"fx": fx, "holder": holder, "__name__": "indirect_type_calls"}
        exec(compile(source, filename, "exec"), namespace)
        return fx.struct(namespace["Record"])

    assert make(_declare_record(1)).__cache_signature__() == make(_declare_record(9)).__cache_signature__()


def test_source_snapshot_requires_inspectable_members():
    namespace = {}
    exec("def read(self): return self.x", namespace)
    with pytest.raises(OSError):
        fx.struct(type("Record", (), {"__annotations__": {"x": fx.Int32}, "read": namespace["read"]}))


def test_field_wrapper_keys_include_type_size_and_alignment():
    item = fx.Struct["x" : fx.Int32]
    variants = (
        fx.Array[item, 2],
        fx.Array[item, 4],
        fx.Array[item, 2, 16],
        fx.Align[item, 8],
        fx.Align[item, 16],
        fx.Storage[item],
    )
    assert len({fx.Struct["data":dtype].__cache_signature__() for dtype in variants}) == len(variants)


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
# Part 3 — What can be a field
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
# Part 4 — Nesting
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
# Part 5 — Closure over the protocols
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

    @pytest.mark.parametrize("dtype, shape", [(fx.Float32, (4,)), (fx.Uint32, (2, 2))])
    def test_round_trip_preserves_field_metadata(self, dtype, shape):
        """A `Vector` field keeps its shape/dtype through the exemplar."""

        def body(a: fx.Int32):
            value = WithVector(scalar=a, vector=fx.Vector.filled(shape, 1, dtype))
            rebuilt = construct_from_ir_values(type(value), value, extract_to_ir_values(value))
            assert isinstance(rebuilt.vector, fx.Vector)
            assert rebuilt.vector.dtype is dtype
            assert rebuilt.vector.shape == shape

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
# Part 6 — Compile-time fields
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
# Part 7 — JIT and kernel boundaries
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
