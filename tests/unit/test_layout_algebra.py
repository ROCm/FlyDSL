#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""
Layout algebra tests using the Fly dialect API with static types.

Each test corresponds to a specific cell in the reference layout-algebra notebook.
"""

import sys
from fractions import Fraction

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly
from flydsl._mlir.passmanager import PassManager
from flydsl.compiler import jit_function

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


FLY_PIPELINE = (
    "builtin.module(fly-canonicalize,fly-layout-lowering,fly-canonicalize,convert-fly-to-rocdl,canonicalize,cse)"
)


@pytest.fixture
def frontend_only_jit(monkeypatch):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "rocm")
    monkeypatch.setenv("ARCH", "gfx942")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setattr(jit_function, "_flydsl_key", lambda: "test-flydsl-key")

    def compile_layout_pipeline(cls, module, **_kwargs):
        pm = PassManager.parse(FLY_PIPELINE)
        pm.run(module.operation)
        assert module.operation.verify()
        return module

    monkeypatch.setattr(jit_function.MlirCompiler, "compile", classmethod(compile_layout_pipeline))


def _assert_size(layout, expected):
    assert fx.get_scalar(fx.size(layout)) == expected


def _source_ir(jit_fn):
    last_compiled = jit_fn._last_compiled
    assert last_compiled is not None
    return last_compiled[1].source_ir


# ==============================================================================
# 1. Basic layout construction & size (Cells 1-3)
# ==============================================================================


def test_basic_size(frontend_only_jit):
    """make_layout((3,9):(1,3)) => size = 27"""

    @flyc.jit
    def build():
        _assert_size(fx.make_layout((3, 9), (1, 3)), 27)

    build()


def test_nested_layout_size(frontend_only_jit):
    """(9,(4,8)):(59,(13,1)) => size = 288"""

    @flyc.jit
    def build():
        _assert_size(fx.make_layout((9, (4, 8)), (59, (13, 1))), 288)

    build()


def test_shape_stride_type_nested_spec_printing(frontend_only_jit):
    """Nested shape/stride types print in tuple form."""

    @flyc.jit
    def build():
        fx.make_int_tuple((9, (4, 8)))
        fx.make_int_tuple((59, (13, 1)))
        fx.make_layout((9, (4, 8)), (59, (13, 1)))

    def check(ir):
        compact_ir = ir.replace(" ", "")
        assert "!fly.int_tuple<(9,(4,8))>" in compact_ir
        assert "!fly.int_tuple<(59,(13,1))>" in compact_ir

    build()
    check(_source_ir(build))


# ==============================================================================
# 2. Coalesce (Cells 4, 5, 7)
# ==============================================================================


def test_coalesce_basic(frontend_only_jit):
    """Cell 4: coalesce((3,(1,9)):(1,(9,3))) => size preserved = 27"""

    @flyc.jit
    def build():
        _assert_size(fly.coalesce(fx.make_layout((3, (1, 9)), (1, (9, 3)))), 27)

    build()


def test_coalesce_dynamic_stride(frontend_only_jit):
    """Cell 4 dynamic: verify mixed static/dynamic survives lowering."""

    @flyc.jit
    def build(runtime_stride: fx.Int32):
        layout = fx.make_layout((2, (1, 6)), (1, (runtime_stride, 2)))
        _assert_size(fly.coalesce(layout), 12)

    build(3)


# ==============================================================================
# 3. Composition (Cells 9, 11, 13)
# ==============================================================================


def test_composition_basic(frontend_only_jit):
    """Cell 9: (6,9):(19,69) o (6,3):(3,1) => size = 18"""

    @flyc.jit
    def build():
        _assert_size(fx.composition(fx.make_layout((6, 9), (19, 69)), fx.make_layout((6, 3), (3, 1))), 18)

    build()


def test_composition_static_vs_dynamic(frontend_only_jit):
    """Cell 11: Static composition + dynamic composition both lower successfully.

    Static: (5,15):(19,51) o (3,5):(1,5) => size = 15.
    Dynamic: function-arg layouts lower without error.
    """

    @flyc.jit
    def build_static():
        _assert_size(fx.composition(fx.make_layout((5, 15), (19, 51)), fx.make_layout((3, 5), (1, 5))), 15)

    @flyc.jit
    def build_dynamic(
        a_shape0: fx.Int32,
        a_shape1: fx.Int32,
        a_stride0: fx.Int32,
        a_stride1: fx.Int32,
        b_shape0: fx.Int32,
        b_shape1: fx.Int32,
        b_stride0: fx.Int32,
        b_stride1: fx.Int32,
    ):
        a = fx.make_layout((a_shape0, a_shape1), (a_stride0, a_stride1))
        b = fx.make_layout((b_shape0, b_shape1), (b_stride0, b_stride1))
        fx.size(fx.composition(a, b))

    build_static()
    build_dynamic(5, 15, 19, 51, 3, 5, 1, 5)


def test_composition_bymode(frontend_only_jit):
    """Cell 13: By-mode composition using make_tile."""

    @flyc.jit
    def build():
        layout = fx.make_layout((9, (4, 8)), (59, (13, 1)))
        tile_m0 = fx.make_layout((3,), (3,))
        tile_m1 = fx.make_layout((2, 4), (1, 8))
        tiler = fx.make_tile(tile_m0, tile_m1)
        _assert_size(fx.logical_divide(layout, tiler), 288)

    build()


def test_composition_with_tuple(frontend_only_jit):
    """Composition preserves a singleton tuple rather than returning a scalar."""

    @flyc.jit
    def build():
        result = fx.composition(fx.make_layout((4,), (1,)), fx.make_layout((2,), (1,)))
        _assert_size(result, 2)
        assert str(result.type).replace(" ", "") == "!fly.layout<(2):(1)>"

    build()


# ==============================================================================
# 4. Complement
# ==============================================================================


def test_complement_simple_rank_1(frontend_only_jit):
    """complement(3:1, 12) => size = 4"""

    @flyc.jit
    def build():
        _assert_size(fx.complement(fx.make_layout((3,), (1,)), codomain_size=fx.make_int_tuple(12)), 4)

    build()


def test_complement_simple_rank_2(frontend_only_jit):
    """complement((3,2):(2,1), 12) => size = 2"""

    @flyc.jit
    def build():
        _assert_size(fx.complement(fx.make_layout((3, 2), (2, 1)), codomain_size=fx.make_int_tuple(12)), 2)

    build()


def test_complement_rank_2_error(frontend_only_jit):
    """Rank-2 non-injective complement: (3,2):(1,2).

    The modes overlap, so complement reports the error at op construction, as
    CuTe rejects it with a static assertion.
    """

    @flyc.jit
    def build():
        fx.complement(fx.make_layout((3, 2), (1, 2)), codomain_size=fx.make_int_tuple(12))

    with pytest.raises(ir.MLIRError, match="non-injective-layout"):
        build()


def test_complement_rank_1_error(frontend_only_jit):
    """Rank-1 non-injective complement: 3:0.

    Fly dialect does NOT raise on non-injective layouts.
    """

    @flyc.jit
    def build():
        _assert_size(fx.complement(fx.make_layout((3,), (0,)), codomain_size=fx.make_int_tuple(12)), 12)

    build()


@pytest.mark.skip(reason="fly.complement with dynamic rank-2 stride crashes in C++ op construction")
def test_complement_rank_2_dynamic_stride_error(frontend_only_jit):
    """Rank-2 complement with dynamic stride: lowering should still succeed."""

    @flyc.jit
    def build(runtime_stride: fx.Int32):
        tiler = fx.make_layout((3, 2), (runtime_stride, 1))
        fx.size(fx.complement(tiler, 12))

    build(2)


def test_complement_with_divide(frontend_only_jit):
    """logical_divide(12:1, 3:1) uses complement internally => size = 12"""

    @flyc.jit
    def build():
        _assert_size(fx.logical_divide(fx.make_layout((12,), (1,)), fx.make_layout((3,), (1,))), 12)

    build()


# ==============================================================================
# 5. Divide Operations (Cells 15, 17, 19, 21, 23)
# ==============================================================================


@pytest.mark.parametrize(
    "operation, expected",
    [
        ("logical_divide", "((4),4):((1),4)"),
        ("zipped_divide", "((4),4):((1),4)"),
        ("tiled_divide", "((4),4):((1),4)"),
        ("flat_divide", "(4,4):(1,4)"),
    ],
)
def test_divide_1d(frontend_only_jit, operation, expected):
    """A tuple divisor keeps its value mode profile; flat_divide unpacks it."""
    divide = getattr(fx, operation)

    @flyc.jit
    def build():
        result = divide(fx.make_layout((16,), (1,)), fx.make_layout((4,), (1,)))
        _assert_size(result, 16)
        assert str(result.type).replace(" ", "") == f"!fly.layout<{expected}>"

    build()


def test_logical_divide_2d(frontend_only_jit):
    """Cell 17: (4,8):(1,4) / (2,4):(1,2) => size = 32"""

    @flyc.jit
    def build():
        _assert_size(fx.logical_divide(fx.make_layout((4, 8), (1, 4)), fx.make_layout((2, 4), (1, 2))), 32)

    build()


def test_zipped_divide(frontend_only_jit):
    """Cell 19: zipped_divide preserves size = 32"""

    @flyc.jit
    def build():
        _assert_size(fx.zipped_divide(fx.make_layout((4, 8), (1, 4)), fx.make_layout((2, 4), (1, 2))), 32)

    build()


def test_tiled_divide(frontend_only_jit):
    """Cell 21: tiled_divide preserves size = 32"""

    @flyc.jit
    def build():
        _assert_size(fx.tiled_divide(fx.make_layout((4, 8), (1, 4)), fx.make_layout((2, 4), (1, 2))), 32)

    build()


def test_flat_divide(frontend_only_jit):
    """Cell 23: flat_divide preserves size = 32"""

    @flyc.jit
    def build():
        _assert_size(fx.flat_divide(fx.make_layout((4, 8), (1, 4)), fx.make_layout((2, 4), (1, 2))), 32)

    build()


# ==============================================================================
# 6. Product Operations (Cells 25, 27, 29)
# ==============================================================================


@pytest.mark.parametrize(
    "operation, expected",
    [
        ("logical_product", "((8),(4)):((1),(8))"),
        ("zipped_product", "((8),(4)):((1),(8))"),
        ("tiled_product", "((8),4):((1),8)"),
        ("flat_product", "(8,4):(1,8)"),
        ("blocked_product", "((8,4)):((1,8))"),
        ("raked_product", "((4,8)):((8,1))"),
    ],
)
def test_product_1d(frontend_only_jit, operation, expected):
    """Tuple operands keep their profiles; tiled_product unpacks the tile mode and flat_product both."""
    product = getattr(fx, operation)

    @flyc.jit
    def build():
        result = product(fx.make_layout((8,), (1,)), fx.make_layout((4,), (1,)))
        _assert_size(result, 32)
        assert str(result.type).replace(" ", "") == f"!fly.layout<{expected}>"

    build()


def test_tile_to_shape_1d(frontend_only_jit):
    """tile_to_shape keeps the outer mode produced by blocked_product."""

    @flyc.jit
    def build():
        result = fx.tile_to_shape(fx.make_layout((4,), (1,)), (16,), (0,))
        _assert_size(result, 16)
        assert str(result.type).replace(" ", "") == "!fly.layout<((4,4)):((1,4))>"

    build()


def test_blocked_raked_product(frontend_only_jit):
    """Cell 27: (3,6):(6,1) * (4,5):(1,4) => size = 360"""

    @flyc.jit
    def build():
        _assert_size(fx.blocked_product(fx.make_layout((3, 6), (6, 1)), fx.make_layout((4, 5), (1, 4))), 360)

    build()


def test_zipped_tiled_flat_product(frontend_only_jit):
    """Cell 29: flat_product (3,6):(6,1) * (4,5):(1,4) => size = 360"""

    @flyc.jit
    def build():
        _assert_size(fx.flat_product(fx.make_layout((3, 6), (6, 1)), fx.make_layout((4, 5), (1, 4))), 360)

    build()


# ==============================================================================
# Basis
# ==============================================================================


def test_e_construction_and_validation():
    b = fx.E(0)
    assert isinstance(b, fx.Basis)
    assert b.value == 1 and b.modes == [0]
    assert fx.E() == 1
    # E accepts the three mode syntaxes: a single int, variadic ints, or a sequence.
    assert fx.E(0).modes == [0]
    assert fx.E(0, 1).modes == [0, 1]
    assert fx.E([0, 1]).modes == [0, 1]
    assert fx.E((0, 1)).modes == [0, 1]
    assert fx.E(0) == fx.Basis(1, [0])
    # a scaled coefficient is built via Basis directly
    assert fx.Basis(2, [0]).value == 2


def test_basis_mul():
    """The Python-frontend Basis scales its coefficient, keeping the modes."""
    assert fx.E(0) * 2 == fx.Basis(2, [0])
    assert 3 * fx.E(0) == fx.Basis(3, [0])  # __rmul__ is commutative for scalars
    assert fx.Basis(2, [0]) * 4 == fx.Basis(8, [0])
    assert (fx.Basis(2, [0, 1]) * fx.Int32(5)) == fx.Basis(10, [0, 1])  # modes preserved
    assert (fx.Basis(2, [0]) * 3).modes == [0]  # mul never touches the modes


def test_basis_hash():
    """Basis is hashable; equal Basis hash equal (usable in sets / dict keys)."""
    assert hash(fx.E(0)) == hash(fx.Basis(1, [0]))  # equal objects -> equal hash
    assert hash(fx.Basis(2, [0])) == hash(fx.Basis(2, [0]))
    assert len({fx.Basis(1, [0]), fx.Basis(1, [0])}) == 1  # dedup in a set
    d = {fx.Basis(2, [0]): "x"}
    assert d[fx.Basis(2, [0])] == "x"  # dict lookup by value + modes


def test_basis_repr_matches_mlir_asm():
    """repr(Basis) matches the MLIR asm form of a basis leaf."""
    assert repr(fx.Basis(2, [0])) == "2E0"
    assert repr(fx.E(0)) == "1E0"
    assert repr(fx.Basis(1, [0, 1])) == "1E0E1"


def test_basis_int_tuple_type_printing(frontend_only_jit):
    @flyc.jit
    def build():
        fx.make_stride(fx.E(0), fx.E(1))
        fx.make_int_tuple(fx.Basis(2, [0]))
        fx.make_identity_layout((4, 8))

    def check(ir):
        assert "2E0" in ir  # scaled coefficient
        # the explicit basis stride and the identity layout emit the same stride
        assert ir.count("(1E0,1E1)") >= 2

    build()
    check(_source_ir(build))


def test_get_scalar_returns_basis(frontend_only_jit):
    """get_scalar wraps a static basis leaf into a Python Basis."""

    @flyc.jit
    def build():
        leaf = fx.make_int_tuple(fx.Basis(2, [0]))
        assert leaf.type.is_leaf_basis
        b = fx.get_scalar(leaf)
        assert isinstance(b, fx.Basis)
        assert b.value == 2 and b.modes == [0]

        leaf = fx.make_int_tuple(fx.Basis(fx.Int32(4), [0]))
        b = fx.get_scalar(leaf)
        assert isinstance(b, fx.Basis)
        assert b.value == 4 and b.modes == [0]

    build()


def test_get_leaves_with_basis(frontend_only_jit):
    """get_leaves flattens a (basis, basis) tuple into Basis leaves."""

    @flyc.jit
    def build():
        stride = fx.make_stride(fx.E(0), fx.make_stride(fx.E(1), fx.E(2)))
        assert fx.get_leaves(stride) == (fx.Basis(1, [0]), fx.Basis(1, [1]), fx.Basis(1, [2]))
        assert stride.to_py_value() == (fx.Basis(1, [0]), (fx.Basis(1, [1]), fx.Basis(1, [2])))

    build()


def test_get_leaves_none_leaf(frontend_only_jit):
    """A ``*`` leaf is read off the type as None; only runtime leaves are extracted."""

    @flyc.jit
    def build(n: fx.Int32):
        coord = fx.make_coord(None, 2, n)
        leaves = fx.get_leaves(coord)
        assert len(leaves) == 3 and leaves[0] is None and leaves[1] == 2
        assert len(fx.get_leaves(coord, dynamic_only=True)) == 1
        assert coord.to_py_value()[:2] == (None, 2)

    build(4)


def test_basis_dynamic_coefficient(frontend_only_jit):
    """A basis coefficient may be a runtime i32 value (a dynamic, non-static leaf)."""

    @flyc.jit
    def build(coeff: fx.Int32):
        leaf = fx.make_int_tuple(fx.Basis(coeff, [0]))
        assert leaf.type.is_leaf_basis
        assert not leaf.type.is_static
        assert leaf.type.get_leaf_as_basis.modes == [0]
        # get_scalar and get_leaves reconstruct a dynamic basis leaf the
        # same way: a Basis whose coefficient is the runtime value and whose
        # modes come from the type (get_leaves = get + get_scalar).
        sc = fx.get_scalar(leaf)
        assert isinstance(sc, fx.Basis) and sc.modes == [0]
        stride = fx.make_stride(fx.Basis(coeff, [0]), fx.E(1))
        leaves = fx.get_leaves(stride)
        assert isinstance(leaves[0], fx.Basis) and leaves[0].modes == [0]
        assert leaves[1] == fx.Basis(1, [1])
        # dynamic_only=True also reconstructs the dynamic basis leaf as a Basis
        # (the static E(1) leaf is dropped).
        dyn = fx.get_leaves(stride, dynamic_only=True)
        assert len(dyn) == 1 and isinstance(dyn[0], fx.Basis) and dyn[0].modes == [0]

    build(2)


def test_make_tile_dynamic_layout(frontend_only_jit):
    """A dynamic layout mode makes fly.make_tile take its dynamic leaves; the tiler lowers."""

    @flyc.jit
    def build(n: fx.Int32):
        layout = fx.make_layout((8, 16), (1, 8))
        fx.logical_divide(layout, (fx.make_layout(n, 1), None))

    build(4)
    ir = _source_ir(build)
    assert "fly.make_tile" in ir
    assert "!fly.tile<[?:1|*]>" in ir


def test_tile_unpack_dynamic_layout(frontend_only_jit):
    """Tile.unpack rebuilds a dynamic layout mode from the tile's leaves; repacked, it lowers."""

    @flyc.jit
    def build(n: fx.Int32):
        tile = fx.make_tile(fx.make_layout(n, 1), None)
        mode, rest = tile.unpack()
        assert str(mode.type) == "!fly.layout<?:1>" and rest is None
        layout = fx.make_layout((8, 16), (1, 8))
        fx.logical_divide(layout, fx.make_tile(mode, rest))

    build(4)
    assert "fly.static : !fly.layout<?:1>" not in _source_ir(build)


# ==============================================================================
# Fraction (ratio) leaves
# ==============================================================================


def test_ratio_type_bindings():
    """IntTupleType takes a Fraction leaf and reads it back as one; TileType rejects it."""
    with ir.Context() as ctx:
        ctx.load_all_available_dialects()
        half = fly.IntTupleType.get(Fraction(1, 2))
        assert str(half) == "!fly.int_tuple<1/2>"
        assert half.is_ratio and half.static_value == Fraction(1, 2)
        with pytest.raises(ValueError, match="static integer leaf"):
            half.get_static_leaf_int
        # Python reduces 4/2 to 2, an integer leaf rather than a ratio.
        two = fly.IntTupleType.get(Fraction(4, 2))
        assert str(two) == "!fly.int_tuple<2>" and not two.is_ratio
        assert two.get_static_leaf_int == 2 and type(two.static_value) is int
        # The numerator and the denominator are each an int32.
        bound = fly.IntTupleType.get(Fraction(-(2**31), 2**31 - 1))
        assert str(bound) == "!fly.int_tuple<-2147483648/2147483647>"
        for out_of_range in (Fraction(2**31, 3), Fraction(1, 2**31)):
            with pytest.raises(ValueError, match="ratio is outside the representable range"):
                fly.IntTupleType.get(out_of_range)
        # A tile mode is an integer extent; a ratio has no tile meaning.
        with pytest.raises(ValueError, match="expected int, None, tuple, or LayoutType"):
            fly.TileType.get([Fraction(1, 2), 4])


def test_ratio_python_helpers():
    """rank/depth and the Python Basis treat a Fraction as a scalar."""
    assert fx.rank(Fraction(1, 2)) == 1 and fx.depth(Fraction(1, 2)) == 0
    assert fx.depth((Fraction(1, 2), (3, Fraction(1, 3)))) == 2
    assert fx.E(0) * Fraction(1, 2) == fx.Basis(Fraction(1, 2), [0])
    assert Fraction(3, 2) * fx.E(1) == fx.Basis(Fraction(3, 2), [1])
    assert fx.E(0) * Fraction(1, 2) * 2 == fx.E(0)
    assert repr(fx.Basis(Fraction(-3, 4), [1])) == "-3/4E1"  # the MLIR asm form


def test_ratio_int_tuple_extractors(frontend_only_jit):
    """Every int-tuple constructor takes a Fraction; every extractor returns one."""

    @flyc.jit
    def build(n: fx.Int32):
        t = fx.make_int_tuple((Fraction(1, 2), (3, Fraction(-3, 4))))
        assert str(t.type) == "!fly.int_tuple<(1/2,(3,-3/4))>"
        assert t.to_py_value() == (Fraction(1, 2), (3, Fraction(-3, 4)))
        assert fx.get_leaves(t) == (Fraction(1, 2), 3, Fraction(-3, 4))
        assert fx.get_scalar(t[0]) == Fraction(1, 2)
        # An integer leaf beside a ratio is still read back as an int.
        assert type(fx.get_scalar(t[1][0])) is int
        assert fx.get_scalar(fx.size(t)) == Fraction(-9, 8)
        for make in (fx.make_shape, fx.make_stride, fx.make_coord):
            assert str(make(Fraction(1, 2), 4).type) == "!fly.int_tuple<(1/2,4)>"

        mixed = fx.make_int_tuple((n, Fraction(1, 2)))
        assert str(mixed.type) == "!fly.int_tuple<(?,1/2)>"
        assert fx.get_leaves(mixed)[1] == Fraction(1, 2)
        assert len(fx.get_leaves(mixed, dynamic_only=True)) == 1

    build(4)


def test_ratio_stride_layout(frontend_only_jit):
    """(4,8):(1/2,2) through construction, mapping, and the layout algebra."""

    @flyc.jit
    def build():
        layout = fx.make_layout((4, 8), (Fraction(1, 2), 2))
        assert str(layout.type) == "!fly.layout<(4,8):(1/2,2)>"
        assert fx.get_stride(layout).to_py_value() == (Fraction(1, 2), 2)
        _assert_size(layout, 32)
        assert fx.get_scalar(fx.cosize(layout)) == Fraction(33, 2)  # 3 * 1/2 + 7 * 2 + 1
        # An even coordinate in mode 0 maps to an integer, an odd one between two.
        assert type(fx.get_scalar(fx.crd2idx((2, 1), layout))) is int
        assert fx.get_scalar(fx.crd2idx((1, 0), layout)) == Fraction(1, 2)
        assert fx.get_scalar(fx.crd2idx(5, layout)) == Fraction(5, 2)
        assert str(fx.coalesce(layout).type) == "!fly.layout<32:1/2>"
        assert str(fx.composition(layout, fx.make_layout(4, 1)).type) == "!fly.layout<4:1/2>"
        assert str(fx.logical_divide(layout, (2, None)).type) == "!fly.layout<((2,2),8):((1/2,1),2)>"
        # Half a 16-bit element is one 8-bit element.
        assert str(fx.recast_layout(layout, 16, 8).type) == "!fly.layout<(4,8):(1,4)>"

    build()


def test_ratio_int_tuple_ops(frontend_only_jit):
    """Static arithmetic is exact and folds a whole result back to an int."""

    @flyc.jit
    def build():
        half = Fraction(1, 2)
        assert fx.get_scalar(fx.int_tuple_add(half, Fraction(1, 3))) == Fraction(5, 6)
        assert type(fx.get_scalar(fx.int_tuple_sub(half, half))) is int
        assert type(fx.get_scalar(fx.int_tuple_mul(half, 4))) is int
        assert str(fx.int_tuple_mul((half, 3), (6, Fraction(1, 3))).type) == "!fly.int_tuple<(3,1)>"
        assert str(fx.int_tuple_product_each(((half, 4), 3)).type) == "!fly.int_tuple<(2,3)>"

        t = fx.make_int_tuple((half, 4, Fraction(3, 2)))
        assert str(fx.slice(t, (None, 2, None)).type) == "!fly.int_tuple<(1/2,3/2)>"
        assert str(fx.select(t, [2, 0]).type) == "!fly.int_tuple<(3/2,1/2)>"
        assert str(fx.append(t, Fraction(5, 3)).type) == "!fly.int_tuple<(1/2,4,3/2,5/3)>"

    build()


def test_ratio_dynamic_coordinate(frontend_only_jit):
    """A ratio stride takes a dynamic coordinate known to be a multiple of its denominator."""

    @flyc.jit
    def build(n: fx.Int32):
        even = fx.int_tuple_mul(n, 2)
        assert str(even.type) == "!fly.int_tuple<?{div=2}>"
        assert str(fx.crd2idx(even, fx.make_layout(8, Fraction(1, 2))).type) == "!fly.int_tuple<?>"
        assert str(fx.crd2idx(even, fx.make_layout(8, Fraction(3, 2))).type) == "!fly.int_tuple<?{div=3}>"

    build(4)


@pytest.mark.parametrize(
    "operation, error",
    [
        (lambda n: fx.int_tuple_div(Fraction(1, 2), 2), "unsupported-ratio-operation"),
        (lambda n: fx.complement(fx.make_layout(4, Fraction(1, 2)), 64), "unsupported-ratio-operation"),
        (lambda n: fx.int_tuple_add(Fraction(1, 2), n), "dynamic-ratio-unsupported"),
        # n is not known to be even, so n * 1/2 may not be an integer.
        (lambda n: fx.crd2idx(n, fx.make_layout(8, Fraction(1, 2))), "dynamic-ratio-unsupported"),
    ],
    ids=["div", "complement", "dynamic_add", "odd_dynamic_coord"],
)
def test_ratio_unsupported_operation(frontend_only_jit, operation, error):
    """An operation without a ratio rule reports it at op construction."""

    @flyc.jit
    def build(n: fx.Int32):
        operation(n)

    with pytest.raises(ir.MLIRError, match=error):
        build(4)


def test_ratio_basis_coefficient(frontend_only_jit):
    """A basis leaf may scale by a Fraction; extractors return it in the Basis."""

    @flyc.jit
    def build():
        stride = fx.make_stride(fx.E(0) * Fraction(1, 2), Fraction(3, 2) * fx.E(1))
        assert str(stride.type) == "!fly.int_tuple<(1/2E0,3/2E1)>"
        assert stride.to_py_value() == (fx.Basis(Fraction(1, 2), [0]), fx.Basis(Fraction(3, 2), [1]))
        assert isinstance(fx.get_scalar(stride[0]).value, Fraction)
        # A whole coefficient is read back as an int.
        assert type(fx.get_scalar(fx.make_int_tuple(fx.Basis(Fraction(4, 2), [1]))).value) is int

        layout = fx.make_layout((4, 8), (fx.E(0) * Fraction(1, 2), fx.E(1)))
        assert fx.crd2idx((2, 3), layout).to_py_value() == (1, 3)
        assert fx.get_scalar(fx.crd2idx((1, 0), layout)) == fx.Basis(Fraction(1, 2), [0])

    build()


# ==============================================================================
# IntTupleLike ops on Layout (regression for issue #713)
# ==============================================================================


def test_int_tuple_like_ops_on_layout(frontend_only_jit):
    """`get_`/`take`/`select`/`group`/`coalesce` and `layout[i]` accept a Layout.

    Regression for issue #713: permissive int-tuple coercion must pass a Layout
    value through unchanged instead of rebuilding it as an IntTuple.
    """

    @flyc.jit
    def build():
        layout = fx.make_layout((128, 64), (1, 128))
        assert str(fx.get_(layout, 0).type) == "!fly.layout<128:1>"
        assert str(layout[0].type) == "!fly.layout<128:1>"
        assert str(fx.select(layout, [1, 0]).type) == "!fly.layout<(64,128):(128,1)>"
        assert str(fx.take(layout, 0, 1).type) == "!fly.layout<128:1>"
        assert str(fx.group(layout, 0, 2).type) == "!fly.layout<((128,64)):((1,128))>"
        assert str(fx.coalesce(layout).type) == "!fly.layout<8192:1>"

    build()


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
