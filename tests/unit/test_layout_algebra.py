#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""
Layout algebra tests using the Fly dialect API with static types.

Each test corresponds to a specific cell in the reference layout-algebra notebook.
"""

import sys

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
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
    """4:1 o 2:1 => size = 2"""

    @flyc.jit
    def build():
        _assert_size(fx.composition(fx.make_layout((4,), (1,)), fx.make_layout((2,), (1,))), 2)

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

    Fly dialect does NOT raise on non-injective layouts (unlike the legacy dialect).
    Verify it runs without crash and returns a result.
    """

    @flyc.jit
    def build():
        _assert_size(fx.complement(fx.make_layout((3, 2), (1, 2)), codomain_size=fx.make_int_tuple(12)), 0)

    # Fly returns a result (doesn't error); just verify no crash.
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


def test_logical_divide_1d(frontend_only_jit):
    """Cell 15: 16:1 / 4:1 => size = 16"""

    @flyc.jit
    def build():
        _assert_size(fx.logical_divide(fx.make_layout((16,), (1,)), fx.make_layout((4,), (1,))), 16)

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


def test_logical_product_1d(frontend_only_jit):
    """Cell 25: (8):(1) * (4):(1) => size = 32"""

    @flyc.jit
    def build():
        _assert_size(fx.logical_product(fx.make_layout((8,), (1,)), fx.make_layout((4,), (1,))), 32)

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
