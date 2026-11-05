# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""ThrVal pins for the gfx1250 block-scaled WMMA A/B layouts (issue #1323).

No GPU. V_WMMA_SCALE applies E8M0 scale byte j to the registers the ISA assigns
to K block j, so the A/B ThrVal layouts must match the CDNA5 ISA (7.12.6.1)
lane/slot -> K mapping exactly:

  FP4/FP6: lanes 0-15 hold K 0-31, 64-95; lanes 16-31 hold K 32-63, 96-127
  FP8:     8-wide K blocks interleaved between the two lane groups

Reference (M, K) space is column-major with stride (1, 16).
"""

from __future__ import annotations

import numpy as np
import pytest

import flydsl  # noqa: F401  preload comgr
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir._mlir_libs._mlirDialectsFlyROCDL import MmaOpGFX1250_WMMAType
from flydsl._mlir.dialects import fly_rocdl  # noqa: F401  register dialect
from flydsl._mlir.dialects.fly import MmaAtomType
from flydsl.compiler import jit_function
from flydsl.expr.rocdl import cdna5

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]

_FP4_FP6_AB = "!fly.layout<((16,2),(32,2)):((1,512),(16,1024))>"
_FP8_AB = "!fly.layout<((16,2),(8,8)):((1,128),(16,256))>"

_FP4_FP6 = [fx.Float4E2M1FN, fx.Float6E2M3FN, fx.Float6E3M2FN]
_FP8 = [fx.Float8E4M3FN, fx.Float8E5M2]


def _isa_k(lane, v, elem):
    g = lane // 16
    if elem in _FP4_FP6:
        return 32 * g + v % 32 + 64 * (v // 32)
    return 8 * g + v % 8 + 16 * (v // 8)


def _scale_atom_type(elem_a, elem_b=None, m=16):
    elem_b = elem_a if elem_b is None else elem_b
    return MmaAtomType.get(cdna5.WMMAScale(m, 16, 128, elem_a.ir_type, elem_b.ir_type, ir.F32Type.get()))


@pytest.fixture
def gfx1250_frontend(monkeypatch):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "rocm")
    monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx1250")
    monkeypatch.setenv("ARCH", "gfx1250")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setattr(jit_function, "_flydsl_key", lambda: "test-flydsl-key-g1250")

    def compile_only(cls, module, **_kwargs):
        assert module.operation.verify()
        return module

    monkeypatch.setattr(jit_function.MlirCompiler, "compile", classmethod(compile_only))


def _ab_positions(elem_a, elem_b=None):
    """Evaluate the A/B ThrVal layouts at every (lane, slot): returns two 32x64 int arrays."""
    elem_b = elem_a if elem_b is None else elem_b
    pos_a, pos_b = np.zeros((32, 64), np.int64), np.zeros((32, 64), np.int64)

    @flyc.jit
    def pin():
        atom = fx.make_mma_atom(cdna5.WMMAScale(16, 16, 128, elem_a, elem_b))
        layout_a = atom.layout_A_tv
        layout_b = atom.layout_B_tv
        # range() becomes scf.for; unroll so crd2idx stays a static leaf.
        for lane in fx.range_constexpr(32):
            for v in fx.range_constexpr(64):
                pos_a[lane, v] = layout_a((lane % 16, lane // 16), v).get_static_leaf_int
                pos_b[lane, v] = layout_b((lane % 16, lane // 16), v).get_static_leaf_int

    pin()
    return pos_a, pos_b


@pytest.mark.parametrize("elem", _FP4_FP6 + _FP8, ids=lambda e: e.__name__)
def test_wmma_scale_16x16x128_printed(elem):
    with ir.Context(), ir.Location.unknown():
        atom = _scale_atom_type(elem)
        want = _FP4_FP6_AB if elem in _FP4_FP6 else _FP8_AB
        assert str(atom.tv_layout_a) == want
        assert str(atom.tv_layout_b) == want
        assert str(atom.tv_layout_c) == "!fly.layout<((16,2),8):((16,8),1)>"


def test_wmma_scale_32x16x128_fp4_printed():
    with ir.Context(), ir.Location.unknown():
        atom = _scale_atom_type(fx.Float4E2M1FN, m=32)
        # A keeps the provisional 32-row layout; B is the 16-row FP4 layout.
        assert str(atom.tv_layout_a) == "!fly.layout<(32,128):(1,32)>"
        assert str(atom.tv_layout_b) == _FP4_FP6_AB
        assert str(atom.tv_layout_c) == "!fly.layout<((16,2),16):((16,16),1)>"


@pytest.mark.parametrize("elem_a,elem_b", [(fx.Float4E2M1FN, fx.Float8E4M3FN), (fx.Float8E4M3FN, fx.Float4E2M1FN)])
def test_wmma_scale_mixed_operands_pick_layout_per_operand(elem_a, elem_b):
    with ir.Context(), ir.Location.unknown():
        atom = _scale_atom_type(elem_a, elem_b)
        assert str(atom.tv_layout_a) == (_FP4_FP6_AB if elem_a in _FP4_FP6 else _FP8_AB)
        assert str(atom.tv_layout_b) == (_FP4_FP6_AB if elem_b in _FP4_FP6 else _FP8_AB)


@pytest.mark.parametrize(
    "k,elem,acc,want",
    [
        (32, "i4", "i32", "!fly.layout<((16,2),(8,2)):((1,128),(16,256))>"),
        (64, "f8E4M3FN", "f32", "!fly.layout<((16,2),(8,4)):((1,128),(16,256))>"),
        (128, "f8E4M3FN", "f32", _FP8_AB),
    ],
    ids=["iu4_k32", "fp8_k64", "fp8_k128"],
)
def test_unscaled_wmma_shared_layout_unchanged(k, elem, acc, want):
    with ir.Context(), ir.Location.unknown():
        ty = ir.Type.parse(elem)
        atom = MmaAtomType.get(MmaOpGFX1250_WMMAType.get(16, 16, k, ty, ty, ir.Type.parse(acc)))
        assert str(atom.tv_layout_a) == want
        assert str(atom.tv_layout_b) == want


@pytest.mark.parametrize("elem", _FP4_FP6 + _FP8, ids=lambda e: e.__name__)
def test_wmma_scale_every_lane_slot_matches_isa(gfx1250_frontend, elem):
    pos_a, pos_b = _ab_positions(elem)
    for lane in range(32):
        for v in range(64):
            assert pos_a[lane, v] % 16 == lane % 16, (lane, v)
            assert pos_a[lane, v] // 16 == _isa_k(lane, v, elem), (lane, v)
    np.testing.assert_array_equal(pos_b, pos_a)
    # Bijective over the 16x128 tile.
    assert sorted(pos_a.ravel().tolist()) == list(range(16 * 128))


def _emulate_wmma_scale(regs_a, regs_b, scale_a, scale_b, elem):
    """CPU model of V_WMMA_SCALE: the hardware reads lane/slot (l, v) as row l%16,
    K = ISA k(l, v), and scales it by that row's E8M0 byte for K block k // 32."""
    a = np.zeros((16, 128))
    b = np.zeros((16, 128))
    for lane in range(32):
        for v in range(64):
            k = _isa_k(lane, v, elem)
            a[lane % 16, k] = regs_a[lane, v] * 2.0 ** scale_a[lane % 16, k // 32]
            b[lane % 16, k] = regs_b[lane, v] * 2.0 ** scale_b[lane % 16, k // 32]
    return a @ b.T


@pytest.mark.parametrize("elem", _FP4_FP6, ids=lambda e: e.__name__)
@pytest.mark.parametrize("varying", [False, True], ids=["uniform_scales", "varying_scales"])
def test_wmma_scale_model_matches_reference(gfx1250_frontend, elem, varying):
    rng = np.random.default_rng(1323)
    a = rng.integers(-6, 7, (16, 128)).astype(np.float64)
    b = rng.integers(-6, 7, (16, 128)).astype(np.float64)
    scale_a = rng.integers(-3, 4, (16, 4)) if varying else np.zeros((16, 4), np.int64)
    scale_b = rng.integers(-3, 4, (16, 4)) if varying else np.zeros((16, 4), np.int64)
    ref = (a * 2.0 ** np.repeat(scale_a, 32, axis=1)) @ (b * 2.0 ** np.repeat(scale_b, 32, axis=1)).T

    # Load registers the way make_fragment_A/B + make_tiled_copy_A/B would: by the atom's layout.
    pos_a, pos_b = _ab_positions(elem)
    regs_a = a.ravel(order="F")[pos_a]
    regs_b = b.ravel(order="F")[pos_b]
    np.testing.assert_array_equal(_emulate_wmma_scale(regs_a, regs_b, scale_a, scale_b, elem), ref)
