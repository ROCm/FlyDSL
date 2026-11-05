# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Host construction, verifier rejection, and ThrVal pins for gfx120x SWMMAC.

No GPU required. Reject paths use ``ir.Type.parse`` because ``Type.get`` asserts
on verify failure.
"""

from __future__ import annotations

import pytest

import flydsl  # noqa: F401  preload comgr
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir._mlir_libs._mlirDialectsFlyROCDL import MmaOpGFX120X_SWMMACType
from flydsl._mlir.dialects import fly_rocdl  # noqa: F401  register dialect
from flydsl._mlir.dialects.fly import MmaAtomType
from flydsl.compiler import jit_function

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


def _parse(src: str):
    return ir.Type.parse(src)


def test_swmmac_fp_and_integer_construct():
    with ir.Context(), ir.Location.unknown():
        f16 = ir.F16Type.get()
        bf16 = ir.BF16Type.get()
        fp8 = ir.Float8E4M3FNType.get()
        bf8 = ir.Float8E5M2Type.get()
        f32 = ir.F32Type.get()
        i4 = ir.IntegerType.get_signless(4)
        si4 = ir.IntegerType.get_signed(4)
        ui4 = ir.IntegerType.get_unsigned(4)
        i8 = ir.IntegerType.get_signless(8)
        si8 = ir.IntegerType.get_signed(8)
        ui8 = ir.IntegerType.get_unsigned(8)
        i32 = ir.IntegerType.get_signless(32)

        for elem_a, elem_b in (
            (f16, f16),
            (bf16, bf16),
            (fp8, fp8),
            (fp8, bf8),
            (bf8, fp8),
            (bf8, bf8),
        ):
            t = MmaOpGFX120X_SWMMACType.get(16, 16, 32, elem_a, elem_b, f32)
            assert "16x16x32" in str(t)
            assert _parse(str(t)) == t

        for k, elem_a, elem_b in (
            (32, i8, i8),
            (32, si8, ui8),
            (32, i4, i4),
            (64, si4, ui4),
        ):
            t = MmaOpGFX120X_SWMMACType.get(16, 16, k, elem_a, elem_b, i32, sign_a=True, sign_b=False, clamp=True)
            assert f"16x16x{k}" in str(t)
            assert "signA = true" in str(t)
            assert "clamp = true" in str(t)
            assert _parse(str(t)) == t


@pytest.mark.parametrize(
    "src,expected_msg",
    [
        (
            "!fly_rocdl.gfx120x.swmmac<32x16x32, (f16, f16) -> f32, signA = false, signB = false, clamp = false>",
            r"GFX120X SWMMAC requires M=N=16, got 32x16",
        ),
        (
            "!fly_rocdl.gfx120x.swmmac<16x16x16, (f16, f16) -> f32, signA = false, signB = false, clamp = false>",
            r"GFX120X SWMMAC floating-point forms require M=N=16, K=32, got 16x16x16",
        ),
        (
            "!fly_rocdl.gfx120x.swmmac<16x16x64, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false>",
            r"GFX120X SWMMAC floating-point forms require M=N=16, K=32, got 16x16x64",
        ),
        (
            "!fly_rocdl.gfx120x.swmmac<16x16x32, (f16, bf16) -> f32, signA = false, signB = false, clamp = false>",
            r"unsupported GFX120X SWMMAC configuration",
        ),
        (
            "!fly_rocdl.gfx120x.swmmac<16x16x32, (bf16, bf16) -> f32, signA = true, signB = false, clamp = false>",
            r"floating-point path does not accept signA/signB/clamp",
        ),
        (
            "!fly_rocdl.gfx120x.swmmac<16x16x16, (i8, i8) -> i32, signA = false, signB = false, clamp = false>",
            r"GFX120X SWMMAC IU8 requires K=32, got 16x16x16",
        ),
        (
            "!fly_rocdl.gfx120x.swmmac<16x16x16, (i4, i4) -> i32, signA = false, signB = false, clamp = false>",
            r"GFX120X SWMMAC IU4 requires K=32 or K=64, got 16x16x16",
        ),
    ],
)
def test_swmmac_verifier_diagnostics(src: str, expected_msg: str):
    with ir.Context(), ir.Location.unknown():
        with pytest.raises(Exception, match=expected_msg):
            _parse(src)


@pytest.mark.parametrize("elem,k", [("i4", 32), ("i4", 64), ("i8", 32)])
@pytest.mark.parametrize("acc", ["si32", "ui32"])
def test_swmmac_rejects_signed_accumulator_types(elem: str, k: int, acc: str):
    with ir.Context(), ir.Location.unknown():
        with pytest.raises(Exception, match="unsupported GFX120X SWMMAC configuration"):
            _parse(
                f"!fly_rocdl.gfx120x.swmmac<16x16x{k}, ({elem}, {elem}) -> {acc}, "
                "signA = false, signB = false, clamp = false>"
            )


def test_swmmac_arch_dispatch_and_kwargs(monkeypatch):
    with ir.Context(), ir.Location.unknown():
        monkeypatch.setattr("flydsl.expr.rocdl.universal.get_rocm_arch", lambda: "gfx1201")
        t = fx.rocdl.SWMMAC(
            16,
            16,
            32,
            fx.Float8E4M3FN,
            fx.Float32,
            elem_ty_b=fx.Float8E5M2,
        )
        assert "f8E4M3FN, f8E5M2" in str(t)

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            fx.rocdl.SWMMAC(16, 16, 32, fx.Float16, fx.Float32, mod_c=1)

        for unsupported_arch in ("gfx942", "gfx1100", "gfx1250"):
            monkeypatch.setattr("flydsl.expr.rocdl.universal.get_rocm_arch", lambda a=unsupported_arch: a)
            with pytest.raises(ValueError, match="SWMMAC is not available on target arch"):
                fx.rocdl.SWMMAC(16, 16, 32, fx.Float16, fx.Float32)


@pytest.fixture
def gfx120_frontend(monkeypatch):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "rocm")
    monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx1201")
    monkeypatch.setenv("ARCH", "gfx1201")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setattr(jit_function, "_flydsl_key", lambda: "test-flydsl-key-swmmac")

    def compile_only(cls, asm, **_kwargs):
        module = ir.Module.parse(asm)
        assert module.operation.verify()
        return module

    monkeypatch.setattr(jit_function.MlirCompiler, "compile", classmethod(compile_only))


def _pin_printed_layouts(k: int, elem_a, elem_b, acc, *, sign: bool = False):
    t = MmaOpGFX120X_SWMMACType.get(16, 16, k, elem_a, elem_b, acc, sign_a=sign, sign_b=sign)
    atom = MmaAtomType.get(t)
    return str(atom.tv_layout_a), str(atom.tv_layout_b), str(atom.tv_layout_c)


def test_swmmac_k32_thrval_rdna4_contiguous(gfx120_frontend):
    with ir.Context(), ir.Location.unknown():
        f16 = ir.F16Type.get()
        f32 = ir.F32Type.get()
        a, b, c = _pin_printed_layouts(32, f16, f16, f32)
        assert a == "!fly.layout<((16,2),8):((1,128),16)>"
        assert b == "!fly.layout<((16,2),16):((1,256),16)>"
        assert c == "!fly.layout<((16,2),8):((16,8),1)>"
        # Must not use gfx1250's 8-element block-interleaved (8,2) layout.
        assert "(8,2)" not in b

    @flyc.jit
    def pin():
        atom = fx.make_mma_atom(fx.rocdl.SWMMAC(16, 16, 32, fx.Float16, fx.Float32))
        layout_a = atom.layout_A_tv
        layout_b = atom.layout_B_tv
        layout_c = atom.layout_C_tv
        for lane in fx.range_constexpr(32):
            l16, g = lane % 16, lane // 16
            for v in fx.range_constexpr(8):
                pos_a = layout_a((l16, g), v).get_static_leaf_int
                assert pos_a == l16 + (8 * g + v) * 16
                pos_c = layout_c((l16, g), v).get_static_leaf_int
                assert pos_c == l16 * 16 + 8 * g + v
            for v in fx.range_constexpr(16):
                pos_b = layout_b((l16, g), v).get_static_leaf_int
                assert pos_b == l16 + (16 * g + v) * 16

    pin()


def test_swmmac_iu4_k64_thrval_rdna4_contiguous(gfx120_frontend):
    with ir.Context(), ir.Location.unknown():
        i4 = ir.IntegerType.get_signless(4)
        i32 = ir.IntegerType.get_signless(32)
        a, b, c = _pin_printed_layouts(64, i4, i4, i32, sign=True)
        assert a == "!fly.layout<((16,2),16):((1,256),16)>"
        assert b == "!fly.layout<((16,2),32):((1,512),16)>"
        assert c == "!fly.layout<((16,2),8):((16,8),1)>"
        # Must not use gfx1250's 8-element block-interleaved (8,2)/(8,4) layouts.
        assert "(8,2)" not in a
        assert "(8,4)" not in b

    @flyc.jit
    def pin():
        atom = fx.make_mma_atom(fx.rocdl.SWMMAC(16, 16, 64, fx.Int4, fx.Int32, sign_a=True, sign_b=True))
        layout_a = atom.layout_A_tv
        layout_b = atom.layout_B_tv
        assert layout_b((0, 0), 16).get_static_leaf_int // 16 == 16
        assert layout_b((0, 1), 0).get_static_leaf_int // 16 == 32
        for lane in fx.range_constexpr(32):
            l16, g = lane % 16, lane // 16
            for v in fx.range_constexpr(16):
                pos_a = layout_a((l16, g), v).get_static_leaf_int
                assert pos_a == l16 + (16 * g + v) * 16
            for v in fx.range_constexpr(32):
                pos_b = layout_b((l16, g), v).get_static_leaf_int
                assert pos_b == l16 + (32 * g + v) * 16

    pin()
