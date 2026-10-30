# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Host construction + ThrVal pins for gfx120 integer WMMA.

No GPU. Do not call ``fx.rocdl.WMMA()`` against the real ``get_rocm_arch()``.
Reject paths must use ``ir.Type.parse`` — ``MmaOpGFX120X_WMMAType.get`` asserts
on verify failure.
"""

from __future__ import annotations

import pytest

import flydsl  # noqa: F401  preload comgr
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir._mlir_libs._mlirDialectsFlyROCDL import MmaOpGFX120X_WMMAType
from flydsl._mlir.dialects import fly_rocdl  # noqa: F401  register dialect
from flydsl._mlir.dialects.fly import MmaAtomType
from flydsl.compiler import jit_function

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]

_IU4_K16 = "!fly_rocdl.gfx120x.wmma<16x16x16, (i4, i4) -> i32, signA = true, signB = true, clamp = false>"
_IU8_K16 = "!fly_rocdl.gfx120x.wmma<16x16x16, (i8, i8) -> i32, signA = false, signB = false, clamp = false>"
_IU4_K32 = "!fly_rocdl.gfx120x.wmma<16x16x32, (i4, i4) -> i32, signA = true, signB = true, clamp = false>"
_FP_K16 = "!fly_rocdl.gfx120x.wmma<16x16x16, (f16, f16) -> f32, signA = false, signB = false, clamp = false>"
_FP_K32 = "!fly_rocdl.gfx120x.wmma<16x16x32, (bf16, bf16) -> f32, signA = false, signB = false, clamp = false>"
_FP_SIGN = "!fly_rocdl.gfx120x.wmma<16x16x16, (bf16, bf16) -> f32, signA = true, signB = false, clamp = false>"


def _parse(src: str):
    return ir.Type.parse(src)


def test_iu4_k16_constructs_signed_operands():
    with ir.Context(), ir.Location.unknown():
        # Parse first: Type.get() aborts the process on verify failure.
        parsed = _parse(_IU4_K16)
        assert "i4" in str(parsed)
        i4 = ir.IntegerType.get_signless(4)
        si4 = ir.IntegerType.get_signed(4)
        ui4 = ir.IntegerType.get_unsigned(4)
        i32 = ir.IntegerType.get_signless(32)
        for a, b, acc in (
            (i4, i4, i32),
            (si4, ui4, i32),
        ):
            t = MmaOpGFX120X_WMMAType.get(16, 16, 16, a, b, acc, sign_a=True, sign_b=True)
            assert "16x16x16" in str(t)
            assert "signA = true" in str(t)
            atom = MmaAtomType.get(t)
            assert atom is not None


def test_iu8_k16_constructs():
    with ir.Context(), ir.Location.unknown():
        parsed = _parse(_IU8_K16)
        i8 = ir.IntegerType.get_signless(8)
        i32 = ir.IntegerType.get_signless(32)
        t = MmaOpGFX120X_WMMAType.get(16, 16, 16, i8, i8, i32)
        assert "16x16x16" in str(t)
        assert parsed == t


def test_iu4_k32_constructs():
    with ir.Context(), ir.Location.unknown():
        parsed = _parse(_IU4_K32)
        assert "16x16x32" in str(parsed)
        i4 = ir.IntegerType.get_signless(4)
        i32 = ir.IntegerType.get_signless(32)
        t = MmaOpGFX120X_WMMAType.get(16, 16, 32, i4, i4, i32, sign_a=True, sign_b=True)
        assert "16x16x32" in str(t)


def test_fp_k16_constructs_fp_k32_keeps_old_string():
    with ir.Context(), ir.Location.unknown():
        f16 = ir.F16Type.get()
        f32 = ir.F32Type.get()
        t = MmaOpGFX120X_WMMAType.get(16, 16, 16, f16, f16, f32)
        assert _parse(_FP_K16) == t
        with pytest.raises(Exception, match=r"GFX120X WMMA floating-point forms require M=N=K=16, got 16x16x32"):
            _parse(_FP_K32)


def test_iu4_signed_is_not_fp_sign_reject():
    with ir.Context(), ir.Location.unknown():
        _parse(_IU4_K16)
        with pytest.raises(Exception, match="floating-point path does not accept signA/signB/clamp"):
            _parse(_FP_SIGN)


@pytest.fixture
def gfx120_frontend(monkeypatch):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "rocm")
    monkeypatch.setenv("FLYDSL_GPU_ARCH", "gfx1201")
    monkeypatch.setenv("ARCH", "gfx1201")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setattr(jit_function, "_flydsl_key", lambda: "test-flydsl-key-g1")

    def compile_only(cls, asm, **_kwargs):
        module = ir.Module.parse(asm)
        assert module.operation.verify()
        return module

    monkeypatch.setattr(jit_function.MlirCompiler, "compile", classmethod(compile_only))


def _pin_printed_layouts(k, elem_a, elem_b, acc, *, sign=False):
    t = MmaOpGFX120X_WMMAType.get(16, 16, k, elem_a, elem_b, acc, sign_a=sign, sign_b=sign)
    atom = MmaAtomType.get(t)
    return str(atom.tv_layout_a), str(atom.tv_layout_b), str(atom.tv_layout_c)


def test_iu4_k16_printed_and_coords(gfx120_frontend):
    with ir.Context(), ir.Location.unknown():
        i4 = ir.IntegerType.get_signless(4)
        i32 = ir.IntegerType.get_signless(32)
        a, b, c = _pin_printed_layouts(16, i4, i4, i32, sign=True)
        assert a == "!fly.layout<((16,2),8):((1,128),16)>"
        assert b == a
        assert c == "!fly.layout<((16,2),8):((16,8),1)>"
        assert "128" in a and ",8)" in c

    @flyc.jit
    def pin():
        atom = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 16, fx.Int4, fx.Int32, sign_a=True, sign_b=True))
        layout_a = atom.layout_A_tv
        layout_b = atom.layout_B_tv
        layout_c = atom.layout_C_tv
        # range() becomes scf.for; unroll so crd2idx stays a static leaf.
        for lane in fx.range_constexpr(32):
            l16, g = lane % 16, lane // 16
            for v in fx.range_constexpr(8):
                pos_a = layout_a((l16, g), v).get_static_leaf_int
                assert pos_a == l16 + (8 * g + v) * 16
                assert layout_b((l16, g), v).get_static_leaf_int == pos_a
                pos_c = layout_c((l16, g), v).get_static_leaf_int
                assert pos_c == l16 * 16 + 8 * g + v

    pin()


def test_iu4_k32_rdna4_not_gfx1250(gfx120_frontend):
    with ir.Context(), ir.Location.unknown():
        i4 = ir.IntegerType.get_signless(4)
        i32 = ir.IntegerType.get_signless(32)
        a, b, c = _pin_printed_layouts(32, i4, i4, i32, sign=True)
        assert a == "!fly.layout<((16,2),16):((1,256),16)>"
        assert b == a
        assert c == "!fly.layout<((16,2),8):((16,8),1)>"
        assert "(8,2)" not in a
        assert ",128)" not in a

    @flyc.jit
    def pin():
        atom = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 32, fx.Int4, fx.Int32, sign_a=True, sign_b=True))
        layout_a = atom.layout_A_tv
        k_g0_v8 = layout_a((0, 0), 8).get_static_leaf_int // 16
        k_g1_v0 = layout_a((0, 1), 0).get_static_leaf_int // 16
        assert k_g0_v8 == 8
        assert k_g1_v0 == 16
        for lane in fx.range_constexpr(32):
            l16, g = lane % 16, lane // 16
            for v in fx.range_constexpr(16):
                pos = layout_a((l16, g), v).get_static_leaf_int
                assert pos % 16 == l16
                assert pos // 16 == 16 * g + v

    pin()


def test_iu8_k16_matches_iu4_k16_thrval(gfx120_frontend):
    with ir.Context(), ir.Location.unknown():
        i4 = ir.IntegerType.get_signless(4)
        i8 = ir.IntegerType.get_signless(8)
        i32 = ir.IntegerType.get_signless(32)
        a4, _, _ = _pin_printed_layouts(16, i4, i4, i32)
        a8, _, _ = _pin_printed_layouts(16, i8, i8, i32)
        assert a4 == a8 == "!fly.layout<((16,2),8):((1,128),16)>"

    @flyc.jit
    def pin():
        a4 = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 16, fx.Int4, fx.Int32)).layout_A_tv
        a8 = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 16, fx.Int8, fx.Int32)).layout_A_tv
        for lane in fx.range_constexpr(32):
            l16, g = lane % 16, lane // 16
            for v in fx.range_constexpr(8):
                assert a4((l16, g), v).get_static_leaf_int == a8((l16, g), v).get_static_leaf_int

    pin()


def test_rdna3_int8_gemm_gfx120_message_drops_no_integer_wmma(monkeypatch):
    monkeypatch.setattr("kernels.gemm.rdna3_int8_gemm.get_rocm_arch", lambda: "gfx1201")
    from kernels.gemm.rdna3_int8_gemm import create_wmma_int8_gemm_module

    with pytest.raises(RuntimeError, match=r"rdna3_int8_gemm requires gfx11\*") as ei:
        create_wmma_int8_gemm_module(64, 64, 64, in_dtype="int8", out_dtype="i32")
    msg = str(ei.value)
    assert "v16 WMMA ABI" in msg
    assert "has no integer WMMA" not in msg


@pytest.mark.parametrize("elem,k", [("i4", 16), ("i4", 32), ("i8", 16)])
@pytest.mark.parametrize("acc", ["si32", "ui32"])
def test_integer_wmma_rejects_signed_accumulator_types(elem, k, acc):
    with ir.Context(), ir.Location.unknown():
        with pytest.raises(Exception, match="unsupported GFX120X WMMA configuration"):
            _parse(
                f"!fly_rocdl.gfx120x.wmma<16x16x{k}, ({elem}, {elem}) -> {acc}, "
                "signA = false, signB = false, clamp = false>"
            )


@pytest.mark.parametrize("packed_k", [4, 8, 16, 32])
def test_int4_lds_stride_bank_distribution(packed_k):
    from kernels.gemm.rdna4_int4_gemm import _padded_stride

    stride = _padded_stride(packed_k)
    assert stride >= packed_k
    assert stride - packed_k < 8
    counts = [0] * 32
    for row in range(16):
        for col in range(4):
            counts[(row * stride + col) % 32] += 1
    assert counts == [2] * 32


@pytest.mark.parametrize("ncopy", [1, 2, 3, 4, 8, 16])
@pytest.mark.parametrize("k_iters", [1, 2, 4, 8])
def test_int4_pipeline_schedule_retires_copies(ncopy, k_iters):
    from kernels.gemm.rdna4_int4_gemm import _pipeline_schedule

    ops, inflight = _pipeline_schedule(ncopy, k_iters)
    live = {}
    loads = []
    stores = []
    wmma_steps = []
    for kind, arg, slot in ops:
        if kind == "load":
            assert slot not in live
            live[slot] = arg
            loads.append(arg)
            assert len(live) <= inflight
        elif kind == "store":
            assert live.pop(slot) == arg
            stores.append(arg)
        else:
            assert kind == "wmma"
            wmma_steps.append(arg)
    assert not live
    assert loads == stores == list(range(ncopy))
    assert wmma_steps == list(range(k_iters))
