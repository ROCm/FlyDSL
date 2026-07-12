#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Backend-agnostic construction tests for the gfx1250 MX-scale WMMA MMA atom
and the 2D TDM copy atom.

These only build the FlyROCDL atom *types* via the Python factories in
``flydsl.expr.rocdl.universal`` and check textual round-trip + verifier
behavior; no GPU is required.
"""

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]

from flydsl._mlir import ir  # noqa: E402


def _ctx():
    ctx = ir.Context()
    return ctx


def test_wmma_scale_type_roundtrip():
    with _ctx(), ir.Location.unknown():
        from flydsl._mlir.dialects import fly_rocdl  # noqa: F401  (register dialect)
        from flydsl.expr.rocdl import universal as U

        f8 = ir.Float8E4M3FNType.get()
        f4 = ir.Float4E2M1FNType.get()
        f32 = ir.F32Type.get()

        t_fp8 = U.WMMAScale(16, 16, 128, f8, f8, f32)
        assert "gfx1250.wmma_scale" in str(t_fp8)
        assert "opselA = 0, opselB = 0" in str(t_fp8)
        assert ir.Type.parse(str(t_fp8)) == t_fp8

        t_fp4 = U.WMMAScale(16, 16, 128, f4, f4, f32, opsel_a=1, opsel_b=2)
        assert "opselA = 1, opselB = 2" in str(t_fp4)
        assert ir.Type.parse(str(t_fp4)) == t_fp4


def test_tdm2d_type_roundtrip():
    with _ctx(), ir.Location.unknown():
        from flydsl._mlir.dialects import fly_rocdl  # noqa: F401
        from flydsl.expr.rocdl import universal as U

        t = U.TDM2D(1)
        assert "gfx1250.tdm_2d" in str(t)
        assert ir.Type.parse(str(t)) == t

        t2 = U.TDM2D(8, pad_interval=64, pad_amount=8, cache_modifier=2)
        assert "warps = 8" in str(t2)
        assert "pad = 64, 8" in str(t2)
        assert ir.Type.parse(str(t2)) == t2
