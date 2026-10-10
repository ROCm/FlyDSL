#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""GFX120X SWMMAC atom wiring probe.

Sparse product kernels are out of scope; this checks ROCDL intrinsic names,
the Python helper export, and the C++ emit path. FileCheck coverage lives in
``tests/mlir/Conversion/swmmac_gfx120x.mlir``.
"""

import os
import re
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


def test_rocdl_python_exposes_swmmac_intrinsics() -> None:
    from flydsl.runtime.device import get_rocm_arch

    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx120"):
        pytest.skip(f"SWMMAC binding probe is gfx120x-only, got {arch!r}")
    from flydsl._mlir.dialects import rocdl

    for name in (
        "swmmac_f32_16x16x32_f16",
        "swmmac_f32_16x16x32_bf16",
        "swmmac_i32_16x16x32_iu8",
        "swmmac_i32_16x16x32_iu4",
        "swmmac_i32_16x16x64_iu4",
        "swmmac_f32_16x16x32_fp8_fp8",
        "swmmac_f32_16x16x32_fp8_bf8",
        "swmmac_f32_16x16x32_bf8_fp8",
        "swmmac_f32_16x16x32_bf8_bf8",
    ):
        assert hasattr(rocdl, name), name
    # Same-type sparse accumulators exist in ROCDL but are not emitted by FlyDSL.
    assert hasattr(rocdl, "swmmac_f16_16x16x32_f16")
    assert hasattr(rocdl, "swmmac_bf16_16x16x32_bf16")


def test_flydsl_swmmac_helper_export_and_cpp_emit() -> None:
    from flydsl.runtime.device import get_rocm_arch

    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx120"):
        pytest.skip(f"SWMMAC helper probe is gfx120x-only, got {arch!r}")
    import flydsl.expr.rocdl as rocdl_mod
    from flydsl._mlir._mlir_libs._mlirDialectsFlyROCDL import MmaOpGFX120X_SWMMACType

    assert hasattr(rocdl_mod, "SWMMAC")
    assert MmaOpGFX120X_SWMMACType is not None
    path = os.path.join(_REPO_ROOT, "lib/Dialect/FlyROCDL/GFX120X/MmaAtom.cpp")
    text = open(path, encoding="utf-8").read()
    assert "MmaOpGFX120X_SWMMACType" in text
    assert "swmmac_f32_16x16x32_f16" in text
    assert "swmmac_i32_16x16x64_iu4" in text
    assert not re.search(r"swmmac_f16_16x16x32_f16::create", text)
    assert not re.search(r"swmmac_bf16_16x16x32_bf16::create", text)
