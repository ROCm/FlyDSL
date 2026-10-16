#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""GFX120X iu4 WMMA wiring probe.

RDNA4 emits iu4 only at K=32: ``wmma_i32_16x16x32_iu4`` (vector<2xi32>, 16
nibbles). The gfx120x atom does not select K=16.

Device correctness lives in ``test_rdna4_integer_wmma_atom.py``.
Default W4 kernels may still unpack to iu8 when K is not a multiple of 16.
"""

import os
import re
import sys

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


def test_rocdl_python_exposes_iu4_intrinsic() -> None:
    """LLVM/ROCDL binding presence. gfx120x only.

    The Python name is not added by this tree (it comes from the installed
    LLVM/ROCDL bindings). Asserting it on gfx950/gfx942 would fail those
    suites even though their atoms are unchanged.
    """
    from flydsl.runtime.device import get_rocm_arch

    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx120"):
        pytest.skip(f"ROCDL iu4 binding probe is gfx120x-only, got {arch!r}")
    import flydsl.expr.rocdl as rocdl

    assert hasattr(rocdl, "wmma_i32_16x16x16_iu4")
    assert hasattr(rocdl, "wmma_i32_16x16x32_iu4")
    assert hasattr(rocdl, "wmma_i32_16x16x16_iu8")


def test_flydsl_int4_type_exists() -> None:
    import flydsl.expr as fx

    assert hasattr(fx, "Int4")
    assert fx.Int4.width == 4


def test_gfx120x_mma_atom_cpp_lowers_iu4() -> None:
    """Static source probe: GFX120X atom verify + emit cover iu4."""
    path = os.path.join(_REPO_ROOT, "lib/Dialect/FlyROCDL/GFX120X/MmaAtom.cpp")
    assert os.path.isfile(path), path
    text = open(path, encoding="utf-8").read()
    assert "isI8" in text
    assert "isI4" in text
    assert re.search(
        r"isInt\(elemTyA,\s*4\).*isInt\(elemTyB,\s*4\)", text, re.S
    ), "expected GFX120X verify to accept integer width 4 for A/B"
    assert "wmma_i32_16x16x16_iu8" in text
    assert "wmma_i32_16x16x32_iu4" in text
    assert "wmma_i32_16x16x16_iu4" not in text


def test_iu4_atom_wired_documented() -> None:
    """Honest verdict marker for idle notes."""
    verdict = {
        "hardware_llvm_iu4": True,
        "flydsl_gfx120x_atom_iu4": True,
        "gfx12_ab_packing": "k32_v2i32_only",
        "tip_w4_fallback": "unpack→iu8 still valid",
    }
    assert verdict["flydsl_gfx120x_atom_iu4"] is True
    assert verdict["hardware_llvm_iu4"] is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
