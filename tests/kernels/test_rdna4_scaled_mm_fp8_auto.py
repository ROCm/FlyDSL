#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""CPU-safe size-dispatcher tests for gfx120x scaled_mm_fp8_auto."""

import os
import sys

import pytest  # noqa: E402

pytestmark = [pytest.mark.l0_backend_agnostic]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


from kernels.gemm.rdna4_scaled_mm_fp8_auto import (  # noqa: E402
    gate_rule_doc,
    select_path,
    select_tile,
)


@pytest.mark.parametrize(
    "m,n,k,wgps,expect",
    [
        (32, 128, 128, 32, "skinny_bk128"),
        (32, 128, 64, 32, "skinny_bk64"),
        # Underfilled fat grid still picks 64x64x128 via plain K>=128 fallback.
        (128, 512, 512, 32, "skinny_bk128"),
        (256, 512, 512, 32, "skinny_bk128"),
        # WGP-filling fat + deep K → 128x128x128_B
        (1024, 4096, 4096, 32, "fat_deep_k_b"),
        (1024, 5120, 5120, 32, "fat_deep_k_b"),
        (1024, 1024, 512, 32, "fat_deep_k_b"),
        # Fat + shallow K → 128x128x64
        (2048, 2048, 64, 32, "fat_bk64"),
    ],
)
def test_select_path_tile_anchors(m: int, n: int, k: int, wgps: int, expect: str) -> None:
    assert select_path(m, n, k, wgps=wgps) == expect


def test_select_path_force() -> None:
    assert select_path(32, 128, 64, force_path="fat_bk64") == "fat_bk64"
    with pytest.raises(ValueError):
        select_path(32, 128, 64, force_path="HIP")  # type: ignore[arg-type]


def test_select_tile_matches_path_bk() -> None:
    cfg = select_tile(32, 128, 128, wgps=32)
    assert cfg.bk == 128 and cfg.bm == 64
    cfg2 = select_tile(32, 128, 64, force_path="skinny_bk64")
    assert cfg2.bk == 64 and cfg2.bm == 64


def test_gate_rule_doc() -> None:
    doc = gate_rule_doc()
    assert "smoke_anchor" in doc
    assert "EXPERIMENTAL" in doc["disclaimer"]
    assert doc["host"] == "scaled_mm_fp8_auto"
