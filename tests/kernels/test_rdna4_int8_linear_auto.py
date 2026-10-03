#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""CPU-safe tests for the gfx120x int8_linear_auto size rule."""

import os
import sys

import pytest  # noqa: E402

pytestmark = [pytest.mark.l0_backend_agnostic]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


from kernels.gemm.rdna4_int8_linear_auto import (  # noqa: E402
    DEFAULT_M_A_MAX,
    DEFAULT_MK_A_MAX,
    gate_rule_doc,
    select_int8_kernel,
)


@pytest.mark.parametrize(
    "m,n,k,expect",
    [
        (32, 128, 64, "w8a16"),  # K < 256
        (128, 256, 128, "w8a16"),  # K < 256
        (128, 256, 256, "iu8"),  # K >= 256
        (256, 512, 512, "iu8"),
        (1024, 4096, 4096, "iu8"),
        (4096, 3072, 3072, "iu8"),
    ],
)
def test_select_int8_kernel_anchors(m: int, n: int, k: int, expect: str) -> None:
    assert select_int8_kernel(m, n, k) == expect


def test_select_int8_kernel_force() -> None:
    assert select_int8_kernel(4096, 3072, 3072, force_kernel="w8a16") == "w8a16"
    assert select_int8_kernel(32, 64, 64, force_kernel="iu8") == "iu8"
    with pytest.raises(ValueError):
        select_int8_kernel(32, 64, 64, force_kernel="HIP")  # type: ignore[arg-type]


def test_auto_rule_doc() -> None:
    doc = gate_rule_doc()
    from kernels.gemm.rdna4_int8_linear_dispatch import DEFAULT_K_IU8_MIN

    assert doc.get("k_iu8_min", doc.get("default_k_iu8_min")) == DEFAULT_K_IU8_MIN == 256
    assert DEFAULT_M_A_MAX == 0 and DEFAULT_MK_A_MAX == 0
    assert (
        doc.get("auto_module") == "kernels.gemm.rdna4_int8_linear_auto"
        or doc.get("gated_module") == "kernels.gemm.rdna4_int8_linear_auto"
    )
    assert (
        "Experimental" in doc.get("disclaimer", "")
        or "experimental" in doc.get("disclaimer", "").lower()
        or "EXPERIMENTAL" in doc.get("disclaimer", "")
        or "Measured" in doc.get("disclaimer", "")
    )
