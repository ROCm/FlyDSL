# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Default gfx120x int8 linear entry: size dispatch -> W8A16 or iu8.

DEFAULT entry. Applies the measured K-gate in rdna4_int8_linear_dispatch
(K >= 256 → iu8; else W8A16; measured 2026-09-30), then runs
W8A16 or iu8. Prefer this over calling one
pipeline alone unless you already know the shape class.

Override with force_kernel="w8a16"|"iu8". Shorthand names: size
dispatcher / gate / _gated.
"""

from kernels.gemm.rdna4_int8_linear_dispatch import (
    DEFAULT_K_IU8_MIN,
    DEFAULT_M_A_MAX,
    DEFAULT_M_W8A16_MAX,
    DEFAULT_MK_A_MAX,
    DEFAULT_MK_W8A16_MAX,
    KernelChoice,
    dispatch_rule_doc,
    int8_linear_auto,
    int8_linear_dispatched,
    select_int8_kernel,
    select_int8_kernel_for_tensors,
)

KERNEL_NAME = "rdna4_int8_linear_auto"


def gate_rule_doc(**kwargs) -> dict[str, object]:
    """Structured dispatcher description (includes auto-module fields)."""
    doc = dispatch_rule_doc(**kwargs)
    doc["auto_module"] = "kernels.gemm.rdna4_int8_linear_auto"
    doc["gated_module"] = doc["auto_module"]  # legacy test key
    doc["rule_source"] = "kernels.gemm.rdna4_int8_linear_dispatch"
    doc.setdefault(
        "disclaimer",
        "gfx120x size dispatcher; not a general FlyDSL autotune API. "
        "Shipped DEFAULT on gfx120x; measured K-gate 2026-09-30 (gfx120x_autotune_tables).",
    )
    # legacy key names tests may still assert
    if "m_a_max" not in doc and "m_w8a16_max" in doc:
        doc["m_a_max"] = doc["m_w8a16_max"]
        doc["mk_a_max"] = doc["mk_w8a16_max"]
    return doc


auto_rule_doc = gate_rule_doc

__all__ = [
    "KERNEL_NAME",
    "DEFAULT_M_A_MAX",
    "DEFAULT_MK_A_MAX",
    "DEFAULT_K_IU8_MIN",
    "DEFAULT_M_W8A16_MAX",
    "DEFAULT_MK_W8A16_MAX",
    "KernelChoice",
    "select_int8_kernel",
    "select_int8_kernel_for_tensors",
    "gate_rule_doc",
    "auto_rule_doc",
    "dispatch_rule_doc",
    "int8_linear_dispatched",
    "int8_linear_auto",
]

# Thin alias: gate/_gated shorthand for size-dispatcher callers.
int8_linear_gated = int8_linear_auto
