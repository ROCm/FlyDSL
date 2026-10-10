# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Target-neutral DSL enums."""

from enum import Enum

from .._mlir.dialects.fly import AtomicOp

__all__ = [
    "AtomicOp",
    "AtomicOrdering",
    "SyncScope",
]


class AtomicOrdering(Enum):
    """Atomic orderings from the LLVM memory model."""

    NotAtomic = "not_atomic"
    Unordered = "unordered"
    Monotonic = "monotonic"
    Acquire = "acquire"
    Release = "release"
    AcqRel = "acq_rel"
    SeqCst = "seq_cst"


class SyncScope:
    """LLVM target-neutral sync scopes.

    Target-specific scopes (e.g. AMDGPU ``agent`` / ``workgroup``) live in
    ``flydsl.expr.rocdl.enum.SyncScope``.
    """

    System = ""
    SingleThread = "singlethread"
