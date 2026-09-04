# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""ROCDL/AMDGPU DSL enums."""

from ..._mlir.dialects.fly_rocdl import TargetAddressSpace

__all__ = [
    "AddressSpace",
    "MemoryOrder",
    "TargetAddressSpace",
    "SyncScope",
]

AddressSpace = TargetAddressSpace


class MemoryOrder:
    """LLVM-compatible memory orderings accepted by ROCDL memory ops."""

    NotAtomic = "not_atomic"
    Unordered = "unordered"
    Monotonic = "monotonic"
    Acquire = "acquire"
    Release = "release"
    SequentiallyConsistent = "seq_cst"


class SyncScope:
    """AMDGPU-specific sync scopes.

    Each field is the literal LLVM sync-scope string for the AMDGPU memory
    model.
    """

    Agent = "agent"
    Workgroup = "workgroup"
    Wavefront = "wavefront"
    OneAs = "one-as"
    AgentOneAs = "agent-one-as"
    WorkgroupOneAs = "workgroup-one-as"
    WavefrontOneAs = "wavefront-one-as"
    SingleThreadOneAs = "singlethread-one-as"
