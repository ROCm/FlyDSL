# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""ROCDL/AMDGPU DSL enums."""

from ..._mlir.dialects.fly_rocdl import TargetAddressSpace

AddressSpace = TargetAddressSpace


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
