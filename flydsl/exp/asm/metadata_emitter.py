# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Metadata Emitter for AMDGCN kernels.

This module handles the emission of assembler directives for kernel
prologue (before instructions) and epilogue (after instructions).

These are NOT instructions - they are assembler directives like:
- .amdgcn_target
- .amdhsa_kernel / .end_amdhsa_kernel
- .amdgpu_metadata

The kernel body instructions are handled by KernelGenerator.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │  MetadataEmitter.emit_prologue()                            │
    │  - .amdgcn_target, .amdhsa_kernel, etc.                     │
    └─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  KernelGenerator.generate()                                 │
    │  - All instructions (including kernargs, s_endpgm)          │
    └─────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
    ┌─────────────────────────────────────────────────────────────┐
    │  MetadataEmitter.emit_epilogue()                            │
    │  - .amdgpu_metadata                                         │
    └─────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass
from typing import List, Tuple

from .utils import normalize_wg_size
from .abi import system_vgpr_workitem_id_from_wg_size


def get_register_granularity(targetid: str) -> Tuple[int, int]:
    """Get VGPR and SGPR allocation granularity for target."""
    # Canonical rule for current targets:
    # - gfx94x/gfx95x: VGPRs allocated in groups of 8, SGPRs in groups of 8
    # - older GFX9: VGPRs in groups of 4, SGPRs in groups of 8
    if any(t in targetid for t in ["gfx940", "gfx942", "gfx950"]):
        return 8, 8
    return 4, 8


@dataclass
class KernelMetadata:
    """Metadata about a kernel for prologue/epilogue generation."""

    name: str
    targetid: str
    codeobj: str = "5"

    # Workgroup configuration
    wg_size: Tuple[int, int, int] = (64, 1, 1)
    subgroup_size: int = 64

    # Workgroup ID requirements
    needs_wgid_x: bool = False
    needs_wgid_y: bool = False
    needs_wgid_z: bool = False

    # Resource usage (filled in after compilation)
    vgprs_used: int = 0
    sgprs_used: int = 0
    agprs_used: int = 0
    lds_size_bytes: int = 0

    # Kernel arguments
    num_args: int = 0

    # Kernel argument preloading (Code Object v5+)
    # Number of dwords (SGPRs) to preload from kernarg segment
    # For GEMM with 3 pointers (A, B, C): 3 * 2 = 6 SGPRs
    kernarg_preload_length: int = 0

    # SRD upper word constant
    SRD127_96: int = 0x20000


class MetadataEmitter:
    """
    Emits assembler directives for kernel prologue and epilogue.

    This class handles only metadata/directives, not instructions.
    Instructions (including kernargs loading, s_endpgm) are handled
    by KernelGenerator through the kernel IR.
    """

    def __init__(self, metadata: KernelMetadata):
        self.metadata = metadata
        self._lines: List[str] = []

    def emit_prologue(self) -> List[str]:
        """
        Emit kernel prologue with metadata directives.

        Returns list of assembly lines for the prologue.
        """
        m = self.metadata
        lines = []

        # Target specification
        lines.append(f'.amdgcn_target "amdgcn-amd-amdhsa--{m.targetid}"')
        lines.append(".text")
        lines.append(f".protected {m.name}")
        lines.append(f".globl {m.name}")
        lines.append(".p2align 8")
        lines.append(f".type {m.name},@function\n")

        # Kernel descriptor section
        lines.append(".section .rodata,#alloc")
        lines.append(".p2align 6")
        lines.append(f".amdhsa_kernel {m.name}")

        # User SGPR count (2 for kernarg ptr + preloaded args)
        user_sgpr_count = 2 + m.kernarg_preload_length
        lines.append(f"  .amdhsa_user_sgpr_count {user_sgpr_count}")

        # User SGPR configuration (following LLVM's order)
        lines.append("  .amdhsa_user_sgpr_dispatch_ptr 0")
        lines.append("  .amdhsa_user_sgpr_queue_ptr 0")
        lines.append("  .amdhsa_user_sgpr_kernarg_segment_ptr 1")
        lines.append("  .amdhsa_user_sgpr_dispatch_id 0")

        # Kernel argument preloading (Code Object v5+)
        # Preloads kernel args directly into SGPRs, saving ~100 cycles at kernel start
        # Only emit these directives when:
        # 1. Preload length is > 0 (preloading is enabled)
        # 2. Code object version >= 5 (preloading is supported)
        codeobj_version = int(m.codeobj) if m.codeobj.isdigit() else 0
        if m.kernarg_preload_length > 0 and codeobj_version >= 5:
            lines.append(
                f"  .amdhsa_user_sgpr_kernarg_preload_length {m.kernarg_preload_length}"
            )
            lines.append("  .amdhsa_user_sgpr_kernarg_preload_offset 0")

        lines.append("  .amdhsa_user_sgpr_private_segment_size 0")
        lines.append("  .amdhsa_uses_dynamic_stack 0")
        lines.append("  .amdhsa_enable_private_segment 0")

        # Resource usage (placeholders to be patched later)
        lines.append("  .amdhsa_accum_offset 0")  # patched later
        lines.append("  .amdhsa_next_free_vgpr 0")  # patched later
        lines.append("  .amdhsa_next_free_sgpr 0")  # patched later
        lines.append("  .amdhsa_group_segment_fixed_size 0")  # patched later
        lines.append("  .amdhsa_private_segment_fixed_size 0")

        # Workgroup ID requests
        lines.append(
            f"  .amdhsa_system_sgpr_workgroup_id_x {1 if m.needs_wgid_x else 0}"
        )
        lines.append(
            f"  .amdhsa_system_sgpr_workgroup_id_y {1 if m.needs_wgid_y else 0}"
        )
        lines.append(
            f"  .amdhsa_system_sgpr_workgroup_id_z {1 if m.needs_wgid_z else 0}"
        )

        # Workitem ID VGPRs
        system_vgpr_workitem_id = self._get_system_vgpr_workitem_id()
        lines.append(f"  .amdhsa_system_vgpr_workitem_id {system_vgpr_workitem_id}")

        # Float mode
        lines.append("  .amdhsa_float_denorm_mode_32 3")
        lines.append("  .amdhsa_float_denorm_mode_16_64 3")

        lines.append(".end_amdhsa_kernel")
        lines.append(".text\n")

        # SRD constant
        lines.append("# SRD upper word (gfx9xx): data_format=4 => 0x20000")
        lines.append(f".set Srd127_96, {m.SRD127_96}\n")

        # Kernel label
        lines.append(f"{m.name}:")

        return lines

    def emit_epilogue(self) -> List[str]:
        """
        Emit kernel epilogue with YAML metadata.

        Returns list of assembly lines for the epilogue.
        """
        m = self.metadata
        lines = []

        # Empty line after s_endpgm
        lines.append("")

        # Build YAML args
        args_yaml = []
        for i in range(m.num_args):
            args_yaml.append(
                f"""      - .name: arg{i}_ptr
        .size: 8
        .offset: {i*8}
        .value_kind: global_buffer
        .value_type: i8*"""
            )
        args_yaml_string = "\n".join(args_yaml)

        amdhsa_minor = "2" if m.codeobj == "5" else "1"
        wg_x, wg_y, wg_z = normalize_wg_size(m.wg_size)

        metadata = f"""
.amdgpu_metadata
---
amdhsa.version:
  - 1
  - {amdhsa_minor}
amdhsa.kernels:
  - .name: {m.name}
    .symbol: '{m.name}.kd'
    .language: OpenCL C
    .language_version:
      - 1
      - 2
    .kernarg_segment_size: {m.num_args * 8}
    .group_segment_fixed_size: {m.lds_size_bytes}
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .wavefront_size: {m.subgroup_size}
    .sgpr_count: {m.sgprs_used}
    .vgpr_count: {m.vgprs_used}
    .max_flat_workgroup_size: {wg_x * wg_y * wg_z}
    .args:
{args_yaml_string}
...
.end_amdgpu_metadata"""

        lines.append(metadata)
        return lines

    def _emit_workgroup_id_sgprs(self, lines: List[str]) -> None:
        """Emit workgroup ID SGPR configuration."""
        m = self.metadata
        if m.needs_wgid_x:
            lines.append("  .amdhsa_system_sgpr_workgroup_id_x 1")
        if m.needs_wgid_y:
            lines.append("  .amdhsa_system_sgpr_workgroup_id_y 1")
        if m.needs_wgid_z:
            lines.append("  .amdhsa_system_sgpr_workgroup_id_z 1")

    def _get_system_vgpr_workitem_id(self) -> int:
        """Determine system VGPR workitem ID mode based on workgroup size."""
        return system_vgpr_workitem_id_from_wg_size(self.metadata.wg_size)

    @staticmethod
    def patch_resource_usage(
        lines: List[str],
        vgprs_used: int,
        sgprs_used: int,
        agprs_used: int,
        lds_size_bytes: int,
        targetid: str,
    ) -> List[str]:
        """
        Patch resource usage in the prologue lines.

        This is called after compilation to fill in the actual values.
        """
        vgpr_granularity, sgpr_granularity = get_register_granularity(targetid)

        # Round up to allocation granularity
        vgprs_used = (
            (vgprs_used + vgpr_granularity - 1) // vgpr_granularity
        ) * vgpr_granularity
        sgprs_used = (
            (sgprs_used + sgpr_granularity - 1) // sgpr_granularity
        ) * sgpr_granularity

        # Compute accum_offset
        accum_offset = max(4, vgprs_used)

        # Handle AGPRs
        if agprs_used > 0:
            agprs_used = (
                (agprs_used + vgpr_granularity - 1) // vgpr_granularity
            ) * vgpr_granularity
            total_arch_vgprs = accum_offset + agprs_used
            vgprs_used = max(vgprs_used, total_arch_vgprs)

        # Patch the lines
        txt = "\n".join(lines)
        txt = txt.replace(
            "  .amdhsa_next_free_vgpr 0", f"  .amdhsa_next_free_vgpr {vgprs_used}"
        )
        txt = txt.replace(
            "  .amdhsa_accum_offset 0", f"  .amdhsa_accum_offset {accum_offset}"
        )
        txt = txt.replace(
            "  .amdhsa_next_free_sgpr 0", f"  .amdhsa_next_free_sgpr {sgprs_used}"
        )
        txt = txt.replace(
            "  .amdhsa_group_segment_fixed_size 0",
            f"  .amdhsa_group_segment_fixed_size {lds_size_bytes}",
        )

        return txt.splitlines()


def create_metadata(
    name: str,
    targetid: str,
    codeobj: str = "5",
    wg_size: Tuple[int, int, int] = (64, 1, 1),
    subgroup_size: int = 64,
    needs_wgid: Tuple[bool, bool, bool] = (False, False, False),
    num_args: int = 0,
    kernarg_preload_length: int = 0,
) -> KernelMetadata:
    """Create kernel metadata from parameters.

    Args:
        kernarg_preload_length: Number of SGPRs to preload from kernarg segment.
            For pointer args, use 2 SGPRs per pointer (8 bytes each).
            E.g., GEMM with 3 pointers (A, B, C) = 6 SGPRs.
    """
    return KernelMetadata(
        name=name,
        targetid=targetid,
        codeobj=codeobj,
        wg_size=normalize_wg_size(wg_size),
        subgroup_size=subgroup_size,
        needs_wgid_x=needs_wgid[0],
        needs_wgid_y=needs_wgid[1],
        needs_wgid_z=needs_wgid[2],
        num_args=num_args,
        kernarg_preload_length=kernarg_preload_length,
    )
