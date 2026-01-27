# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Gather-to-Shared (G2S) handling for buffer_load...lds instructions.

This module contains all functionality related to gather_to_lds operations:
- Scheduling
- SRD tracing through memref cast chains
- VGPR offset computation
- The main gather_to_lds handler
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Any, Tuple, TYPE_CHECKING

import sympy

from wave_lang.support.ir_imports import amdgpu_d, MemRefType, VectorType

# Instruction classes removed - using unified emitter
from .utils import parse_vector_type_from_obj, parse_memref_type_from_obj

if TYPE_CHECKING:
    from .kernel_model import KernelInfo
    from .handlers import OperationHandlers

# =============================================================================
# Constants
# =============================================================================

# Default LDS column count for 2D memrefs
_DEFAULT_LDS_COLS = 32

# Default cache swizzle stride (LLVM uses 0x40400000 = stride 64)
_DEFAULT_CACHE_SWIZZLE_STRIDE = 64

# Cache swizzle enable bit (bit 30)
_CACHE_SWIZZLE_ENABLE = 0x40000000

# SRD word2: maximum buffer size
_SRD_MAX_BUFFER_SIZE = 0x7FFFFFFD

# SRD word3: data_format=4 (32-bit), num_format=7, add_tid_enable=1
_SRD_WORD3_LDS = 0x27000

# Element size scale for f16 (bytes per element)
_F16_ELEM_BYTES = 2


# =============================================================================
# Scheduling functions
# =============================================================================


@dataclass
class G2SSchedule:
    """Scheduling info for g2s region."""

    first_g2s_idx: int
    g2s_ops: List[Any]


def analyze_g2s_region(ops: List[Any]) -> Optional[G2SSchedule]:
    """Find gather_to_lds operations and their scheduling info."""
    g2s_ops = []
    first_g2s_idx = None

    for i, op in enumerate(ops):
        if isinstance(op, amdgpu_d.GatherToLDSOp):
            if first_g2s_idx is None:
                first_g2s_idx = i
            g2s_ops.append(op)

    if not g2s_ops:
        return None

    return G2SSchedule(first_g2s_idx=first_g2s_idx, g2s_ops=g2s_ops)


def precreate_g2s_srds(
    schedule: G2SSchedule,
    kernel_info: "KernelInfo",
    handlers: "OperationHandlers",
) -> None:
    """
    Pre-create G2S SRD copies before any buffer_load...lds instructions.

    This must be called BEFORE the loop body is walked to ensure:
    1. All G2S SRD copies are created before the loop
    2. The original kernel argument SRDs are not overwritten during SRD copy
    3. The SRD copies can be reused across loop iterations

    This is critical for correct behavior when the register allocator reuses
    physical registers - we need all SRD copies to be made while the original
    SRDs are still live.
    """
    walker = handlers.walker

    # Initialize the SRD copies cache
    if not hasattr(walker, "_lds_srd_copies"):
        walker._lds_srd_copies = {}

    # Pre-create SRD copies for all unique source memrefs
    seen_memrefs = set()
    for g2s_op in schedule.g2s_ops:
        src_memref_ssa = str(g2s_op.operands[0])
        if src_memref_ssa in seen_memrefs:
            continue
        seen_memrefs.add(src_memref_ssa)

        # Trace to find the original SRD
        orig_srd_regs, original_ssa = handlers.g2s._trace_memref_to_srd(
            src_memref_ssa, kernel_info
        )

        if orig_srd_regs is None:
            continue

        # This will create and cache the G2S SRD copy
        # If already cached (e.g., from a previous call), it will just return the cached copy
        handlers.g2s._get_or_create_srd(orig_srd_regs, src_memref_ssa)


# =============================================================================
# G2SHandler
# =============================================================================


class G2SHandler:
    """Handles gather_to_lds operations for the ASM backend."""

    def __init__(self, handlers: "OperationHandlers"):
        self.handlers = handlers
        self.walker = handlers.walker

    # -------------------------------------------------------------------------
    # SRD tracing
    # -------------------------------------------------------------------------

    def _trace_memref_to_srd(self, memref_ssa: str, kernel_info: "KernelInfo"):
        """Trace through cast chain to find the original memref with an SRD.

        Returns:
            (KRegRange, original_ssa) - virtual SGPR range
        """
        ctx = self.walker.kernel_ctx
        srd_range = ctx.get_srd(memref_ssa)
        if srd_range is not None:
            return srd_range, memref_ssa

        # Trace through cast chain to find original memref
        fat_sources = getattr(self.walker, "_fat_buffer_sources", {})
        cast_sources = getattr(self.walker, "_memref_cast_sources", {})
        current = memref_ssa
        visited = set()
        while current and current not in visited:
            visited.add(current)

            # Check if we have an SRD for this memref
            srd_range = ctx.get_srd(current)
            if srd_range is not None:
                return srd_range, current

            # Check if this is a kernel argument that needs SRD setup
            if current in kernel_info.subspans:
                binding = kernel_info.subspans[current]
                if binding.arg_index >= 0 and binding.memref_info:
                    # Set up SRD for this kernel argument
                    limit_bytes = self.handlers._compute_buffer_size(
                        binding.memref_info
                    )
                    srd_range = ctx.ensure_srd(current, binding.arg_index, limit_bytes)
                    return srd_range, current

            # Follow cast chain
            if current in fat_sources:
                info = fat_sources[current]
                current = (
                    info.get("source_ssa", current) if isinstance(info, dict) else info
                )
            elif current in cast_sources:
                current = cast_sources[current]
            else:
                break
        return None, None

    # -------------------------------------------------------------------------
    # SRD allocation and emission
    # -------------------------------------------------------------------------

    def _allocate_aligned_srd(self):
        """Allocate 4-aligned SGPRs for an SRD.

        Returns:
            KRegRange of 4 virtual SGPRs
        """
        ctx = self.walker.kernel_ctx
        return ctx.program.alloc_sreg_range(4, alignment=4)

    def _compute_cache_swizzle_bits(self, src_memref_ssa: str) -> int:
        """Compute cache swizzle bits from fat_raw_buffer_cast info."""
        fat_buffer_info = getattr(self.walker, "_fat_buffer_sources", {}).get(
            src_memref_ssa
        )
        stride = _DEFAULT_CACHE_SWIZZLE_STRIDE
        if fat_buffer_info and isinstance(fat_buffer_info, dict):
            stride = fat_buffer_info.get(
                "cache_swizzle_stride", _DEFAULT_CACHE_SWIZZLE_STRIDE
            )

        if stride > 0:
            return _CACHE_SWIZZLE_ENABLE | (1 << (int(math.log2(stride)) + 16))
        return 0

    def _emit_srd_copy(
        self,
        new_srd_range,  # KRegRange - new SRD to fill
        orig_srd_range,  # KRegRange - original SRD to copy from
        cache_swizzle_bits: int,
    ) -> None:
        """Emit SRD copy using kernel IR instructions.

        Uses a single pseudo-instruction to define and initialize the SRD.
        The generator will expand this into individual s_mov_b32 instructions.
        """
        from .kernel_ir import KInstr, KImm

        ctx = self.walker.kernel_ctx
        program = ctx.program

        # Emit a single pseudo-instruction that defines the range and carries
        # all the info needed to render the copy
        # Format: _g2s_srd_copy new_range, orig_range, cache_swizzle_bits
        program.emit(
            KInstr(
                "_g2s_srd_copy",
                (new_srd_range,),  # Define the new range
                (
                    orig_srd_range,
                    KImm(cache_swizzle_bits),
                ),  # Source range and swizzle bits
                comment=f"G2S SRD copy",
            )
        )

    def _get_or_create_srd(self, orig_srd_regs: tuple, src_memref_ssa: str):
        """Get cached SRD or create a new one for gather_to_lds.

        Returns:
            In legacy mode: Tuple[int, int, int, int] of physical SGPR indices
            In kernel IR mode: KRegRange of 4 SGPRs
        """
        cache_key = f"lds_srd_copy_{orig_srd_regs[0]}"
        srd_copies = getattr(self.walker, "_lds_srd_copies", None)
        if srd_copies is None:
            self.walker._lds_srd_copies = {}
            srd_copies = self.walker._lds_srd_copies

        if cache_key in srd_copies:
            return srd_copies[cache_key]

        srd_regs = self._allocate_aligned_srd()
        cache_swizzle_bits = self._compute_cache_swizzle_bits(src_memref_ssa)
        self._emit_srd_copy(srd_regs, orig_srd_regs, cache_swizzle_bits)
        srd_copies[cache_key] = srd_regs
        return srd_regs

    # -------------------------------------------------------------------------
    # Offset computation
    # -------------------------------------------------------------------------

    def _compute_vgpr_offset(
        self, index_ssa: str, kernel_info: "KernelInfo", scale_bytes: int
    ):
        """Compute a VGPR containing the offset value, scaled to bytes.

        Returns:
            KVReg (virtual VGPR)
        """
        from .kernel_ir import KInstr, KImm

        value = kernel_info.index_env.get(index_ssa)
        ctx = self.walker.kernel_ctx

        if isinstance(value, int):
            voffset = ctx.vreg()
            ctx.program.emit(
                KInstr(
                    "v_mov_b32",
                    (voffset,),
                    (KImm(value * scale_bytes),),
                    comment="offset",
                )
            )
            return voffset

        if isinstance(value, sympy.Expr):
            scaled_expr = value * scale_bytes if scale_bytes != 1 else value
            return ctx.expr_emitter.get_or_emit(scaled_expr)

        raise ValueError(f"Cannot compute offset for {index_ssa}: value={value}")

    def _compute_vaddr(self, src_index_ssa: str, kernel_info: "KernelInfo") -> int:
        """Compute vaddr VGPR for global memory offset."""
        return self._compute_vgpr_offset(
            src_index_ssa, kernel_info, scale_bytes=_F16_ELEM_BYTES
        )

    # -------------------------------------------------------------------------
    # M0 computation
    # -------------------------------------------------------------------------

    def _get_lds_layout(self, dst_memref_type) -> Tuple[int, int]:
        """Extract LDS columns and element bytes from destination memref type."""
        if not isinstance(dst_memref_type, MemRefType):
            raise ValueError(f"Unsupported destination memref type: {dst_memref_type}")
        shape, _, elem_bytes = parse_memref_type_from_obj(dst_memref_type)
        lds_cols = shape[1] if len(shape) >= 2 else _DEFAULT_LDS_COLS
        return lds_cols, elem_bytes

    def _emit_m0(
        self,
        operation,
        kernel_info: "KernelInfo",
        lds_base_bytes: int,
        lds_cols: int,
        elem_bytes: int,
    ) -> None:
        """Emit M0 register setup for LDS destination."""
        self._emit_m0_kernel_ir(
            operation, kernel_info, lds_base_bytes, lds_cols, elem_bytes
        )

    def _emit_m0_kernel_ir(
        self,
        operation,
        kernel_info: "KernelInfo",
        lds_base_bytes: int,
        lds_cols: int,
        elem_bytes: int,
    ) -> None:
        """Emit M0 register setup using kernel IR."""
        from .kernel_ir import KInstr, KImm, KSpecialReg
        from .instruction_registry import Instruction

        ctx = self.walker.kernel_ctx
        program = ctx.program
        M0 = KSpecialReg("m0")

        # Compute M0 inline from row/col indices
        dst_row = kernel_info.index_env.get(str(operation.operands[3]))
        dst_col = kernel_info.index_env.get(str(operation.operands[4]))

        # Build M0 expression: lds_base + (row * cols + col) * elem_bytes
        row = (
            dst_row if isinstance(dst_row, sympy.Expr) else sympy.Integer(dst_row or 0)
        )
        col = (
            dst_col if isinstance(dst_col, sympy.Expr) else sympy.Integer(dst_col or 0)
        )
        m0_expr = sympy.simplify(lds_base_bytes + (row * lds_cols + col) * elem_bytes)

        # Emit: either constant or expression -> readfirstlane -> M0
        if m0_expr.is_number:
            program.emit(
                KInstr(
                    Instruction.S_MOV_B32,
                    (M0,),
                    (KImm(int(m0_expr)),),
                    comment="LDS offset",
                )
            )
        else:
            # Compute expression to VGPR
            m0_vreg = ctx.expr_emitter.get_or_emit(m0_expr)
            # Read first lane to SGPR
            s_tmp = ctx.sreg()
            program.emit(KInstr(Instruction.V_READFIRSTLANE_B32, (s_tmp,), (m0_vreg,)))
            # Move to M0
            program.emit(KInstr(Instruction.S_MOV_B32, (M0,), (s_tmp,)))

    def _emit_buffer_load(self, transfer_bytes: int, voffset, srd_range) -> None:
        """Emit buffer_load...lds instruction.

        Args:
            transfer_bytes: Number of bytes to transfer (4 or 16)
            voffset: KVReg (virtual VGPR)
            srd_range: KRegRange (virtual SGPR range)
        """
        from .kernel_ir import KInstr, KImm

        ctx = self.walker.kernel_ctx
        program = ctx.program

        # Determine instruction mnemonic
        if transfer_bytes == 4:
            mnemonic = "buffer_load_dword"
        elif transfer_bytes == 16:
            mnemonic = "buffer_load_dwordx4"
        else:
            raise NotImplementedError(
                f"gather_to_lds: transfer_bytes={transfer_bytes} not supported. "
                f"Only 4B (buffer_load_dword) and 16B (buffer_load_dwordx4) are valid."
            )

        # buffer_load_dword_lds vaddr, srd, soffset offen lds
        # In kernel IR, this is a pseudo-instruction that combines buffer_load with LDS
        program.emit(
            KInstr(
                f"{mnemonic}_lds",
                (),  # No destination (LDS is the destination via M0)
                (voffset, srd_range, KImm(0)),  # vaddr, srd, soffset
                comment=f"gather {transfer_bytes}B",
            )
        )

    # -------------------------------------------------------------------------
    # Main handler
    # -------------------------------------------------------------------------

    def handle_gather_to_lds_op(
        self, operation: amdgpu_d.GatherToLDSOp, kernel_info: "KernelInfo"
    ):
        """Handle amdgpu.gather_to_lds - gather from global memory directly to LDS."""
        # Parse transfer type
        transfer_type = operation.attributes["transferType"].value
        if not isinstance(transfer_type, VectorType):
            raise ValueError(f"Unsupported transfer type: {transfer_type}")
        elem_count, elem_bytes, _ = parse_vector_type_from_obj(transfer_type)
        transfer_bytes = elem_count * elem_bytes

        # Get operands
        src_memref_ssa = str(operation.operands[0])
        src_index_ssa = str(operation.operands[1])
        dst_memref_ssa = str(operation.operands[2])

        # Find/create SRD
        orig_srd_regs, original_ssa = self._trace_memref_to_srd(
            src_memref_ssa, kernel_info
        )
        if orig_srd_regs is None:
            self.walker.kernel_ctx.comment(f"ERROR: no SRD found for {src_memref_ssa}")
            return

        srd_regs = self._get_or_create_srd(orig_srd_regs, src_memref_ssa)

        # Emit comment
        self.walker.kernel_ctx.comment(
            f"gather_to_lds: {transfer_bytes}B from {original_ssa} to LDS"
        )

        # Compute vaddr
        voffset = self._compute_vaddr(src_index_ssa, kernel_info)

        # Get LDS layout and emit M0
        lds_base_bytes = self.walker._lds_view_base_bytes.get(dst_memref_ssa, 0)
        lds_cols, elem_bytes = self._get_lds_layout(operation.operands[2].type)
        self._emit_m0(operation, kernel_info, lds_base_bytes, lds_cols, elem_bytes)

        # Emit buffer load
        self._emit_buffer_load(transfer_bytes, voffset, srd_regs)

        # Track g2s LDS views
        g2s_views = getattr(self.walker, "_g2s_lds_views", None)
        if g2s_views is None:
            self.walker._g2s_lds_views = set()
        self.walker._g2s_lds_views.add(dst_memref_ssa)
