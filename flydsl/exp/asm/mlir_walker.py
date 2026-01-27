# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
MLIR IR Walker for ASM code generation.

Implements LLVM-style vmcnt optimization by pre-computing M0 base values
BEFORE barriers, then using scalar operations for M0 assignment.
"""

from wave_lang.support.ir_imports import (
    affine_d,
    amdgpu_d,
    arith_d,
    func_d,
    gpu_d,
    memref_d,
    rocdl_d,
    scf_d,
    stream_d,
    vector_d,
)

from .kernel_model import KernelInfo
from .handlers import OperationHandlers
from .gather_to_shared import analyze_g2s_region, precreate_g2s_srds
from .kernel_compilation_context import KernelCompilationContext


class IRWalker:
    def __init__(self, kernel_ctx: KernelCompilationContext):
        """
        Initialize IRWalker with kernel compilation context.

        Args:
            kernel_ctx: KernelCompilationContext - the source of truth for
                        instruction emission and register allocation.
        """
        self.kernel_ctx = kernel_ctx

        # Supporting fields
        self.last_vmem_ticket = None  # Used for wait count computation
        self._lds_view_base_bytes: dict[str, int] = {}  # LDS view offsets
        # Initialize operation handlers
        self.handlers = OperationHandlers(self)

    @property
    def unified(self):
        """Return the unified emitter for instruction emission."""
        return self.kernel_ctx.unified

    def interpret_func(self, fn: func_d.FuncOp) -> KernelInfo:
        kernel_info = KernelInfo(name=fn.sym_name.value)

        entry_block = fn.entry_block
        kernel_info.arg_ssa_order = [str(arg) for arg in entry_block.arguments]

        # NOTE: translation_info parsing is centralized in mlir_analysis and used
        # by KernelModuleCompiler. Avoid re-parsing here to prevent duplication.
        #
        # Source wg/subgroup from the kernel compilation context (single source of truth).
        if getattr(self.kernel_ctx, "wg_size", None):
            kernel_info.wg_size = tuple(self.kernel_ctx.wg_size)
        if getattr(self.kernel_ctx, "subgroup_size", None):
            kernel_info.subgroup_size = int(self.kernel_ctx.subgroup_size)

        # Update kernel context with actual bounds from MLIR
        # This enables correct algebraic simplifications based on workgroup size
        if self.kernel_ctx is not None:
            self.kernel_ctx.update_bounds_from_kernel_info(kernel_info)

        # Walk operations and fill environment + accesses
        self._walk_block(fn.entry_block, kernel_info)

        return kernel_info

    def _walk_block(self, block, kernel_info: KernelInfo):
        """Walk operations in a block and dispatch to handlers.

        For g2s regions: dispatches ops before first g2s, pre-computes M0,
        then dispatches remaining ops.
        """
        ops = list(block.operations)

        # Analyze for g2s scheduling
        schedule = analyze_g2s_region(ops)

        if schedule is None:
            # No g2s ops - dispatch all ops sequentially
            for op in ops:
                self._dispatch_operation(op, kernel_info)
        else:
            # Dispatch ops before first g2s (populates index_env)
            for i in range(schedule.first_g2s_idx):
                self._dispatch_operation(ops[i], kernel_info)

            # Pre-create G2S SRD copies to ensure they're allocated before the loop
            # This prevents the second SRD copy from overwriting the first's source
            # Skip if already inside a loop (handled by handle_scf_for_op)
            if not getattr(self, "_inside_loop", False):
                precreate_g2s_srds(schedule, kernel_info, self.handlers)

            # Dispatch remaining ops
            for i in range(schedule.first_g2s_idx, len(ops)):
                self._dispatch_operation(ops[i], kernel_info)

    def _dispatch_operation(self, operation, kernel_info: KernelInfo):
        """Dispatch a single operation to its appropriate handler."""
        if isinstance(operation, arith_d.ConstantOp):
            self.handlers.handle_arith_constant_op(operation, kernel_info)
        elif isinstance(operation, arith_d.AddIOp):
            self.handlers.handle_arith_addi_op(operation, kernel_info)
        elif isinstance(operation, arith_d.MulIOp):
            self.handlers.handle_arith_muli_op(operation, kernel_info)
        elif isinstance(operation, arith_d.IndexCastOp):
            self.handlers.handle_arith_index_cast_op(operation, kernel_info)
        elif isinstance(operation, gpu_d.ThreadIdOp):
            self.handlers.handle_gpu_thread_id_op(operation, kernel_info)
        elif isinstance(operation, gpu_d.BlockIdOp):
            self.handlers.handle_gpu_block_id_op(operation, kernel_info)
        elif isinstance(operation, affine_d.AffineApplyOp):
            self.handlers.handle_affine_apply_op(operation, kernel_info)
        elif isinstance(operation, vector_d.LoadOp):
            self.handlers.handle_vector_load_op(operation, kernel_info)
        elif isinstance(operation, vector_d.StoreOp):
            self.handlers.handle_vector_store_op(operation, kernel_info)
        elif isinstance(operation, amdgpu_d.MFMAOp):
            self.handlers.handle_mfma_op(operation, kernel_info)
        elif isinstance(operation, amdgpu_d.LDSBarrierOp):
            self.handlers.handle_lds_barrier_op(operation, kernel_info)
        elif isinstance(operation, memref_d.ViewOp):
            self.handlers.handle_view_op(operation, kernel_info)
        elif isinstance(operation, memref_d.AllocOp):
            self.handlers.handle_alloc_op(operation, kernel_info)
        elif isinstance(operation, memref_d.CastOp):
            self.handlers.handle_memref_cast_op(operation, kernel_info)
        elif isinstance(operation, memref_d.ReinterpretCastOp):
            self.handlers.handle_memref_reinterpret_cast_op(operation, kernel_info)
        elif isinstance(operation, stream_d.BindingSubspanOp):
            self.handlers.handle_stream_binding_subspan_op(operation, kernel_info)
        elif isinstance(operation, scf_d.ForOp):
            self.handlers.handle_scf_for_op(operation, kernel_info)
        elif isinstance(operation, vector_d.ExtractStridedSliceOp):
            self.handlers.handle_vector_extract_strided_slice_op(operation, kernel_info)
        # Critical operations for gather_to_lds support
        elif isinstance(operation, amdgpu_d.GatherToLDSOp):
            self.handlers.g2s.handle_gather_to_lds_op(operation, kernel_info)
        elif isinstance(operation, amdgpu_d.FatRawBufferCastOp):
            self.handlers.handle_fat_raw_buffer_cast_op(operation, kernel_info)
        elif isinstance(operation, rocdl_d.ReadfirstlaneOp):
            self.handlers.handle_readfirstlane_op(operation, kernel_info)
        elif isinstance(operation, gpu_d.SubgroupBroadcastOp):
            self.handlers.handle_subgroup_broadcast_op(operation, kernel_info)
        elif isinstance(operation, rocdl_d.SWaitcntOp):
            self.handlers.handle_s_waitcnt_op(operation, kernel_info)
        else:
            pass
