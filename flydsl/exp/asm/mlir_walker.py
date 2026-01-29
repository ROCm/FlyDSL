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

from .ir_imports import (
    affine_d,
    amdgpu_d,
    arith_d,
    func_d,
    gpu_d,
    math_d,
    memref_d,
    rocdl_d,
    scf_d,
    stream_d,
    vector_d,
    MemRefType,
)

from .kernel_model import KernelInfo
from .handlers import OperationHandlers
from .gather_to_shared import analyze_g2s_region, precreate_g2s_srds
from .kernel_compilation_context import KernelCompilationContext
from .handlers_shared import ssa
from .kernel_model import BindingUse, MemRefInfo
from .utils import parse_memref_type_from_obj


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

    def interpret_func(self, fn) -> KernelInfo:
        # func.func uses `entry_block`; gpu.func does not in current bindings.
        entry_block = (
            fn.entry_block
            if hasattr(fn, "entry_block")
            else fn.operation.regions[0].blocks[0]
        )

        kernel_name = (
            fn.sym_name.value
            if hasattr(fn, "sym_name") and hasattr(fn.sym_name, "value")
            else str(fn)
        )
        kernel_info = KernelInfo(name=kernel_name)

        kernel_info.arg_ssa_order = [ssa(arg) for arg in entry_block.arguments]

        # Map kernel arguments to runtime registers.
        #
        # The ASM backend prologue loads each kernarg as a 64-bit quantity into an
        # SGPR pair. For scalar (non-ptr/non-memref) arguments that need the low
        # 32 bits for index arithmetic, we materialize a VGPR copy.
        #
        # IMPORTANT: Skip memref and pointer types - these are handled by
        # handlers_memory.py (handle_llvm_ptr_op, etc.) which directly uses the
        # SGPR pair for buffer resource descriptors.
        try:
            from .kernel_ir import KInstr, KSReg
            from .instruction_registry import Instruction

            for arg_idx, arg in enumerate(entry_block.arguments):
                # Skip memref and pointer types - handled by handlers_memory.py
                if isinstance(arg.type, MemRefType):
                    continue
                type_str = str(arg.type)
                if "ptr" in type_str or "memref" in type_str.lower():
                    continue
                    
                pair = self.kernel_ctx.get_kernarg_pair(int(arg_idx))
                if pair is None or not isinstance(pair.base_reg, KSReg):
                    continue
                v = self.kernel_ctx.vreg()
                self.kernel_ctx.program.emit(
                    KInstr(
                        Instruction.V_MOV_B32,
                        defs=(v,),
                        uses=(pair.base_reg,),
                        comment=f"kernarg{arg_idx} lo",
                    )
                )
                self.kernel_ctx.ssa_to_reg[ssa(arg)] = (v,)
        except Exception:
            pass

        # Direct memref argument mapping (GPU dialect compatibility).
        for arg_idx, arg in enumerate(entry_block.arguments):
            if isinstance(arg.type, MemRefType):
                memref_ssa = ssa(arg)
                binding_use = kernel_info.subspans.setdefault(
                    memref_ssa, BindingUse(memref_ssa, arg_idx)
                )
                if not binding_use.memref_info:
                    shape, strides, elem_bytes = parse_memref_type_from_obj(arg.type)
                    binding_use.memref_info = MemRefInfo(shape, strides, elem_bytes)

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
        self._walk_block(entry_block, kernel_info)

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
        elif isinstance(operation, arith_d.SubIOp):
            self.handlers.handle_arith_subi_op(operation, kernel_info)
        elif isinstance(operation, arith_d.AddFOp):
            self.handlers.handle_arith_addf_op(operation, kernel_info)
        elif isinstance(operation, arith_d.RemUIOp):
            self.handlers.handle_arith_remui_op(operation, kernel_info)
        elif isinstance(operation, arith_d.DivUIOp):
            self.handlers.handle_arith_divui_op(operation, kernel_info)
        elif isinstance(operation, arith_d.MulIOp):
            self.handlers.handle_arith_muli_op(operation, kernel_info)
        elif isinstance(operation, arith_d.IndexCastOp):
            self.handlers.handle_arith_index_cast_op(operation, kernel_info)
        elif isinstance(operation, arith_d.CmpIOp):
            self.handlers.handle_arith_cmpi_op(operation, kernel_info)
        elif isinstance(operation, arith_d.SelectOp):
            self.handlers.handle_arith_select_op(operation, kernel_info)
        elif isinstance(operation, arith_d.MulFOp) or str(operation.operation.name) == "arith.mulf" or str(getattr(operation, "name", "")) == "arith.mulf":
            self.handlers.handle_arith_mulf_op(operation, kernel_info)
        elif isinstance(operation, arith_d.SubFOp):
            self.handlers.handle_arith_subf_op(operation, kernel_info)
        elif isinstance(operation, arith_d.MaximumFOp):
            self.handlers.handle_arith_maximumf_op(operation, kernel_info)
        elif isinstance(operation, arith_d.MinimumFOp):
            self.handlers.handle_arith_minimumf_op(operation, kernel_info)
        elif isinstance(operation, arith_d.DivFOp):
            self.handlers.handle_arith_divf_op(operation, kernel_info)
        elif hasattr(arith_d, "DivSIOp") and isinstance(operation, arith_d.DivSIOp):
            self.handlers.handle_arith_divsi_op(operation, kernel_info)
        elif hasattr(arith_d, "RemSIOp") and isinstance(operation, arith_d.RemSIOp):
            self.handlers.handle_arith_remsi_op(operation, kernel_info)
        elif isinstance(operation, arith_d.ExtFOp) or str(operation.operation.name) == "arith.extf" or str(getattr(operation, "name", "")) == "arith.extf":
            self.handlers.handle_arith_extf_op(operation, kernel_info)
        elif isinstance(operation, arith_d.TruncFOp) or str(operation.operation.name) == "arith.truncf" or str(getattr(operation, "name", "")) == "arith.truncf":
            self.handlers.handle_arith_truncf_op(operation, kernel_info)
        elif isinstance(operation, arith_d.BitcastOp):
            self.handlers.handle_arith_bitcast_op(operation, kernel_info)
        elif hasattr(arith_d, "XOrIOp") and isinstance(operation, arith_d.XOrIOp):
            self.handlers.handle_arith_xori_op(operation, kernel_info)
        elif hasattr(arith_d, "ShRUIOp") and isinstance(operation, arith_d.ShRUIOp):
            self.handlers.handle_arith_shrui_op(operation, kernel_info)
        elif hasattr(arith_d, "AndIOp") and isinstance(operation, arith_d.AndIOp):
            self.handlers.handle_arith_andi_op(operation, kernel_info)
        elif hasattr(arith_d, "OrIOp") and isinstance(operation, arith_d.OrIOp):
            self.handlers.handle_arith_ori_op(operation, kernel_info)
        elif hasattr(arith_d, "ShLIOp") and isinstance(operation, arith_d.ShLIOp):
            self.handlers.handle_arith_shli_op(operation, kernel_info)
        elif isinstance(operation, gpu_d.ThreadIdOp):
            self.handlers.handle_gpu_thread_id_op(operation, kernel_info)
        elif isinstance(operation, gpu_d.BlockIdOp):
            self.handlers.handle_gpu_block_id_op(operation, kernel_info)
        elif isinstance(operation, gpu_d.BlockDimOp):
            self.handlers.handle_gpu_block_dim_op(operation, kernel_info)
        elif affine_d is not None and isinstance(operation, affine_d.AffineApplyOp):
            self.handlers.handle_affine_apply_op(operation, kernel_info)
        elif isinstance(operation, vector_d.LoadOp):
            self.handlers.handle_vector_load_op(operation, kernel_info)
        elif isinstance(operation, vector_d.StoreOp):
            self.handlers.handle_vector_store_op(operation, kernel_info)
        elif hasattr(vector_d, "ShuffleOp") and isinstance(operation, vector_d.ShuffleOp):
            self.handlers.handle_vector_shuffle_op(operation, kernel_info)
        elif hasattr(vector_d, "ReductionOp") and isinstance(operation, vector_d.ReductionOp):
            self.handlers.handle_vector_reduction_op(operation, kernel_info)
        elif (
            (hasattr(vector_d, "BroadcastOp") and isinstance(operation, vector_d.BroadcastOp))
            or str(operation.operation.name) == "vector.broadcast"
        ):
            self.handlers.handle_vector_broadcast_op(operation, kernel_info)
        elif (
            (hasattr(vector_d, "BitCastOp") and isinstance(operation, vector_d.BitCastOp))
            or (hasattr(vector_d, "BitcastOp") and isinstance(operation, vector_d.BitcastOp))
            or str(operation.operation.name) == "vector.bitcast"
        ):
            self.handlers.handle_vector_bitcast_op(operation, kernel_info)
        elif (
            (hasattr(vector_d, "ExtractOp") and isinstance(operation, vector_d.ExtractOp))
            or str(operation.operation.name) == "vector.extract"
        ):
            self.handlers.handle_vector_extract_op(operation, kernel_info)
        elif (
            (hasattr(vector_d, "FromElementsOp") and isinstance(operation, vector_d.FromElementsOp))
            or str(operation.operation.name) == "vector.from_elements"
        ):
            self.handlers.handle_vector_from_elements_op(operation, kernel_info)
        elif (
            hasattr(memref_d, "ExtractAlignedPointerAsIndexOp")
            and isinstance(operation, memref_d.ExtractAlignedPointerAsIndexOp)
        ):
            self.handlers.handle_memref_extract_aligned_pointer_as_index_op(
                operation, kernel_info
            )
        elif str(operation.operation.name) == "llvm.inttoptr":
            self.handlers.handle_llvm_inttoptr_op(operation, kernel_info)
        elif str(operation.operation.name) == "llvm.call_intrinsic":
            self.handlers.handle_llvm_call_intrinsic_op(operation, kernel_info)
        elif str(operation.operation.name) == "rocdl.make.buffer.rsrc":
            self.handlers.handle_rocdl_make_buffer_rsrc_op(operation, kernel_info)
        elif str(operation.operation.name) == "rocdl.raw.ptr.buffer.load":
            self.handlers.handle_rocdl_raw_ptr_buffer_load_op(operation, kernel_info)
        elif str(operation.operation.name) == "rocdl.raw.ptr.buffer.store":
            self.handlers.handle_rocdl_raw_ptr_buffer_store_op(operation, kernel_info)
        elif str(operation.operation.name) == "rocdl.ds_bpermute":
            self.handlers.handle_rocdl_ds_bpermute_op(operation, kernel_info)
        elif str(operation.operation.name).startswith("rocdl.mfma."):
            self.handlers.handle_rocdl_mfma_op(operation, kernel_info)
        elif str(operation.operation.name) in ("rocdl.sched.barrier", "rocdl.sched.group.barrier"):
            self.handlers.handle_rocdl_sched_barrier_op(operation, kernel_info)
        elif isinstance(operation, scf_d.YieldOp):
            self.handlers.handle_scf_yield_op(operation, kernel_info)
        elif isinstance(operation, amdgpu_d.MFMAOp):
            self.handlers.handle_mfma_op(operation, kernel_info)
        elif isinstance(operation, amdgpu_d.LDSBarrierOp):
            self.handlers.handle_lds_barrier_op(operation, kernel_info)
        elif isinstance(operation, gpu_d.BarrierOp):
            self.handlers.handle_barrier_op(operation, kernel_info)
        elif isinstance(operation, gpu_d.ShuffleOp):
            self.handlers.handle_gpu_shuffle_op(operation, kernel_info)
        elif isinstance(operation, memref_d.ViewOp):
            self.handlers.handle_view_op(operation, kernel_info)
        elif hasattr(memref_d, "GetGlobalOp") and isinstance(operation, memref_d.GetGlobalOp):
            self.handlers.handle_memref_get_global_op(operation, kernel_info)
        elif isinstance(operation, memref_d.AllocOp):
            self.handlers.handle_alloc_op(operation, kernel_info)
        elif isinstance(operation, memref_d.AllocaOp):
            self.handlers.handle_memref_alloca_op(operation, kernel_info)
        elif isinstance(operation, memref_d.CastOp):
            self.handlers.handle_memref_cast_op(operation, kernel_info)
        elif isinstance(operation, memref_d.ReinterpretCastOp):
            self.handlers.handle_memref_reinterpret_cast_op(operation, kernel_info)
        elif isinstance(operation, memref_d.LoadOp):
            self.handlers.handle_memref_load_op(operation, kernel_info)
        elif isinstance(operation, memref_d.StoreOp):
            self.handlers.handle_memref_store_op(operation, kernel_info)
        elif isinstance(operation, stream_d.BindingSubspanOp):
            self.handlers.handle_stream_binding_subspan_op(operation, kernel_info)
        elif isinstance(operation, scf_d.ForOp):
            self.handlers.handle_scf_for_op(operation, kernel_info)
        elif isinstance(operation, scf_d.IfOp):
            self.handlers.handle_scf_if_op(operation, kernel_info)
        elif math_d is not None and isinstance(operation, math_d.Exp2Op):
            self.handlers.handle_math_exp2_op(operation, kernel_info)
        elif math_d is not None and isinstance(operation, math_d.SqrtOp):
            self.handlers.handle_math_sqrt_op(operation, kernel_info)
        elif math_d is not None and isinstance(operation, math_d.RsqrtOp):
            self.handlers.handle_math_rsqrt_op(operation, kernel_info)
        elif math_d is not None and isinstance(operation, math_d.AbsFOp):
            self.handlers.handle_math_absf_op(operation, kernel_info)
        elif math_d is not None and isinstance(operation, math_d.CopySignOp):
            self.handlers.handle_math_copysign_op(operation, kernel_info)
        elif isinstance(operation, arith_d.FPToSIOp):
            self.handlers.handle_arith_fptosi_op(operation, kernel_info)
        elif isinstance(operation, arith_d.SIToFPOp):
            self.handlers.handle_arith_sitofp_op(operation, kernel_info)
        elif isinstance(operation, vector_d.ExtractStridedSliceOp):
            self.handlers.handle_vector_extract_strided_slice_op(operation, kernel_info)
        # Critical operations for gather_to_lds support
        elif hasattr(amdgpu_d, "GatherToLDSOp") and isinstance(
            operation, amdgpu_d.GatherToLDSOp
        ):
            self.handlers.g2s.handle_gather_to_lds_op(operation, kernel_info)
        elif hasattr(amdgpu_d, "FatRawBufferCastOp") and isinstance(
            operation, amdgpu_d.FatRawBufferCastOp
        ):
            self.handlers.handle_fat_raw_buffer_cast_op(operation, kernel_info)
        elif rocdl_d is not None and isinstance(operation, rocdl_d.ReadfirstlaneOp):
            self.handlers.handle_readfirstlane_op(operation, kernel_info)
        elif hasattr(gpu_d, "SubgroupBroadcastOp") and isinstance(
            operation, gpu_d.SubgroupBroadcastOp
        ):
            self.handlers.handle_subgroup_broadcast_op(operation, kernel_info)
        elif rocdl_d is not None and isinstance(operation, rocdl_d.SWaitcntOp):
            self.handlers.handle_s_waitcnt_op(operation, kernel_info)
        else:
            pass
