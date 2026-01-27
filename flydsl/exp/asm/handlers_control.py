# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .ir_imports import arith_d, amdgpu_d, memref_d, rocdl_d, scf_d

from .handlers_shared import *
from .kernel_model import KernelInfo


class _ControlFlowHandlers:
    def handle_scf_for_op(self, operation: scf_d.ForOp, kernel_info: KernelInfo):
        """
        Handle scf.for operations - emit loop assembly code.

        Args:
            operation: The scf.for operation
            kernel_info: Kernel information for context
        """
        # Extract loop bounds
        lower_bound_ssa = ssa(operation.lowerBound)
        upper_bound_ssa = ssa(operation.upperBound)
        step_ssa = ssa(operation.step)

        # Get bounds from index_env (should be constants)
        if lower_bound_ssa not in kernel_info.index_env:
            raise ValueError(
                f"Loop lower bound {lower_bound_ssa} not found in index_env"
            )
        if upper_bound_ssa not in kernel_info.index_env:
            raise ValueError(
                f"Loop upper bound {upper_bound_ssa} not found in index_env"
            )
        if step_ssa not in kernel_info.index_env:
            raise ValueError(f"Loop step {step_ssa} not found in index_env")
        lower_bound = kernel_info.index_env[lower_bound_ssa]
        upper_bound = kernel_info.index_env[upper_bound_ssa]
        step = kernel_info.index_env[step_ssa]

        # Pre-create G2S SRDs BEFORE the loop starts
        # This is critical for correctness: if G2S operations are in the loop body,
        # we need to create all SRD copies before the loop header is emitted.
        # Otherwise, the SRD copy for matrix B can overwrite the original SRD for
        # matrix A, causing incorrect memory accesses in subsequent loop iterations.
        from .gather_to_shared import analyze_g2s_region, precreate_g2s_srds

        loop_body = operation.body
        loop_ops = list(loop_body.operations)
        g2s_schedule = analyze_g2s_region(loop_ops)
        if g2s_schedule is not None:
            # Pre-create G2S SRDs (these must be created before the loop)
            precreate_g2s_srds(g2s_schedule, kernel_info, self)

        # Kernel IR mode: use virtual registers
        from .kernel_ir import KVReg

        ctx = self.walker.kernel_ctx

        # Begin loop structure with virtual registers
        loop_ctx = ctx.begin_loop(lower_bound, upper_bound, step)

        # Get induction variable and map it to the loop counter SGPR
        loop_body = operation.body
        induction_var = loop_body.arguments[0]
        induction_var_ssa = ssa(induction_var)
        counter_sreg = loop_ctx["counter_sreg"]

        # Store mapping from SSA induction variable to SGPR
        # Store the virtual SGPR directly - the utils.py to_sympy() handles KSReg
        kernel_info.index_env[induction_var_ssa] = counter_sreg
        loop_ctx["induction_var_ssa"] = induction_var_ssa

        # Allocate and initialize VGPRs for iter_args (accumulators)
        num_iter_args = len(loop_body.arguments) - 1  # Exclude induction var
        iter_arg_ranges = ctx.alloc_accumulators(num_iter_args)

        # Track in SSA->reg map
        for i, arg in enumerate(loop_body.arguments[1:]):
            arg_ssa = ssa(arg)
            quad = iter_arg_ranges[i]
            # Store as tuple of individual regs for compatibility
            regs = tuple(KVReg(quad.base_reg.id + j) for j in range(4))
            ctx.ssa_to_reg[arg_ssa] = regs

        loop_ctx["iter_arg_ranges"] = iter_arg_ranges

        # Emit loop header
        ctx.emit_loop_header(loop_ctx)

        # Walk loop body (mark as inside loop to prevent duplicate M0/SRD setup)
        self.walker._inside_loop = True
        self.walker._walk_block(loop_body, kernel_info)
        self.walker._inside_loop = False

        # Emit loop latch
        ctx.emit_loop_latch(loop_ctx)

        # End loop
        ctx.end_loop()

        # Map scf.for results to final values of iter_args
        for i, result in enumerate(operation.results):
            result_ssa = ssa(result)
            if i < len(iter_arg_ranges):
                quad = iter_arg_ranges[i]
                regs = tuple(KVReg(quad.base_reg.id + j) for j in range(4))
                ctx.ssa_to_reg[result_ssa] = regs

    def handle_scf_if_op(self, operation: scf_d.IfOp, kernel_info: KernelInfo):
        """Handle scf.if via EXEC masking (per-lane predication).

        This is required for kernels like softmax that use lane0/wave0 writes to LDS.
        We implement only the common pattern where the else-region is empty.
        """
        from .kernel_ir import KInstr, KImm, EXEC

        cond_ssa = ssa(operation.condition)
        # If this condition didn't come from a tracked arith.cmpi, fall back to
        # unconditional 'then' execution (vec_add uses i1 allocas initialized to true).
        cmpi_preds = getattr(self.walker, "_cmpi_predicates", {})
        if cond_ssa not in cmpi_preds:
            then_block = operation.thenRegion.blocks[0]
            self.walker._walk_block(then_block, kernel_info)
            return

        # Emit VCC for this condition (recorded by cmpi handler).
        self._emit_vcc_for(cond_ssa, kernel_info)  # type: ignore[attr-defined]

        # If there is a non-empty else region, we currently don't support it.
        if len(operation.elseRegion.blocks) > 0 and len(list(operation.elseRegion.blocks[0].operations)) > 0:
            raise NotImplementedError("scf.if with non-empty else region is not supported in asm backend yet")

        ctx = self.walker.kernel_ctx

        # Save EXEC.
        saved_exec = ctx.program.alloc_sreg_range(2, alignment=2)
        ctx.program.emit(KInstr("s_mov_b64", defs=(saved_exec,), uses=(EXEC,), comment="save exec"))
        # VCC written by VALU compare may not be immediately visible to SALU consumers.
        # Match ISA software-wait guidance: add 2-cycle delay before using VCC in s_and_b64.
        ctx.program.emit(KInstr("s_nop", defs=(), uses=(KImm(1),), comment="vcc hazard"))
        # exec &= vcc
        from .kernel_ir import VCC
        ctx.program.emit(KInstr("s_and_b64", defs=(EXEC,), uses=(EXEC, VCC), comment="exec &= vcc"))

        # Then-region under masked EXEC.
        then_block = operation.thenRegion.blocks[0]
        self.walker._walk_block(then_block, kernel_info)

        # Restore EXEC.
        ctx.program.emit(KInstr("s_mov_b64", defs=(EXEC,), uses=(saved_exec,), comment="restore exec"))

    # Note: gather_to_lds handlers moved to gather_to_shared.py (G2SMixin)

    def handle_memref_cast_op(
        self, operation: memref_d.CastOp, kernel_info: KernelInfo
    ):
        """Handle memref.cast operations - track source memref mapping.

        MLIR format:
            %result = memref.cast %src : memref<...> to memref<...>
        """
        result_ssa = str(operation.results[0])
        source_ssa = str(operation.operands[0])

        # Track the cast chain for SRD lookup
        if not hasattr(self.walker, "_memref_cast_sources"):
            self.walker._memref_cast_sources = {}
        self.walker._memref_cast_sources[result_ssa] = source_ssa

    def handle_memref_reinterpret_cast_op(
        self, operation: memref_d.ReinterpretCastOp, kernel_info: KernelInfo
    ):
        """Handle memref.reinterpret_cast operations - track source memref mapping.

        MLIR format:
            %result = memref.reinterpret_cast %src to offset: [...], sizes: [...], strides: [...]
                : memref<...> to memref<...>
        """
        result_ssa = str(operation.results[0])
        source_ssa = str(operation.operands[0])

        # Track the cast chain for SRD lookup
        if not hasattr(self.walker, "_memref_cast_sources"):
            self.walker._memref_cast_sources = {}
        self.walker._memref_cast_sources[result_ssa] = source_ssa

    def handle_fat_raw_buffer_cast_op(self, operation, kernel_info: KernelInfo):
        """Handle amdgpu.fat_raw_buffer_cast - track source memref and cache swizzle stride."""
        result_ssa = ssa(operation.results[0])
        source_ssa = ssa(operation.operands[0])

        # Extract cacheSwizzleStride from operand 2 if present
        cache_swizzle_stride = None
        if len(operation.operands) >= 3:
            defining_op = operation.operands[2].owner.opview
            if isinstance(defining_op, arith_d.ConstantOp) and hasattr(
                defining_op.value, "value"
            ):
                cache_swizzle_stride = int(defining_op.value.value)

        # Track for gather_to_lds SRD tracing
        if not hasattr(self.walker, "_fat_buffer_sources"):
            self.walker._fat_buffer_sources = {}
        info = {"source_ssa": source_ssa}
        if cache_swizzle_stride is not None:
            info["cache_swizzle_stride"] = cache_swizzle_stride
        self.walker._fat_buffer_sources[result_ssa] = info

    def handle_readfirstlane_op(self, operation, kernel_info: KernelInfo):
        """Handle rocdl.readfirstlane - propagate value for uniform broadcast.

        The expression is preserved as-is (not evaluated) because each wavefront
        has different tid values. v_readfirstlane is emitted during code generation.
        """
        result_ssa = ssa(operation.results[0])
        source_ssa = ssa(operation.operands[0])

        if source_ssa in kernel_info.index_env:
            kernel_info.index_env[result_ssa] = kernel_info.index_env[source_ssa]

    def handle_s_waitcnt_op(self, operation, kernel_info: KernelInfo):
        """Handle rocdl.s.waitcnt - emit wait count instruction.

        Encoding (gfx9+): bits 0-3 = vmcnt (0 = wait for all, 15 = no wait)
        """
        waitcnt_value = int(operation.bitfield.value)
        vmcnt = waitcnt_value & 0xF  # 4-bit field: 0-15

        # vmcnt=15 means "no wait" (max 4-bit value), so only emit if < 15
        if vmcnt < 15:
            self.walker.unified.s_waitcnt(f"vmcnt({vmcnt})")
            # Notify ticketing system about the wait
            # Always go through kernel_ctx ticketing in the kernel IR pipeline.
            self.walker.kernel_ctx.ticketing.observe_vmem_wait(vmcnt)
