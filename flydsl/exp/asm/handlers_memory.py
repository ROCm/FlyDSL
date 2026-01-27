# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from wave_lang.support.ir_imports import (
    amdgpu_d,
    gpu_d,
    memref_d,
    stream_d,
    vector_d,
)
from wave_lang.kernel.wave.constraints import MMAType

from .handlers_shared import *
from .kernel_model import BindingUse, KernelInfo, MemRefInfo, VecAccess
from .utils import (
    parse_memref_type_from_obj,
    parse_vector_type_from_obj,
    split_const_dynamic,
)


class _MemoryHandlers:
    def handle_vector_load_op(
        self, operation: vector_d.LoadOp, kernel_info: KernelInfo
    ):
        """Handle vector.load operations - track memory accesses and emit load instructions."""
        memref_ssa = str(operation.operands[0])  # memref is first operand
        num_elements, element_bytes, _ = parse_vector_type_from_obj(
            operation.results[0].type
        )
        indices = [
            str(operation.operands[i]) for i in range(1, len(operation.operands))
        ]

        # If memref is not in subspans, it may be LDS (workgroup) memory; handle later in emit

        # Update memref info if not already set
        if memref_ssa in kernel_info.subspans:
            binding_use = kernel_info.subspans[memref_ssa]
            if not binding_use.memref_info:
                try:
                    memref_type_object = operation.operands[0].type
                    shape, strides, element_bytes = parse_memref_type_from_obj(
                        memref_type_object
                    )
                    binding_use.memref_info = MemRefInfo(shape, strides, element_bytes)
                except Exception as e:
                    raise ValueError(
                        f"Cannot parse memref type for load operation: {e}"
                    )

        kernel_info.accesses.append(
            VecAccess("load", memref_ssa, num_elements, element_bytes, indices)
        )

        # Emit load instruction
        self._emit_load_instruction(operation, kernel_info, memref_ssa, indices)

    def handle_vector_extract_strided_slice_op(
        self, operation: vector_d.ExtractStridedSliceOp, kernel_info: KernelInfo
    ):
        """Handle vector.extract_strided_slice operations - extract subset of source registers."""
        # Get source SSA value and its registers
        source_ssa = str(operation.operands[0])
        source_regs = self.walker.kernel_ctx.ssa_to_reg.get(source_ssa)

        if not source_regs:
            # Source not tracked - skip silently
            return

        # Extract offset and size from operation attributes
        offsets = operation.attributes["offsets"]
        sizes = operation.attributes["sizes"]

        # Parse the offset value (should be a single integer for 1D extract)
        offset_val = int(str(offsets).split("[")[1].split("]")[0])
        size_val = int(str(sizes).split("[")[1].split("]")[0])

        # Extract the appropriate subset of registers
        if size_val == 1:
            # Single scalar extract - return just the one register as a tuple
            extracted_reg = source_regs[offset_val]
            result_regs = (extracted_reg,)
        else:
            # Multi-element extract - return a slice
            result_regs = source_regs[offset_val : offset_val + size_val]

        result_ssa = str(operation.result)
        self.walker.kernel_ctx.ssa_to_reg[result_ssa] = result_regs

    def handle_vector_store_op(
        self, operation: vector_d.StoreOp, kernel_info: KernelInfo
    ):
        """Handle vector.store operations - track memory accesses and emit store instructions."""
        memref_ssa = str(
            operation.operands[1]
        )  # memref is second operand (after value)
        num_elements, element_bytes, _ = parse_vector_type_from_obj(
            operation.operands[0].type
        )  # value is first operand
        indices = [
            str(operation.operands[i]) for i in range(2, len(operation.operands))
        ]

        # If memref is not in subspans, it may be LDS (workgroup) memory; handle later in emit

        # Update memref info if not already set
        if memref_ssa in kernel_info.subspans:
            binding_use = kernel_info.subspans[memref_ssa]
            if not binding_use.memref_info:
                try:
                    memref_type_object = operation.operands[1].type
                    shape, strides, element_bytes = parse_memref_type_from_obj(
                        memref_type_object
                    )
                    binding_use.memref_info = MemRefInfo(shape, strides, element_bytes)
                except Exception as e:
                    raise ValueError(
                        f"Cannot parse memref type for store operation: {e}"
                    )

        kernel_info.accesses.append(
            VecAccess("store", memref_ssa, num_elements, element_bytes, indices)
        )

        # Emit store instruction
        self._emit_store_instruction(operation, kernel_info, memref_ssa, indices)

    def handle_stream_binding_subspan_op(
        self, operation: stream_d.BindingSubspanOp, kernel_info: KernelInfo
    ):
        """Handle stream.binding.subspan operations - map memrefs to function arguments."""

        # Subspan is immediately consumed by a reinterpret cast
        users = list(operation.result.uses)
        assert (
            len(users) == 1
        ), f"Expected 1 user for stream.binding.subspan operation, got {users}"
        reinterpret = users[0].owner.operation.opview
        assert isinstance(
            reinterpret, memref_d.ReinterpretCastOp
        ), f"Expected memref.reinterpret_cast operation, got {reinterpret}"

        # map memref SSA -> which function arg index it came from
        source_ssa = str(operation.operands[0])  # function arg SSA
        result_ssa = str(reinterpret.results[0])  # memref SSA
        argument_index = kernel_info.arg_ssa_order.index(source_ssa)
        binding_use = kernel_info.subspans.setdefault(
            result_ssa, BindingUse(result_ssa, argument_index)
        )

        # Extract memref information from the result type
        # This must succeed for SRD setup to work
        memref_type_object = reinterpret.results[0].type
        shape, strides, element_bytes = parse_memref_type_from_obj(memref_type_object)
        binding_use.memref_info = MemRefInfo(shape, strides, element_bytes)

        # Emit SRD setup
        self._emit_srd_setup(operation, kernel_info, result_ssa, argument_index)

    def _compute_buffer_size(self, memref_info):
        """Compute buffer size in bytes from memref shape and element size."""
        if not memref_info.shape:
            # Scalar or unranked: use single element
            return memref_info.elem_bytes
        else:
            # Compute total buffer size: product of all dimensions * element size
            total_elements = 1
            for dim in memref_info.shape:
                total_elements *= dim
            return total_elements * memref_info.elem_bytes

    def _emit_srd_setup(self, operation, kernel_info, memref_ssa, argument_index):
        """Emit SRD setup for a binding subspan operation."""
        binding_use = kernel_info.subspans.get(memref_ssa)
        if not binding_use or not binding_use.memref_info:
            raise ValueError(
                f"Cannot determine memref information for {memref_ssa}. "
                f"SRD setup requires memref shape and element size."
            )

        limit_bytes = self._compute_buffer_size(binding_use.memref_info)

        # In kernel IR mode, SRD setup is deferred to actual load/store operations
        # Just record the subspan info, SRD will be set up lazily
        pass

    def handle_mfma_op(self, operation: amdgpu_d.MFMAOp, kernel_info: KernelInfo):
        """Handle amdgpu.mfma operations - emit MFMA instruction with proper input sourcing."""

        # Get the operand SSA values from the MFMA operation
        # MFMA format: %result = amdgpu.mfma %lhs * %rhs + %acc
        if len(operation.operands) >= 3:
            lhs_ssa = str(operation.operands[0])  # First operand (LHS of multiply)
            rhs_ssa = str(operation.operands[1])  # Second operand (RHS of multiply)
            acc_ssa = str(operation.operands[2])  # Third operand (accumulator)

            # Kernel IR mode: use virtual registers

            ctx = self.walker.kernel_ctx

            # Get operand registers from kernel context
            lhs_regs = ctx.ssa_to_reg.get(lhs_ssa)
            rhs_regs = ctx.ssa_to_reg.get(rhs_ssa)
            acc_regs = ctx.ssa_to_reg.get(acc_ssa)

            if lhs_regs and rhs_regs:
                # Dispatch to correct MFMA based on mma_type from compile options
                mma_type = ctx.mma_type

                if mma_type == MMAType.F32_16x16x32_F16:
                    # 16x16x32 MFMA (requires 4 VGPRs per operand = 8 x f16)
                    if len(lhs_regs) != 4 or len(rhs_regs) != 4:
                        raise ValueError(
                            f"MFMA 16x16x32 requires 4 VGPRs per operand. "
                            f"Got lhs={len(lhs_regs)} (ssa={lhs_ssa}), "
                            f"rhs={len(rhs_regs)} (ssa={rhs_ssa}), "
                            f"acc_ssa={acc_ssa}."
                        )

                    result_regs = ctx.emit_mfma_f32_16x16x32_f16(
                        lhs_regs,
                        rhs_regs,
                        acc_regs if acc_regs and len(acc_regs) == 4 else None,
                    )
                elif mma_type == MMAType.F32_16x16x16_F16 or mma_type is None:
                    # 16x16x16 MFMA (default, requires 2 VGPRs per operand = 4 x f16)
                    if len(lhs_regs) != 2 or len(rhs_regs) != 2:
                        raise ValueError(
                            f"MFMA 16x16x16 requires 2 VGPRs per operand. "
                            f"Got lhs={len(lhs_regs)} (ssa={lhs_ssa}), "
                            f"rhs={len(rhs_regs)} (ssa={rhs_ssa}), "
                            f"acc_ssa={acc_ssa}."
                        )

                    result_regs = ctx.emit_mfma_f32_16x16x16_f16(
                        lhs_regs,
                        rhs_regs,
                        acc_regs if acc_regs and len(acc_regs) == 4 else None,
                    )
                else:
                    raise NotImplementedError(
                        f"Unsupported MMA type: {mma_type}. "
                        f"Supported: F32_16x16x16_F16, F32_16x16x32_F16"
                    )

                # Track result in SSA mapping
                result_ssa = str(operation.result)
                ctx.ssa_to_reg[result_ssa] = result_regs

                return

            raise RuntimeError(
                f"MFMA operation inputs not available. "
                f"lhs={lhs_ssa} ({lhs_regs}), rhs={rhs_ssa} ({rhs_regs})"
            )

    def handle_barrier_op(self, operation: gpu_d.BarrierOp, kernel_info: KernelInfo):
        """Handle gpu.barrier operations - emit synchronization barrier."""
        self.walker.unified.s_barrier(comment="workgroup barrier")

    def handle_lds_barrier_op(
        self, operation: amdgpu_d.LDSBarrierOp, kernel_info: KernelInfo
    ):
        """Handle amdgpu.lds_barrier - emit lgkmcnt(0) + s_barrier."""
        self.walker.unified.s_waitcnt(waitcnt="lgkmcnt(0)")
        self.walker.unified.s_barrier(comment="LDS barrier")

    def handle_view_op(self, operation: memref_d.ViewOp, kernel_info: KernelInfo):
        """Handle memref.view operations - capture view base byte offset for LDS-backed memrefs."""
        result_ssa = str(operation.results[0])
        # The offset operand is already in bytes (index into xi8 buffer)
        # Only capture if the offset is a known integer constant in index_env
        base_bytes = None
        for operand in operation.operands:
            key = str(operand)
            if key in kernel_info.index_env and isinstance(
                kernel_info.index_env[key], int
            ):
                base_bytes = kernel_info.index_env[key]
                break
        if base_bytes is not None:
            self.walker._lds_view_base_bytes[result_ssa] = int(base_bytes)

    def handle_alloc_op(self, operation: memref_d.AllocOp, kernel_info: KernelInfo):
        """Handle memref.alloc operations - capture LDS allocation size."""
        # Parse the memref type to get shape and element size
        shape, strides, elem_bytes = parse_memref_type_from_obj(
            operation.results[0].type
        )

        # Compute total LDS allocation size
        if shape:
            total_elements = 1
            for dim in shape:
                total_elements *= dim
            alloc_size_bytes = total_elements * elem_bytes

            # Track the maximum LDS size (in case of multiple allocations)
            kernel_info.lds_size_bytes = max(
                kernel_info.lds_size_bytes, alloc_size_bytes
            )

    def _parse_load_memref_info(self, operation):
        """Parse memref information from a vector.load operation."""
        memref_type_object = operation.operands[0].type
        try:
            shape, strides, element_bytes = parse_memref_type_from_obj(
                memref_type_object
            )
            return MemRefInfo(shape, strides, element_bytes)
        except Exception as e:
            raise ValueError(f"Cannot parse memref type for load operation: {e}")

    def _emit_lds_load(self, operation, kernel_info, memref_ssa, byte_offset_expr):
        """Emit an LDS load operation using MLIR's 2D memref indices.

        Uses the byte_offset_expr computed from MLIR's actual indices rather than
        forcing lane-linear addressing. The MLIR indices already encode the correct
        addressing for both single-wave and multi-wave modes, including any swizzle
        patterns needed for cache efficiency.

        Optimization: When the address has a constant offset component that fits within
        the hardware limit (DS_MAX_OFFSET), we use the ds_read offset field instead of
        computing the full address. This saves a v_add_u32 instruction.

        For offsets exceeding DS_MAX_OFFSET (~8192 bytes on CDNA3/4), we fall back to
        computing the full address without using the offset field.
        """
        import os
        import sympy
        from .utils import split_const_dynamic

        DEBUG_DS_OFFSET = os.environ.get("WAVE_LDS_DSREAD_OFFSET_DEBUG", "0") == "1"

        # Add view base offset if present
        vbase_val = self.walker._lds_view_base_bytes.get(memref_ssa, 0)
        original_byte_offset_expr = byte_offset_expr  # Save for debug

        # Use MLIR-derived expression for all cases (single-wave, multi-wave, g2s, non-g2s)
        # The MLIR index expression already contains the correct addressing formula
        if vbase_val:
            byte_offset_expr = byte_offset_expr + sympy.Integer(vbase_val)

        # Split address into base + constant offset to use ds_read offset field
        # ds_read supports 16-bit offset (0-65535), but we use conservative limits
        const_offset, base_expr = split_const_dynamic(
            byte_offset_expr, max_immediate=65528
        )

        # Max offset for ds_read_b64/ds_read_b128 on CDNA3/CDNA4
        # The ISA spec says 16-bit unsigned (0-65535), which is correct.
        # Previous conservative limit (2040) was causing excessive constant
        # materialization for LDS addresses in the 4096+ range.
        # Testing shows 8192 works correctly for GEMM kernels.
        DS_MAX_OFFSET = 8192  # Increased to cover typical LDS offset ranges

        # Determine load size from operation result type
        num_elements, element_bytes, _ = parse_vector_type_from_obj(
            operation.results[0].type
        )
        total_bytes = num_elements * element_bytes

        # Alignment depends on load size
        DS_ALIGN = 16 if total_bytes == 16 else 8

        if DEBUG_DS_OFFSET:
            print(f"[DS_OFFSET_DEBUG] memref={memref_ssa[:60]}...")
            print(f"[DS_OFFSET_DEBUG]   vbase_val={vbase_val}")
            print(f"[DS_OFFSET_DEBUG]   original_expr={original_byte_offset_expr}")
            print(f"[DS_OFFSET_DEBUG]   after_vbase_expr={byte_offset_expr}")
            print(
                f"[DS_OFFSET_DEBUG]   const_offset={const_offset}, base_expr={base_expr}"
            )

        # Kernel IR mode: emit LDS load with virtual registers
        from .kernel_ir import KInstr, KImm, KVReg

        ctx = self.walker.kernel_ctx

        # Determine if we can use the offset field
        has_dynamic_base = len(base_expr.free_symbols) > 0

        if not has_dynamic_base:
            # Pure constant address - materialize it
            addr_vreg = ctx.vreg()
            ctx.program.emit(
                KInstr(
                    "v_mov_b32",
                    (addr_vreg,),
                    (KImm(int(byte_offset_expr)),),
                    comment=f"LDS addr = {byte_offset_expr}",
                )
            )
            lds_offset = 0
        elif 0 <= const_offset <= DS_MAX_OFFSET and const_offset % DS_ALIGN == 0:
            # Can use offset field - compute only the base expression
            # Use a fresh scope to avoid CSE issues with different memrefs
            with ctx.expr_emitter.scope("lds_base"):
                addr_vreg = ctx.expr_emitter.get_or_emit(base_expr)
            lds_offset = const_offset
            if DEBUG_DS_OFFSET:
                print(
                    f"[DS_OFFSET_DEBUG]   -> USING_OFFSET: addr={addr_vreg}, offset={lds_offset}"
                )
        else:
            # Offset out of range or not aligned - compute full address
            addr_vreg = ctx.expr_emitter.get_or_emit(byte_offset_expr)
            lds_offset = 0

        # Allocate destination registers and emit appropriate ds_read instruction
        # based on load size
        if total_bytes == 16:
            # 128-bit load for 8 x f16 (used with 16x16x32 MFMA)
            dst_range = ctx.vreg_quad()
            ctx.emit_lds_read_b128(dst_range, addr_vreg, lds_offset)
            result_regs = tuple(KVReg(dst_range.base_reg.id + i) for i in range(4))
        elif total_bytes == 8:
            # 64-bit load for 4 x f16 (used with 16x16x16 MFMA)
            dst_range = ctx.vreg_pair()
            ctx.emit_lds_read_b64(dst_range, addr_vreg, lds_offset)
            result_regs = (
                KVReg(dst_range.base_reg.id),
                KVReg(dst_range.base_reg.id + 1),
            )
        else:
            raise NotImplementedError(
                f"LDS load of {total_bytes} bytes not supported. "
                f"Expected 8 (ds_read_b64) or 16 (ds_read_b128) bytes."
            )

        # Track in SSA mapping as tuple of KVReg
        result_ssa = str(operation.results[0])
        ctx.ssa_to_reg[result_ssa] = result_regs

    def _ensure_global_load_srd(self, kernel_info, memref_ssa):
        """Ensure SRD is set up for a global load."""
        # Kernel IR mode: use kernel_ctx SRD tracking
        if memref_ssa in self.walker.kernel_ctx.srd_ranges:
            return

        binding_use = kernel_info.subspans[memref_ssa]
        if not binding_use.memref_info:
            raise ValueError(
                f"Cannot determine memref information for {memref_ssa}. "
                f"SRD setup requires memref shape and element size."
            )

        limit_bytes = self._compute_buffer_size(binding_use.memref_info)
        arg_idx = binding_use.arg_index if binding_use.arg_index >= 0 else 0
        self.walker.kernel_ctx.ensure_srd(memref_ssa, arg_idx, limit_bytes)

    def _parse_vector_load_type(self, operation):
        """Parse vector type from load operation result."""
        try:
            num_elements, element_bytes, _ = parse_vector_type_from_obj(
                operation.results[0].type
            )
            return num_elements * element_bytes
        except Exception as e:
            raise ValueError(f"Cannot parse vector type for global load: {e}")

    def _emit_buffer_load_and_track(
        self, operation, kernel_info, memref_ssa, vector_bytes, voffset_v, instoffset
    ):
        """Emit buffer load instruction and track loaded registers and ticket."""
        result_ssa = str(operation.results[0])

        # Kernel IR mode: emit via kernel_ctx with virtual registers
        from .kernel_ir import KVReg

        # voffset_v might be a physical index; convert to virtual reg
        if isinstance(voffset_v, int):
            voffset = KVReg(voffset_v)  # Treat as virtual for now
        else:
            voffset = voffset_v

        loaded_ranges = self.walker.kernel_ctx.emit_buffer_load(
            memref_ssa, vector_bytes, voffset, instoffset
        )

        # Convert ranges to tuple of individual registers for ssa_to_reg storage
        if len(loaded_ranges) == 1:
            # Single range (pair or quad)
            base = loaded_ranges[0].base_reg
            count = loaded_ranges[0].count
            regs_tuple = tuple(KVReg(base.id + i) for i in range(count))
        else:
            # Multiple ranges - flatten into single tuple
            regs_tuple = []
            for rng in loaded_ranges:
                base = rng.base_reg
                regs_tuple.extend(KVReg(base.id + i) for i in range(rng.count))
            regs_tuple = tuple(regs_tuple)

        self.walker.kernel_ctx.ssa_to_reg[result_ssa] = regs_tuple

    def _emit_global_load(self, operation, kernel_info, memref_ssa, byte_offset_expr):
        """Emit a global buffer load operation."""
        self._ensure_global_load_srd(kernel_info, memref_ssa)

        # Split constant/dynamic and materialize dynamic part via cached emitter (CSE)
        const_offset, dynamic_expr = split_const_dynamic(byte_offset_expr)

        # Kernel IR mode: allocate virtual registers
        from .kernel_ir import KInstr, KImm
        from .instruction_registry import Instruction

        # Compute voffset in kernel IR
        voffset_v = self.walker.kernel_ctx.vreg()

        if dynamic_expr == 0 or (
            hasattr(dynamic_expr, "is_zero") and dynamic_expr.is_zero
        ):
            # No dynamic part: set voffset to 0
            self.walker.kernel_ctx.program.emit(
                KInstr(
                    Instruction.V_MOV_B32,
                    (voffset_v,),
                    (KImm(0),),
                    comment="voffset = 0",
                )
            )
            instoffset = const_offset
        else:
            # Dynamic part: use expression emitter to compute voffset
            # The expression emitter caches results so the same expression
            # returns the same vreg (CSE)
            expr_emitter = self.walker.kernel_ctx.expr_emitter
            voffset_v = expr_emitter.get_or_emit(dynamic_expr)
            instoffset = const_offset

        vector_bytes = self._parse_vector_load_type(operation)
        self._emit_buffer_load_and_track(
            operation, kernel_info, memref_ssa, vector_bytes, voffset_v, instoffset
        )

    def _is_lds_memref(self, operation):
        """Check if the memref has LDS (workgroup) address space."""
        memref_type = operation.operands[0].type
        # Check if the memref has #gpu.address_space<workgroup> attribute
        if (
            hasattr(memref_type, "memory_space")
            and memref_type.memory_space is not None
        ):
            # Convert memory_space attribute to string and check for "workgroup"
            memory_space_str = str(memref_type.memory_space)
            return "workgroup" in memory_space_str.lower()
        return False

    def _emit_load_instruction(self, operation, kernel_info, memref_ssa, indices):
        """Emit load instruction for a vector.load operation derived purely from indices."""
        from .utils import build_memref_byte_offset_expr

        # Parse memref info and build byte offset expression
        memref_info = self._parse_load_memref_info(operation)
        byte_offset_expr = build_memref_byte_offset_expr(
            indices, kernel_info, memref_info
        )

        # Check address space to determine LDS vs global
        if self._is_lds_memref(operation):
            # LDS load path (workgroup address space)
            self._emit_lds_load(operation, kernel_info, memref_ssa, byte_offset_expr)
            return

        # Global buffer load path
        self._emit_global_load(operation, kernel_info, memref_ssa, byte_offset_expr)

    def _parse_store_type_info(self, operation):
        """Parse memref and vector type information from a vector.store operation."""
        # Get memref info
        memref_type_object = operation.operands[1].type
        try:
            shape, strides, element_bytes = parse_memref_type_from_obj(
                memref_type_object
            )
            memref_info = MemRefInfo(shape, strides, element_bytes)
        except Exception as e:
            raise ValueError(f"Cannot parse memref type for store operation: {e}")

        # Get vector type info
        value_vector_type = operation.operands[0].type
        try:
            num_elements, elem_bytes, _ = parse_vector_type_from_obj(value_vector_type)
            vector_bytes = num_elements * elem_bytes
        except Exception as e:
            raise ValueError(f"Cannot parse vector type for store value: {e}")

        return memref_info, value_vector_type, num_elements, vector_bytes

    def _emit_lds_store(
        self,
        kernel_info,
        memref_ssa,
        value_vector_type,
        indices,
        memref_info,
        vector_bytes,
    ):
        """Emit an LDS store operation."""
        import sympy
        from .kernel_ir import KVReg, KRegRange, KInstr
        from .utils import build_memref_byte_offset_expr

        ctx = self.walker.kernel_ctx

        # Compute LDS address, adding view base offset if present
        byte_offset_expr = build_memref_byte_offset_expr(
            indices, kernel_info, memref_info
        )
        # Add view base offset for this specific memref (each matrix has different base)
        vbase_val = self.walker._lds_view_base_bytes.get(memref_ssa, 0)
        if vbase_val:
            byte_offset_expr = byte_offset_expr + sympy.Integer(vbase_val)
        addr_vreg = ctx.expr_emitter.get_or_emit(byte_offset_expr)

        # Wait for any pending VMEM loads
        ctx.program.emit(
            KInstr(
                "s_waitcnt", (), ("vmcnt(0)",), comment="wait for VMEM before LDS store"
            )
        )

        # Get source registers from SSA mapping (these are KVReg objects)
        src_regs = self._current_store_regs

        # Build a properly aligned KRegRange for the source
        if vector_bytes == 4:
            # Single register
            src_vreg = src_regs[0] if isinstance(src_regs, (tuple, list)) else src_regs
            ctx.program.emit(
                KInstr(
                    "ds_write_b32",
                    (),
                    (addr_vreg, src_vreg),
                    comment=f"LDS store 4B to {memref_ssa}",
                )
            )
        elif vector_bytes == 8:
            # Register pair (must be 64-bit aligned)
            if isinstance(src_regs, (tuple, list)) and len(src_regs) >= 2:
                # Create aligned range from the source registers
                base_id = (
                    src_regs[0].id if isinstance(src_regs[0], KVReg) else src_regs[0]
                )
                src_range = KRegRange(KVReg(base_id), 2, alignment=2)
            else:
                raise ValueError(
                    f"Expected 2 registers for ds_write_b64, got {src_regs}"
                )
            ctx.emit_lds_write_b64(addr_vreg, src_range)
        elif vector_bytes == 16:
            # Register quad (must be 128-bit aligned)
            if isinstance(src_regs, (tuple, list)) and len(src_regs) >= 4:
                base_id = (
                    src_regs[0].id if isinstance(src_regs[0], KVReg) else src_regs[0]
                )
                src_range = KRegRange(KVReg(base_id), 4, alignment=4)
            else:
                raise ValueError(
                    f"Expected 4 registers for ds_write_b128, got {src_regs}"
                )
            ctx.emit_lds_write_b128(addr_vreg, src_range)
        else:
            raise NotImplementedError(
                f"LDS stores of {vector_bytes} bytes not supported"
            )

    def _ensure_global_store_srd(self, kernel_info, memref_ssa):
        """Ensure SRD is set up for a global store."""
        binding_use = kernel_info.subspans[memref_ssa]

        # Kernel IR mode: use kernel_ctx SRD tracking
        if memref_ssa in self.walker.kernel_ctx.srd_ranges:
            return

        if not binding_use.memref_info:
            raise ValueError(
                f"Cannot determine memref information for {memref_ssa}. "
                f"SRD setup requires memref shape and element size."
            )

        limit_bytes = self._compute_buffer_size(binding_use.memref_info)
        arg_idx = binding_use.arg_index if binding_use.arg_index >= 0 else 0
        self.walker.kernel_ctx.ensure_srd(memref_ssa, arg_idx, limit_bytes)

    def _emit_global_store(
        self,
        kernel_info,
        memref_ssa,
        value_vector_type,
        indices,
        memref_info,
        num_elements,
        vector_bytes,
    ):
        """Emit a global buffer store operation."""
        # Kernel IR mode: use virtual registers
        from .kernel_ir import KInstr, KImm, KVReg, KRegRange
        from .utils import build_element_byte_offset_exprs
        from .instruction_registry import Instruction

        # Get expression emitter - loop-invariant expressions are cached globally,
        # loop-varying expressions are never cached, so no cache clearing needed.
        expr_emitter = self.walker.kernel_ctx.expr_emitter

        # Compute address - allocate virtual voffset
        byte_exprs = build_element_byte_offset_exprs(
            value_vector_type, indices, kernel_info, memref_info
        )
        const_offset, dynamic_expr = split_const_dynamic(byte_exprs[0])

        # Compute voffset in kernel IR (store path)
        voffset_v = self.walker.kernel_ctx.vreg()

        if dynamic_expr == 0 or (
            hasattr(dynamic_expr, "is_zero") and dynamic_expr.is_zero
        ):
            self.walker.kernel_ctx.program.emit(
                KInstr(
                    Instruction.V_MOV_B32,
                    (voffset_v,),
                    (KImm(0),),
                    comment="voffset = 0",
                )
            )
            instoffset = const_offset
        else:
            # Dynamic part: use expression emitter to compute voffset
            voffset_v = expr_emitter.get_or_emit(dynamic_expr)
            instoffset = const_offset

        # IMPORTANT: Wait for pending loads BEFORE setting up store SRD
        # Otherwise we overwrite the load SRD while loads are still in flight.
        # Use ticketing to emit only once per store phase, not per-store.
        ticketing = self.walker.kernel_ctx.ticketing
        if ticketing._vmem_last_ticket >= 0:
            # Only emit wait if we haven't already waited for all VMEM
            threshold = ticketing.compute_vmem_wait(0)  # Wait for all pending
            if threshold is not None:
                self.walker.kernel_ctx.program.emit(
                    KInstr(
                        Instruction.S_WAITCNT,
                        (),
                        (f"vmcnt({threshold})",),
                        comment="wait for VMEM before store SRD setup",
                    )
                )

        # Now it's safe to set up the store SRD (may reuse same physical regs)
        self._ensure_global_store_srd(kernel_info, memref_ssa)

        # Get source registers from ssa_to_reg
        src_regs = self._current_store_regs
        if isinstance(src_regs, tuple) and len(src_regs) > 0:
            # Convert to KRegRange(s) for the store
            # Group registers into quads (16 bytes each) for vectorized stores
            num_regs = len(src_regs)

            if vector_bytes <= 4:
                # Single dword
                first_reg = src_regs[0]
                if isinstance(first_reg, KVReg):
                    src_range = KRegRange(first_reg, 1)
                else:
                    src_range = KRegRange(KVReg(first_reg), 1)
                src_ranges = (src_range,)
            elif vector_bytes == 8:
                # Pair (dwordx2)
                first_reg = src_regs[0]
                if isinstance(first_reg, KVReg):
                    src_range = KRegRange(first_reg, 2)
                else:
                    src_range = KRegRange(KVReg(first_reg), 2)
                src_ranges = (src_range,)
            else:
                # Multiple quads (16+ bytes)
                # Each quad is 4 VGPRs = 16 bytes
                num_quads = (vector_bytes + 15) // 16
                src_ranges = []
                for q in range(num_quads):
                    base_idx = q * 4
                    if base_idx < num_regs:
                        first_reg = src_regs[base_idx]
                        if isinstance(first_reg, KVReg):
                            src_range = KRegRange(first_reg, 4)
                        else:
                            src_range = KRegRange(KVReg(first_reg), 4)
                        src_ranges.append(src_range)
                src_ranges = tuple(src_ranges)

            self.walker.kernel_ctx.emit_buffer_store(
                memref_ssa, src_ranges, voffset_v, instoffset
            )

    def _is_lds_store_memref(self, operation):
        """Check if the store destination memref has LDS (workgroup) address space."""
        memref_type = operation.operands[1].type  # For stores, memref is operand[1]
        # Check if the memref has #gpu.address_space<workgroup> attribute
        if (
            hasattr(memref_type, "memory_space")
            and memref_type.memory_space is not None
        ):
            # Convert memory_space attribute to string and check for "workgroup"
            memory_space_str = str(memref_type.memory_space)
            return "workgroup" in memory_space_str.lower()
        return False

    def _emit_store_instruction(self, operation, kernel_info, memref_ssa, indices):
        """Emit store instruction for a vector.store operation derived purely from indices."""
        # Parse type information
        memref_info, value_vector_type, num_elements, vector_bytes = (
            self._parse_store_type_info(operation)
        )

        # Get the SSA value being stored (first operand)
        value_ssa = str(operation.operands[0])

        # Look up the registers containing the value to store
        value_regs = self.walker.kernel_ctx.ssa_to_reg.get(value_ssa)
        if not value_regs:
            raise RuntimeError(
                f"Store operation references SSA value {value_ssa} but it's not in kernel_ctx.ssa_to_reg. "
                f"Available: {list(self.walker.kernel_ctx.ssa_to_reg.keys())}"
            )

        # Store value_regs for extraction in subsequent methods
        self._current_store_regs = value_regs

        # Check address space to determine LDS vs global
        if self._is_lds_store_memref(operation):
            # LDS store path (workgroup address space)
            self._emit_lds_store(
                kernel_info,
                memref_ssa,
                value_vector_type,
                indices,
                memref_info,
                vector_bytes,
            )
            return

        # Global buffer store path
        # SRD setup happens inside _emit_global_store after waitcnt
        self._emit_global_store(
            kernel_info,
            memref_ssa,
            value_vector_type,
            indices,
            memref_info,
            num_elements,
            vector_bytes,
        )
