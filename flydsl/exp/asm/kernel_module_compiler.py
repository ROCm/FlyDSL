# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

from .kernel_compilation_context import KernelCompilationContext

if TYPE_CHECKING:
    from wave_lang.kernel.wave.constraints import MMAType


@dataclass
class KernelModuleCompiler:
    """
    Module-level kernel compiler that generates complete .s assembly files.

    This class handles the full compilation pipeline from MLIR to assembly:
    1. Parse MLIR and extract kernel metadata
    2. Create KernelCompilationContext
    3. Walk MLIR operations and emit to kernel IR
    4. Run register allocation
    5. Generate complete assembly (prologue + body + epilogue + metadata)

    Uses MetadataEmitter for prologue/epilogue generation (single source of truth).

    Usage:
        compiler = KernelModuleCompiler(targetid="gfx942", codeobj="5")
        asm = compiler.compile_mlir_string(mlir_text)
    """

    targetid: str = "gfx942"
    codeobj: str = "5"
    mma_type: Optional["MMAType"] = None

    def compile_mlir_string(self, mlir_text: str) -> str:
        """
        Compile MLIR text to complete AMDGCN assembly.

        Args:
            mlir_text: MLIR module text

        Returns:
            Complete assembly text ready for assembler
        """
        from wave_lang.support.ir_imports import Context, Module, func_d, MemRefType
        from .mlir_walker import IRWalker
        from .metadata_emitter import MetadataEmitter, create_metadata
        from .mlir_analysis import (
            walk_ops_recursively,
            detect_needed_workgroup_ids,
            extract_translation_info,
            should_skip_function,
        )

        def _is_64bit_pointer_type(mlir_type) -> bool:
            """Check if an MLIR type is a 64-bit pointer-like type.

            Returns True for:
            - MemRefType (buffer pointers)
            - !stream.binding (IREE stream bindings, also 64-bit)

            Returns False for scalar types (i32, f32, i64, f64, etc.)
            """
            # MemRefType is a 64-bit pointer
            if isinstance(mlir_type, MemRefType):
                return True
            # !stream.binding is also a 64-bit pointer (IREE stream dialect)
            type_str = str(mlir_type)
            if "stream.binding" in type_str:
                return True
            return False

        def _validate_kernargs_for_preloading(fn, kernel_name: str) -> None:
            """Validate that all kernel arguments are 64-bit pointer-like types.

            Kernel argument preloading assumes all args are 64-bit pointers
            (2 SGPRs each). Raises ValueError if any argument doesn't meet this.

            Accepted types: MemRefType, !stream.binding
            Rejected types: scalars (i32, f32, i64, f64, etc.)
            """
            for i, arg in enumerate(fn.entry_block.arguments):
                if not _is_64bit_pointer_type(arg.type):
                    raise ValueError(
                        f"Kernel argument preloading requires all arguments to be "
                        f"64-bit pointers (MemRefType or stream.binding). Argument "
                        f"{i} of kernel '{kernel_name}' has type {arg.type}. Either "
                        f"target a non-gfx95* architecture or ensure all arguments "
                        f"are memory references."
                    )

        all_lines: List[str] = []

        with Context() as ctx:
            ctx.allow_unregistered_dialects = True
            module = Module.parse(mlir_text)

            for fn in walk_ops_recursively(module.operation):
                if not isinstance(fn, func_d.FuncOp):
                    continue

                kernel_name = fn.sym_name.value

                # Skip non-kernel functions (async wrappers, benchmark scaffolding)
                if should_skip_function(fn):
                    continue

                num_args = len(list(fn.entry_block.arguments))

                # Extract kernel metadata
                ti = extract_translation_info(fn)
                wg_size, subgroup_size = ti.wg_size, ti.subgroup_size

                # Detect workgroup ID needs
                needs_wgid_x, needs_wgid_y, needs_wgid_z = detect_needed_workgroup_ids(
                    fn
                )

                # Create metadata for prologue/epilogue (via MetadataEmitter)
                # Kernel argument preloading: 2 SGPRs per pointer arg.
                # This tells hardware to preload kernel args into SGPRs at
                # kernel start (s[2:3], s[4:5], etc.), reducing latency.
                #
                # Requirements for preloading:
                # - Target must support preloading (gfx9xx, specifically gfx95* for MI350X)
                # - Code object version must be >= 5 (preloading added in COv5)
                # - All kernel args must be 64-bit pointers (2 SGPRs each)
                #
                # On gfx950/MI350X, preloading provides ~20-30% speedup for
                # small kernels by eliminating s_load latency at kernel start.
                #
                # The implementation follows LLVM's pattern exactly:
                # 1. s_load into preload locations (s[2:3], s[4:5], etc.)
                # 2. s_waitcnt
                # 3. s_branch to 256-byte aligned entry point
                # 4. .p2align 8 for alignment
                # 5. Copy from preload locations to SRD ranges
                # 6. Rest of kernel code

                # Gate preloading by target and code object version
                codeobj_version = int(self.codeobj) if self.codeobj.isdigit() else 0
                # NOTE: Do NOT enable preloading for all gfx9* targets.
                # Restrict to gfx95* (MI350X/gfx950 family) until validated more broadly.
                target_supports_preload = self.targetid.startswith("gfx95")
                use_preloading = target_supports_preload and codeobj_version >= 5

                # Validate that all kernel args are 64-bit pointers (MemRefType)
                # before enabling preloading. This enforces the 2-SGPRs-per-arg assumption.
                if use_preloading:
                    _validate_kernargs_for_preloading(fn, kernel_name)

                # Maximum preloadable: 16 SGPRs = 8 pointer args (hardware limit)
                MAX_PRELOAD_SGPRS = 16
                kernarg_preload_length = num_args * 2 if use_preloading else 0
                if kernarg_preload_length > MAX_PRELOAD_SGPRS:
                    # Exceeds hardware limit; disable preloading for this kernel
                    kernarg_preload_length = 0
                    use_preloading = False
                metadata = create_metadata(
                    name=kernel_name,
                    targetid=self.targetid,
                    codeobj=self.codeobj,
                    wg_size=wg_size,
                    subgroup_size=subgroup_size,
                    needs_wgid=(needs_wgid_x, needs_wgid_y, needs_wgid_z),
                    num_args=num_args,
                    kernarg_preload_length=kernarg_preload_length,
                )

                # Emit prologue (assembler directives)
                meta_emitter = MetadataEmitter(metadata)
                prologue_lines = meta_emitter.emit_prologue()

                # Create kernel context with proper thread ID bounds
                num_waves = max(
                    1, wg_size[0] * wg_size[1] * wg_size[2] // subgroup_size
                )
                kernel_ctx = KernelCompilationContext(
                    use_flat_tid=(num_waves > 1),
                    use_workgroup_ids=(needs_wgid_x, needs_wgid_y, needs_wgid_z),
                    tid_ub_x=wg_size[0],
                    tid_ub_y=wg_size[1],
                    tid_ub_z=wg_size[2] if len(wg_size) > 2 else 1,
                    subgroup_size=subgroup_size,
                    wg_size=wg_size,
                    mma_type=self.mma_type,
                    use_kernarg_preloading=use_preloading,
                    num_kernargs=num_args,
                    kernel_name=kernel_name,
                )

                # Emit kernarg loading at the start of kernel IR
                kernel_ctx.emit_kernargs(num_args)

                # Walk MLIR and emit to kernel IR
                walker = IRWalker(kernel_ctx)
                kernel_info = walker.interpret_func(fn)

                # Finalize kernel IR (adds s_endpgm, runs allocation, renders)
                body_lines, stats = kernel_ctx.finalize()

                # Get LDS size from kernel_info
                lds_size_bytes = getattr(kernel_info, "lds_size_bytes", 0)

                # Patch prologue with actual resource values
                patched_prologue = MetadataEmitter.patch_resource_usage(
                    prologue_lines,
                    stats.peak_vgprs,
                    stats.peak_sgprs,
                    getattr(stats, "peak_agprs", 0),
                    lds_size_bytes,
                    self.targetid,
                )

                # Emit epilogue (YAML metadata)
                metadata.vgprs_used = stats.peak_vgprs
                metadata.sgprs_used = stats.peak_sgprs
                metadata.agprs_used = getattr(stats, "peak_agprs", 0)
                metadata.lds_size_bytes = lds_size_bytes
                epilogue_lines = meta_emitter.emit_epilogue()

                # Combine all lines: prologue + body + epilogue
                all_lines.extend(patched_prologue)
                all_lines.extend(body_lines)
                all_lines.extend(epilogue_lines)

        return "\n".join(all_lines)
