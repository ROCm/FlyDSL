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
        from .ir_imports import Context, Module, MemRefType, func_d, gpu_d, arith_d
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

        def _can_preload_kernargs(fn, kernel_name: str) -> bool:
            """Return True if all kernel arguments are 64-bit pointer-like.

            Kernel argument preloading assumes all args are 64-bit pointers
            (2 SGPRs each). GPU-dialect frontends may include scalar parameters,
            in which case preloading must be disabled.

            Accepted types: MemRefType, !stream.binding
            Rejected types: scalars (i32, f32, i64, f64, etc.)
            """
            entry_block = (
                fn.entry_block
                if hasattr(fn, "entry_block")
                else fn.operation.regions[0].blocks[0]
            )
            for _i, arg in enumerate(entry_block.arguments):
                if not _is_64bit_pointer_type(arg.type):
                    return False
            return True

        def _attrs_to_dict(attrs):
            return {a.name: a.attr for a in attrs}

        def _try_eval_index_const(v) -> Optional[int]:
            # Best-effort: resolve `arith.constant` to Python int.
            try:
                if v is None or v.owner is None:
                    return None
                opv = v.owner.opview
                if isinstance(opv, arith_d.ConstantOp):
                    import re

                    m = re.match(r"^-?\d+", str(opv.value).strip())
                    if m:
                        return int(m.group(0))
            except Exception:
                return None
            return None

        def _infer_launch_workgroup_sizes(module_op) -> dict[tuple[str, str], tuple[int, int, int]]:
            """Infer workgroup sizes from `gpu.launch_func` call sites.

            Returns a map keyed by (gpu_module_sym_name, gpu_func_sym_name).
            """
            inferred: dict[tuple[str, str], tuple[int, int, int]] = {}
            for op in walk_ops_recursively(module_op.operation):
                if not isinstance(op, gpu_d.LaunchFuncOp):
                    continue
                attrs = _attrs_to_dict(op.attributes)
                kernel_attr = attrs.get("kernel", None)
                if kernel_attr is None:
                    continue
                # Printed like: @vec_kernels::@vec_add
                kernel_ref = str(kernel_attr)
                if "::" in kernel_ref:
                    mod_ref, fn_ref = kernel_ref.split("::", 1)
                else:
                    mod_ref, fn_ref = "", kernel_ref
                gpu_mod = mod_ref.lstrip("@")
                gpu_fn = fn_ref.lstrip("@")

                # Operand order (iree gpu.launch_func):
                #   0..2: grid sizes (blocks in)
                #   3..5: block sizes (threads in)
                #   6.. : kernel operands
                if len(op.operands) < 6:
                    continue
                bx = _try_eval_index_const(op.operands[3])
                by = _try_eval_index_const(op.operands[4])
                bz = _try_eval_index_const(op.operands[5])
                if bx is None:
                    continue
                inferred[(gpu_mod, gpu_fn)] = (bx, by or 1, bz or 1)
            return inferred

        all_lines: List[str] = []

        with Context() as ctx:
            ctx.allow_unregistered_dialects = True
            module = Module.parse(mlir_text)

            inferred_wg_sizes = _infer_launch_workgroup_sizes(module)

            for fn in walk_ops_recursively(module.operation):
                is_func_kernel = isinstance(fn, func_d.FuncOp)
                is_gpu_kernel = isinstance(fn, gpu_d.GPUFuncOp)
                if not (is_func_kernel or is_gpu_kernel):
                    continue

                # func.func exposes `sym_name`; gpu.func may only carry it as an
                # attribute in some bindings.
                if hasattr(fn, "sym_name") and hasattr(fn.sym_name, "value"):
                    kernel_name = fn.sym_name.value
                else:
                    fn_attrs = _attrs_to_dict(fn.attributes)
                    sym = fn_attrs.get("sym_name")
                    if sym is None:
                        kernel_name = str(fn)
                    else:
                        kernel_name = (
                            sym.value if hasattr(sym, "value") else str(sym).strip('"')
                        )

                # Skip non-kernel functions (async wrappers, benchmark scaffolding)
                if should_skip_function(fn):
                    continue

                # For `func.func` kernels, require translation_info (Wave/IREE pipeline).
                # If missing, treat as a wrapper (e.g. a host stub like `__call__`).
                if is_func_kernel:
                    fn_attrs = _attrs_to_dict(fn.attributes)
                    if "translation_info" not in fn_attrs:
                        continue

                # For GPU dialect kernels, require `gpu.kernel` marker.
                parent_name = ""
                if is_gpu_kernel:
                    fn_attrs = _attrs_to_dict(fn.attributes)
                    if "gpu.kernel" not in fn_attrs:
                        continue
                    parent_mod = fn.operation.parent.opview
                    if isinstance(parent_mod, gpu_d.GPUModuleOp):
                        parent_name = parent_mod.sym_name.value

                entry_block = (
                    fn.entry_block
                    if hasattr(fn, "entry_block")
                    else fn.operation.regions[0].blocks[0]
                )
                num_args = len(list(entry_block.arguments))

                # Extract kernel metadata.
                #
                # - For func.func: require translation_info.
                # - For gpu.func: infer wg_size from launch sites and default subgroup_size=64.
                if is_gpu_kernel:
                    wg_size = inferred_wg_sizes.get((parent_name, kernel_name))
                    if wg_size is None:
                        # Fallback: match by function name only.
                        for (_mod_name, fn_name), wgs in inferred_wg_sizes.items():
                            if fn_name == kernel_name:
                                wg_size = wgs
                                break
                    if wg_size is None and len(inferred_wg_sizes) == 1:
                        wg_size = next(iter(inferred_wg_sizes.values()))
                    if wg_size is None:
                        wg_size = (256, 1, 1)
                    subgroup_size = 64
                else:
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

                # Only enable preloading when all args are pointer-like.
                if use_preloading and not _can_preload_kernargs(fn, kernel_name):
                    use_preloading = False

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
