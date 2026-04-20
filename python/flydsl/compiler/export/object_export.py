# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""AOT export: dump compiled FlyDSL kernels to relocatable ELF .o files.

The .o file contains:
- x86 host wrapper code (calls mgpuModuleLoad, mgpuLaunchKernel)
- Embedded GPU binary (HSACO fatbin) inside a gpu.binary-derived data section
- Metadata globals (function_name, version, arch) for introspection

Link with: -lfly_jit_runtime -lmlir_c_runner_utils

Implementation:
- Two-phase symbol prefixing via MLIR IR tree walk:
    Phase 1 (collect): scan all ops, collect defined function names and
        gpu.binary names that need prefixing.
    Phase 2 (rewrite): walk again, rename definitions and all references.
- Metadata encoded via llvm.GlobalOp using MLIR Python API.
- In-process ExecutionEngine with PIC + object dump.

FlyDSL's compiled IR retains gpu.binary + gpu.launch_func (the gpu-to-llvm
lowering happens during ExecutionEngine creation, not in the pass pipeline).
We prefix the gpu.binary sym_name so that the auto-generated constructor/
destructor symbols become {prefix}_kernels_load, {prefix}_kernels_module,
avoiding link-time collisions when linking multiple exported .o files.
"""

import os
import tempfile
from typing import TYPE_CHECKING

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm

if TYPE_CHECKING:
    from ..jit_executor import CompiledArtifact

# Suffix constants for metadata global names
_META_ENTRY = "entry_name"
_META_VER = "fly_version"
_META_ARCH = "gpu_arch"


def _collect_defined_names(module: ir.Module):
    """Phase 1: Scan the module and collect names that need prefixing.

    Returns:
        func_defs: set of llvm.func names that have a body (not declarations)
        gpu_bin_map: dict mapping gpu.binary sym_name -> None (placeholder for new name)
    """
    func_defs = set()
    gpu_bin_names = set()

    def visitor(op):
        # llvm.func with a body = defined function (not a declaration)
        if op.name == "llvm.func":
            regions = op.opview.operation.regions
            if len(regions) > 0 and len(regions[0].blocks) > 0:
                func_defs.add(op.attributes["sym_name"].value)
        # gpu.binary = embedded GPU kernel module
        elif op.name == "gpu.binary" and "sym_name" in op.attributes:
            gpu_bin_names.add(op.attributes["sym_name"].value)
        return ir.WalkResult.ADVANCE

    module.operation.walk(visitor)
    return func_defs, gpu_bin_names


def _rewrite_symbols(module: ir.Module, prefix: str, func_defs: set, gpu_bin_names: set):
    """Phase 2: Walk the module and rewrite all symbols that were collected.

    Handles:
    - llvm.func definitions: rename sym_name
    - llvm.call: rename callee if it references a defined function
    - llvm.mlir.addressof: rename global_name if it references a defined function
    - llvm.mlir.global_ctors / global_dtors: rename entries
    - gpu.binary: rename sym_name
    - gpu.launch_func: update kernel reference to match renamed gpu.binary
    """
    gpu_rename_map = {name: f"{prefix}_{name}" for name in gpu_bin_names}

    def _prefix(name):
        return f"{prefix}_{name}"

    def _rename_ref_list(attr_list, known_names):
        """Rename FlatSymbolRefAttr entries in an ArrayAttr if they're in known_names."""
        return ir.ArrayAttr.get([
            ir.FlatSymbolRefAttr.get(_prefix(ref.value))
            if ref.value in known_names else ref
            for ref in attr_list
        ])

    def rewriter(op):
        op_name = op.name

        # --- Function definitions ---
        if op_name == "llvm.func":
            regions = op.opview.operation.regions
            if len(regions) > 0 and len(regions[0].blocks) > 0:
                cur = op.attributes["sym_name"].value
                if cur in func_defs:
                    op.attributes["sym_name"] = ir.StringAttr.get(_prefix(cur))

        # --- GPU binary module name ---
        elif op_name == "gpu.binary" and "sym_name" in op.attributes:
            cur = op.attributes["sym_name"].value
            if cur in gpu_rename_map:
                op.attributes["sym_name"] = ir.StringAttr.get(gpu_rename_map[cur])

        # --- GPU kernel launch reference ---
        elif op_name == "gpu.launch_func" and "kernel" in op.attributes:
            parts = op.attributes["kernel"].value  # e.g. ['kernels', 'k_0']
            if parts and parts[0] in gpu_rename_map:
                op.attributes["kernel"] = ir.SymbolRefAttr.get(
                    [gpu_rename_map[parts[0]]] + parts[1:]
                )

        # --- Call sites ---
        elif op_name == "llvm.call" and "callee" in op.attributes:
            target = op.attributes["callee"].value
            if target in func_defs:
                op.attributes["callee"] = ir.FlatSymbolRefAttr.get(_prefix(target))

        # --- Address-of references ---
        elif op_name == "llvm.mlir.addressof" and "global_name" in op.attributes:
            ref = op.attributes["global_name"].value
            if ref in func_defs:
                op.attributes["global_name"] = ir.FlatSymbolRefAttr.get(_prefix(ref))

        # --- Global constructors / destructors ---
        elif op_name == "llvm.mlir.global_ctors" and "ctors" in op.attributes:
            op.attributes["ctors"] = _rename_ref_list(op.attributes["ctors"], func_defs)
        elif op_name == "llvm.mlir.global_dtors" and "dtors" in op.attributes:
            op.attributes["dtors"] = _rename_ref_list(op.attributes["dtors"], func_defs)

        return ir.WalkResult.ADVANCE

    module.operation.walk(rewriter)


def _clone_and_prefix(ir_source, prefix: str) -> ir.Module:
    """Clone the compiled MLIR module into a fresh context and prefix all symbols.

    Args:
        ir_source: The compiled MLIR module (ir.Module or str).
        prefix: Prefix for all defined symbols.

    Returns:
        A new ir.Module in a fresh context with prefixed symbols.
    """
    ir_text = str(ir_source) if not isinstance(ir_source, str) else ir_source
    ctx = ir.Context()
    with ctx:
        ctx.load_all_available_dialects()
        module = ir.Module.parse(ir_text)
        func_defs, gpu_bin_names = _collect_defined_names(module)
        _rewrite_symbols(module, prefix, func_defs, gpu_bin_names)
    return module


def _add_metadata(module: ir.Module, prefix: str, entry_name: str, arch: str, version: str):
    """Add metadata globals to the module as null-terminated string constants.

    Creates:
    - {prefix}_function_name: prefixed entry function name
    - {prefix}_version: FlyDSL version string
    - {prefix}_arch: target GPU architecture
    """
    entries = {
        f"{prefix}_{_META_ENTRY}": entry_name + "\0",
        f"{prefix}_{_META_VER}": version + "\0",
        f"{prefix}_{_META_ARCH}": arch + "\0",
    }
    with module.context, ir.Location.unknown():
        with ir.InsertionPoint(module.body):
            for name, val in entries.items():
                llvm.GlobalOp(
                    sym_name=name,
                    global_type=ir.Type.parse(f"!llvm.array<{len(val)} x i8>"),
                    linkage=ir.Attribute.parse("#llvm.linkage<external>"),
                    value=ir.StringAttr.get(val),
                )


def dump_to_object(
    artifact: "CompiledArtifact",
    function_prefix: str,
    arch: str = "",
) -> bytes:
    """Export a CompiledArtifact as a relocatable ELF .o file.

    Process:
    1. Clone module into a fresh context, prefix all symbols
    2. Add metadata globals
    3. Create ExecutionEngine with PIC + object dump
    4. Write to temp file and return bytes

    Args:
        artifact: A compiled kernel artifact.
        function_prefix: Prefix for all exported symbols.
        arch: Target GPU architecture. Auto-detected if empty.

    Returns:
        ELF .o file contents as bytes.
    """
    from flydsl._mlir.execution_engine import ExecutionEngine
    from ..jit_executor import _resolve_runtime_libs

    if not arch:
        from ...runtime.device import get_rocm_arch
        try:
            arch = get_rocm_arch()
        except Exception:
            arch = "unknown"

    import flydsl

    module = _clone_and_prefix(artifact._ir_text, function_prefix)

    with module.context:
        _add_metadata(
            module, function_prefix,
            f"{function_prefix}_{artifact._entry}",
            arch, flydsl.__version__,
        )

        engine = ExecutionEngine(
            module,
            opt_level=3,
            shared_libs=_resolve_runtime_libs(),
            enable_object_dump=True,
            enable_pic=True,
        )

        fd, tmp_path = tempfile.mkstemp(suffix=".o")
        os.close(fd)
        try:
            engine.dump_to_object_file(tmp_path)
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            os.unlink(tmp_path)
