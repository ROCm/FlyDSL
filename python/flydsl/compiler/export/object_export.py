# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""AOT export: dump compiled kernels to relocatable ELF .o files.

Two-phase symbol prefixing (collect then rewrite) avoids link-time
collisions when multiple exported .o files are linked together.
"""

import os
import tempfile
from typing import TYPE_CHECKING

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm

if TYPE_CHECKING:
    from ..jit_executor import CompiledArtifact

_META_ENTRY = "entry_name"
_META_VER = "fly_version"
_META_ARCH = "gpu_arch"


def _collect_defined_names(module: ir.Module):
    """Collect llvm.func definitions and gpu.binary names that need prefixing."""
    func_defs = set()
    gpu_bin_names = set()

    def visitor(op):
        if op.name == "llvm.func":
            regions = op.opview.operation.regions
            if len(regions) > 0 and len(regions[0].blocks) > 0:
                func_defs.add(op.attributes["sym_name"].value)
        elif op.name == "gpu.binary" and "sym_name" in op.attributes:
            gpu_bin_names.add(op.attributes["sym_name"].value)
        return ir.WalkResult.ADVANCE

    module.operation.walk(visitor)
    return func_defs, gpu_bin_names


def _rewrite_symbols(module: ir.Module, prefix: str, func_defs: set, gpu_bin_names: set):
    """Rename all collected symbols with the given prefix."""
    gpu_rename_map = {name: f"{prefix}_{name}" for name in gpu_bin_names}

    def _prefix(name):
        return f"{prefix}_{name}"

    def _rename_ref_list(attr_list, known_names):
        return ir.ArrayAttr.get([
            ir.FlatSymbolRefAttr.get(_prefix(ref.value))
            if ref.value in known_names else ref
            for ref in attr_list
        ])

    def rewriter(op):
        op_name = op.name

        if op_name == "llvm.func":
            regions = op.opview.operation.regions
            if len(regions) > 0 and len(regions[0].blocks) > 0:
                cur = op.attributes["sym_name"].value
                if cur in func_defs:
                    op.attributes["sym_name"] = ir.StringAttr.get(_prefix(cur))

        elif op_name == "gpu.binary" and "sym_name" in op.attributes:
            cur = op.attributes["sym_name"].value
            if cur in gpu_rename_map:
                op.attributes["sym_name"] = ir.StringAttr.get(gpu_rename_map[cur])

        elif op_name == "gpu.launch_func" and "kernel" in op.attributes:
            parts = op.attributes["kernel"].value
            if parts and parts[0] in gpu_rename_map:
                op.attributes["kernel"] = ir.SymbolRefAttr.get(
                    [gpu_rename_map[parts[0]]] + parts[1:]
                )

        elif op_name == "llvm.call" and "callee" in op.attributes:
            target = op.attributes["callee"].value
            if target in func_defs:
                op.attributes["callee"] = ir.FlatSymbolRefAttr.get(_prefix(target))

        elif op_name == "llvm.mlir.addressof" and "global_name" in op.attributes:
            ref = op.attributes["global_name"].value
            if ref in func_defs:
                op.attributes["global_name"] = ir.FlatSymbolRefAttr.get(_prefix(ref))

        elif op_name == "llvm.mlir.global_ctors" and "ctors" in op.attributes:
            op.attributes["ctors"] = _rename_ref_list(op.attributes["ctors"], func_defs)
        elif op_name == "llvm.mlir.global_dtors" and "dtors" in op.attributes:
            op.attributes["dtors"] = _rename_ref_list(op.attributes["dtors"], func_defs)

        return ir.WalkResult.ADVANCE

    module.operation.walk(rewriter)


def _clone_and_prefix(ir_source, prefix: str) -> ir.Module:
    """Clone module into a fresh context and prefix all defined symbols."""
    ir_text = str(ir_source) if not isinstance(ir_source, str) else ir_source
    ctx = ir.Context()
    with ctx:
        ctx.load_all_available_dialects()
        module = ir.Module.parse(ir_text)
        func_defs, gpu_bin_names = _collect_defined_names(module)
        _rewrite_symbols(module, prefix, func_defs, gpu_bin_names)
    return module


def _add_metadata(module: ir.Module, prefix: str, entry_name: str, arch: str, version: str):
    """Add metadata globals (entry_name, version, arch) as string constants."""
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
    output_path: str = "",
    arch: str = "",
) -> bytes:
    """Export a CompiledArtifact as a relocatable ELF .o file (bytes).

    If output_path is set, also writes the .o to that path.
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

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            engine.dump_to_object_file(output_path)
            with open(output_path, "rb") as f:
                return f.read()
        else:
            fd, tmp_path = tempfile.mkstemp(suffix=".o")
            os.close(fd)
            try:
                engine.dump_to_object_file(tmp_path)
                with open(tmp_path, "rb") as f:
                    return f.read()
            finally:
                os.unlink(tmp_path)
