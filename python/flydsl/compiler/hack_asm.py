# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Run a hand-edited ``.s`` in place of the generated device code (``FLYDSL_HACK_UT_ASM``).

Debug/UT only.  This closes the loop on the ISA dump: ``FLYDSL_DUMP_IR`` writes a
``.s``, you edit it, and pointing ``FLYDSL_HACK_UT_ASM`` at the result makes an
otherwise unmodified test or benchmark execute the hand-tuned assembly.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from .._mlir import ir
from ..utils import env
from .diagnostics import DSLCompileError


def _infer_kernel_names_from_s(isa_text: str) -> set:
    """Extract kernel names from the ``.amdhsa_kernel`` descriptors in a GCN ``.s``."""
    names = set()
    for line in isa_text.splitlines():
        stripped = line.strip()
        if stripped.startswith(".amdhsa_kernel "):
            name = stripped[len(".amdhsa_kernel ") :].strip()
            if name:
                names.add(name)
    return names


def _assemble_isa_to_hsaco(asm_path: Path, arch: str) -> bytes:
    """Assemble a GCN ``.s`` into an HSA code object with the ROCm toolchain's clang.

    clang drives the same LLVM MC assembler and ``ld.lld`` that
    ``gpu-module-to-binary`` reaches internally, so the object is byte-identical to
    the one the compiler would have produced from that ISA.  The triple has to spell
    the ``unknown`` environment explicitly: anything shorter normalizes to a target id
    that does not match the ``.amdgcn_target`` MLIR writes into the dump.  Assembling
    against the wrong ``arch`` trips that same check, which is the arch guard.
    """
    clang = Path(os.environ.get("ROCM_PATH") or "/opt/rocm") / "llvm" / "bin" / "clang"
    if not clang.is_file():
        raise DSLCompileError(f"FLYDSL_HACK_UT_ASM needs the ROCm toolchain's clang, which is missing: {clang}")

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "hacked.hsaco"
        proc = subprocess.run(
            # fmt: off
            [
                str(clang), "-x", "assembler",
                "-target", "amdgcn-amd-amdhsa-unknown", f"-mcpu={arch}",
                "-o", str(out), str(asm_path),
            ],
            # fmt: on
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise DSLCompileError(f"FLYDSL_HACK_UT_ASM={asm_path}: assembling the ISA failed:\n{proc.stderr.strip()}")
        return out.read_bytes()


def substitute_hacked_asm(module: ir.Module, *, arch: str, func_name: str, module_kernel_names) -> None:
    """Replace the compiled device object with the ISA at ``FLYDSL_HACK_UT_ASM``.

    The ``.s`` is assembled with the ROCm toolchain's clang and spliced into the
    existing ``gpu.binary``.  Launch geometry and kernel lookup are unaffected: the
    host side still takes grid/block/smem from the traced ``gpu.launch_func`` and
    resolves the kernel by name, so the ISA must keep the original kernel symbol.
    """
    from .._mlir._mlir_libs._mlirDialectsGPU import ObjectAttr
    from .._mlir.dialects._gpu_enum_gen import CompilationTarget

    asm_path = Path(env.debug.hack_ut_asm.strip()).expanduser()
    if not asm_path.is_file():
        raise DSLCompileError(f"FLYDSL_HACK_UT_ASM points at a missing file: {asm_path}")

    # Refuse to substitute into an unrelated kernel: silently compiling the
    # original would report the compiler's numbers as if they were the hand-edit's.
    asm_kernel_names = _infer_kernel_names_from_s(asm_path.read_text(encoding="utf-8"))
    matched = asm_kernel_names & set(module_kernel_names)
    if not matched:
        raise DSLCompileError(
            f"FLYDSL_HACK_UT_ASM={asm_path}: none of the kernel symbols in the .s "
            f"({sorted(asm_kernel_names)}) match the kernels compiled for {func_name!r} "
            f"({sorted(module_kernel_names)}); refusing to substitute"
        )

    obj_bytes = _assemble_isa_to_hsaco(asm_path, arch)

    binaries = [op.operation for op in module.body.operations if op.operation.name == "gpu.binary"]
    if len(binaries) != 1:
        raise DSLCompileError(f"expected exactly one gpu.binary in the compiled module, found {len(binaries)}")
    gpu_binary = binaries[0]
    objects = ir.ArrayAttr(gpu_binary.attributes["objects"])
    if len(objects) != 1:
        raise DSLCompileError(f"expected exactly one gpu.object on {gpu_binary.name}, found {len(objects)}")
    obj_attr = ObjectAttr(objects[0])

    # format=Binary, never Assembly: FlyLLVMTranslation routes Assembly objects to
    # mgpuModuleLoadJIT, which is not implemented for HIP.  clang produced a linked
    # code object, which hipModuleLoadData takes directly.  The kernel metadata from
    # the original compilation is dropped rather than carried over stale.
    new_obj = ObjectAttr.get(
        obj_attr.target,
        int(CompilationTarget.Binary),
        obj_bytes,
        obj_attr.properties,
        None,
        context=module.context,
    )
    gpu_binary.attributes["objects"] = ir.ArrayAttr.get([new_obj])
    print(
        f"[flydsl.compile] FLYDSL_HACK_UT_ASM: substituted {len(obj_bytes)} bytes for "
        f"{sorted(matched)} from {asm_path}"
    )
