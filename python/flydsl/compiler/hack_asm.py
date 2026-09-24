# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Run a hand-edited ``.s`` in place of the generated device code (``FLYDSL_HACK_UT_ASM``).

Debug/UT only.  This closes the loop on the ISA dump: ``FLYDSL_DUMP_IR`` writes a
``.s``, you edit it, and pointing ``FLYDSL_HACK_UT_ASM`` at the result makes an
otherwise unmodified test or benchmark execute the hand-tuned assembly.

The override is scoped by kernel symbol, and the ``.s`` must cover every kernel in a
module it applies to.  One caveat has no guard: constexpr specializations of a kernel
share a symbol, so one ``.s`` is substituted into all of them -- override one
specialization at a time.
"""

from __future__ import annotations

import atexit
import hashlib
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


# Assembling is ~100ms; a process compiling N kernels must not pay it N times.  Keyed on
# the .s content, so an edit between compilations takes effect and never serves stale.
_ASSEMBLE_CACHE: dict = {}


class _MatchLog:
    """What the ``.s`` matched across the process, and the exit warning it drives.

    A run that never matches anything is a typo in a kernel name, not a no-op, so it
    is reported at exit rather than passing silently.  One object rather than three
    module globals so a test swaps a single attribute to isolate itself.
    """

    def __init__(self):
        self.matched: set = set()
        self.skipped: set = set()
        self.hooked = False

    def hook_once(self) -> None:
        """Register the exit warning on first use."""
        if not self.hooked:
            atexit.register(self.report)
            self.hooked = True

    def report(self) -> None:
        if self.matched or not self.skipped:
            return
        print(
            f"[flydsl.compile] WARNING: FLYDSL_HACK_UT_ASM never matched a compiled kernel. "
            f"The .s declares no symbol among {sorted(self.skipped)}, so every kernel ran "
            f"the compiler's own code -- these are NOT the hand-edit's numbers."
        )


_MATCH_LOG = _MatchLog()


def _assemble_isa_to_hsaco(asm_path: Path, arch: str, digest: str) -> bytes:
    """Assemble a GCN ``.s`` into an HSA code object with the ROCm toolchain's clang.

    Memoized on ``digest`` (the file's content hash) so a process compiling many
    kernels pays clang once, and an edit between compilations is never served stale.

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

    cached = _ASSEMBLE_CACHE.get((digest, arch))
    if cached is not None:
        return cached

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
        obj = out.read_bytes()

    _ASSEMBLE_CACHE[(digest, arch)] = obj
    return obj


def substitute_hacked_asm(module: ir.Module, *, arch: str, func_name: str, module_kernel_names) -> bool:
    """Replace the compiled device object with the ISA at ``FLYDSL_HACK_UT_ASM``.

    The ``.s`` is assembled with the ROCm toolchain's clang and spliced into the
    existing ``gpu.binary``.  Launch geometry and kernel lookup are unaffected: the
    host side still takes grid/block/smem from the traced ``gpu.launch_func`` and
    resolves the kernel by name, so the ISA must keep the original kernel symbol.

    Returns whether the substitution applied.  A module whose kernels the ``.s`` does
    not name is left to the compiler and reported as skipped: a benchmark that compiles
    a hand-edited kernel alongside untouched ones has to keep running.  The
    process-wide check that *something* matched is what catches a mistyped symbol.
    """
    from .._mlir._mlir_libs._mlirDialectsGPU import ObjectAttr
    from .._mlir.dialects._gpu_enum_gen import CompilationTarget

    asm_path = Path(env.debug.hack_ut_asm.strip()).expanduser()
    if not asm_path.is_file():
        raise DSLCompileError(f"FLYDSL_HACK_UT_ASM points at a missing file: {asm_path}")

    _MATCH_LOG.hook_once()

    # A module with no kernels cannot be reasoned about: skipping it silently would
    # leave _report_unmatched with nothing to warn about, which is the one outcome this
    # tool must never produce.
    module_kernels = set(module_kernel_names)
    if not module_kernels:
        raise DSLCompileError(
            f"FLYDSL_HACK_UT_ASM={asm_path}: no kernel symbols were found in the module "
            f"compiled for {func_name!r}; refusing to guess whether the override applies"
        )

    # Scope the override to the kernels the .s actually declares.  Substituting into an
    # unrelated kernel would report the compiler's numbers as if they were the
    # hand-edit's; refusing outright would break any process that compiles more than the
    # one kernel under study.  Skip, and let _report_unmatched catch a .s that matched
    # nothing at all.
    isa_text = asm_path.read_text(encoding="utf-8")
    digest = hashlib.sha256(isa_text.encode("utf-8")).hexdigest()
    asm_kernel_names = _infer_kernel_names_from_s(isa_text)
    matched = asm_kernel_names & module_kernels
    if not matched:
        _MATCH_LOG.skipped.update(module_kernels)
        print(
            f"[flydsl.compile] FLYDSL_HACK_UT_ASM: {func_name!r} keeps compiler codegen "
            f"({sorted(module_kernels)} not in the .s)"
        )
        return False

    # The whole gpu.binary is replaced, so a .s that declares only some of the module's
    # kernels would drop the rest from the code object while the host still launches
    # them by name -- a load failure at best, and silently wrong numbers at worst.
    missing = module_kernels - asm_kernel_names
    if missing:
        raise DSLCompileError(
            f"FLYDSL_HACK_UT_ASM={asm_path}: the .s declares {sorted(matched)} but the "
            f"module compiled for {func_name!r} also contains {sorted(missing)}. "
            f"Replacing the binary would drop those kernels; dump and edit a .s that "
            f"covers every kernel in the module."
        )

    obj_bytes = _assemble_isa_to_hsaco(asm_path, arch, digest)

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
    _MATCH_LOG.matched.update(matched)
    print(
        f"[flydsl.compile] FLYDSL_HACK_UT_ASM: substituted {len(obj_bytes)} bytes for "
        f"{sorted(matched)} from {asm_path}"
    )
    if env.debug.hack_ut_asm_save:
        # The exact bytes that were substituted, for llvm-objdump or a profiler.  Next to
        # the .s rather than in the dump dir: re-running with FLYDSL_DUMP_IR would write
        # a fresh dump there and could clobber an .s edited in place.
        out = asm_path.with_suffix(".hsaco")
        out.write_bytes(obj_bytes)
        print(f"[flydsl.compile] FLYDSL_HACK_UT_ASM: saved code object -> {out}")
    return True
