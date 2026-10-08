#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Patch a JIT cache entry to run a hand-edited ``.s`` (debug only).

Answers "what if these few instructions were different" without a compiler
rebuild.  With ``$C`` a scratch cache directory that starts empty:

    FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=dumps python test_x.py      # writes NN_final_isa.s
    FLYDSL_RUNTIME_CACHE_DIR=$C python test_x.py                 # caches the compiler's artifact
    vim dumps/<kernel>/NN_final_isa.s
    python3 scripts/patch_cache_asm.py $C dumps/<kernel>/NN_final_isa.s
    FLYDSL_RUNTIME_CACHE_DIR=$C FLYDSL_RUNTIME_RUN_ONLY=1 python test_x.py

A cache entry keeps the compiled module as MLIR text and rebuilds its
ExecutionEngine from that text on load, so replacing the ``gpu.binary`` object
in the text is enough.  RUN_ONLY turns a cache miss into an error instead of a
silent recompile.

Run only the case under study in both the dump and the caching step.  Constexpr
specializations of a kernel share its symbol, so the dump directory keeps only
the last one compiled, and their cache entries cannot be told apart: the tool
refuses a ``.s`` that matches more than one, because patching the wrong one would
leave the run executing the compiler's code with nothing to show for it.
"""

import argparse
import os
import pickle
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_LAUNCH_RE = re.compile(r"gpu\.launch_func\b[^@\n]*@[\w$.]+::@([\w$.]+)")
_CHIP_RE = re.compile(r'chip\s*=\s*"([^"]+)"')


def kernel_names_in_isa(isa_text: str) -> set:
    """Kernel symbols declared by ``.amdhsa_kernel`` in a GCN ``.s``."""
    names = set()
    for line in isa_text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == ".amdhsa_kernel":
            names.add(parts[1])
    return names


def launched_kernels(ir_text: str) -> set:
    """Kernel symbols the host code of a cached module launches."""
    return set(_LAUNCH_RE.findall(ir_text))


def assemble(asm_path: Path, arch: str) -> bytes:
    """Assemble a GCN ``.s`` into a linked HSA code object with the ROCm clang.

    The triple spells the ``unknown`` environment so it matches the
    ``.amdgcn_target`` MLIR writes into the dump; a ``.s`` dumped for another
    arch fails that same check instead of being mis-assembled.
    """
    clang = Path(os.environ.get("ROCM_PATH") or "/opt/rocm") / "llvm" / "bin" / "clang"
    if not clang.is_file():
        sys.exit(f"patch_cache_asm: the ROCm clang is missing: {clang}")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "patched.hsaco"
        cmd = [str(clang), "-x", "assembler", "-target", "amdgcn-amd-amdhsa-unknown", f"-mcpu={arch}"]
        proc = subprocess.run([*cmd, "-o", str(out), str(asm_path)], capture_output=True, text=True)
        if proc.returncode != 0:
            sys.exit(f"patch_cache_asm: assembling {asm_path} failed:\n{proc.stderr.strip()}")
        return out.read_bytes()


def load_entries(cache_dir: Path) -> list:
    """``(path, artifact, kernels)`` for every entry under a cache root."""
    entries = []
    for path in sorted(cache_dir.rglob("*.pkl")):
        with open(path, "rb") as f:
            artifact = pickle.load(f)
        entries.append((path, artifact, launched_kernels(artifact._ir_text)))
    return entries


def replace_binary(ir_text: str, isa_path: Path) -> tuple:
    """Assemble ``isa_path`` for the module's chip and swap it into its single ``gpu.binary``.

    Returns ``(new_ir_text, code_object)``.
    """
    from flydsl._mlir import ir
    from flydsl._mlir._mlir_libs._mlirDialectsGPU import ObjectAttr
    from flydsl._mlir.dialects._gpu_enum_gen import CompilationTarget
    from flydsl.compiler.jit_function import _create_mlir_context

    # The same context CompiledArtifact parses the text back with.
    with _create_mlir_context():
        module = ir.Module.parse(ir_text)
        binaries = [op.operation for op in module.body.operations if op.operation.name == "gpu.binary"]
        if len(binaries) != 1:
            sys.exit(f"patch_cache_asm: expected one gpu.binary in the entry, found {len(binaries)}")
        objects = ir.ArrayAttr(binaries[0].attributes["objects"])
        if len(objects) != 1:
            sys.exit(f"patch_cache_asm: expected one gpu.object in the gpu.binary, found {len(objects)}")
        old = ObjectAttr(objects[0])
        chip = _CHIP_RE.search(str(old.target))
        if chip is None:
            sys.exit(f"patch_cache_asm: no chip in the entry's target {old.target}")

        code_object = assemble(isa_path, chip.group(1))
        # Binary, never Assembly: an Assembly object would be loaded through
        # mgpuModuleLoadJIT, which HIP does not implement.  The original kernel
        # metadata is dropped rather than carried over stale.
        new = ObjectAttr.get(old.target, int(CompilationTarget.Binary), code_object, old.properties, None)
        binaries[0].attributes["objects"] = ir.ArrayAttr.get([new])
        return str(module), code_object


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cache_dir", type=Path, help="scratch JIT cache root (FLYDSL_RUNTIME_CACHE_DIR)")
    ap.add_argument("asm", type=Path, help="hand-edited .s, as dumped by FLYDSL_DUMP_IR")
    ap.add_argument("--save-hsaco", action="store_true", help="also write the code object next to the .s")
    args = ap.parse_args(argv)

    cache_dir = args.cache_dir.expanduser().resolve()
    # A patched entry is served to every later run that hits its key, so it must
    # not land in the cache ordinary runs use.  Entries are found recursively, so
    # a parent or a subdirectory of the default cache reaches it too.
    default = (Path.home() / ".flydsl" / "cache").resolve()
    if cache_dir.is_relative_to(default) or default.is_relative_to(cache_dir):
        sys.exit(f"patch_cache_asm: {cache_dir} overlaps the default cache {default}; use a scratch directory")
    asm = args.asm.expanduser()
    asm_kernels = kernel_names_in_isa(asm.read_text(encoding="utf-8"))
    if not asm_kernels:
        sys.exit(f"patch_cache_asm: {asm} declares no .amdhsa_kernel")

    entries = load_entries(cache_dir)
    matching = [e for e in entries if e[2] & asm_kernels]

    def listing(es):
        return "\n".join(f"  {p}  {sorted(k)}" for p, _, k in es) or "  (no entries)"

    if not matching:
        sys.exit(
            f"patch_cache_asm: no cache entry under {cache_dir} launches {sorted(asm_kernels)}; "
            f"entries found:\n{listing(entries)}"
        )
    if len(matching) > 1:
        sys.exit(
            f"patch_cache_asm: {sorted(asm_kernels)} matches {len(matching)} entries, which cannot be told apart "
            f"(constexpr specializations share a kernel symbol). Re-populate an empty scratch cache running only "
            f"the case you dumped:\n{listing(matching)}"
        )
    path, artifact, kernels = matching[0]

    # The whole gpu.binary is replaced, so kernels the .s omits would vanish from
    # the code object while the host still launches them by name.
    missing = kernels - asm_kernels
    if missing:
        sys.exit(
            f"patch_cache_asm: {path} also launches {sorted(missing)}, which {asm} does not declare; "
            f"replacing the binary would drop them"
        )

    from flydsl.compiler.jit_function import JitCacheManager

    artifact._ir_text, code_object = replace_binary(artifact._ir_text, asm)
    JitCacheManager._write_cache_file(path, artifact)
    print(f"patched {path}: {len(code_object)} bytes for {sorted(kernels)} from {asm}")
    if args.save_hsaco:
        out = asm.with_suffix(".hsaco")
        out.write_bytes(code_object)
        print(f"saved code object -> {out}")


if __name__ == "__main__":
    main()
