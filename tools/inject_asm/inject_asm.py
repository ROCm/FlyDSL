#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Run a test against hand-edited assembly through a JIT cache entry (debug only).

Answers "what if these few instructions were different" without a compiler
rebuild.  ``prepare`` runs an unmodified test command twice into a fresh
workspace, once to dump the ISA and once to cache the compiled kernel, then
copies the dump to ``original.s`` and ``edited.s``.  ``run`` injects
``edited.s`` into the cached entry and reruns the command with
FLYDSL_RUNTIME_RUN_ONLY=1, which turns a cache miss into an error instead of a
silent recompile:

    python3 tools/inject_asm/inject_asm.py prepare WS -- python -m pytest test_x.py -k case
    vim WS/edited.s
    python3 tools/inject_asm/inject_asm.py run WS              # the edit
    python3 tools/inject_asm/inject_asm.py run WS --baseline   # original.s, same conditions

A cache entry keeps the compiled module as MLIR text and rebuilds its
ExecutionEngine from that text on load, so replacing the ``gpu.binary`` object
in the text is enough.  ``inject`` does that one step on any cache directory.

The command must compile exactly one JIT function in one specialization.
Constexpr specializations of a kernel share its symbol, so the dump keeps only
the last one compiled and their cache entries cannot be told apart; patching the
wrong one would leave the run executing the compiler's code with nothing to show
for it.

The ``.s`` is assembled with the ROCm clang unless ``--clang`` names another,
and linked with the ROCm ``ld.lld`` as the compiler does.  FlyDSL itself
assembles with the LLVM it was built with, which is usually newer.  On gfx942
and gfx950 the ROCm clang still reproduces the compiler's code object byte for
byte, and ``prepare`` checks that; a dump for gfx11 or later uses directives it
cannot read, so pass the clang of the LLVM FlyDSL was built with.
"""

import argparse
import json
import os
import pickle
import re
import shlex
import shutil
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


def _rocm_path() -> Path:
    """The toolkit gpu-module-to-binary links with (``ROCDL::getROCMPath``)."""
    for var in ("ROCM_PATH", "ROCM_ROOT", "ROCM_HOME"):
        if os.environ.get(var):
            return Path(os.environ[var])
    return Path("/opt/rocm")


def _toolchain(clang: Path = None) -> tuple:
    """The clang that assembles and the toolkit ``ld.lld`` that links; exits if either is missing."""
    rocm = _rocm_path()
    # Resolved: Path("./clang") prints as "clang", which subprocess would look up on PATH.
    clang = clang.expanduser().resolve() if clang else rocm / "llvm" / "bin" / "clang"
    lld = rocm / "llvm" / "bin" / "ld.lld"
    for tool in (clang, lld):
        if not tool.is_file():
            sys.exit(f"inject_asm: {tool} is missing")
    return clang, lld


def assemble(asm_path: Path, arch: str, clang: Path = None) -> bytes:
    """Assemble a GCN ``.s`` into a linked HSA code object the way the compiler does.

    gpu-module-to-binary assembles in process and links with ``ld.lld -shared``
    from the ROCm toolkit.  Here ``clang`` (the ROCm one by default) only
    assembles and that same ``ld.lld`` links: a clang that links by itself uses
    the lld beside it, which can lay the object out differently even when the code
    is identical.  The triple spells the ``unknown`` environment so it matches the
    ``.amdgcn_target`` MLIR writes into the dump; a ``.s`` dumped for another arch
    fails that same check instead of being mis-assembled.
    """
    clang, lld = _toolchain(clang)
    with tempfile.TemporaryDirectory() as tmp:
        obj, out = Path(tmp) / "patched.o", Path(tmp) / "patched.hsaco"
        asm_only = [clang, "-x", "assembler", "-target", "amdgcn-amd-amdhsa-unknown", f"-mcpu={arch}", "-c"]
        steps = [
            ("assembling", [*asm_only, "-o", obj, asm_path]),
            ("linking", [lld, "-shared", obj, "-o", out]),
        ]
        for what, cmd in steps:
            proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
            if proc.returncode != 0:
                sys.exit(f"inject_asm: {what} {asm_path} failed:\n{proc.stderr.strip()}")
        return out.read_bytes()


def load_entries(cache_dir: Path) -> list:
    """``(path, artifact, kernels)`` for every entry under a cache root."""
    entries = []
    for path in sorted(cache_dir.rglob("*.pkl")):
        with open(path, "rb") as f:
            artifact = pickle.load(f)
        entries.append((path, artifact, launched_kernels(artifact._ir_text)))
    return entries


def replace_binary(ir_text: str, isa_path: Path, clang: Path = None) -> tuple:
    """Assemble ``isa_path`` for the module's chip and swap it into its single ``gpu.binary``.

    Returns ``(new_ir_text, code_object, original_code_object)``.
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
            sys.exit(f"inject_asm: expected one gpu.binary in the entry, found {len(binaries)}")
        objects = ir.ArrayAttr(binaries[0].attributes["objects"])
        if len(objects) != 1:
            sys.exit(f"inject_asm: expected one gpu.object in the gpu.binary, found {len(objects)}")
        old = ObjectAttr(objects[0])
        chip = _CHIP_RE.search(str(old.target))
        if chip is None:
            sys.exit(f"inject_asm: no chip in the entry's target {old.target}")

        code_object = assemble(isa_path, chip.group(1), clang)
        # Binary, never Assembly: an Assembly object would be loaded through
        # mgpuModuleLoadJIT, which HIP does not implement.  The original kernel
        # metadata is dropped rather than carried over stale.
        new = ObjectAttr.get(old.target, int(CompilationTarget.Binary), code_object, old.properties, None)
        binaries[0].attributes["objects"] = ir.ArrayAttr.get([new])
        return str(module), code_object, bytes(old.object)


def _refuse_default_cache(cache_dir: Path) -> None:
    # A patched entry is served to every later run that hits its key, so it must
    # not land in the cache ordinary runs use.  Entries are found recursively, so
    # a parent or a subdirectory of the default cache reaches it too.
    default = (Path.home() / ".flydsl" / "cache").resolve()
    if cache_dir.is_relative_to(default) or default.is_relative_to(cache_dir):
        sys.exit(f"inject_asm: {cache_dir} overlaps the default cache {default}; use a scratch directory")


def inject(cache_dir: Path, asm: Path, clang: Path = None, check: bool = False, save_hsaco: bool = False):
    """Patch the one entry under ``cache_dir`` that ``asm`` names, and return its path.

    With ``check`` nothing is written; the result is whether ``asm`` reassembles to
    the entry's code object byte for byte.
    """
    cache_dir = cache_dir.expanduser().resolve()
    _refuse_default_cache(cache_dir)
    asm = asm.expanduser()
    asm_kernels = kernel_names_in_isa(asm.read_text(encoding="utf-8"))
    if not asm_kernels:
        sys.exit(f"inject_asm: {asm} declares no .amdhsa_kernel")

    entries = load_entries(cache_dir)
    matching = [e for e in entries if e[2] & asm_kernels]

    def listing(es):
        return "\n".join(f"  {p}  {sorted(k)}" for p, _, k in es) or "  (no entries)"

    if not matching:
        sys.exit(
            f"inject_asm: no cache entry under {cache_dir} launches {sorted(asm_kernels)}; "
            f"entries found:\n{listing(entries)}"
        )
    if len(matching) > 1:
        sys.exit(
            f"inject_asm: {sorted(asm_kernels)} matches {len(matching)} entries, which cannot be told apart "
            f"(constexpr specializations share a kernel symbol). Re-populate an empty scratch cache running only "
            f"the case you dumped:\n{listing(matching)}"
        )
    path, artifact, kernels = matching[0]

    # The whole gpu.binary is replaced, so kernels the .s omits would vanish from
    # the code object while the host still launches them by name.
    missing = kernels - asm_kernels
    if missing:
        sys.exit(
            f"inject_asm: {path} also launches {sorted(missing)}, which {asm} does not declare; "
            f"replacing the binary would drop them"
        )

    from flydsl.compiler.jit_function import JitCacheManager

    new_ir_text, code_object, original = replace_binary(artifact._ir_text, asm, clang)
    if check:
        return code_object == original

    artifact._ir_text = new_ir_text
    JitCacheManager._write_cache_file(path, artifact)
    print(f"patched {path}: {len(code_object)} bytes for {sorted(kernels)} from {asm}")
    if save_hsaco:
        out = asm.with_suffix(".hsaco")
        out.write_bytes(code_object)
        print(f"saved code object -> {out}")
    return path


# Knobs that change what a run does with the cache.  Each workspace run sets the
# ones it needs and drops the rest: an exported FLYDSL_DUMP_IR, for one, would
# stop the caching run from writing its entry.
_RUN_KNOBS = (
    "FLYDSL_DUMP_IR",
    "FLYDSL_DUMP_DIR",
    "FLYDSL_RUNTIME_CACHE_DIR",
    "FLYDSL_RUNTIME_ENABLE_CACHE",
    "FLYDSL_RUNTIME_RUN_ONLY",
)


def _run_command(config: dict, drop: tuple = (), **knobs) -> int:
    env = {k: v for k, v in os.environ.items() if k not in _RUN_KNOBS + drop}
    env.update(knobs)
    print(f"[inject_asm] {shlex.join(config['cmd'])}", flush=True)
    return subprocess.run(config["cmd"], cwd=config["cwd"], env=env).returncode


def prepare(ws: Path, cmd: list, clang: Path = None) -> None:
    """Dump and cache ``cmd`` into a fresh workspace, and set up ``original.s`` / ``edited.s``."""
    ws = ws.expanduser().resolve()
    if ws.exists() and any(ws.iterdir()):
        sys.exit(f"inject_asm: {ws} is not empty; prepare needs a fresh workspace")
    _refuse_default_cache(ws / "cache")
    # Before the command runs twice, which can take a while, rather than at the check after it.
    resolved_clang, _ = _toolchain(clang)
    ws.mkdir(parents=True, exist_ok=True)
    config = {"cmd": cmd, "cwd": os.getcwd(), "clang": str(resolved_clang) if clang else None}

    # FLYDSL_DUMP_IR turns the cache off, so the dump and the entry come from two
    # runs.  Running the identical command keeps them the same compilations.
    dump, cache = ws / "dump", ws / "cache"
    for label, knobs in (
        ("dump", {"FLYDSL_DUMP_IR": "1", "FLYDSL_DUMP_DIR": str(dump)}),
        ("cache", {"FLYDSL_RUNTIME_CACHE_DIR": str(cache), "FLYDSL_RUNTIME_ENABLE_CACHE": "1"}),
    ):
        if _run_command(config, **knobs) != 0:
            sys.exit(f"inject_asm: the {label} run failed; prepare needs a command that passes unmodified")

    # Count cache entries, not dumps: a dump directory is named after its JIT
    # function, so two functions of the same name leave one dump between them.
    entries = load_entries(cache)
    if len(entries) != 1:
        found = "\n".join(f"  {p.relative_to(ws)}  {sorted(k)}" for p, _, k in entries) or "  (none)"
        sys.exit(
            f"inject_asm: the command wrote {len(entries)} cache entries; prepare needs exactly one. "
            f"Narrow it to a single JIT function and specialization (e.g. pytest -k). Cache entries:\n{found}"
        )
    ((entry, _, kernels),) = entries
    isas = [s for s in sorted(dump.rglob("*_final_isa.s")) if kernel_names_in_isa(s.read_text()) == kernels]
    if len(isas) != 1:
        sys.exit(f"inject_asm: found {len(isas)} dumps declaring {sorted(kernels)} under {dump}; expected one")
    isa = isas[0]

    original, edited = ws / "original.s", ws / "edited.s"
    shutil.copyfile(isa, original)
    shutil.copyfile(isa, edited)
    original.chmod(0o444)
    (ws / "prepare.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"[inject_asm] {sorted(kernel_names_in_isa(isa.read_text()))} -> {entry.relative_to(ws)}")

    try:
        same = inject(cache, original, clang, check=True)
    except SystemExit as failure:
        sys.exit(
            f"{failure.code}\ninject_asm: prepare again in an empty workspace; if the clang cannot read this dump "
            f"(one for gfx11 or later, with the ROCm clang), pass --clang naming the clang of the LLVM FlyDSL was "
            f"built with."
        )
    if same:
        print("[inject_asm] check passed: original.s reassembles to the compiler's code object byte for byte")
    else:
        # run --baseline reassembles original.s with the same clang, so run vs
        # --baseline stays a fair comparison; the compiler's own numbers do not.
        print(
            "[inject_asm] WARNING: original.s does not reassemble to the compiler's code object byte for byte. "
            "Compare `run` with `run --baseline`, not with the compiler's numbers, or prepare again with --clang "
            "naming the clang of the LLVM FlyDSL was built with."
        )
    print(f"edit  {edited}\nthen  python3 {Path(__file__).resolve()} run {ws}")


def run(ws: Path, baseline: bool = False) -> int:
    """Inject the workspace's ``edited.s`` (or ``original.s``) and rerun its command under RUN_ONLY."""
    ws = ws.expanduser().resolve()
    try:
        config = json.loads((ws / "prepare.json").read_text())
    except FileNotFoundError:
        sys.exit(f"inject_asm: {ws} has no prepare.json; run prepare first")
    clang = Path(config["clang"]) if config["clang"] else None
    inject(ws / "cache", ws / ("original.s" if baseline else "edited.s"), clang)
    # run exists to execute.  Under COMPILE_ONLY a cache hit returns before the code
    # object is even loaded, so the command would pass having run nothing.  prepare
    # keeps it: preparing on a machine without the target GPU is a real use.
    return _run_command(
        config, drop=("COMPILE_ONLY",), FLYDSL_RUNTIME_CACHE_DIR=str(ws / "cache"), FLYDSL_RUNTIME_RUN_ONLY="1"
    )


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else [str(a) for a in argv]
    # The test command follows "--".  Split it off before argparse, whose
    # REMAINDER would also swallow our own options after the workspace.
    cmd = []
    if "--" in argv:
        cut = argv.index("--")
        argv, cmd = argv[:cut], argv[cut + 1 :]
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="dump and cache a test command into a fresh workspace")
    p.add_argument("workspace", type=Path)
    p.add_argument("--clang", type=Path, help="clang to assemble with (default: the ROCm clang)")

    r = sub.add_parser("run", help="inject the workspace's edited.s and rerun its command under RUN_ONLY")
    r.add_argument("workspace", type=Path)
    r.add_argument("--baseline", action="store_true", help="inject the unedited original.s instead")

    i = sub.add_parser("inject", help="patch one cache entry with a .s; what prepare and run build on")
    i.add_argument("cache_dir", type=Path, help="scratch JIT cache root (FLYDSL_RUNTIME_CACHE_DIR)")
    i.add_argument("asm", type=Path, help="hand-edited .s, as dumped by FLYDSL_DUMP_IR")
    i.add_argument("--save-hsaco", action="store_true", help="also write the code object next to the .s")
    i.add_argument("--clang", type=Path, help="clang to assemble with (default: the ROCm clang)")
    i.add_argument(
        "--check",
        action="store_true",
        help="compare the reassembled .s with the entry's code object instead of patching it",
    )
    args = ap.parse_args(argv)

    if (args.command == "prepare") != bool(cmd):
        ap.error("prepare, and only prepare, takes the test command after --")
    if args.command == "prepare":
        prepare(args.workspace, cmd, args.clang)
    elif args.command == "run":
        return run(args.workspace, args.baseline)
    elif args.check:
        if not inject(args.cache_dir, args.asm, args.clang, check=True):
            sys.exit(
                f"inject_asm: {args.asm} reassembles to a code object that differs from the entry's. Edits would "
                f"be measured against a different baseline; pass --clang with the LLVM FlyDSL was built with."
            )
        print(f"check passed: {args.asm} reassembles to the entry's code object byte for byte")
    else:
        entry = inject(args.cache_dir, args.asm, args.clang, save_hsaco=args.save_hsaco)
        print(f"run with: FLYDSL_RUNTIME_CACHE_DIR={entry.parent.parent} FLYDSL_RUNTIME_RUN_ONLY=1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
