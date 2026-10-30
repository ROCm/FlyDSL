# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import ctypes
import fcntl
import hashlib
import inspect
import json
import os
import pickle
import threading
import time
from collections import namedtuple
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .._mlir import ir
from .._mlir.dialects import func
from .._mlir.passmanager import PassManager
from ..expr.meta import tracing_context
from ..expr.typing import Constexpr, Stream
from ..utils import env, log
from ..utils.file import atomic_write
from .ast_rewriter import ASTRewriter
from .backends import get_backend, get_ktrace_backend
from .diagnostics import (
    DSLCompileError,
    diag_records_from_mlir_error,
    dsl_ir_diagnostics,
    install_excepthook,
    warn_annotation_value_mismatch,
    warn_invalid_annotations,
)
from .jit_argument import (
    DLTensorJitArg,
    JitArgumentRegistry,
    PointerJitArg,
    TorchTensorJitArg,
    convert_to_jit_arguments,
    is_type_param_annotation,
    resolve_signature,
)
from .jit_executor import CallState, CompiledArtifact
from .kernel_function import (
    CompilationContext,
    create_gpu_module,
    effective_fastmath_hint,
    func_def_location,
    get_gpu_module_body,
    merge_compile_hints,
)
from .link_utils import _append_link_lib_options_to_attach_targets, _format_link_lib_options
from .protocol import (
    JitArgument,
    c_abi_spec,
    construct_from_ir_values,
    get_ir_types,
)

EXTRA_SOURCE_DIRS: List[str] = []

CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "currsize", "disk_size"])


class FileLock:
    """fcntl-based file lock supporting shared and exclusive modes."""

    def __init__(self, path, *, exclusive=True, timeout=30):
        self._path = str(path)
        self._exclusive = exclusive
        self._timeout = timeout
        self._fd = None

    def __enter__(self):
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
        self._fd = fd
        op = fcntl.LOCK_EX if self._exclusive else fcntl.LOCK_SH
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                fcntl.flock(fd, op | fcntl.LOCK_NB)
                return self
            except (OSError, BlockingIOError):
                if time.monotonic() >= deadline:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                    self._fd = None
                    raise RuntimeError(
                        f"Timed out waiting for {'exclusive' if self._exclusive else 'shared'} "
                        f"lock on {self._path} after {self._timeout}s"
                    )
                time.sleep(0.05)

    def __exit__(self, *exc):
        fd = self._fd
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(fd)
            except OSError:
                pass
            self._fd = None
        return False


def _create_mlir_context(*, load_dialects=True):
    """Create an ``ir.Context`` with multithreading disabled.

    Disabling multithreading avoids LLVM global-state races when multiple
    processes or threads compile concurrently through the same MLIR install.
    """
    ctx = ir.Context()
    ctx.enable_multithreading(False)
    if load_dialects:
        ctx.load_all_available_dialects()
    return ctx


# Environment variables that influence code generation *independently of the
# resolved GPUTarget*. Their current values enter the hot cache key so
# cross-process / cross-config artifacts don't collide.
_CACHE_INVALIDATING_ENV_VARS = (
    "FLYDSL_COMPILE_OPT_LEVEL",
    "FLYDSL_COMPILE_BACKEND",
    "FLYDSL_COMPILE_LLVM_DIR",
    "LLVM_GFX1250_ANY_XNACK_CODEGEN",
    "FLYDSL_DEBUG_ENABLE_DEBUG_INFO",
    "FLYDSL_EXTRA_SOURCE_DIRS",
    "FLYDSL_KTRACE_ENABLE",
    "FLYDSL_KTRACE_BLOCKS",
    "FLYDSL_KTRACE_BUFFER_BYTES",
)


# os._Environ keeps the live mapping in a plain dict (``_data``) keyed by the
# OS-encoded bytes of each name; mutations to os.environ update it in place.
# Reading it with pre-encoded keys skips os.environ.get's per-call key encoding,
# which is ~5x faster on this hot path. Guarded: if the internal layout is
# absent (non-posix / future CPython) we fall back to the public API. Values go
# through os.fsdecode so the fast and slow paths — and thus two processes
# sharing a cache dir — produce byte-for-byte identical keys.
try:
    _ENABLE_ENV_FAST_READ = isinstance(os.environ._data, dict)
except Exception:
    _ENABLE_ENV_FAST_READ = False
_CACHE_INVALIDATING_ENV_VARS_ENCODED = tuple((n, os.fsencode(n)) for n in _CACHE_INVALIDATING_ENV_VARS)


def _cache_invalidating_env_values() -> tuple:
    # Re-read on every call: users may mutate os.environ mid-process (e.g.
    # toggling FLYDSL_COMPILE_OPT_LEVEL between runs) and any caching here
    # would freeze the first observed values into every subsequent cache key.
    if _ENABLE_ENV_FAST_READ:
        data = os.environ._data
        return tuple((n, os.fsdecode(data[b]) if b in data else "") for n, b in _CACHE_INVALIDATING_ENV_VARS_ENCODED)
    return tuple((n, os.environ.get(n, "")) for n in _CACHE_INVALIDATING_ENV_VARS)


def _flydsl_key() -> str:
    extra = list(EXTRA_SOURCE_DIRS)
    env_extra = os.environ.get("FLYDSL_EXTRA_SOURCE_DIRS", "")
    if env_extra:
        extra.extend(d.strip() for d in env_extra.split(":") if d.strip())
    return _flydsl_key_cached(_use_external_binary_codegen(), env.compile.llvm_dir, tuple(extra))


def _is_native_library(name: str) -> bool:
    # Same selection as cmake/WriteNativeDigest.cmake: *.so, *.so.*, *.a.
    return name.endswith((".so", ".a")) or ".so." in name


def _native_library_digests(libs_dir: Path) -> List[str]:
    """``name:sha256`` of every native library in ``libs_dir``, sorted by name.

    Uses the table the build bakes into ``_native_digest.py``; only if it is
    missing or lists other files (an unsupported manual edit) are the libraries
    hashed here, which costs several hundred milliseconds.
    """
    names = sorted(
        entry.name
        for entry in os.scandir(libs_dir)
        if _is_native_library(entry.name) and entry.is_file(follow_symlinks=False)
    )
    try:
        from .._mlir._mlir_libs._native_digest import NATIVE_SHA256
    except ImportError:
        NATIVE_SHA256 = None
    if NATIVE_SHA256 is not None and sorted(NATIVE_SHA256) == names:
        return [f"{name}:{NATIVE_SHA256[name]}" for name in names]

    log().warning(
        f"No build-time digest matches the native libraries in {libs_dir}; hashing them at runtime. "
        "Rebuild FlyDSL (scripts/build.sh) to restore the fast path."
    )
    digests = []
    for name in names:
        h = hashlib.sha256()
        with open(libs_dir / name, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        digests.append(f"{name}:{h.hexdigest()}")
    return digests


@lru_cache(maxsize=4)
def _flydsl_key_cached(use_external_binary: bool, llvm_dir: str, extra_source_dirs: tuple = ()) -> str:
    """Compute a hash fingerprint of the entire FlyDSL compiler toolchain.

    Covers:
      1. All Python source files under flydsl.compiler.*, flydsl.expr.*,
         flydsl.extension.*, flydsl.runtime.*, flydsl.utils.*
      2. Every native library in ``_mlir/_mlir_libs`` (build-time digests)
      3. flydsl.__version__

    Any change to compiler code, pass pipeline, runtime wrappers, or C++
    bindings will produce a different key, invalidating stale disk caches.
    """
    import flydsl

    contents = []

    flydsl_root = Path(flydsl.__file__).resolve().parent

    # 1) Hash all Python source files in key sub-packages.
    for package_name in ("compiler", "expr", "extension", "runtime", "utils"):
        pkg_path = flydsl_root / package_name
        if not pkg_path.is_dir():
            continue
        for py_file in sorted(pkg_path.rglob("*.py")):
            with open(py_file, "rb") as f:
                source_hash = hashlib.sha256(f.read()).hexdigest()
            relative_path = py_file.relative_to(flydsl_root).as_posix()
            contents.append(f"{relative_path}:{source_hash}")

    p = flydsl_root / "__init__.py"
    if p.is_file():
        with open(p, "rb") as f:
            contents.append(hashlib.sha256(f.read()).hexdigest())

    # 2) Native libraries (C++ passes, runtime wrappers, bindings).
    contents.extend(_native_library_digests(flydsl_root / "_mlir" / "_mlir_libs"))

    # 3) Hash .py files in extra source directories (downstream fingerprint).
    for src_dir in extra_source_dirs:
        src_path = Path(src_dir)
        if src_path.is_dir():
            for py_file in sorted(src_path.rglob("*.py")):
                with open(py_file, "rb") as f:
                    contents.append(hashlib.sha256(f.read()).hexdigest())

    contents.append(f"external_binary_codegen={use_external_binary}")
    if use_external_binary:
        from .external_llvm import external_llvm_fingerprint

        contents.append(external_llvm_fingerprint(llvm_dir or None))

    key = f"flydsl:{flydsl.__version__}-" + "-".join(contents)
    log().debug(f"flydsl_key: {hashlib.sha256(key.encode()).hexdigest()[:16]}")
    return key


def _use_external_binary_codegen() -> bool:
    return bool(env.compile.llvm_dir.strip())


def _stage_label_from_fragment(fragment: str) -> str:
    """Make a stable, filename-friendly label from a pipeline fragment."""
    import re as _re

    base = fragment.strip()
    if base.startswith("gpu.module(") and base.endswith(")"):
        base = base[len("gpu.module(") : -1].strip()
    base = base.split("{", 1)[0].strip()
    base = _re.sub(r"[^0-9A-Za-z]+", "_", base).strip("_").lower()
    return base or "stage"


def _dump_ir(stage: str, *, dump_dir: Path, asm: str) -> Path:
    """Write one compilation stage's MLIR assembly to a .mlir file."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    out = dump_dir / f"{stage}.mlir"
    out.write_text(asm, encoding="utf-8")
    return out


def _extract_isa_text(mlir_asm: str) -> str:
    """Extract human-readable ISA from MLIR gpu.binary assembly attribute.

    The ``gpu-module-to-binary{format=isa}`` pass embeds the ISA inside an MLIR
    attribute like ``assembly = "..."`` with MLIR string escapes (``\\0A`` for
    newline, ``\\09`` for tab, ``\\22`` for double-quote).  This function
    locates that string and un-escapes it so the output is a normal ``.s`` file.
    """
    import re as _re

    m = _re.search(r'assembly\s*=\s*"', mlir_asm)
    if not m:
        return mlir_asm

    start = m.end()
    # Walk forward to find the closing unescaped quote.
    i = start
    chars = []
    while i < len(mlir_asm):
        ch = mlir_asm[i]
        if ch == '"':
            break
        if ch == "\\" and i + 1 < len(mlir_asm):
            nxt = mlir_asm[i + 1]
            if nxt == "\\":
                chars.append("\\")
                i += 2
                continue
            if nxt == '"':
                chars.append('"')
                i += 2
                continue
            # MLIR hex escape: \XX
            if i + 3 <= len(mlir_asm):
                hex_str = mlir_asm[i + 1 : i + 3]
                try:
                    chars.append(chr(int(hex_str, 16)))
                    i += 3
                    continue
                except ValueError:
                    pass
        chars.append(ch)
        i += 1

    return "".join(chars)


def _dump_isa(*, dump_dir: Path, ctx: ir.Context, asm: str, verify: bool, stage_name: str = "15_final_isa"):
    """Best-effort dump of final GPU ISA/assembly (.s).

    Runs ``gpu-module-to-binary{format=isa}`` on a *cloned* module so the
    main compilation is not affected.  The raw ISA text is extracted from the
    MLIR ``assembly = "..."`` attribute and written as a clean ``.s`` file.
    """
    try:
        mod = ir.Module.parse(asm, context=ctx)
        di_pass = (
            "ensure-debug-info-scope-on-llvm-func{emission-kind=LineTablesOnly}," if env.debug.enable_debug_info else ""
        )
        pm = PassManager.parse(
            f'builtin.module({di_pass}gpu-module-to-binary{{format=isa opts="{"-g" if env.debug.enable_debug_info else ""}" section= toolkit=}})',
            context=ctx,
        )
        pm.enable_verifier(bool(verify))
        pm.run(mod.operation)

        raw_mlir = mod.operation.get_asm(enable_debug_info=False)
        isa_text = _extract_isa_text(raw_mlir)

        dump_dir.mkdir(parents=True, exist_ok=True)
        out = dump_dir / f"{stage_name}.s"
        out.write_text(isa_text, encoding="utf-8")
        return out
    except Exception as exc:
        log().debug(f"[dump_isa] failed: {exc}")
        return None


def _infer_kernel_names_from_asm(asm: str) -> list:
    """Extract gpu.func kernel names from MLIR assembly."""
    names = []
    for line in asm.splitlines():
        if "gpu.func @" not in line or " kernel" not in line:
            continue
        try:
            after = line.split("gpu.func @", 1)[1]
            name = after.split("(", 1)[0].strip()
            if name:
                names.append(name)
        except Exception:
            pass
    return names


def _sanitize_path_component(s: str) -> str:
    import re as _re

    s = str(s).strip()
    return _re.sub(r"[^A-Za-z0-9_.-]+", "_", s) if s else "unknown"


def _extract_llvm_ir(module: ir.Module):
    """Extract LLVM IR text from the gpu.module inside *module* (must already be in LLVM dialect)."""
    try:
        from .._mlir._mlir_libs._mlirDialectsLLVM import translate_module_to_llvmir

        for op in module.body.operations:
            if op.operation.name == "gpu.module":
                return translate_module_to_llvmir(op.operation)
        return None
    except Exception as exc:
        log().debug(f"[extract_llvm_ir] failed: {exc}")
        return None


@dataclass
class PipelineConfig:
    """Result of :func:`_pipeline_fragments_for_mode`."""

    fragments: list
    pre_binary: Optional[list]
    binary_fragment: Optional[str]
    llvm_opts: Optional[dict]
    external: bool


def _pipeline_fragments_for_mode(backend, *, compile_hints: dict) -> PipelineConfig:
    """Return pipeline configuration including optional external split."""
    llvm_opts = compile_hints.get("llvm_options")
    if _use_external_binary_codegen():
        pre_binary_fragments, binary_fragment = backend.external_binary_pipeline_fragments(compile_hints=compile_hints)
        return PipelineConfig(
            fragments=[*pre_binary_fragments, binary_fragment],
            pre_binary=pre_binary_fragments,
            binary_fragment=binary_fragment,
            llvm_opts=llvm_opts,
            external=True,
        )

    fragments = backend.pipeline_fragments(compile_hints=compile_hints)
    return PipelineConfig(
        fragments=fragments,
        pre_binary=None,
        binary_fragment=None,
        llvm_opts=llvm_opts,
        external=False,
    )


def _run_pipeline(module: ir.Module, fragments: list, *, verifier: bool, print_after_all: bool) -> None:
    """Parse and run a comma-joined pass pipeline on *module*."""
    pipeline = f"builtin.module({','.join(fragments)})"
    pm = PassManager.parse(pipeline)
    pm.enable_verifier(verifier)
    pm.enable_ir_printing(print_after_all=print_after_all)
    _run_pass_manager(pm, module)


def _run_pass_manager(pm: PassManager, module: ir.Module) -> None:
    """Run *pm* on *module*, raising a failure as a DSLCompileError that points at the user's source."""
    with dsl_ir_diagnostics(module.context) as diags:
        try:
            pm.run(module.operation)
        except ir.MLIRError as exc:
            # PassManager.run captures the errors into the MLIRError before any other handler sees them.
            records = diag_records_from_mlir_error(exc)
            raise DSLCompileError(str(exc), diagnostics=records) from exc
        except Exception as exc:
            raise DSLCompileError(str(exc), diagnostics=diags) from exc


def _artifact_cache_key(asm: str, cfg: PipelineConfig, backend) -> str:
    """Identify the complete lowering input within this process.

    ``asm`` is the prepared module text, omitting source locations when DWARF
    is disabled. Lowering retains locations for diagnostics. Do not elide
    constants/resources or canonicalize away attributes. ``backend.hash()``
    may carry live toolchain configuration, so it is read per request; the loaded
    compiler files cannot change in-process, so only :func:`_disk_cache_key`
    fingerprints them.
    """
    target = backend.target
    config = {
        "target": [target.backend, target.arch, target.warp_size],
        "backend": backend.hash(),
        "pipeline": cfg.fragments,
        "llvm_options": cfg.llvm_opts,
        "env": _cache_invalidating_env_values(),
    }
    key = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())
    key.update(asm.encode())
    return key.hexdigest()


def _disk_cache_key(artifact_key: str) -> str:
    """Namespace an artifact key by the compiler toolchain for cross-process reuse."""
    return hashlib.sha256(f"{_flydsl_key()}\n{artifact_key}".encode()).hexdigest()


class MlirCompiler:
    @classmethod
    def prepare(cls, module: ir.Module, *, backend, compile_hints: dict, link_libs=None):
        """Render the exact lowering input: hinted module text and resolved pipeline.

        The text follows the cache location policy. On a miss, :meth:`compile`
        parses the same prepared module with source locations for diagnostics.
        """
        try:
            module.operation.verify()
        except ir.MLIRError as exc:
            records = diag_records_from_mlir_error(exc)
            raise DSLCompileError("MLIR verification failed", diagnostics=records) from exc

        backend.lower_compile_hints(module, compile_hints=compile_hints)
        asm = module.operation.get_asm(enable_debug_info=env.debug.enable_debug_info, assume_verified=True)
        cfg = _pipeline_fragments_for_mode(backend, compile_hints=compile_hints)
        if cfg.external and link_libs:
            raise RuntimeError(
                "FLYDSL_COMPILE_LLVM_DIR external codegen does not support extern link_libs yet; "
                "use embedded codegen for kernels that require #fly.explicit_module."
            )
        if link_libs:
            link_opt = _format_link_lib_options(link_libs)
            cfg.fragments, found = _append_link_lib_options_to_attach_targets(cfg.fragments, link_opt)
            if not found:
                raise RuntimeError("link_libs specified but no attach-target fragment found in pipeline")
        return asm, cfg

    @classmethod
    def compile(cls, asm: str, *, func_name: str, config: PipelineConfig) -> ir.Module:
        """Lower the text and configuration resolved by :meth:`prepare`."""
        module = ir.Module.parse(asm)
        fragments = config.fragments
        pre_binary_fragments = config.pre_binary
        binary_fragment = config.binary_fragment
        llvm_opts = config.llvm_opts
        external_binary = config.external

        from .llvm_options import llvm_options as _llvm_options

        _llvm_ctx = _llvm_options(llvm_opts) if llvm_opts else nullcontext()

        if env.debug.print_origin_ir:
            log().info(f"Origin IR: \n{module}")

        # FLYDSL_DEBUG_DUMP_ASM takes the FLYDSL_DUMP_IR path but writes only the final ISA.
        dump_ir = env.debug.dump_ir
        dump_enabled = dump_ir or env.debug.dump_asm
        dump_dir = Path(env.debug.dump_dir).resolve()

        if dump_enabled and not dump_ir and external_binary:
            raise RuntimeError(
                "FLYDSL_DEBUG_DUMP_ASM is not supported with "
                "FLYDSL_COMPILE_LLVM_DIR external codegen; use FLYDSL_DUMP_IR=1 "
                "to inspect pre-binary/final MLIR, or run external LLVM tools directly for ISA dumps."
            )

        with _llvm_ctx:
            if dump_enabled:
                asm = module.operation.get_asm(enable_debug_info=True)
                kernel_names = _infer_kernel_names_from_asm(asm)
                subdir = kernel_names[0] if len(kernel_names) == 1 else (func_name or "module")
                dump_dir = dump_dir / _sanitize_path_component(subdir)
                if dump_ir:
                    print(f"[flydsl.compile] FLYDSL_DUMP_IR=1 dir={dump_dir}")

                    out = _dump_ir("00_origin", dump_dir=dump_dir, asm=asm)
                    print(f"[flydsl.compile] dump 00_origin -> {out}")

                asm_for_isa = None
                llir = None
                stage_num_base = 1
                dump_fragments = pre_binary_fragments if external_binary else fragments
                for idx, frag in enumerate(dump_fragments):
                    if frag.strip().startswith("gpu-module-to-binary"):
                        llir = _extract_llvm_ir(module)

                    stage_num = stage_num_base + idx
                    stage_name = f"{stage_num:02d}_{_stage_label_from_fragment(frag)}"
                    pm = PassManager.parse(f"builtin.module({frag})")
                    pm.enable_verifier(env.debug.enable_verifier)
                    _run_pass_manager(pm, module)

                    stage_asm = module.operation.get_asm(enable_debug_info=True)
                    if dump_ir:
                        out = _dump_ir(stage_name, dump_dir=dump_dir, asm=stage_asm)
                        print(f"[flydsl.compile] dump {stage_name} -> {out}")

                    if frag.strip() == "reconcile-unrealized-casts":
                        asm_for_isa = stage_asm

                next_stage = stage_num_base + len(dump_fragments)
                if external_binary:
                    from .external_llvm import run_external_binary_codegen

                    llir = _extract_llvm_ir(module)
                    stage_name = f"{next_stage:02d}_external_binary"
                    run_external_binary_codegen(
                        module,
                        binary_fragment,
                        llvm_options=llvm_opts,
                        work_dir=dump_dir,
                        stage_prefix=stage_name,
                    )
                    module.operation.verify()
                    print(f"[flydsl.compile] dump {stage_name}_input -> {dump_dir / f'{stage_name}_input.mlir'}")
                    print(
                        f"[flydsl.compile] dump {stage_name}_external_output -> "
                        f"{dump_dir / f'{stage_name}_external_output.mlir'}"
                    )
                    print(f"[flydsl.compile] dump {stage_name}_output -> {dump_dir / f'{stage_name}_output.mlir'}")
                    next_stage += 1

                if llir is not None:
                    ll_name = f"{next_stage:02d}_llvm_ir"
                    if dump_ir:
                        (dump_dir / f"{ll_name}.ll").write_text(llir, encoding="utf-8")
                        print(f"[flydsl.compile] dump {ll_name} -> {dump_dir / f'{ll_name}.ll'}")
                    next_stage += 1

                if asm_for_isa is not None:
                    if not external_binary:
                        isa_stage = f"{next_stage:02d}_final_isa"
                        isa_out = _dump_isa(
                            dump_dir=dump_dir,
                            ctx=module.context,
                            asm=asm_for_isa,
                            verify=env.debug.enable_verifier,
                            stage_name=isa_stage,
                        )
                        if isa_out is not None:
                            print(f"[flydsl.compile] dump {isa_stage} -> {isa_out}")
                    else:
                        print("[flydsl.compile] ISA dump skipped (external LLVM mode)")
            else:
                if external_binary:
                    from .external_llvm import run_external_binary_codegen

                    _run_pipeline(
                        module,
                        pre_binary_fragments,
                        verifier=env.debug.enable_verifier,
                        print_after_all=env.debug.print_after_all,
                    )

                    run_external_binary_codegen(
                        module,
                        binary_fragment,
                        llvm_options=llvm_opts,
                    )
                    module.operation.verify()
                else:
                    _run_pipeline(
                        module,
                        fragments,
                        verifier=env.debug.enable_verifier,
                        print_after_all=env.debug.print_after_all,
                    )

        return module


class JitCacheManager:
    """Directory-based cache manager with multi-process safety.

    Cache directory structure:
        {cache_root}/{func_name}/
            {artifact_key}.pkl  - serialized compiled kernel
            {artifact_key}.lock - per-key advisory lock file

    All disk reads use shared (reader) locks; writes use exclusive locks
    and same-directory atomic replacement to prevent partial reads.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.memory_cache: Dict[str, Any] = {}
        self._hits = 0
        self._misses = 0

    def _cache_file(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.pkl"

    def _lock_file(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.lock"

    @staticmethod
    def _write_cache_file(cache_file: Path, value: Any) -> None:
        with atomic_write(cache_file, mode="wb") as output:
            pickle.dump(value, output)

    def get(self, cache_key: str) -> Optional[Any]:
        if cache_key in self.memory_cache:
            self._hits += 1
            return self.memory_cache[cache_key]

        cache_file = self._cache_file(cache_key)
        if cache_file.exists():
            lock_path = self._lock_file(cache_key)
            try:
                with FileLock(lock_path, exclusive=False, timeout=30):
                    if not cache_file.exists():
                        self._misses += 1
                        return None
                    with open(cache_file, "rb") as f:
                        value = pickle.load(f)
                self.memory_cache[cache_key] = value
                self._hits += 1
                log().debug(f"Cache hit from disk: {cache_file.name}")
                return value
            except Exception as e:
                log().warning(f"Failed to load cache {cache_file}: {e}")
        self._misses += 1
        return None

    def set(self, cache_key: str, value: Any) -> None:
        self.memory_cache[cache_key] = value
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self._cache_file(cache_key)
        lock_path = self._lock_file(cache_key)
        try:
            with FileLock(lock_path, exclusive=True, timeout=30):
                if cache_file.exists():
                    log().debug(f"Cache already exists, skipping write: {cache_file.name}")
                    return
                self._write_cache_file(cache_file, value)
            log().debug(f"Cache saved: {cache_file.name}")
        except Exception as e:
            log().warning(f"Failed to save cache {cache_file}: {e}")

    @contextmanager
    def compile_lock(self, cache_key: str, *, accept_cached=None):
        """Acquire an exclusive compile lock, re-check disk, yield (existing_or_None, writer_or_None).

        If *existing* is not None, another process already wrote the artifact
        and *writer* is None.  Otherwise *writer* is a callable that performs
        an atomic write under the already-held lock (no re-locking).
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self._lock_file(cache_key)
        cache_file = self._cache_file(cache_key)

        with FileLock(lock_path, exclusive=True, timeout=600):
            # Re-check disk under exclusive lock.
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        value = pickle.load(f)
                except Exception:
                    # Corrupt cache — remove so writer can overwrite.
                    try:
                        cache_file.unlink()
                    except OSError:
                        pass
                else:
                    if accept_cached is None or accept_cached(value):
                        self.memory_cache[cache_key] = value
                        self._hits += 1
                        yield (value, None)
                        return

            # Cache miss — provide a writer that writes under the already-held lock.
            def _writer(value):
                self._write_cache_file(cache_file, value)
                self.memory_cache[cache_key] = value

            yield (None, _writer)

    def cache_info(self) -> CacheInfo:
        disk_count = 0
        if self.cache_dir.exists():
            disk_count = sum(1 for _ in self.cache_dir.glob("*.pkl"))
        return CacheInfo(
            hits=self._hits,
            misses=self._misses,
            currsize=len(self.memory_cache),
            disk_size=disk_count,
        )

    def __contains__(self, cache_key: str) -> bool:
        return cache_key in self.memory_cache or self._cache_file(cache_key).exists()


def _resolve_jit_arg_type(arg, annotation):
    """Resolve the JitArgument type for an argument, using the same dispatch
    logic as convert_to_jit_arguments.  Returns the type (not an instance)."""
    from .jit_argument import JitArgumentRegistry

    if isinstance(annotation, type) and issubclass(annotation, JitArgument):
        return annotation
    if isinstance(arg, JitArgument):
        return type(arg)
    constructor, _ = JitArgumentRegistry.get(type(arg))
    return constructor


def _build_call_spec(sig, args_tuple):
    """Build reusable ABI slot specifications without retaining argument values."""

    slot_specs = []
    has_user_stream = False

    for i, (param_name, param) in enumerate(sig.parameters.items()):
        annotation = param.annotation

        if annotation is not inspect.Parameter.empty and Constexpr.is_constexpr_annotation(annotation):
            continue

        if annotation is not inspect.Parameter.empty and is_type_param_annotation(annotation):
            continue

        if getattr(annotation, "_is_stream_param", False):
            has_user_stream = True

        arg = args_tuple[i]

        jit_arg_type = _resolve_jit_arg_type(arg, annotation)
        if jit_arg_type is None:
            raise TypeError(
                f"@flyc.jit argument {param_name!r} of type {type(arg).__name__} is not a "
                f"registered JitArgument type and cannot be packed for host dispatch."
            )

        inst = arg if isinstance(arg, jit_arg_type) else jit_arg_type(arg)
        for ctype, fill in c_abi_spec(inst):
            slot_specs.append((i, ctype, fill))

    # Auto-stream: NULL ptr selects HIP default stream when no user stream arg.
    if not has_user_stream:
        slot_specs.append((-1, ctypes.c_void_p, None))

    return tuple(slot_specs)


def _call_state_from_spec(call_spec, artifact):
    if not artifact._ktrace_traced:
        return CallState(call_spec, artifact._get_func_exe())

    from ..expr import ktrace as _ktrace

    _ktrace.require_compatible_buffer(artifact)

    trace_backend = get_ktrace_backend()
    slot_specs = (*call_spec, (-1, ctypes.c_void_p, None))
    presets = {len(call_spec): trace_backend.ensure_buffer().device_ptr}
    return CallState(slot_specs, artifact._get_func_exe(), presets, trace_backend.require_dispatchable_buffer)


def _build_call_state(sig, args_tuple, artifact):
    """Build a :class:`CallState` for fast repeated dispatch."""
    return _call_state_from_spec(_build_call_spec(sig, args_tuple), artifact)


def _has_live_device_argument(value) -> bool:
    """Recognize device buffers without querying or initializing a runtime."""
    if isinstance(value, PointerJitArg):
        return value.pointer.value is not None
    if isinstance(value, TorchTensorJitArg):
        return value.torch_tensor.device.type == "cuda"
    if isinstance(value, DLTensorJitArg):
        return bool(value.dladaptor.data_ptr) and value.dladaptor.address_space == 1

    from ..expr.struct import _runtime_fields, is_struct_type

    if is_struct_type(type(value)):
        return any(_has_live_device_argument(field_value) for _name, field_value in _runtime_fields(value))
    constructor, _dsl_type = JitArgumentRegistry.get(type(value))
    return constructor is TorchTensorJitArg and value.device.type == "cuda"


def _is_compile_placeholder(value, *, allow_null_pointers=False) -> bool:
    """Return whether ``value`` can specialize compilation but not be launched."""
    if isinstance(value, PointerJitArg):
        return value.pointer.value is None and not allow_null_pointers
    if isinstance(value, TorchTensorJitArg):
        return value.torch_tensor.device.type != "cuda"
    if isinstance(value, DLTensorJitArg):
        return not value.dladaptor.data_ptr or value.dladaptor.address_space != 1
    if isinstance(value, Stream):
        return False

    from ..expr.struct import _runtime_fields, is_struct_type

    if is_struct_type(type(value)):
        return any(
            _is_compile_placeholder(field_value, allow_null_pointers=allow_null_pointers)
            for _name, field_value in _runtime_fields(value)
        )
    constructor, _dsl_type = JitArgumentRegistry.get(type(value))
    return constructor is TorchTensorJitArg and value.device.type != "cuda"


@dataclass
class _TraceResult:
    module: ir.Module
    link_libs: Optional[list]
    post_load_processors: list
    extern_linked: bool
    param_names: List[str]
    jit_args: list
    has_user_stream: bool
    has_ktrace_buf: bool


class JitFunction:
    def __init__(self, func: Callable, compile_hints: Optional[dict] = None):
        install_excepthook()
        self.func = ASTRewriter.transform(func)
        self.compile_hints = dict(compile_hints) if compile_hints is not None else {}
        self.cache_manager = None
        self._sig = None  # lazy: set on first call
        self._has_self_param = False  # lazy: set in _ensure_sig
        self._backend_target = None  # refreshed for each compilation request
        self._mem_cache = {}
        self._last_compiled = None  # (cache_key, CompiledArtifact) for compile()

    def __get__(self, obj, objtype=None):
        # when used as a method on a class bind the owning instance as the
        # first positional argument.
        if obj is None:
            return self
        return partial(self.__call__, obj)

    def _effective_compile_hints(self):
        """Resolve persistent defaults and the current thread-local overlay."""
        return merge_compile_hints(self.compile_hints, CompilationContext.get_compile_hints())

    def _split_self(self, args):
        """Ensure the signature is resolved and split a leading ``self`` off ``args``."""
        self._ensure_sig()
        if not self._has_self_param:
            return None, args
        if not args:
            raise TypeError(f"{self.func.__name__}() missing 'self' argument")
        return args[0], args[1:]

    def cache_info(self) -> Optional[CacheInfo]:
        """Return cache statistics, or ``None`` if the disk cache is disabled."""
        if self.cache_manager is None:
            return None
        return self.cache_manager.cache_info()

    def _ensure_sig(self):
        """Initialize signature + param metadata on first call (not at decoration time)."""
        if self._sig is not None:
            return
        full_sig = resolve_signature(self.func)
        params = list(full_sig.parameters.values())

        self._has_self_param = bool(params) and params[0].name == "self"
        if self._has_self_param:
            self._sig = full_sig.replace(parameters=params[1:])
        else:
            self._sig = full_sig

        # Definition-time annotation validity check (once per function, signature-only).
        warn_invalid_annotations(self._sig, context="@jit")

    def _ensure_cache_manager(self):
        run_only = env.runtime.run_only
        if run_only and (env.debug.dump_ir or env.debug.dump_asm):
            raise ValueError(
                "FLYDSL_RUNTIME_RUN_ONLY=1 is incompatible with FLYDSL_DUMP_IR=1 or FLYDSL_DEBUG_DUMP_ASM=1: "
                "run-only mode skips the MLIR pass pipeline that would produce IR dumps."
            )
        cache_root = env.runtime.cache_dir
        if not cache_root and run_only:
            raise RuntimeError("FLYDSL_RUNTIME_RUN_ONLY=1 but FLYDSL_RUNTIME_CACHE_DIR is empty.")
        if not cache_root or not (env.runtime.enable_cache or run_only):
            self.cache_manager = None
            return
        cache_dir = Path(cache_root) / self.func.__name__
        if self.cache_manager is None or self.cache_manager.cache_dir != cache_dir:
            self.cache_manager = JitCacheManager(cache_dir)

    def _resolve_arguments(self, bound_args) -> None:
        """Replace runtime call values in ``bound_args`` with their ``JitArgument``.

        ``Constexpr[T]`` and ``Type[T]`` values are left as-is: tracing bakes them
        into the IR, which alone identifies the artifact.
        """
        for name, arg in bound_args.items():
            ann = self._sig.parameters[name].annotation
            if ann is not inspect.Parameter.empty:
                if Constexpr.is_constexpr_annotation(ann) or is_type_param_annotation(ann):
                    continue
            if isinstance(arg, JitArgument):
                continue
            if isinstance(ann, type) and issubclass(ann, JitArgument):
                bound_args[name] = ann(arg)
                continue
            ctor, _ = JitArgumentRegistry.get(type(arg))
            if ctor is None:
                raise TypeError(
                    f"{name}: {type(arg).__name__} is neither a JitArgument nor has a registered constructor."
                )
            bound_args[name] = ctor(arg)

    def _trace(self, ctx, sig, bound, bound_self, *, execute: bool) -> "_TraceResult":
        """Trace the complete launcher and kernels without lowering or executing."""
        param_names, jit_args, dsl_types, constexpr_values = convert_to_jit_arguments(sig, bound)
        for pname, dsl_type in zip(param_names, dsl_types):
            ann = sig.parameters[pname].annotation
            if ann is not inspect.Parameter.empty and isinstance(ann, type) and not issubclass(dsl_type, ann):
                warn_annotation_value_mismatch(pname, ann, dsl_type, context="@jit")
        has_user_stream = _ensure_stream_arg(jit_args)
        has_ktrace_buf = _ensure_ktrace_buffer_arg(jit_args, allocate=execute and not env.compile.compile_only)
        ir_types = get_ir_types(jit_args)
        loc = func_def_location(self.func, ctx)

        log().info(f"jit_args={jit_args}")
        log().info(f"dsl_types={dsl_types}")

        module = ir.Module.create(loc=loc)
        module.operation.attributes["gpu.container_module"] = ir.UnitAttr.get()
        _attach_ktrace_config(module, has_ktrace_buf)

        with ir.InsertionPoint(module.body), loc:
            backend = get_backend()
            gpu_module = create_gpu_module("kernels", targets=backend.gpu_module_targets())

            func_op = func.FuncOp(self.func.__name__, (ir_types, []))
            func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            entry_block = func_op.add_entry_block()

            with CompilationContext.create() as comp_ctx:
                comp_ctx.gpu_module_op = gpu_module
                comp_ctx.gpu_module_body = get_gpu_module_body(gpu_module)

                with ir.InsertionPoint(entry_block):
                    ir_args = list(func_op.regions[0].blocks[0].arguments)
                    if has_ktrace_buf:
                        comp_ctx.ktrace_buf_arg = ir_args[-1]
                    if not has_user_stream:
                        comp_ctx.stream_arg = ir_args[-2] if has_ktrace_buf else ir_args[-1]
                    user_jit_args = jit_args[: len(param_names)]
                    dsl_args = construct_from_ir_values(dsl_types, user_jit_args, ir_args)
                    log().info(f"dsl_args={dsl_args}")
                    named_args = dict(zip(param_names, dsl_args))
                    named_args.update(constexpr_values)
                    with tracing_context(
                        self.func,
                        fastmath=effective_fastmath_hint(CompilationContext.get_compile_hints()),
                    ):
                        if bound_self is not None:
                            self.func(bound_self, **named_args)
                        else:
                            self.func(**named_args)
                    func.ReturnOp([])

        link_libs = list(comp_ctx.link_libs) if comp_ctx.link_libs else None
        post_load_processors = list(comp_ctx.post_load_processors)
        extern_linked = bool(link_libs or post_load_processors)
        if extern_linked and _use_external_binary_codegen():
            raise RuntimeError(
                "FLYDSL_COMPILE_LLVM_DIR external codegen does not support extern-linked kernels yet; "
                "use embedded codegen for kernels that require #fly.explicit_module."
            )
        if extern_linked:
            gpu_module.offloadingHandler = ir.Attribute.parse("#fly.explicit_module")
            if "targets" in gpu_module.operation.attributes:
                del gpu_module.operation.attributes["targets"]

        return _TraceResult(
            module=module,
            link_libs=link_libs,
            post_load_processors=post_load_processors,
            extern_linked=extern_linked,
            param_names=param_names,
            jit_args=jit_args,
            has_user_stream=has_user_stream,
            has_ktrace_buf=has_ktrace_buf,
        )

    def __call__(self, *args, **kwargs):
        return self._compile_or_execute(args, kwargs, execute=True)

    def _compile_only(self, *args, **kwargs):
        """Compile one specialization without creating an engine or running it."""
        return self._compile_or_execute(args, kwargs, execute=False)

    def _compile_or_execute(self, args, kwargs, *, execute: bool, return_compiled: bool = False):
        if ir.Context.current is not None:
            return self.func(*args, **kwargs)

        bound_self, args = self._split_self(args)
        sig = self._sig
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        effective_hints = self._effective_compile_hints()
        backend = get_backend()
        self._backend_target = backend.target

        self._ensure_cache_manager()
        self._resolve_arguments(bound.arguments)
        args_tuple = tuple(bound.arguments.values())

        if return_compiled:
            has_device_args = any(_has_live_device_argument(value) for value in args_tuple)
            execute = not any(
                _is_compile_placeholder(value, allow_null_pointers=has_device_args) for value in args_tuple
            )
        if execute:
            from ..runtime.device_runtime import ensure_compile_runtime_pairing_from_env

            ensure_compile_runtime_pairing_from_env(backend.target.backend)

        from ..expr import ktrace as _ktrace
        from ..expr.ktrace_common import EventNameConflict, name_assignment_lock, name_table

        def accept_cached(artifact):
            try:
                _ktrace.restore_names(artifact)
            except EventNameConflict:
                if env.runtime.run_only:
                    raise RuntimeError(
                        "ktrace: cached event IDs conflict with another live binary, but "
                        "FLYDSL_RUNTIME_RUN_ONLY=1 prevents rebuilding. Prepare the traced "
                        "kernels together once with run-only mode disabled."
                    ) from None
                log().debug("ktrace: rebuilding a cached specialization with conflicting event IDs")
                return False
            return True

        tracing = effective_hints.get("ktrace", env.ktrace.enable)
        names_lock = name_assignment_lock() if tracing else nullcontext()
        with names_lock, _create_mlir_context() as ctx, CompilationContext.compile_hints(effective_hints):
            # Every ordinary JIT/compile request traces before ANY artifact hit.
            # Only an explicit CompiledFunction skips tracing on invocation.
            traced = self._trace(ctx, sig, bound, bound_self, execute=execute)
            asm, config = MlirCompiler.prepare(
                traced.module,
                backend=backend,
                compile_hints=effective_hints,
                link_libs=traced.link_libs,
            )
            cache_key = _artifact_cache_key(asm, config, backend)
            # Link libraries and Python post-load initializers live outside the IR:
            # reuse such artifacts only in-process, never from disk.
            dumping = env.debug.dump_ir or env.debug.dump_asm
            manager = None if traced.extern_linked or dumping else self.cache_manager
            compiled_func = self._mem_cache.get(cache_key)
            disk_key = None if manager is None else _disk_cache_key(cache_key)
            if compiled_func is None and manager is not None:
                compiled_func = manager.get(disk_key)
            if compiled_func is not None and not accept_cached(compiled_func):
                compiled_func = None

            if compiled_func is None:
                if env.runtime.run_only:
                    reason = (
                        "it links external libraries or post-load initializers, which are never cached on disk"
                        if traced.extern_linked
                        else f"no artifact {disk_key} in {manager.cache_dir}"
                    )
                    raise RuntimeError(
                        f"FLYDSL_RUNTIME_RUN_ONLY=1 but no usable AOT cache for {self.func.__name__}: {reason}. "
                        "Run-only traces current Python code but never lowers or generates binaries."
                    )
                lock = (
                    manager.compile_lock(disk_key, accept_cached=accept_cached)
                    if manager
                    else nullcontext((None, None))
                )
                with lock as (existing, writer):
                    compiled_func = existing
                    if compiled_func is None:
                        if traced.has_ktrace_buf:
                            # Pin event IDs this process already assigned. They stay out of the
                            # key: accept_cached rejects a cached binary whose IDs conflict.
                            _attach_ktrace_config(traced.module, True, known_names=name_table())
                        # Keep source locations for pass diagnostics even when DWARF is off.
                        # The key omits them unless debug info affects the generated artifact.
                        source_ir = traced.module.operation.get_asm(enable_debug_info=True, assume_verified=True)
                        compiled_module = MlirCompiler.compile(source_ir, func_name=self.func.__name__, config=config)
                        compiled_func = CompiledArtifact(
                            compiled_module,
                            self.func.__name__,
                            source_ir,
                            post_load_processors=traced.post_load_processors,
                            link_libs=traced.link_libs,
                            uses_explicit_module=traced.extern_linked,
                        )
                        _capture_ktrace_names(compiled_func, traced.has_ktrace_buf, compiled_module)
                        compiled_func._ktrace_traced = traced.has_ktrace_buf
                        if writer:
                            try:
                                writer(compiled_func)
                            except Exception as e:
                                log().warning(f"Failed to write compile cache: {e}")

            self._mem_cache[cache_key] = compiled_func
            self._last_compiled = (cache_key, compiled_func)

        # OUTSIDE MLIR context/locks: engine initialization and kernel launch.
        if env.compile.compile_only:
            return None
        if not execute:
            return self._make_compiled_function(cache_key, bound, None) if return_compiled else None
        # A traced call rebuilds its adapter (well under 1% of the trace); only a
        # CompiledFunction keeps one to skip tracing.
        state = _build_call_state(sig, args_tuple, compiled_func)
        result = state(args_tuple)
        return self._make_compiled_function(cache_key, bound, state) if return_compiled else result

    def _make_compiled_function(self, cache_key, bound, call_state):
        """Reuse this dispatch's cache lookup and ABI; never prepare export IR here."""
        artifact = self._mem_cache[cache_key]
        compiled = CompiledFunction(call_state, artifact)
        if call_state is None:
            compiled._call_spec = _build_call_spec(self._sig, tuple(bound.arguments.values()))
        compiled._backend = self._backend_target.backend
        compiled._aot_spec, compiled._aot_error = _prepare_aot_signature(self, bound, artifact)
        return compiled


def _ensure_stream_arg(jit_args: list) -> bool:
    """Ensure jit_args contains a Stream argument.  If the user's function
    already declares ``stream: fx.Stream``, return True (user-supplied).
    Otherwise append a default ``Stream(None)`` and return False."""
    if any(isinstance(a, Stream) for a in jit_args):
        return True
    jit_args.append(Stream(None))
    return False


def _attach_ktrace_config(module, traced: bool, known_names=None) -> None:
    if not traced:
        return
    from ..utils import env

    hints = CompilationContext.get_compile_hints()
    blocks = (hints.get("ktrace_blocks", env.ktrace.blocks) or "").strip()
    get_ktrace_backend().configure_module(
        module,
        target_arch=get_backend().target.arch,
        known_names=known_names,
        blocks=blocks,
    )


def _ktrace_names_from_module(module) -> dict | None:
    try:
        attrs = module.operation.attributes
        if "fly_ktrace.event_names" not in attrs:
            return None
        return {
            str(named.name): int(str(named.attr).split(":")[0].strip()) for named in attrs["fly_ktrace.event_names"]
        }
    except Exception as exc:  # noqa: BLE001
        log().warning(f"ktrace: could not read the event-name table from the module: {exc}")
        return None


def _capture_ktrace_names(artifact, traced: bool, module) -> None:
    if not traced:
        return
    from ..expr.ktrace_common import merge_names

    artifact._ktrace_buffer_config = get_ktrace_backend().buffer_config()
    artifact._ktrace_names = _ktrace_names_from_module(module) or {}
    merge_names(artifact._ktrace_names)


def _ensure_ktrace_buffer_arg(jit_args: list, *, allocate: bool = True) -> bool:
    from ..expr import ktrace as _ktrace

    if not _ktrace.tracing_enabled():
        return False
    from ..expr.numeric import Int8
    from ..utils import env
    from .jit_argument import PointerJitArg

    address = None
    if allocate and not env.compile.compile_only:
        address = ctypes.c_void_p(get_ktrace_backend().ensure_buffer().device_ptr)
    jit_args.append(PointerJitArg(Int8, address))
    return True


def jit(func: Optional[Callable] = None) -> JitFunction:
    """JIT decorator for host launcher functions."""
    if func is None:
        return lambda f: JitFunction(f)
    return JitFunction(func)


class CompiledFunction:
    """Pre-compiled callable returned by ``flyc.compile()``.

    Calls remain positional-only and reuse a fixed specialization without
    retracing or checking changed Python captures. Request a new specialization
    with ``flyc.compile()`` after changing compile-time state.

    Placeholder-only arguments and non-device tensors defer runtime
    initialization until the first call, allowing CPU-only compilation and
    :meth:`export_to_c` without retracing the launcher.
    """

    __slots__ = (
        "_call_state",
        "_call_spec",
        "_call_lock",
        "_runtime_ready",
        "_backend",
        "_keepalive",
        "_aot_compiled",
        "_aot_error",
        "_aot_spec",
    )

    def __init__(self, call_state, keepalive):
        self._call_state = call_state
        self._call_spec = None
        self._call_lock = threading.Lock()
        self._runtime_ready = call_state is not None
        self._backend = None
        self._keepalive = keepalive  # prevent GC of CompiledArtifact / ExecutionEngine
        self._aot_compiled = None
        self._aot_error = None
        self._aot_spec = None

    def __call__(self, *args):
        state = self._call_state
        if not self._runtime_ready:
            with self._call_lock:
                if not self._runtime_ready:
                    from ..runtime.device_runtime import ensure_compile_runtime_pairing_from_env

                    ensure_compile_runtime_pairing_from_env(self._backend)
                    state = self._call_state
                    if state is None:
                        state = _call_state_from_spec(self._call_spec, self._keepalive)
                        self._call_state = state
                    self._runtime_ready = True
        state = self._call_state
        assert state is not None
        return state(args)

    def max_blocks_per_cu(self) -> int:
        """Max resident blocks of this kernel per CU on the current device, without launching it."""
        return self._keepalive.max_blocks_per_cu()

    def export_to_c(self, file_path: str, file_name: str, function_prefix: str = "") -> None:
        """Export this specialization as a standalone ``.o`` and C header.

        The object embeds its backend adapter and only needs the backend's
        system runtime libraries when linked into an executable or shared
        library; it does not require FlyDSL or Python at deployment time.
        Host export currently requires 64-bit little-endian Linux ELF tools.
        Export IR and ABI validation are prepared lazily on the first export.
        """
        self._get_aot_compiled()._export_to_c(file_path, file_name, function_prefix)

    def _get_aot_compiled(self):
        """Parse and validate export IR once, only when an export is requested."""
        with self._call_lock:
            if self._aot_error is not None:
                error_type, message = self._aot_error
                raise error_type(message)
            if self._aot_compiled is None:
                if self._aot_spec is None:
                    raise RuntimeError("flyc.compile(): no AOT export artifact is available")
                from ._aot import _aot_from_jit_artifact

                abi_spec, target = self._aot_spec
                self._aot_compiled = _aot_from_jit_artifact(
                    self._keepalive, abi_spec=abi_spec, backend=target.backend, arch=target.arch
                )
            return self._aot_compiled


def _prepare_aot_signature(jf, bound, artifact):
    """Snapshot this callable's ABI without retaining buffers or parsing IR."""
    if getattr(artifact, "_ktrace_traced", False):
        return None, (NotImplementedError, "ktrace-instrumented kernels cannot be exported to C")
    try:
        from ._aot import _snapshot_abi

        abi_spec = _snapshot_abi(jf._sig, bound.arguments)
        return (abi_spec, jf._backend_target), None
    except Exception as exc:
        error_type = (
            type(exc) if isinstance(exc, (TypeError, ValueError, NotImplementedError, RuntimeError)) else RuntimeError
        )
        return None, (error_type, str(exc))


def _unwrap_jit_function(func):
    if isinstance(func, JitFunction):
        return func, None
    if isinstance(func, partial):
        method = func.func
        owner = getattr(method, "__self__", None)
        if (
            isinstance(owner, JitFunction)
            and getattr(method, "__func__", None) is JitFunction.__call__
            and len(func.args) == 1
            and not func.keywords
        ):
            return owner, func.args[0]
    return None, None


def _compile_impl(func, *args, **kwargs) -> Optional[CompiledFunction]:
    """Pre-compile a ``@flyc.jit`` function, returning a fast callable.

    Usage::

        compiled_fn = flyc.compile(launch_gemm, c, a, b, sa, sb, M, N, stream)

        # Hot loop — minimal dispatch overhead (~5 µs):
        for ...:
            compiled_fn(c, a, b, sa, sb, M, N, stream)

    Compilation arguments may be positional or keyword arguments. Live device
    arguments retain the existing initial launch, including when optional pointer
    arguments are null. Null pointer wrappers without any live device buffers,
    and non-device tensors, are treated as compile placeholders, which keeps
    the compile/export path independent of a device runtime. The returned
    :class:`CompiledFunction` keeps its existing positional-only fast path.

    Constexpr values are baked in at compile time and ignored on subsequent
    calls; only runtime values (data pointers, scalars, stream) may change.

    When ``COMPILE_ONLY`` is enabled, compilation and cache persistence still
    run, but no execution engine is materialized and ``None`` is returned.
    """
    jf, bound_self = _unwrap_jit_function(func)
    if jf is None:
        raise TypeError(f"flyc.compile() expects a @flyc.jit function, got {type(func).__name__}")
    if bound_self is not None:
        args = (bound_self, *args)

    return jf._compile_or_execute(args, kwargs, execute=True, return_compiled=True)


class CompileCallable:
    """Subscriptable compile callable.

    Usage::

        flyc.compile(launch, *args)                          # no hints
        flyc.compile[{"fast_fp_math": True}](launch, *args)  # with hints
        flyc.compile[hints](launch)                          # deferred
    """

    def __init__(self, compile_hints: Optional[dict] = None):
        self._compile_hints = compile_hints

    def __getitem__(self, hints: dict) -> "CompileCallable":
        """``flyc.compile[{...}]`` → new CompileCallable with compile hints."""
        if not isinstance(hints, dict):
            raise TypeError(f"flyc.compile[...] expects a dict of compile hints, got {type(hints).__name__}")
        return CompileCallable(compile_hints=hints)

    def __call__(self, func, *args, **kwargs):
        jit_func, _bound_self = _unwrap_jit_function(func)
        if self._compile_hints and jit_func is not None:
            jit_func.compile_hints = {**jit_func.compile_hints, **self._compile_hints}
        if not args and not kwargs:
            # No args → just return the (hinted) function for deferred compilation
            return func
        return _compile_impl(func, *args, **kwargs)


compile = CompileCallable()
