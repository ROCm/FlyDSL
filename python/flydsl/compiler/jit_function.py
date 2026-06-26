# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import ctypes
import fcntl
import hashlib
import inspect
import os
import pickle
import pkgutil
import tempfile
import time
import types
from collections import namedtuple
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .._mlir import ir
from .._mlir.dialects import func
from .._mlir.passmanager import PassManager
from ..expr.meta import tracing_context
from ..expr.typing import Constexpr, Stream
from ..utils import env, log
from .ast_rewriter import ASTRewriter
from .backends import compile_backend_name, get_backend
from .diagnostics import (
    DSLCompileError,
    diag_records_from_mlir_error,
    dsl_ir_diagnostics,
    install_excepthook,
    warn_annotation_value_mismatch,
    warn_invalid_annotations,
)
from .jit_argument import convert_to_jit_arguments, is_type_param_annotation, resolve_signature
from .jit_executor import CallState, CompiledArtifact
from .kernel_function import (
    CompilationContext,
    KernelFunction,
    create_gpu_module,
    func_def_location,
    get_gpu_module_body,
)
from .link_utils import _append_link_lib_options_to_attach_targets, _format_link_lib_options
from .protocol import (
    JitArgument,
    c_abi_spec,
    cache_signature,
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


# Sentinel distinct from a real ``None`` snapshot value (a since-deleted global).
_NOT_IN_BASELINE = object()


# Environment variables that influence code generation *independently of the
# resolved GPUTarget*. Their current values enter the hot cache key so
# cross-process / cross-config artifacts don't collide.
_CACHE_INVALIDATING_ENV_VARS = (
    "FLYDSL_COMPILE_OPT_LEVEL",
    "FLYDSL_COMPILE_BACKEND",
    "FLYDSL_COMPILE_LLVM_DIR",
    "FLYDSL_DEBUG_ENABLE_DEBUG_INFO",
    "FLYDSL_EXTRA_SOURCE_DIRS",
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


def _snapshot_global_value(val, *, stable, _path=()):
    """Summarize a captured global for the cache key / drift detection.

    ``stable=True`` is cross-process-stable (no ``id()``, which is randomized per
    process) so two processes importing the same module produce the same key —
    suitable for folding into cache keys. ``stable=False`` includes ``id()`` so
    in-process drift detection catches rebinding even when type/value look equal.
    ``_path`` carries the ids of containers currently being walked, breaking reference cycles.

    Scalars and builtin containers (tuple/list/dict/set) are summarized **by
    value**, recursively, in both modes — so different contents produce different
    keys (cross-process) and in-place mutation is detected (in-process).
    """
    if isinstance(val, (int, float, bool, str, bytes, type(None))):
        return ("scalar", val)
    if isinstance(val, (tuple, list, set, frozenset, dict)):
        if id(val) in _path:
            return ("cycle", type(val).__qualname__)
        _path = _path + (id(val),)
        kind = type(val).__qualname__
        if isinstance(val, dict):
            # sort by repr for a canonical, comparison-safe order (mixed key types)
            items = sorted(
                (
                    (
                        _snapshot_global_value(k, stable=stable, _path=_path),
                        _snapshot_global_value(v, stable=stable, _path=_path),
                    )
                    for k, v in val.items()
                ),
                key=repr,
            )
            return ("dict", tuple(items))
        elems = (_snapshot_global_value(v, stable=stable, _path=_path) for v in val)
        if isinstance(val, (set, frozenset)):
            # sets are unordered: sort by repr for a canonical order
            return (kind, tuple(sorted(elems, key=repr)))
        return (kind, tuple(elems))
    if callable(val):
        if stable:
            # qualname+module is stable across processes; repr would bake in
            # <0x...> addresses for many callables.
            qualname = getattr(val, "__qualname__", None) or getattr(val, "__name__", "?")
            return ("callable", getattr(val, "__module__", "?"), qualname)
        return ("callable", id(val), repr(val))
    # Opaque object: stable collapses to type; drift keeps id to catch rebinding.
    return ("obj", type(val).__qualname__) if stable else ("obj", id(val), type(val).__qualname__)


def _discover_global_refs(func, owner_cls=None) -> List[Tuple[str, str, dict]]:
    """Statically discover which module globals ``func`` and its (jit / kernel /
    same-dir user) dependencies *read*.

    The result depends only on the code objects in the dependency tree — NOT on
    the current values of those globals — so it is stable across calls and can be
    memoized per ``(func, owner_cls)``. The recursive walk (co_names + closures +
    class members) therefore runs once instead of on every JIT call; the hot path
    only re-reads the discovered values via :func:`_snapshot_refs`.

    Returns a sorted list of ``(name, module_name, globals_dict)`` triples. The
    snapshot is keyed on ``(name, module_name)`` so the same identifier appearing
    in different modules doesn't collide. JitFunction / KernelFunction deps are
    always recursed into (regardless of source location); plain Python helpers are
    filtered by ``_is_user_function`` (same directory as the root file) to avoid
    pulling stdlib / third-party globals into the snapshot.

    ``owner_cls`` enables recursing into ``self.helper`` style method calls so
    helpers attached to the owning class also have their globals tracked.
    """
    from .kernel_function import KernelFunction

    try:
        rootFile = inspect.getfile(func)
    except (TypeError, OSError):
        rootFile = ""
    refs: Dict[Tuple[str, str], dict] = {}
    visited: Set[int] = set()

    def _walk(f, cls=None):
        if id(f) in visited:
            return
        visited.add(id(f))
        f_globals = getattr(f, "__globals__", {})
        mod_name = f_globals.get("__name__", "?")
        for name in f.__code__.co_names:
            key = (name, mod_name)
            if name in f_globals and key not in refs:
                val = f_globals[name]
                refs[key] = f_globals
                underlying = _get_underlying_func(val)
                if underlying is not None and (
                    isinstance(val, (JitFunction, KernelFunction)) or _is_user_function(underlying, rootFile)
                ):
                    _walk(underlying)
            # Also recurse through class-member helpers (self.helper(...))
            if cls is not None:
                try:
                    obj = getattr(cls, name)
                except AttributeError:
                    obj = None
                underlying = _get_underlying_func(obj) if obj is not None else None
                if underlying is not None and _is_user_function(underlying, rootFile):
                    _walk(underlying, cls=cls)
        if f.__code__.co_freevars and getattr(f, "__closure__", None):
            for _cname, cell in zip(f.__code__.co_freevars, f.__closure__):
                try:
                    val = cell.cell_contents
                except ValueError:
                    continue
                underlying = _get_underlying_func(val)
                if underlying is not None:
                    _walk(underlying)

    _walk(func, cls=owner_cls or _owner_class_from_func(func))
    return [(name, mod_name, refs[(name, mod_name)]) for (name, mod_name) in sorted(refs)]


def _snapshot_refs(refs: List[Tuple[str, str, dict]], *, stable: bool) -> Dict[Tuple[str, str], Any]:
    """Read the *current* values of pre-discovered global refs into a snapshot.

    Cheap O(N) dict reads with no recursion — this is what runs on every JIT call.
    ``stable=True`` returns cross-process-stable values (no ``id()``), suitable for
    folding into cache keys. ``stable=False`` returns id-bearing values used for
    in-process drift detection.
    """
    out: Dict[Tuple[str, str], Any] = {}
    for name, mod_name, var_dict in refs:
        if name in var_dict:
            out[(name, mod_name)] = _snapshot_global_value(var_dict[name], stable=stable)
    return out


def _flydsl_key() -> str:
    extra = list(EXTRA_SOURCE_DIRS)
    env_extra = os.environ.get("FLYDSL_EXTRA_SOURCE_DIRS", "")
    if env_extra:
        extra.extend(d.strip() for d in env_extra.split(":") if d.strip())
    return _flydsl_key_cached(_use_external_binary_codegen(), env.compile.llvm_dir, tuple(extra))


@lru_cache(maxsize=4)
def _flydsl_key_cached(use_external_binary: bool, llvm_dir: str, extra_source_dirs: tuple = ()) -> str:
    """Compute a hash fingerprint of the entire FlyDSL compiler toolchain.

    Covers:
      1. All Python source files under flydsl.compiler.*, flydsl.expr.*,
         flydsl.runtime.*, flydsl.utils.*
      2. Native shared libraries (_mlirDialectsFly*.so, libFly*.so, libfly_jit_runtime.so,
         libmlir_rocm_runtime.so)
      3. flydsl.__version__

    Any change to compiler code, pass pipeline, runtime wrappers, or C++
    bindings will produce a different key, invalidating stale disk caches.
    """
    import flydsl

    contents = []

    flydsl_root = Path(flydsl.__file__).resolve().parent

    # 1) Hash all Python source files in key sub-packages.
    pkg_prefixes = [
        (str(flydsl_root / "compiler"), "flydsl.compiler."),
        (str(flydsl_root / "expr"), "flydsl.expr."),
        (str(flydsl_root / "runtime"), "flydsl.runtime."),
        (str(flydsl_root / "utils"), "flydsl.utils."),
    ]
    for pkg_path, prefix in pkg_prefixes:
        if not os.path.isdir(pkg_path):
            continue
        for lib in pkgutil.walk_packages([pkg_path], prefix=prefix):
            try:
                spec = lib.module_finder.find_spec(lib.name)
                if spec and spec.origin and os.path.isfile(spec.origin):
                    with open(spec.origin, "rb") as f:
                        contents.append(hashlib.sha256(f.read()).hexdigest())
            except Exception:
                pass

    p = flydsl_root / "__init__.py"
    if p.is_file():
        with open(p, "rb") as f:
            contents.append(hashlib.sha256(f.read()).hexdigest())

    # 2) Hash native shared libraries (C++ passes, runtime wrappers, bindings).
    backend = get_backend()
    mlir_libs_dir = flydsl_root / "_mlir" / "_mlir_libs"
    if mlir_libs_dir.is_dir():
        for pattern in backend.native_lib_patterns():
            for so_file in sorted(mlir_libs_dir.glob(pattern)):
                h = hashlib.sha256()
                with open(so_file, "rb") as f:
                    while True:
                        chunk = f.read(1024 * 1024)
                        if not chunk:
                            break
                        h.update(chunk)
                contents.append(h.hexdigest())

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

    key = f"flydsl:{flydsl.__version__}:{backend.hash()}-" + "-".join(contents)
    log().debug(f"flydsl_key: {hashlib.sha256(key.encode()).hexdigest()[:16]}")
    return key


def _use_external_binary_codegen() -> bool:
    return bool(env.compile.llvm_dir.strip())


def _get_underlying_func(obj):
    if isinstance(obj, KernelFunction):
        # Prefer the pre-AST-rewrite func: its closure / co_names still
        # reference helper callables, which is what the cache-key dependency
        # collector needs to walk to detect helper source changes.  Fallback
        # to `_func` for older KernelFunction instances without the field.
        return getattr(obj, "_original_func", obj._func)
    if isinstance(obj, JitFunction):
        return getattr(obj, "_original_func", obj.func)
    if isinstance(obj, types.MethodType):
        return obj.__func__
    if isinstance(obj, types.FunctionType):
        return obj
    return None


def _get_func_source(func) -> str:
    try:
        return inspect.getsource(func)
    except OSError:
        return func.__code__.co_code.hex()


def _is_user_function(func, rootFile):
    try:
        funcFile = inspect.getfile(func)
    except (TypeError, OSError):
        return False
    return os.path.dirname(os.path.abspath(funcFile)) == os.path.dirname(os.path.abspath(rootFile))


def _owner_class_from_func(func):
    qualname = getattr(func, "__qualname__", "")
    parts = qualname.split(".")[:-1]
    if not parts or "<locals>" in parts:
        return None

    obj = func.__globals__.get(parts[0])
    for part in parts[1:]:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj if isinstance(obj, type) else None


def _collect_class_member_dependency_sources(
    func,
    rootFile,
    owner_cls,
    visited: Set[int],
) -> List[str]:
    sources = []

    for name in func.__code__.co_names:
        try:
            obj = getattr(owner_cls, name)
        except AttributeError:
            continue

        underlying = _get_underlying_func(obj)
        if underlying is None or id(underlying) in visited:
            continue
        if not _is_user_function(underlying, rootFile):
            continue

        visited.add(id(underlying))
        sources.append(f"class:{owner_cls.__qualname__}.{name}:{_get_func_source(underlying)}")
        sources.extend(_collect_dependency_sources(underlying, rootFile, visited, owner_cls=owner_cls))

    return sources


def _collect_closure_scalar_vals(func, visited_ids: Optional[Set[int]] = None) -> List[str]:
    """Recursively collect scalar closure values from func and all callable deps in its closure.

    This ensures that compile-time parameters captured by nested @kernel functions
    (e.g. tile_m, tile_n, waves_per_eu inside a KernelFunction._func) are included
    in the cache key even when the outer @jit launcher does not reference them directly.
    """
    if visited_ids is None:
        visited_ids = set()
    if id(func) in visited_ids:
        return []
    visited_ids.add(id(func))

    vals = []
    if not (func.__code__.co_freevars and getattr(func, "__closure__", None)):
        return vals

    for name, cell in zip(func.__code__.co_freevars, func.__closure__):
        try:
            val = cell.cell_contents
        except ValueError:
            continue
        if isinstance(val, (int, float, bool, str, type(None), tuple)):
            vals.append(f"{name}={val!r}")
        else:
            # Recurse into callable deps (KernelFunction, JitFunction, plain functions)
            underlying = _get_underlying_func(val)
            if underlying is not None and id(underlying) not in visited_ids:
                nested = _collect_closure_scalar_vals(underlying, visited_ids)
                # Prefix with the closure var name to avoid collisions across nesting levels
                vals.extend(f"via:{name}:{v}" for v in nested)

    return vals


def _collect_dependency_sources(
    func,
    rootFile,
    visited: Optional[Set[int]] = None,
    owner_cls=None,
) -> List[str]:
    from .kernel_function import KernelFunction

    if visited is None:
        visited = set()
    sources = []

    def _emit(prefix: str, name: str, val, underlying):
        # JitFunction has its own manager_key — use it as a stable identity
        # so cross-directory deps are tracked without dragging in the full
        # source file (which would force ad-hoc same-dir filtering).
        if isinstance(val, JitFunction):
            val._ensure_cache_manager()
            sources.append(f"{prefix}jit:{name}:{val.manager_key}")
            return False  # do not recurse: manager_key already covers transitive deps
        sources.append(f"{prefix}{name}:{_get_func_source(underlying)}")
        return True  # recurse to pick up nested helpers

    # 1) Scan global name references (co_names → __globals__)
    for name in func.__code__.co_names:
        obj = func.__globals__.get(name)
        underlying = _get_underlying_func(obj)
        if underlying is None or id(underlying) in visited:
            continue
        # Always include JitFunction/KernelFunction regardless of source location;
        # plain helpers stay behind the same-dir filter to avoid stdlib pollution.
        is_jit_like = isinstance(obj, (JitFunction, KernelFunction))
        if not is_jit_like and not _is_user_function(underlying, rootFile):
            continue
        visited.add(id(underlying))
        should_recurse = _emit("", name, obj, underlying)
        if should_recurse:
            sources.extend(_collect_dependency_sources(underlying, rootFile, visited))

    # 2) Scan closure variables (co_freevars → __closure__) for callable
    #    dependencies.  This catches @flyc.kernel functions defined in an
    #    enclosing scope and captured by the @flyc.jit launcher via closure.
    if func.__code__.co_freevars and getattr(func, "__closure__", None):
        for name, cell in zip(func.__code__.co_freevars, func.__closure__):
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            underlying = _get_underlying_func(val)
            if underlying is None or id(underlying) in visited:
                continue
            visited.add(id(underlying))
            should_recurse = _emit("closure:", name, val, underlying)
            if should_recurse:
                sources.extend(_collect_dependency_sources(underlying, rootFile, visited))

    owner_cls = owner_cls or _owner_class_from_func(func)
    if owner_cls is not None:
        sources.extend(_collect_class_member_dependency_sources(func, rootFile, owner_cls, visited))

    return sources


def _jit_function_cache_key(func: Callable, owner_cls=None) -> str:
    parts = []
    parts.append(_flydsl_key())
    parts.append(_get_func_source(func))
    try:
        rootFile = inspect.getfile(func)
    except (TypeError, OSError):
        rootFile = ""
    depSources = _collect_dependency_sources(func, rootFile, owner_cls=owner_cls)
    depSources.sort()
    parts.extend(depSources)

    # Collect scalar closure values recursively — this covers compile-time parameters
    # (tile_m, tile_n, waves_per_eu, etc.) captured directly by the @jit launcher OR
    # indirectly via nested @kernel / helper functions, without requiring an explicit
    # _cache_tag tuple in every kernel factory function.
    all_closure_vals = sorted(_collect_closure_scalar_vals(func))
    if all_closure_vals:
        parts.append("closure_vals:" + ",".join(all_closure_vals))

    combined = "\n".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:32]


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


def _pipeline_fragments_for_mode(backend) -> PipelineConfig:
    """Return pipeline configuration including optional external split."""
    from .kernel_function import CompilationContext

    hints = CompilationContext.get_compile_hints()
    llvm_opts = hints.get("llvm_options")
    if _use_external_binary_codegen():
        pre_binary_fragments, binary_fragment = backend.external_binary_pipeline_fragments(compile_hints=hints)
        return PipelineConfig(
            fragments=[*pre_binary_fragments, binary_fragment],
            pre_binary=pre_binary_fragments,
            binary_fragment=binary_fragment,
            llvm_opts=llvm_opts,
            external=True,
        )

    fragments = backend.pipeline_fragments(compile_hints=hints)
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
    with dsl_ir_diagnostics(module.context) as diags:
        try:
            pm.run(module.operation)
        except Exception as exc:
            raise DSLCompileError(str(exc), diagnostics=diags) from exc


class MlirCompiler:
    @classmethod
    def compile(
        cls, module: ir.Module, *, arch: str = "", func_name: str = "", link_libs: Optional[list] = None
    ) -> ir.Module:
        try:
            module.operation.verify()
        except ir.MLIRError as exc:
            raise DSLCompileError("MLIR verification failed", diagnostics=diag_records_from_mlir_error(exc)) from exc

        backend = get_backend(arch=arch)

        module = ir.Module.parse(module.operation.get_asm(enable_debug_info=env.debug.enable_debug_info))
        cfg = _pipeline_fragments_for_mode(backend)
        fragments = cfg.fragments
        pre_binary_fragments = cfg.pre_binary
        binary_fragment = cfg.binary_fragment
        llvm_opts = cfg.llvm_opts
        external_binary = cfg.external

        if external_binary and link_libs:
            raise RuntimeError(
                "FLYDSL_COMPILE_LLVM_DIR external codegen does not support extern link_libs yet; "
                "use embedded codegen for kernels that require #fly.explicit_module."
            )

        if link_libs:
            link_opt = _format_link_lib_options(link_libs)
            fragments, found_attach_target = _append_link_lib_options_to_attach_targets(fragments, link_opt)
            if not found_attach_target:
                raise RuntimeError("link_libs specified but no attach-target fragment found in pipeline")

        from .llvm_options import llvm_options as _llvm_options

        _llvm_ctx = _llvm_options(llvm_opts) if llvm_opts else nullcontext()

        if env.debug.print_origin_ir:
            log().info(f"Origin IR: \n{module}")

        dump_enabled = env.debug.dump_ir
        dump_dir = Path(env.debug.dump_dir).resolve()

        with _llvm_ctx:
            if dump_enabled:
                asm = module.operation.get_asm(enable_debug_info=True)
                kernel_names = _infer_kernel_names_from_asm(asm)
                subdir = kernel_names[0] if len(kernel_names) == 1 else (func_name or "module")
                dump_dir = dump_dir / _sanitize_path_component(subdir)
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
                    with dsl_ir_diagnostics(module.context) as diags:
                        try:
                            pm.run(module.operation)
                        except Exception as exc:
                            raise DSLCompileError(str(exc), diagnostics=diags) from exc

                    stage_asm = module.operation.get_asm(enable_debug_info=True)
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

                    if env.debug.dump_asm:
                        raise RuntimeError(
                            "FLYDSL_DEBUG_DUMP_ASM is not supported with "
                            "FLYDSL_COMPILE_LLVM_DIR external codegen; use FLYDSL_DUMP_IR=1 "
                            "to inspect pre-binary/final MLIR, or run external LLVM tools directly for ISA dumps."
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
        {cache_root}/{func_name}_{manager_key}/
            {cache_key}.pkl  - serialized compiled kernel
            {cache_key}.lock - per-key advisory lock file

    All disk reads use shared (reader) locks; writes use exclusive locks
    with atomic ``tempfile`` + ``os.rename`` to prevent partial reads.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.memory_cache: Dict[str, Any] = {}
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _safe_key(cache_key: str) -> str:
        return hashlib.sha256(cache_key.encode()).hexdigest()[:16]

    def _cache_file(self, cache_key: str) -> Path:
        return self.cache_dir / f"{self._safe_key(cache_key)}.pkl"

    def _lock_file(self, cache_key: str) -> Path:
        return self.cache_dir / f"{self._safe_key(cache_key)}.lock"

    @staticmethod
    def _atomic_write(cache_file: Path, value: Any) -> None:
        """Write *value* atomically via tempfile + rename."""
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(cache_file.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(value, f)
            os.rename(tmp, str(cache_file))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

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
                self._atomic_write(cache_file, value)
            log().debug(f"Cache saved: {cache_file.name}")
        except Exception as e:
            log().warning(f"Failed to save cache {cache_file}: {e}")

    @contextmanager
    def compile_lock(self, cache_key: str):
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
                    self.memory_cache[cache_key] = value
                    self._hits += 1
                    yield (value, None)
                    return
                except Exception:
                    # Corrupt cache — remove so writer can overwrite.
                    try:
                        cache_file.unlink()
                    except OSError:
                        pass

            # Cache miss — provide a writer that writes under the already-held lock.
            def _writer(value):
                self._atomic_write(cache_file, value)
                self.memory_cache[cache_key] = value

            yield (None, _writer)

    def load_all(self) -> int:
        if not self.cache_dir.exists():
            return 0
        count = 0
        for cache_file in sorted(self.cache_dir.glob("*.pkl")):
            lock_path = cache_file.with_suffix(".lock")
            try:
                with FileLock(lock_path, exclusive=False, timeout=30):
                    with open(cache_file, "rb") as f:
                        pickle.load(f)
                count += 1
            except Exception:
                pass
        log().debug(f"Found {count} cached entries in {self.cache_dir}")
        return count

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


def _build_call_state(sig, args_tuple, func_exe):
    """Build a CallState for fast repeated dispatch.

    Resolves each parameter's JitArgument type using the same registry as
    convert_to_jit_arguments, then asks it for a reusable slot specification.
    This ensures a single source of truth for argument packing.
    """

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

    return CallState(slot_specs, func_exe)


class JitFunction:
    def __init__(self, func: Callable, compile_hints: Optional[dict] = None):
        install_excepthook()
        # Same rationale as KernelFunction._original_func: ASTRewriter.transform
        # mutates `func.__code__` in place, after which the JIT cache walker
        # (`_get_underlying_func`) can no longer see closure-captured helpers
        # via the original co_names / co_freevars.  Snapshot the pre-rewrite
        # func here so the walker can recover those references.
        import types as _types

        _orig_code = func.__code__
        self._original_func = _types.FunctionType(
            _orig_code,
            func.__globals__,
            name=func.__name__,
            argdefs=func.__defaults__,
            closure=func.__closure__,
        )
        self._original_func.__kwdefaults__ = func.__kwdefaults__
        self._original_func.__qualname__ = func.__qualname__
        self._original_func.__module__ = func.__module__
        self.func = ASTRewriter.transform(func)
        self.compile_hints = dict(compile_hints) if compile_hints is not None else {}
        self.manager_key = None
        self._manager_owner_cls = None
        self.cache_manager = None
        self._call_state_cache = {}  # cache_key -> CallState
        self._sig = None  # lazy: set on first call
        self._has_self_param = False  # lazy: set in _ensure_sig
        self._backend_target = None  # lazy: GPUTarget resolved once in _ensure_sig
        self._mem_cache = {}
        self._last_compiled = None  # (cache_key, CompiledArtifact) for compile()
        self._extern_linkage_keys = set()

        # owner_cls -> first-compile snapshot of the used globals; RAISE on any
        # later drift. Keyed by owner_cls (like the refs/prefix caches below) so a
        # JIT method reused across owner classes drift-checks each class against
        # its own baseline instead of the first owner's.
        self._used_global_vals: Dict[Any, Dict[Tuple[str, str], Any]] = {}
        # owner_cls -> discovered global refs. The (recursive) discovery is a pure
        # function of the dependency tree's code objects, so it's memoized here and
        # the hot path only re-snapshots the current values.
        self._global_refs_cache: Dict[Any, List[Tuple[str, str, dict]]] = {}
        # owner_cls -> pre-built ``("_globals_", ...)`` cache-key segment. The
        # globals folded into the key cannot change without the drift check
        # raising first, so the segment is built once and reused on every cache
        # hit (matching Triton, which never re-snapshots globals per launch).
        self._globals_prefix_cache: Dict[Any, tuple] = {}

    def __get__(self, obj, objtype=None):
        # when used as a method on a class bind the owning instance as the
        # first positional argument.
        if obj is None:
            return self
        return partial(self.__call__, obj)

    def _get_global_refs(self, owner_cls=None) -> List[Tuple[str, str, dict]]:
        """Memoized global-ref discovery (see :func:`_discover_global_refs`)."""
        cache = self._global_refs_cache
        if owner_cls not in cache:
            cache[owner_cls] = _discover_global_refs(self.func, owner_cls)
        return cache[owner_cls]

    def _check_globals_drift(self, owner_cls=None) -> None:
        """Raise if any captured global value has changed since first compile."""
        baseline = self._used_global_vals[owner_cls]
        for name, mod_name, var_dict in self._get_global_refs(owner_cls):
            key = (name, mod_name)
            old = baseline.get(key, _NOT_IN_BASELINE)
            if old is _NOT_IN_BASELINE:
                continue
            new = _snapshot_global_value(var_dict[name], stable=False) if name in var_dict else None
            if new != old:
                raise RuntimeError(
                    f"FlyDSL: global '{name}' (module '{mod_name}') used by @flyc.jit "
                    f"'{self.func.__name__}' changed since first compile "
                    f"(old={old!r}, new={new!r}). Capturing live globals is unsafe; "
                    "restart the process or avoid mutating the global. Prefer passing "
                    "the value as an explicit argument so it participates in the cache key."
                )

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
        self._backend_target = get_backend().target  # frozen dataclass, stable

        # Definition-time annotation validity check (once per function, signature-only).
        warn_invalid_annotations(self._sig, context="@jit")

    def _ensure_cache_manager(self, owner_cls=None):
        if self.manager_key is not None and self._manager_owner_cls is owner_cls:
            return
        self._manager_owner_cls = owner_cls
        self.manager_key = _jit_function_cache_key(self.func, owner_cls=owner_cls)

        run_only = env.runtime.run_only
        if run_only and env.debug.dump_ir:
            raise ValueError(
                "FLYDSL_RUNTIME_RUN_ONLY=1 is incompatible with FLYDSL_DUMP_IR=1: "
                "run-only mode skips the MLIR pass pipeline that would produce IR dumps."
            )

        need_cache = env.runtime.enable_cache or run_only
        if not need_cache:
            self.cache_manager = None
            return

        cache_root = env.runtime.cache_dir
        if not cache_root:
            if run_only:
                raise RuntimeError("FLYDSL_RUNTIME_RUN_ONLY=1 but FLYDSL_RUNTIME_CACHE_DIR is empty.")
            self.cache_manager = None
            return

        cache_dir = Path(cache_root) / f"{self.func.__name__}_{self.manager_key}"
        self.cache_manager = JitCacheManager(cache_dir)
        self.cache_manager.load_all()

    def _resolve_and_make_cache_key(self, bound_args):
        """Resolve raw call values into JitArgument instances *in place* and
        build the tuple cache key from them.

        Side effect: entries in ``bound_args`` whose annotation is neither
        ``Constexpr[T]`` nor ``Type[T]`` are replaced with their resolved
        ``JitArgument`` instance (e.g. ``int`` → ``Int32``,

        * Annotation-driven (``Constexpr[T]`` / ``Type[T]``): value or type
          baked directly into the key, ``bound_args`` left untouched.
        * JitArgument-driven: the call value is (or is wrapped into) a
          ``JitArgument`` and its ``cache_signature()`` is appended.
        """
        from .jit_argument import JitArgumentRegistry

        sig = self._sig
        # Re-read env vars on every call.
        key_parts = [("_env_", _cache_invalidating_env_values()), ("_target_", self._backend_target)]
        if self.compile_hints:
            key_parts.append(("_hints_", tuple(sorted((k, str(v)) for k, v in self.compile_hints.items()))))

        for name, arg in bound_args.items():
            param = sig.parameters.get(name)
            ann = param.annotation if param else inspect.Parameter.empty

            if ann is not inspect.Parameter.empty:
                if Constexpr.is_constexpr_annotation(ann):
                    key_parts.append((name, Constexpr.value_signature(arg)))
                    continue
                if is_type_param_annotation(ann):
                    key_parts.append((name, arg))
                    continue

            if isinstance(arg, JitArgument):
                jit_arg = arg
            elif isinstance(ann, type) and issubclass(ann, JitArgument):
                jit_arg = ann(arg)
            else:
                ctor, _ = JitArgumentRegistry.get(type(arg))
                if ctor is None:
                    raise TypeError(
                        f"{name}: {type(arg).__name__} is neither a JitArgument nor has a registered "
                        f"constructor; cannot derive cache signature."
                    )
                jit_arg = ctor(arg)

            bound_args[name] = jit_arg
            key_parts.append((name, cache_signature(jit_arg)))

        return tuple(key_parts)

    def _globals_key_prefix(self, owner_cls=None) -> tuple:
        """Memoized ``("_globals_", ...)`` cache-key segment for ``owner_cls``.

        Built once from the first-call snapshot and reused on every cache hit.
        Safe to memoize: any change to a folded global is caught by the drift
        check, which raises before a stale segment could be used.

        The stable snapshot is folded in so two processes that observe different
        values for the same captured global cannot share an artifact —
        cross-process silent stale hits are prevented even though in-process drift
        detection (which raises) never runs in the second process.
        """
        cache = self._globals_prefix_cache
        if owner_cls not in cache:
            stable = _snapshot_refs(self._get_global_refs(owner_cls), stable=True)
            cache[owner_cls] = (("_globals_", tuple(sorted(stable.items()))),) if stable else ()
        return cache[owner_cls]

    def _build_full_cache_key(self, bound_arguments, *, owner_cls=None, bound_self=None):
        """Build the complete cache key: arg signatures + stable globals snapshot + self type."""
        cache_key = self._globals_key_prefix(owner_cls) + self._resolve_and_make_cache_key(bound_arguments)
        if bound_self is not None:
            cache_key = (("_self_type_", type(bound_self)),) + cache_key
        return cache_key

    @staticmethod
    def _cache_key_to_str(cache_key) -> str:
        """Convert tuple cache key to string for disk cache."""
        return str(cache_key)

    def __call__(self, *args, **kwargs):
        if ir.Context.current is not None:
            return self.func(*args, **kwargs)

        self._ensure_sig()

        bound_self = None
        if self._has_self_param:
            if not args:
                raise TypeError(f"{self.func.__name__}() missing 'self' argument")
            bound_self, args = args[0], args[1:]
        owner_cls = type(bound_self) if bound_self is not None else None
        self._ensure_cache_manager(owner_cls)

        # snapshot the used globals on first compile (per owner_cls) and RAISE on
        # any later change.
        if owner_cls not in self._used_global_vals:
            self._used_global_vals[owner_cls] = _snapshot_refs(self._get_global_refs(owner_cls), stable=False)
        else:
            self._check_globals_drift(owner_cls)

        sig = self._sig
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        cache_key = self._build_full_cache_key(bound.arguments, owner_cls=owner_cls, bound_self=bound_self)

        args_tuple = tuple(bound.arguments.values())

        # Compile/runtime pairing at JIT entry (not in CompiledArtifact / ExecutionEngine init).
        from ..runtime.device_runtime import ensure_compile_runtime_pairing_from_env

        ensure_compile_runtime_pairing_from_env(compile_backend_name())

        # Fast path: reuse pre-built CallState (no ctypes alloc, no DLPack)
        call_state = self._call_state_cache.get(cache_key)
        if call_state is not None:
            if env.compile.compile_only:
                return None
            return call_state(args_tuple)

        # Normal path: check in-process cache first, then optional disk cache.
        # In run_only mode the disk cache is read regardless of enable_cache, since
        # AOT-only execution treats the on-disk cache as the deployment artifact.
        run_only = env.runtime.run_only
        use_disk_cache = env.runtime.enable_cache or run_only
        allow_disk_cache = use_disk_cache and cache_key not in self._extern_linkage_keys
        _rejected_link_libs = False
        cached_func = self._mem_cache.get(cache_key)
        if cached_func is None and allow_disk_cache and not env.debug.dump_ir:
            str_key = self._cache_key_to_str(cache_key)
            cached_func = self.cache_manager.get(str_key) if self.cache_manager else None
            if cached_func is not None and getattr(cached_func, "_link_libs", None):
                _rejected_link_libs = True
                cached_func = None
            if cached_func is not None:
                self._mem_cache[cache_key] = cached_func

        if cached_func is not None:
            if env.compile.compile_only:
                return None
            # Build CallState via JitArgument registry (same dispatch as compile path)
            state = _build_call_state(
                sig,
                args_tuple,
                cached_func._get_func_exe(),
            )
            self._call_state_cache[cache_key] = state
            return state(args_tuple)

        if run_only:
            cdir = getattr(self.cache_manager, "cache_dir", None)
            cdir_exists = cdir.exists() if cdir is not None else False
            msg = (
                f"FLYDSL_RUNTIME_RUN_ONLY=1 but no usable AOT cache for "
                f"{self.func.__name__}: "
                f"manager_key={self.manager_key}, "
                f"cache_key={self._cache_key_to_str(cache_key)[:96]}..., "
                f"cache_dir={cdir} (exists={cdir_exists})"
            )
            if _rejected_link_libs:
                msg += (
                    "; note: a cached artifact was found but rejected because "
                    "it contains external link libraries (extern-linked kernels "
                    "are not cached to disk)"
                )
            raise RuntimeError(msg)

        _hints_ctx = CompilationContext.compile_hints(self.compile_hints) if self.compile_hints else nullcontext()

        compiled_func = None  # will be set inside lock or compile path

        # Determine whether to use compile_lock for cross-process safety.
        _use_compile_lock = use_disk_cache and self.cache_manager and not env.debug.dump_ir
        if _use_compile_lock:
            str_key = self._cache_key_to_str(cache_key)
            _compile_lock_ctx = self.cache_manager.compile_lock(str_key)
        else:
            _compile_lock_ctx = nullcontext((None, None))

        with _compile_lock_ctx as (_lock_result, _cache_writer):

            if _lock_result is not None and not getattr(_lock_result, "_link_libs", None):
                # Cache hit after waiting for another process to compile.
                compiled_func = _lock_result
                self._mem_cache[cache_key] = compiled_func
                self._last_compiled = (cache_key, compiled_func)
            else:
                with _create_mlir_context() as ctx, _hints_ctx:
                    param_names, jit_args, dsl_types, constexpr_values = convert_to_jit_arguments(sig, bound)
                    # Per-call value/annotation consistency check.
                    for pname, dsl_type in zip(param_names, dsl_types):
                        ann = sig.parameters[pname].annotation
                        if (
                            ann is not inspect.Parameter.empty
                            and isinstance(ann, type)
                            and not issubclass(dsl_type, ann)
                        ):
                            warn_annotation_value_mismatch(pname, ann, dsl_type, context="@jit")
                    has_user_stream = _ensure_stream_arg(jit_args)
                    ir_types = get_ir_types(jit_args)
                    loc = func_def_location(self.func, ctx)

                    log().info(f"jit_args={jit_args}")
                    log().info(f"dsl_types={dsl_types}")

                    module = ir.Module.create(loc=loc)
                    module.operation.attributes["gpu.container_module"] = ir.UnitAttr.get()

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
                                if not has_user_stream:
                                    comp_ctx.stream_arg = ir_args[-1]
                                user_jit_args = jit_args[: len(param_names)]
                                dsl_args = construct_from_ir_values(dsl_types, user_jit_args, ir_args)
                                log().info(f"dsl_args={dsl_args}")
                                named_args = dict(zip(param_names, dsl_args))
                                named_args.update(constexpr_values)
                                # Bound the call-site boundary at the jit body.
                                with tracing_context(self.func):
                                    if bound_self is not None:
                                        self.func(bound_self, **named_args)
                                    else:
                                        self.func(**named_args)
                                func.ReturnOp([])

                    original_ir = module.operation.get_asm(enable_debug_info=True)

                    # Extern-symbol integration is carried entirely via
                    # CompilationContext: each ExternFunction populates
                    # link_libs and post_load_processors at declaration time,
                    # so the JIT path depends on no framework-specific import.
                    link_libs = list(comp_ctx.link_libs) if comp_ctx.link_libs else None
                    post_load_processors = list(comp_ctx.post_load_processors)
                    extern_linked = bool(link_libs or post_load_processors)
                    if extern_linked and _use_external_binary_codegen():
                        raise RuntimeError(
                            "FLYDSL_COMPILE_LLVM_DIR external codegen does not support extern-linked kernels yet; "
                            "use embedded codegen for kernels that require #fly.explicit_module."
                        )
                    if extern_linked:
                        self._extern_linkage_keys.add(cache_key)
                        # Switch to explicit Python-side module loading so
                        # post_load_processors can receive hipModule_t handles.
                        # Also clear targets set at construction: the backend attach-target
                        # pass is the sole source when link_libs is used; duplicating
                        # targets can make the runtime pick an object without extern libs.
                        gpu_module.offloadingHandler = ir.Attribute.parse("#fly.explicit_module")
                        if "targets" in gpu_module.operation.attributes:
                            del gpu_module.operation.attributes["targets"]

                    compiled_module = MlirCompiler.compile(
                        module,
                        arch=backend.target.arch,
                        func_name=self.func.__name__,
                        link_libs=link_libs,
                    )

                    compiled_func = CompiledArtifact(
                        compiled_module,
                        self.func.__name__,
                        original_ir,
                        post_load_processors=post_load_processors,
                        link_libs=link_libs,
                        uses_explicit_module=extern_linked,
                    )

                    # Always keep a reference to the latest compilation result so
                    # flyc.compile() can retrieve it even when caching is disabled.
                    self._last_compiled = (cache_key, compiled_func)

                    # Keep compiled artifacts alive within the process even when disk
                    # cache is disabled. This preserves code object lifetime for
                    # profiler/roctracer teardown and enables fast same-process reuse.
                    self._mem_cache[cache_key] = compiled_func
                    if _cache_writer and not extern_linked:
                        try:
                            _cache_writer(compiled_func)
                        except Exception as e:
                            log().warning(f"Failed to write compile cache: {e}")

        # OUTSIDE lock: engine init + kernel launch
        if env.compile.compile_only:
            print(f"[flydsl] COMPILE_ONLY=1, compilation succeeded (arch={get_backend().target.arch})")
            return None

        # The in-process CompiledArtifact cache above owns the ExecutionEngine/
        # code object, so the function pointer remains valid even when disk
        # cache is off.
        state = _build_call_state(
            sig,
            args_tuple,
            compiled_func._get_func_exe(),
        )
        self._call_state_cache[cache_key] = state
        return state(args_tuple)


def _ensure_stream_arg(jit_args: list) -> bool:
    """Ensure jit_args contains a Stream argument.  If the user's function
    already declares ``stream: fx.Stream``, return True (user-supplied).
    Otherwise append a default ``Stream(None)`` and return False."""
    if any(isinstance(a, Stream) for a in jit_args):
        return True
    jit_args.append(Stream(None))
    return False


def jit(func: Optional[Callable] = None) -> JitFunction:
    """JIT decorator for host launcher functions."""
    if func is None:
        return lambda f: JitFunction(f)
    return JitFunction(func)


class CompiledFunction:
    """Pre-compiled callable returned by ``flyc.compile()``.

    All MLIR compilation, signature analysis, and argument metadata resolution
    happen once at ``compile()`` time.  The ``__call__`` hot path does only:

    1. Update pre-allocated ctypes storage (data_ptr / scalar extraction)
    2. Invoke the JIT'd C function pointer

    No ``inspect.Signature.bind``, no ``_resolve_and_make_cache_key``, no cache lookup.
    Accepts **positional arguments only** (same count and order as the
    original ``@flyc.jit`` function).
    """

    __slots__ = ("_call_state", "_keepalive")

    def __init__(self, call_state, keepalive):
        self._call_state = call_state
        self._keepalive = keepalive  # prevent GC of CompiledArtifact / ExecutionEngine

    def __call__(self, *args):
        return self._call_state(args)


def _compile_impl(func, *args) -> CompiledFunction:
    """Pre-compile a ``@flyc.jit`` function, returning a fast callable.

    Usage::

        compiled_fn = flyc.compile(launch_gemm, c, a, b, sa, sb, M, N, stream)

        # Hot loop — minimal dispatch overhead (~5 µs):
        for ...:
            compiled_fn(c, a, b, sa, sb, M, N, stream)

    All arguments (including ``stream``) must be **positional**.
    The returned :class:`CompiledFunction` also accepts only positional args.

    Constexpr values are baked in at compile time and ignored on subsequent
    calls; only runtime values (data pointers, scalars, stream) may change.
    """
    if not isinstance(func, JitFunction):
        raise TypeError(f"flyc.compile() expects a @flyc.jit function, got {type(func).__name__}")

    jf = func

    jf(*args)

    # Retrieve the CallState (already built by __call__ above).
    sig = jf._sig  # guaranteed initialized after __call__
    bound = sig.bind(*args)
    bound.apply_defaults()
    cache_key = jf._build_full_cache_key(bound.arguments)
    args_tuple = tuple(bound.arguments.values())

    # Look up the CompiledArtifact.  We must hold a direct reference to it
    # because it owns the ExecutionEngine and GPU code objects — if it gets
    # GC'd, the function pointer inside CallState becomes dangling.
    artifact = jf._mem_cache.get(cache_key)
    if artifact is None:
        # Cache disabled — retrieve from _last_compiled (set unconditionally
        # by __call__).  This does not alter __call__'s caching semantics.
        last = jf._last_compiled
        if last is not None and last[0] == cache_key:
            artifact = last[1]
    if artifact is None:
        raise RuntimeError("flyc.compile(): compilation succeeded but no cached artifact found.")

    call_state = jf._call_state_cache.get(cache_key)
    if call_state is None:
        call_state = _build_call_state(sig, args_tuple, artifact._get_func_exe())

    return CompiledFunction(call_state, artifact)


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

    def __call__(self, func, *args):
        if self._compile_hints and isinstance(func, JitFunction):
            func.compile_hints = {**func.compile_hints, **self._compile_hints}
        if not args:
            # No args → just return the (hinted) function for deferred compilation
            return func
        return _compile_impl(func, *args)


compile = CompileCallable()
