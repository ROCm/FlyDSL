# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import ctypes
import hashlib
import inspect
import os
import pickle
import pkgutil
import threading
import types
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from .._mlir import ir
from .._mlir.dialects import func
from .._mlir.passmanager import PassManager
from ..runtime.device import get_rocm_arch, is_rdna_arch
from ..expr.typing import Stream
from ..utils import env, log
from .ast_rewriter import ASTRewriter
from .jit_argument import convert_to_jit_arguments
from .jit_executor import CompiledArtifact
from .kernel_function import (
    CompilationContext,
    FuncLocationTracker,
    KernelFunction,
    create_gpu_module,
    get_gpu_module_body,
)
from .protocol import fly_types, fly_pointers, fly_construct



@lru_cache(maxsize=1)
def _flydsl_key() -> str:
    """Compute a hash fingerprint of the entire FlyDSL compiler toolchain.

    Covers:
      1. All Python source files under flydsl.compiler.*, flydsl.expr.*,
         flydsl.runtime.*, flydsl.lang.*, flydsl.utils.*
      2. Native shared libraries (_fly*.so, libFly*.so, libfly_jit_runtime.so,
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
        (str(flydsl_root / "lang"), "flydsl.lang."),
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

    # Also hash flydsl/__init__.py and _version.py.
    for name in ("__init__.py", "_version.py"):
        p = flydsl_root / name
        if p.is_file():
            with open(p, "rb") as f:
                contents.append(hashlib.sha256(f.read()).hexdigest())

    # 2) Hash native shared libraries (C++ passes, runtime wrappers, bindings).
    mlir_libs_dir = flydsl_root / "_mlir" / "_mlir_libs"
    if mlir_libs_dir.is_dir():
        so_patterns = [
            "_fly*.so",
            "_fly_rocdl*.so",
            "libFly*.so",
            "libfly_jit_runtime.so",
            "libmlir_rocm_runtime.so",
            "_mlirRegisterEverything*.so",
        ]
        for pattern in so_patterns:
            for so_file in sorted(mlir_libs_dir.glob(pattern)):
                h = hashlib.sha256()
                with open(so_file, "rb") as f:
                    while True:
                        chunk = f.read(1024 * 1024)
                        if not chunk:
                            break
                        h.update(chunk)
                contents.append(h.hexdigest())

    key = f"flydsl:{flydsl.__version__}-" + "-".join(contents)
    log().debug(f"flydsl_key: {hashlib.sha256(key.encode()).hexdigest()[:16]}")
    return key


def _get_underlying_func(obj):
    if isinstance(obj, KernelFunction):
        return obj._func
    if isinstance(obj, JitFunction):
        return obj.func
    if isinstance(obj, types.FunctionType):
        return obj
    return None


def _is_user_function(func, rootFile):
    try:
        funcFile = inspect.getfile(func)
    except (TypeError, OSError):
        return False
    return os.path.dirname(os.path.abspath(funcFile)) == os.path.dirname(os.path.abspath(rootFile))


def _collect_dependency_sources(func, rootFile, visited: Optional[Set[int]] = None) -> List[str]:
    if visited is None:
        visited = set()
    sources = []
    for name in func.__code__.co_names:
        obj = func.__globals__.get(name)
        underlying = _get_underlying_func(obj)
        if underlying is None or id(underlying) in visited:
            continue
        if not _is_user_function(underlying, rootFile):
            continue
        visited.add(id(underlying))
        try:
            src = inspect.getsource(underlying)
        except OSError:
            src = underlying.__code__.co_code.hex()
        sources.append(f"{name}:{src}")
        sources.extend(_collect_dependency_sources(underlying, rootFile, visited))
    return sources


def _jit_function_cache_key(func: Callable) -> str:
    parts = []
    parts.append(_flydsl_key())
    try:
        source = inspect.getsource(func)
    except OSError:
        source = func.__code__.co_code.hex()
    parts.append(source)
    try:
        rootFile = inspect.getfile(func)
    except (TypeError, OSError):
        rootFile = ""
    depSources = _collect_dependency_sources(func, rootFile)
    depSources.sort()
    parts.extend(depSources)

    if func.__code__.co_freevars and getattr(func, "__closure__", None):
        closure_vals = []
        for name, cell in zip(func.__code__.co_freevars, func.__closure__):
            try:
                val = cell.cell_contents
                if isinstance(val, (int, float, bool, str, type(None), tuple)):
                    closure_vals.append(f"{name}={val!r}")
            except ValueError:
                pass
        if closure_vals:
            parts.append("closure:" + ",".join(closure_vals))

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
        pm = PassManager.parse(
            "builtin.module(gpu-module-to-binary{format=isa opts= section= toolkit=})",
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


class MlirCompiler:
    @staticmethod
    def _pipeline_fragments(*, chip: str) -> list:
        wave64 = "false" if is_rdna_arch(chip) else "true"
        return [
            "gpu-kernel-outlining{data-layout-str=}",
            "fly-canonicalize",
            "fly-layout-lowering",
            "convert-fly-to-rocdl",
            "canonicalize",
            f"gpu.module(convert-scf-to-cf,cse,"
            f"convert-gpu-to-rocdl{{chipset={chip} index-bitwidth=0 runtime=HIP use-bare-ptr-memref-call-conv=true}})",
            f"rocdl-attach-target{{O=2 abi=600 chip={chip} correct-sqrt=true daz=false fast=false features= "
            f"finite-only=false module= triple=amdgcn-amd-amdhsa unsafe-math=false wave64={wave64}}}",
            "convert-scf-to-cf",
            "convert-cf-to-llvm",
            "gpu-to-llvm{use-bare-pointers-for-host=true use-bare-pointers-for-kernels=true}",
            "convert-arith-to-llvm",
            "convert-func-to-llvm",
            "reconcile-unrealized-casts",
            "gpu-module-to-binary{format=fatbin}",
        ]

    @classmethod
    def compile(cls, module: ir.Module, *, chip: str = None, func_name: str = "") -> ir.Module:
        module.operation.verify()

        if chip is None:
            chip = env.compile.arch or get_rocm_arch()

        module = ir.Module.parse(module.operation.get_asm(enable_debug_info=env.debug.enable_debug_info))
        fragments = cls._pipeline_fragments(chip=chip)

        if env.debug.print_origin_ir:
            log().info(f"Origin IR: \n{module}")

        dump_enabled = env.debug.dump_ir
        dump_dir = Path(env.debug.dump_dir).resolve()

        if dump_enabled:
            asm = module.operation.get_asm(enable_debug_info=True)
            kernel_names = _infer_kernel_names_from_asm(asm)
            subdir = kernel_names[0] if len(kernel_names) == 1 else (func_name or "module")
            dump_dir = dump_dir / _sanitize_path_component(subdir)
            print(f"[flydsl.compile] FLYDSL_DUMP_IR=1 dir={dump_dir}")

            out = _dump_ir("00_origin", dump_dir=dump_dir, asm=asm)
            print(f"[flydsl.compile] dump 00_origin -> {out}")

            asm_for_isa = None
            stage_num_base = 1
            for idx, frag in enumerate(fragments):
                stage_num = stage_num_base + idx
                stage_name = f"{stage_num:02d}_{_stage_label_from_fragment(frag)}"
                pm = PassManager.parse(f"builtin.module({frag})")
                pm.enable_verifier(env.debug.enable_verifier)
                pm.run(module.operation)

                stage_asm = module.operation.get_asm(enable_debug_info=True)
                out = _dump_ir(stage_name, dump_dir=dump_dir, asm=stage_asm)
                print(f"[flydsl.compile] dump {stage_name} -> {out}")

                if frag.strip() == "reconcile-unrealized-casts":
                    asm_for_isa = stage_asm

            if asm_for_isa is not None:
                isa_stage = f"{stage_num_base + len(fragments):02d}_final_isa"
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
            pipeline = f"builtin.module({','.join(fragments)})"
            pm = PassManager.parse(pipeline)
            pm.enable_verifier(env.debug.enable_verifier)
            pm.enable_ir_printing(print_after_all=env.debug.print_after_all)
            pm.run(module.operation)

        return module


class JitCacheManager:
    """Directory-based cache manager.

    Cache directory structure:
        {cache_root}/{func_name}_{manager_key}/
            {cache_key}.pkl  - serialized compiled kernel

    Each compiled kernel is saved immediately after compilation.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.memory_cache: Dict[str, Any] = {}

    def _cache_file(self, cache_key: str) -> Path:
        safe_key = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
        return self.cache_dir / f"{safe_key}.pkl"

    def get(self, cache_key: str) -> Optional[Any]:
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]

        cache_file = self._cache_file(cache_key)
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    value = pickle.load(f)
                self.memory_cache[cache_key] = value
                log().debug(f"Cache hit from disk: {cache_file.name}")
                return value
            except Exception as e:
                log().warning(f"Failed to load cache {cache_file}: {e}")
        return None

    def set(self, cache_key: str, value: Any) -> None:
        self.memory_cache[cache_key] = value
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self._cache_file(cache_key)
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(value, f)
            log().debug(f"Cache saved: {cache_file.name}")
        except Exception as e:
            log().warning(f"Failed to save cache {cache_file}: {e}")

    def load_all(self) -> int:
        if not self.cache_dir.exists():
            return 0
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, "rb") as f:
                    pickle.load(f)
                count += 1
            except Exception:
                pass
        log().debug(f"Found {count} cached entries in {self.cache_dir}")
        return count

    def __contains__(self, cache_key: str) -> bool:
        return cache_key in self.memory_cache or self._cache_file(cache_key).exists()


def _resolve_jit_arg_type(arg, annotation):
    """Resolve the JitArgument type for an argument, using the same dispatch
    logic as convert_to_jit_arguments.  Returns the type (not an instance)."""
    from .jit_argument import JitArgumentRegistry

    if isinstance(arg, int) and annotation is Stream:
        return Stream
    if hasattr(arg, '__fly_ptrs__'):
        return type(arg)
    constructor, _ = JitArgumentRegistry.get(type(arg))
    return constructor


def _build_call_state(sig, args_tuple, func_exe):
    """Build a CallState for fast repeated dispatch.

    Resolves each parameter's JitArgument type using the same registry as
    convert_to_jit_arguments, then asks it for a reusable slot specification.
    This ensures a single source of truth for argument packing.

    Returns a CallState, or None if any parameter can't be fast-pathed.
    """
    from .jit_argument import _is_constexpr_annotation, _is_type_param_annotation

    slot_specs = []
    has_user_stream = False

    for i, (param_name, param) in enumerate(sig.parameters.items()):
        annotation = param.annotation

        if annotation is not inspect.Parameter.empty and _is_constexpr_annotation(annotation):
            continue

        if annotation is not inspect.Parameter.empty and _is_type_param_annotation(annotation):
            continue

        if getattr(annotation, '_is_stream_param', False):
            has_user_stream = True

        arg = args_tuple[i]

        jit_arg_type = _resolve_jit_arg_type(arg, annotation)
        if jit_arg_type is None or not hasattr(jit_arg_type, '_reusable_slot_spec'):
            return None

        spec = jit_arg_type._reusable_slot_spec(arg)
        if spec is None:
            return None

        ctype, extract = spec

        try:
            extract(arg)
        except (AttributeError, TypeError):
            return None

        slot_specs.append((i, ctype, extract))

    # Auto-stream: append a zero-valued slot for the default (NULL) stream.
    # When no user-declared stream parameter exists, the compiled kernel
    # still expects a stream pointer as the last argument.  A NULL pointer
    # (value 0) selects the HIP default stream.
    if not has_user_stream:
        slot_specs.append((-1, ctypes.c_void_p, None))

    return CallState(slot_specs, func_exe)


class CallState:
    """Pre-allocated state for fast kernel dispatch.

    Built from JitArgument types' _reusable_slot_spec protocol — the same
    types used by convert_to_jit_arguments for the full DLPack path.

    Pre-allocates typed ctypes storage and a packed pointer array at init
    time.  On each call, only updates .value on existing storage objects —
    no ctypes object allocation, no pointer()/cast() calls.  Uses
    thread-local storage for thread safety.
    """

    __slots__ = ('_func_exe', '_spec', '_tls')

    def __init__(self, slot_specs, func_exe):
        self._func_exe = func_exe
        self._spec = slot_specs  # list of (arg_idx, ctype, extract_fn)
        self._tls = threading.local()

    def _init_buffers(self):
        n_ptrs = len(self._spec)
        packed = (ctypes.c_void_p * n_ptrs)()
        storages = []
        updaters = []

        for packed_idx, (arg_idx, ctype, extract) in enumerate(self._spec):
            s = ctype(0)
            packed[packed_idx] = ctypes.addressof(s)
            storages.append(s)
            if extract is not None:
                updaters.append((arg_idx, s, extract))

        self._tls.packed = packed
        self._tls.updaters = updaters
        self._tls._storages = storages

    def __call__(self, args_tuple):
        if not hasattr(self._tls, 'packed'):
            self._init_buffers()

        for arg_idx, storage, extract in self._tls.updaters:
            storage.value = extract(args_tuple[arg_idx])

        return self._func_exe(self._tls.packed)


class JitFunction:
    def __init__(self, func: Callable):
        self.func = ASTRewriter.transform(func)
        self.manager_key = None
        self.cache_manager = None
        self._call_state_cache = {}  # cache_key -> CallState
        self._sig = None  # lazy: set on first call
        self._mem_cache = {}

    def _ensure_sig(self):
        """Initialize signature + param metadata on first call (not at decoration time)."""
        if self._sig is not None:
            return
        self._sig = inspect.signature(self.func)

    def _ensure_cache_manager(self):
        if self.manager_key is not None:
            return
        self.manager_key = _jit_function_cache_key(self.func)
        if not env.runtime.enable_cache:
            return
        cache_root = env.runtime.cache_dir
        if cache_root:
            cache_dir = Path(cache_root) / f"{self.func.__name__}_{self.manager_key}"
            self.cache_manager = JitCacheManager(cache_dir)
            self.cache_manager.load_all()

    @staticmethod
    def _arg_cache_sig(arg, *, runtime=False):
        """Cache signature for a single argument value.

        When runtime=True the parameter's annotation indicates a runtime
        type (has __fly_ptrs__, e.g. Int32, Stream) — value is passed to
        the kernel at launch time and does NOT affect compiled code, so
        only the Python type is recorded.

        Dispatch order:
        1. __cache_signature__ — types that explicitly declare their cache
           identity (e.g. TensorAdaptor includes assumed_align).
        2. Python scalars — value in key when not runtime; type only when
           runtime (annotated as Int32/Stream/etc.).
        3. Tensor-like (dtype+shape) — fallback for bare torch.Tensor.
        4. Everything else — type only.
        """
        if hasattr(arg, "__cache_signature__"):
            return arg.__cache_signature__()
        elif isinstance(arg, (int, float, bool, str)):
            if runtime:
                return type(arg)
            return (type(arg), arg)
        elif hasattr(arg, "dtype") and hasattr(arg, "shape"):
            strides = tuple(arg.stride()) if hasattr(arg, "stride") else ()
            return (arg.dtype, tuple(arg.shape), strides)
        else:
            return type(arg)

    def _make_cache_key(self, bound_args):
        """Build a tuple cache key from bound arguments.

        For parameters annotated with a runtime type (one that implements
        __fly_ptrs__, e.g. Int32, Stream), the value is excluded from the
        key — only the Python type matters.  This prevents unnecessary
        recompilation when only runtime values change.
        """
        from .jit_argument import _is_constexpr_annotation
        sig = self._sig
        key_parts = []
        for name, arg in bound_args.items():
            param = sig.parameters.get(name)
            ann = param.annotation if param else inspect.Parameter.empty

            if ann is not inspect.Parameter.empty and _is_constexpr_annotation(ann):
                key_parts.append((name, type(arg), arg))
                continue

            is_runtime = (ann is not inspect.Parameter.empty
                          and hasattr(ann, '__fly_ptrs__'))
            if isinstance(arg, tuple):
                key_parts.append((name, tuple(
                    self._arg_cache_sig(a) for a in arg)))
            else:
                key_parts.append((name, self._arg_cache_sig(arg, runtime=is_runtime)))
        return tuple(key_parts)

    @staticmethod
    def _cache_key_to_str(cache_key) -> str:
        """Convert tuple cache key to string for disk cache."""
        return str(cache_key)

    def __call__(self, *args, **kwargs):
        if ir.Context.current is not None:
            return self.func(*args, **kwargs)

        self._ensure_sig()
        self._ensure_cache_manager()

        sig = self._sig
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        cache_key = self._make_cache_key(bound.arguments)

        args_tuple = tuple(bound.arguments.values())

        # Fast path: reuse pre-built CallState (no ctypes alloc, no DLPack)
        call_state = self._call_state_cache.get(cache_key)
        if call_state is not None:
            return call_state(args_tuple)

        # Normal path: check caches
        cached_func = None
        use_cache = env.runtime.enable_cache
        if use_cache:
            cached_func = self._mem_cache.get(cache_key)
            if cached_func is None and not env.debug.dump_ir:
                str_key = self._cache_key_to_str(cache_key)
                cached_func = self.cache_manager.get(str_key) if self.cache_manager else None
                if cached_func is not None:
                    self._mem_cache[cache_key] = cached_func

        if cached_func is not None:
            # Build CallState via JitArgument registry (same dispatch as compile path)
            try:
                state = _build_call_state(
                    sig, args_tuple, cached_func._get_func_exe(),
                )
            except Exception:
                state = None
            if state is not None:
                self._call_state_cache[cache_key] = state
                return state(args_tuple)

            # Fallback: run through DLPack (should not happen for static layout)
            log().warning("CallState build failed on cache hit, falling back to DLPack path")
            if not hasattr(self, "_cached_ctx"):
                self._cached_ctx = ir.Context()
                self._cached_ctx.load_all_available_dialects()
            with self._cached_ctx:
                _, jit_args, _, _ = convert_to_jit_arguments(sig, bound)
                _ensure_stream_arg(jit_args)
                return cached_func(*jit_args)

        with ir.Context() as ctx:
            param_names, jit_args, dsl_types, constexpr_values = convert_to_jit_arguments(sig, bound)
            has_user_stream = _ensure_stream_arg(jit_args)
            ir_types = fly_types(jit_args)
            loc = ir.Location.unknown(ctx)

            log().info(f"jit_args={jit_args}")
            log().info(f"dsl_types={dsl_types}")

            module = ir.Module.create(loc=loc)
            module.operation.attributes["gpu.container_module"] = ir.UnitAttr.get()

            func_tracker = FuncLocationTracker(self.func)

            with ir.InsertionPoint(module.body), loc:
                chip = env.compile.arch or get_rocm_arch()
                gpu_module = create_gpu_module("kernels", targets=[f'#rocdl.target<chip = "{chip}">'])

                func_op = func.FuncOp(self.func.__name__, (ir_types, []))
                func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
                entry_block = func_op.add_entry_block()

                with CompilationContext.create(func_tracker) as comp_ctx:
                    comp_ctx.gpu_module_op = gpu_module
                    comp_ctx.gpu_module_body = get_gpu_module_body(gpu_module)

                    with ir.InsertionPoint(entry_block):
                        ir_args = list(func_op.regions[0].blocks[0].arguments)
                        if not has_user_stream:
                            comp_ctx.stream_arg = ir_args[-1]
                        user_jit_args = jit_args[: len(param_names)]
                        dsl_args = fly_construct(dsl_types, user_jit_args, ir_args)
                        log().info(f"dsl_args={dsl_args}")
                        named_args = dict(zip(param_names, dsl_args))
                        named_args.update(constexpr_values)
                        self.func(**named_args)
                        func.ReturnOp([])

            original_ir = module.operation.get_asm(enable_debug_info=True)

            compiled_module = MlirCompiler.compile(module, chip=chip, func_name=self.func.__name__)

            if env.compile.compile_only:
                print(f"[flydsl] COMPILE_ONLY=1, compilation succeeded (arch={chip})")
                return None

            compiled_func = CompiledArtifact(
                compiled_module,
                self.func.__name__,
                original_ir,
            )

            if use_cache:
                self._mem_cache[cache_key] = compiled_func
                if self.cache_manager and not env.debug.dump_ir:
                    str_key = self._cache_key_to_str(cache_key)
                    self.cache_manager.set(str_key, compiled_func)

            result = compiled_func(*jit_args)

            # Build CallState so subsequent calls skip DLPack
            try:
                state = _build_call_state(
                    sig, args_tuple,
                    compiled_func._get_func_exe(),
                )
                if state is not None:
                    self._call_state_cache[cache_key] = state
            except Exception:
                pass

            return result


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
