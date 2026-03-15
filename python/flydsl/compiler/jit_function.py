import ctypes
import hashlib
import inspect
import os
import pickle
import pkgutil
import types
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import torch

from .._mlir import ir
from .._mlir.dialects import func
from .._mlir.passmanager import PassManager
from ..runtime.device import get_rocm_arch
from ..utils import env, log
from .ast_rewriter import ASTRewriter
from ..expr.typing import Constexpr, Stream
from .jit_argument import convert_to_jit_arguments, TensorAdaptor
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
        so_patterns = ["_fly*.so", "_fly_rocdl*.so", "libFly*.so",
                       "libfly_jit_runtime.so", "libmlir_rocm_runtime.so",
                       "_mlirRegisterEverything*.so"]
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
        base = base[len("gpu.module("):-1].strip()
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
        if ch == '\\' and i + 1 < len(mlir_asm):
            nxt = mlir_asm[i + 1]
            if nxt == '\\':
                chars.append('\\')
                i += 2
                continue
            if nxt == '"':
                chars.append('"')
                i += 2
                continue
            # MLIR hex escape: \XX
            if i + 3 <= len(mlir_asm):
                hex_str = mlir_asm[i + 1:i + 3]
                try:
                    chars.append(chr(int(hex_str, 16)))
                    i += 3
                    continue
                except ValueError:
                    pass
        chars.append(ch)
        i += 1

    return ''.join(chars)


def _dump_isa(*, dump_dir: Path, ctx: ir.Context, asm: str,
              verify: bool, stage_name: str = "15_final_isa"):
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
        return [
            "gpu-kernel-outlining{data-layout-str=}",
            "fly-canonicalize",
            "fly-layout-lowering",
            "convert-fly-to-rocdl",
            "canonicalize",
            f"gpu.module(convert-scf-to-cf,cse,"
            f"convert-gpu-to-rocdl{{chipset={chip} index-bitwidth=0 runtime=HIP use-bare-ptr-memref-call-conv=true}})",
            f"rocdl-attach-target{{O=2 abi=600 chip={chip} correct-sqrt=true daz=false fast=false features= "
            f"finite-only=false module= triple=amdgcn-amd-amdhsa unsafe-math=false wave64=true}}",
            "convert-scf-to-cf",
            "convert-cf-to-llvm",
            "fly-gpu-to-llvm{use-bare-pointers-for-host=true use-bare-pointers-for-kernels=true}",
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


def _build_fast_path_descriptor(sig, bound, jit_args, has_user_stream):
    """Build a descriptor for fast pointer extraction on subsequent cache hits.

    Returns a list of (param_name_or_None, kind, n_ptrs) tuples where kind is:
      'tensor'   - torch.Tensor → data_ptr
      'int'      - int → c_int32
      'stream'   - Stream → cuda_stream
      'skip'     - Constexpr/Type param, not passed to JIT
      'auto_stream' - auto-appended Stream(None)

    n_ptrs is how many C pointers this arg contributes (determined from jit_args).
    """
    from .jit_argument import _is_constexpr_annotation, _is_type_param_annotation

    descriptor = []
    jit_idx = 0

    for param_name, value in bound.arguments.items():
        param = sig.parameters[param_name]
        annotation = param.annotation

        if annotation is not inspect.Parameter.empty and _is_constexpr_annotation(annotation):
            descriptor.append((param_name, 'skip', 0))
            continue

        if annotation is not inspect.Parameter.empty and _is_type_param_annotation(annotation):
            descriptor.append((param_name, 'skip', 0))
            continue

        jit_arg = jit_args[jit_idx]
        ptrs = fly_pointers(jit_arg)
        n_ptrs = len(ptrs)

        if isinstance(value, torch.Tensor):
            descriptor.append((param_name, 'tensor', n_ptrs))
        elif isinstance(value, int) and annotation is not Stream:
            descriptor.append((param_name, 'int', n_ptrs))
        elif isinstance(value, Stream) or (isinstance(value, int) and annotation is Stream):
            descriptor.append((param_name, 'stream', n_ptrs))
        elif isinstance(value, torch.cuda.streams.Stream):
            descriptor.append((param_name, 'cuda_stream', n_ptrs))
        elif isinstance(value, TensorAdaptor):
            descriptor.append((param_name, 'tensor_adaptor', n_ptrs))
        else:
            # Unknown type — can't fast-path
            return None

        jit_idx += 1

    if not has_user_stream:
        # Auto-appended Stream(None) at the end
        stream_arg = jit_args[jit_idx]
        ptrs = fly_pointers(stream_arg)
        descriptor.append((None, 'auto_stream', len(ptrs)))

    return descriptor


def _fast_extract_ptrs(descriptor, bound_arguments):
    """Extract C pointers directly from raw Python args using the descriptor.

    Returns a list of ctypes.c_void_p, or None if extraction fails.
    The _storage list keeps Python ctypes objects alive until the JIT call completes.
    """
    c_ptrs = []
    _storage = []

    for param_name, kind, n_ptrs in descriptor:
        if kind == 'skip':
            continue
        elif kind == 'tensor':
            tensor = bound_arguments[param_name]
            # For static-layout tensors (n_ptrs == 1), we just need data_ptr
            # For dynamic-layout tensors (n_ptrs == 2), we can't skip DLTensorAdaptor
            if n_ptrs != 1:
                return None, None
            ptr_storage = ctypes.c_void_p(tensor.data_ptr())
            _storage.append(ptr_storage)
            c_ptrs.append(ctypes.cast(ctypes.pointer(ptr_storage), ctypes.c_void_p))
        elif kind == 'int':
            value = bound_arguments[param_name]
            c_val = ctypes.c_int32(value)
            _storage.append(c_val)
            c_ptrs.append(ctypes.cast(ctypes.pointer(c_val), ctypes.c_void_p))
        elif kind == 'stream':
            value = bound_arguments[param_name]
            if isinstance(value, Stream):
                raw = value.value
            else:
                raw = value
            if isinstance(raw, int):
                ptr_storage = ctypes.c_void_p(raw)
            elif raw is None:
                ptr_storage = ctypes.c_void_p(0)
            else:
                ptr_storage = ctypes.c_void_p(raw.cuda_stream)
            _storage.append(ptr_storage)
            c_ptrs.append(ctypes.cast(ctypes.pointer(ptr_storage), ctypes.c_void_p))
        elif kind == 'cuda_stream':
            stream_obj = bound_arguments[param_name]
            ptr_storage = ctypes.c_void_p(stream_obj.cuda_stream)
            _storage.append(ptr_storage)
            c_ptrs.append(ctypes.cast(ctypes.pointer(ptr_storage), ctypes.c_void_p))
        elif kind == 'tensor_adaptor':
            # TensorAdaptor passed directly — can't skip DLPack
            return None, None
        elif kind == 'auto_stream':
            ptr_storage = ctypes.c_void_p(0)
            _storage.append(ptr_storage)
            c_ptrs.append(ctypes.cast(ctypes.pointer(ptr_storage), ctypes.c_void_p))
        else:
            return None, None

    return c_ptrs, _storage


class JitFunction:
    def __init__(self, func: Callable):
        self.func = ASTRewriter.transform(func)
        self.manager_key = None
        self.cache_manager = None
        self._fast_path_cache = {}  # cache_key -> descriptor

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

    def _make_cache_key(self, bound_args: Dict) -> str:
        key_parts = []
        for name, arg in bound_args.items():
            key_parts.append(f"{name}:{self._get_type_signature(arg)}")
        return "|".join(key_parts)

    def _get_type_signature(self, obj) -> str:
        if hasattr(obj, "__cache_signature__"):
            return obj.__cache_signature__()
        elif hasattr(obj, "dtype") and hasattr(obj, "shape"):
            return f"tensor[{obj.dtype},{obj.shape}]"
        elif isinstance(obj, (int, float, bool, str)):
            return f"{type(obj).__name__}:{obj}"
        return type(obj).__name__

    def __call__(self, *args, **kwargs):
        if ir.Context.current is not None:
            return self.func(*args, **kwargs)

        self._ensure_cache_manager()

        if not hasattr(self, "_sig"):
            self._sig = inspect.signature(self.func)
        sig = self._sig
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        cache_key = self._make_cache_key(bound.arguments)

        cached_func = None
        dump_ir = env.debug.dump_ir
        use_cache = env.runtime.enable_cache
        if not hasattr(self, "_mem_cache"):
            self._mem_cache = {}
        if use_cache:
            compiled_func = self._mem_cache.get(cache_key)
            if compiled_func is not None:
                cached_func = compiled_func
            elif not dump_ir:
                cached_func = self.cache_manager.get(cache_key) if self.cache_manager else None

        if cached_func is not None:
            # Fast path: if we have a descriptor for this cache_key,
            # extract pointers directly without convert_to_jit_arguments
            descriptor = self._fast_path_cache.get(cache_key)
            if descriptor is not None:
                c_ptrs, _storage = _fast_extract_ptrs(descriptor, bound.arguments)
                if c_ptrs is not None:
                    return cached_func.fast_call(c_ptrs)

            # First cache hit or fast path failed: use normal path
            # and record descriptor for next time
            if not hasattr(self, "_cached_ctx"):
                self._cached_ctx = ir.Context()
                self._cached_ctx.load_all_available_dialects()
            with self._cached_ctx:
                _, jit_args, _, _ = convert_to_jit_arguments(sig, bound)
                has_user_stream = _ensure_stream_arg(jit_args)
                # Record descriptor for future fast path
                if descriptor is None:
                    try:
                        desc = _build_fast_path_descriptor(
                            sig, bound, jit_args, has_user_stream
                        )
                        if desc is not None:
                            self._fast_path_cache[cache_key] = desc
                    except Exception:
                        pass  # Silently fall back to normal path
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
                        user_jit_args = jit_args[:len(param_names)]
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
                if self.cache_manager and not dump_ir:
                    self.cache_manager.set(cache_key, compiled_func)

            return compiled_func(*jit_args)


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
