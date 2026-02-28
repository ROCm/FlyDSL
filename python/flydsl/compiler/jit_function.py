import hashlib
import inspect
import os
import pickle
import types
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import flydsl

from .._mlir import ir
from .._mlir.dialects import func
from .._mlir.passmanager import PassManager
from ..utils import env, log
from .ast_rewriter import ASTRewriter
from .jit_argument import convert_to_jit_arguments
from .jit_executor import JitCompiledFunction
from .kernel_function import (
    CompilationContext,
    FuncLocationTracker,
    KernelFunction,
    create_gpu_module,
    get_gpu_module_body,
)
from .protocol import get_ir_types, new_from_ir_values
from flydsl.runtime.device import get_rocm_arch

@lru_cache(maxsize=1)
def _get_llvm_version() -> str:
    _FLYDSL_ROOT = Path(__file__).resolve().parents[4]
    llvm_hash_file = _FLYDSL_ROOT / "cmake" / "llvm-hash.txt"
    if llvm_hash_file.exists():
        llvm_hash = llvm_hash_file.read_text()
    else:
        llvm_hash = "release_version"
    log().debug(f"LLVM version: {llvm_hash}")
    return llvm_hash


@lru_cache(maxsize=1)
def _flydsl_verison_key() -> str:
    return f"flydsl:{flydsl.__version__}|llvm:{_get_llvm_version()}"


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
    parts.append(_flydsl_verison_key())
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
    """Extract pass name from a pipeline fragment for use as a filename."""
    base = fragment.strip()
    if base.startswith("gpu.module(") and base.endswith(")"):
        base = base[len("gpu.module(") : -1].strip()
    return base.split("{", 1)[0].strip()


def _dump_ir_to_file(stage: str, *, dump_dir: Path, asm: str) -> Path:
    dump_dir.mkdir(parents=True, exist_ok=True)
    out = dump_dir / f"{stage}.mlir"
    out.write_text(asm, encoding="utf-8")
    return out


def _dump_isa(*, dump_dir: Path, asm: str, verify: bool) -> Optional[Path]:
    """Best-effort dump of final ISA assembly (.s) from the post-reconcile IR."""
    try:
        from .._mlir.dialects import gpu
        mod = ir.Module.parse(asm)
        pm = PassManager.parse(
            "builtin.module(gpu-module-to-binary{format=isa opts= section= toolkit=})"
        )
        pm.enable_verifier(verify)
        pm.run(mod.operation)
        for op in mod.body:
            if isinstance(op, gpu.BinaryOp):
                objects = list(map(gpu.ObjectAttr, op.objects))
                isa_bytes = objects[-1].object
                out = dump_dir / "final_isa.s"
                out.write_bytes(isa_bytes)
                return out
    except Exception:
        pass
    return None


class MlirCompiler:
    @staticmethod
    def _pipeline_fragments(*, chip: str) -> list:
        return [
            "gpu-kernel-outlining{data-layout-str=}",
            "fly-canonicalize",
            "fly-layout-lowering",
            "convert-fly-to-rocdl",
            "canonicalize",
            f"gpu.module(convert-scf-to-cf,"
            f"convert-gpu-to-rocdl{{chipset={chip} index-bitwidth=0 runtime=HIP use-bare-ptr-memref-call-conv=true}})",
            f"rocdl-attach-target{{O=2 abi=600 chip={chip} correct-sqrt=true daz=false fast=false features= "
            f"finite-only=false module= triple=amdgcn-amd-amdhsa unsafe-math=false wave64=true}}",
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
        if env.debug.dump_ir:
            dump_dir = Path(env.debug.dump_dir)
            if func_name:
                dump_dir = dump_dir / func_name

            out = _dump_ir_to_file("00_original", dump_dir=dump_dir, asm=str(module))
            print(f"[flydsl] DUMP_IR enabled, dir={dump_dir}")
            print(f"[flydsl] dump 00_original -> {out}")

            asm_for_isa = None
            isa_stage_idx = len(fragments) - 1
            for idx, frag in enumerate(fragments, start=1):
                stage_name = f"{idx:02d}_{_stage_label_from_fragment(frag)}"
                pm = PassManager.parse(f"builtin.module({frag})")
                pm.enable_verifier(env.debug.enable_verifier)
                pm.run(module.operation)
                stage_asm = module.operation.get_asm(enable_debug_info=True)
                out = _dump_ir_to_file(stage_name, dump_dir=dump_dir, asm=stage_asm)
                print(f"[flydsl] dump {stage_name} -> {out}")
                if idx == isa_stage_idx:
                    asm_for_isa = stage_asm

            if asm_for_isa is not None:
                isa_out = _dump_isa(
                    dump_dir=dump_dir,
                    asm=asm_for_isa,
                    verify=env.debug.enable_verifier,
                )
                if isa_out is not None:
                    print(f"[flydsl] dump final_isa -> {isa_out}")
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


class JitFunction:
    def __init__(self, func: Callable):
        self.func = ASTRewriter.transform(func)
        self.manager_key = None
        self.cache_manager = None

    def _ensure_cache_manager(self):
        if self.manager_key is not None:
            return
        self.manager_key = _jit_function_cache_key(self.func)
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

        if not hasattr(self, '_sig'):
            self._sig = inspect.signature(self.func)
        sig = self._sig
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        cache_key = self._make_cache_key(bound.arguments)

        if not hasattr(self, '_mem_cache'):
            self._mem_cache = {}
        cached_func = self._mem_cache.get(cache_key)
        if cached_func is None and not env.debug.dump_ir:
            cached_func = self.cache_manager.get(cache_key) if self.cache_manager else None

        if cached_func is not None:
            if not hasattr(self, '_cached_ctx'):
                self._cached_ctx = ir.Context()
                self._cached_ctx.load_all_available_dialects()
            with self._cached_ctx:
                _, jit_args, _, _ = convert_to_jit_arguments(sig, bound)
                return cached_func(*jit_args)

        with ir.Context() as ctx:
            param_names, jit_args, dsl_types, constexpr_values = convert_to_jit_arguments(sig, bound)
            ir_types = get_ir_types(jit_args)
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
                        dsl_args = new_from_ir_values(dsl_types, jit_args, ir_args)
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

            compiled_func = JitCompiledFunction(
                compiled_module,
                self.func.__name__,
                original_ir,
            )

            self._mem_cache[cache_key] = compiled_func
            if self.cache_manager:
                self.cache_manager.set(cache_key, compiled_func)

            return compiled_func(*jit_args)


def jit(func: Optional[Callable] = None) -> JitFunction:
    """JIT decorator for host launcher functions."""
    if func is None:
        return lambda f: JitFunction(f)
    return JitFunction(func)
