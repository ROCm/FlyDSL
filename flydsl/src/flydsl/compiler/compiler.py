"""High-level compilation entrypoint for FLIR modules."""

from __future__ import annotations

import contextlib
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Sequence, Union

from _mlir import ir
from _mlir.passmanager import PassManager

from flydsl.compiler.context import ensure_flir_python_extensions
from flydsl.runtime.device import get_rocm_arch

from .executor import default_shared_libs
from .cache import FileCache, cache_enabled, cache_rebuild_requested, default_key_payload, make_cache_key

if TYPE_CHECKING:
    from .executor import ExecutionEngineExecutor as Executor


@dataclass(frozen=True)
class CompileOptions:
    verify: bool = True
    print_final_module: bool = False
    opt_level: int = 3
    shared_libs: Optional[Sequence[str]] = None
    backend: Literal["execution_engine"] = "execution_engine"


def _pipeline_fragments(
    *,
    chip: str,
    use_bare_ptr_memref_call_conv: bool = False,
    use_bare_pointers_for_host: bool = False,
    use_bare_pointers_for_kernels: bool = False,
) -> list[str]:
    """FLIR compilation pipeline fragments as a plain list of strings.

    Each entry is the content inside `builtin.module(...)` for PassManager.parse.

    NOTE: We intentionally avoid running CSE/DCE *before* `flir-to-standard`.
    On some kernels this changes IR structure enough that later codegen/scheduling
    produces different ISA with measurable perf regressions.
    """
    b2s = lambda b: "true" if bool(b) else "false"
    rocdl_bare_ptr_opt = b2s(use_bare_ptr_memref_call_conv)
    llvm_bare_host_opt = b2s(use_bare_pointers_for_host)
    llvm_bare_kern_opt = b2s(use_bare_pointers_for_kernels)
    return [
        "flir-to-standard",
        "trivial-dce",
        "canonicalize",
        "cse",
        "gpu-kernel-outlining{data-layout-str=}",
        "gpu.module(convert-scf-to-cf)",
        "gpu.module(convert-gpu-to-rocdl{chipset=gfx000 index-bitwidth=0 runtime=unknown "
        + f"use-bare-ptr-memref-call-conv={rocdl_bare_ptr_opt}"
        + "})",
        "gpu.module(reconcile-unrealized-casts)",
        # Keep this as a formatted string so the chip is visible in dumps and matches
        # the non-dump compilation pipeline.
        f"rocdl-attach-target{{O=2 abi=600 chip={chip} correct-sqrt=true daz=false fast=false features= finite-only=false module= triple=amdgcn-amd-amdhsa unsafe-math=false wave64=true}}",
        "gpu-to-llvm{intersperse-sizes-for-kernels=false "
        + f"use-bare-pointers-for-host={llvm_bare_host_opt} "
        + f"use-bare-pointers-for-kernels={llvm_bare_kern_opt}"
        + "}",
        "reconcile-unrealized-casts",
        "gpu-module-to-binary{format=fatbin opts= section= toolkit=}",
    ]


def _stage_label_from_fragment(fragment: str) -> str:
    """Make a stable, filename-friendly label from a pipeline fragment."""
    base = fragment.strip()
    # Prefer the "inner" pass name for gpu.module(...) wrappers.
    if base.startswith("gpu.module(") and base.endswith(")"):
        base = base[len("gpu.module(") : -1].strip()
    # Strip pass options to keep labels stable.
    base = base.split("{", 1)[0].strip()
    # Replace non-alphanumerics with underscores.
    base = re.sub(r"[^0-9A-Za-z]+", "_", base).strip("_").lower()
    return base or "stage"


def _build_pipeline_str(
    *,
    chip: str,
    use_bare_ptr_memref_call_conv: bool = False,
    use_bare_pointers_for_host: bool = False,
    use_bare_pointers_for_kernels: bool = False,
) -> str:
    """Build the full PassManager pipeline string from `_pipeline_fragments`."""
    frags = _pipeline_fragments(
        chip=chip,
        use_bare_ptr_memref_call_conv=use_bare_ptr_memref_call_conv,
        use_bare_pointers_for_host=use_bare_pointers_for_host,
        use_bare_pointers_for_kernels=use_bare_pointers_for_kernels,
    )
    return f"builtin.module({','.join(frags)})"


def _override_gpu_module_targets(module: ir.Module, *, chip: str) -> None:
    """Force all `gpu.module` targets to a consistent ROCm target.

    Some tests/modules set `gpu.module [...]` targets (e.g. `abi=500`) which can
    produce code objects that fail to load on newer GPUs. `flir.compile()`
    owns the target selection.
    """
    ctx = module.context
    # `#rocdl.target` attribute syntax in this MLIR build only reliably carries `chip`
    # (ABI defaults are implicit). More detailed lowering config is handled by the
    # `rocdl-attach-target{...}` pass options in the pipeline.
    target = ir.Attribute.parse(f'#rocdl.target<chip = "{chip}">', context=ctx)
    targets = ir.ArrayAttr.get([target], context=ctx)

    def _cb(op):
        if op.name == "gpu.module":
            op.attributes["targets"] = targets
        return ir.WalkResult.ADVANCE

    module.operation.walk(_cb)


def _env_truthy(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default)
    return str(v).strip().lower() not in {"", "0", "false", "no", "off"}


def _dump_ir(stage: str, *, dump_dir: Path, asm: str) -> Path:
    dump_dir.mkdir(parents=True, exist_ok=True)
    out = dump_dir / f"{stage}.mlir"
    out.write_text(asm, encoding="utf-8")
    return out


def _dump_isa_from_rocdl_module_asm(*, dump_dir: Path, ctx: ir.Context, asm: str, verify: bool) -> Optional[Path]:
    """Best-effort dump final ISA/assembly (.s) for the current GPU module.

    This is only used for debug dumps. It intentionally does not affect the main
    compilation pipeline.
    """
    try:
        from flydsl.dialects.ext.gpu import get_compile_object_bytes
    except Exception:
        return None

    try:
        # Parse a fresh clone so we don't mutate the main compilation module.
        mod = ir.Module.parse(asm, context=ctx)
        pm = PassManager.parse("builtin.module(gpu-module-to-binary{format=isa opts= section= toolkit=})", context=ctx)
        pm.enable_verifier(bool(verify))
        pm.run(mod.operation)
        isa_bytes = get_compile_object_bytes(mod)
        out = dump_dir / "15_final_isa.s"
        out.write_bytes(isa_bytes)
        return out
    except Exception:
        return None


def _sanitize_path_component(s: str) -> str:
    # Keep it human-readable but filesystem-safe.
    s = str(s).strip()
    if not s:
        return "unknown"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _infer_kernel_names_from_asm(asm: str) -> list[str]:
    # MLIR assembly prints GPU kernels as:
    #   gpu.func @kernel_name(...) kernel {
    #
    # The argument list can include `loc(...)` which contains parentheses, so avoid
    # regex matching the full `(...)` range; just parse line-wise.
    names: list[str] = []
    for line in asm.splitlines():
        if "gpu.func @" not in line:
            continue
        if " kernel" not in line:
            continue
        try:
            after = line.split("gpu.func @", 1)[1]
        except Exception:
            continue
        name = after.split("(", 1)[0].strip()
        if name:
            names.append(name)
    return names


def compile(
    flir_module_or_ir: Union[object, ir.Module],
    *,
    verify: bool = True,
    print_final_module: bool = False,
    opt_level: int = 3,
    shared_libs: Optional[Sequence[str]] = None,
    backend: Literal["execution_engine"] = "execution_engine",
    use_bare_ptr_memref_call_conv: bool = False,
    use_bare_pointers_for_host: bool = False,
    use_bare_pointers_for_kernels: bool = False,
) -> Executor:
    """Compile a FLIR module to an Executor.

    Returns an MLIR ExecutionEngine-backed executor.
    """

    # Accept `flir.lang.MlirModule` instances.
    mlir_module = getattr(flir_module_or_ir, "module", None)
    if mlir_module is None:
        mlir_module = flir_module_or_ir
    if not isinstance(mlir_module, ir.Module):
        raise TypeError(f"Expected an MLIR module or flir.lang.MlirModule; got {type(flir_module_or_ir)}")

    ctx = mlir_module.context
    ensure_flir_python_extensions(ctx)

    dump_enabled = _env_truthy("FLIR_DUMP_IR", "0")
    dump_root_dir = Path(os.environ.get("FLIR_DUMP_DIR", "my_ir_dumps")).resolve()
    dump_prefix_base = (
        getattr(flir_module_or_ir, "GPU_MODULE_NAME", None)
        or getattr(flir_module_or_ir, "__name__", None)
        or getattr(getattr(flir_module_or_ir, "__class__", None), "__name__", None)
        or "module"
    )

    # Parse a fresh module for compilation so callers can print/reuse their builder module.
    with ctx:
        asm = mlir_module.operation.get_asm(enable_debug_info=True)
        dump_dir = dump_root_dir
        if dump_enabled:
            kernel_names = _infer_kernel_names_from_asm(asm)
            # If there's exactly one gpu kernel in the module, use it for the subdir.
            # Otherwise fall back to the higher-level module name.
            kernel_dir = kernel_names[0] if len(kernel_names) == 1 else dump_prefix_base
            dump_dir = dump_root_dir / _sanitize_path_component(kernel_dir)
            print(f"[flir.compile] FLIR_DUMP_IR=1 dir={dump_dir}")
        try:
            module = ir.Module.parse(asm, context=ctx)
        except Exception:
            # Some environments/bindings have rare MLIR assembly parse failures when
            # round-tripping with rich debug info. Fall back to a simpler print; and
            # if that still fails, compile in-place (mutating the caller module).
            try:
                asm_no_debug = mlir_module.operation.get_asm(enable_debug_info=False)
                module = ir.Module.parse(asm_no_debug, context=ctx)
            except Exception:
                module = mlir_module

    chip = get_rocm_arch()

    pipeline = _build_pipeline_str(
        chip=chip,
        use_bare_ptr_memref_call_conv=use_bare_ptr_memref_call_conv,
        use_bare_pointers_for_host=use_bare_pointers_for_host,
        use_bare_pointers_for_kernels=use_bare_pointers_for_kernels,
    )

    with ctx:
        _override_gpu_module_targets(module, chip=chip)

        # ------------------------------------------------------------------
        # Cache lookup (post-target override, pre-pipeline).
        # ------------------------------------------------------------------
        cache = None
        cache_key = None
        if cache_enabled():
            try:
                key_payload = default_key_payload(
                    chip=str(chip),
                    pipeline=str(pipeline),
                    input_asm=module.operation.get_asm(enable_debug_info=False),
                )
                cache_key = make_cache_key(key_payload)
                cache = FileCache(key=cache_key)
            except Exception:
                cache = None
                cache_key = None

        # Acquire a per-cache-key process lock across "check -> compile -> put".
        # This prevents multiple processes from concurrently compiling the same key
        # on a cache miss
        lock_cm = cache.lock() if cache is not None else contextlib.nullcontext(None)
        with lock_cm as lock_fd:
            if cache is not None and (not cache_rebuild_requested()):
                cached_asm = cache.get_module_asm()
                if cached_asm:
                    try:
                        cached_mod = ir.Module.parse(cached_asm, context=ctx)
                        if dump_enabled:
                            print(f"[flir.compile] cache hit key={cache_key}")
                        from .executor import ExecutionEngineExecutor as Executor
                        if shared_libs is None:
                            shared_libs = default_shared_libs().as_list()
                        return Executor(cached_mod, opt_level=opt_level, shared_libs=shared_libs)
                    except Exception:
                        # Treat cache parse failures as misses.
                        pass
            if dump_enabled:
                # When dumping is enabled, run the pipeline in stages so each intermediate
                # module state is captured to a file.
                out = _dump_ir(
                    "00_target_overridden",
                    dump_dir=dump_dir,
                    asm=module.operation.get_asm(enable_debug_info=True),
                )
                print(f"[flir.compile] dump 00_target_overridden -> {out}")
                asm_for_isa: Optional[str] = None
                stage_frags = _pipeline_fragments(
                    chip=chip,
                    use_bare_ptr_memref_call_conv=use_bare_ptr_memref_call_conv,
                    use_bare_pointers_for_host=use_bare_pointers_for_host,
                    use_bare_pointers_for_kernels=use_bare_pointers_for_kernels,
                )
                # Keep dump filenames stable vs the historical numbering scheme:
                # 00_target_overridden, then 03..14 for pipeline stages, then 15_final_isa.
                stage_num_base = 3
                for stage_num, frag in enumerate(stage_frags, start=stage_num_base):
                    stage_name = f"{stage_num:02d}_{_stage_label_from_fragment(frag)}"
                    pm = PassManager.parse(f"builtin.module({frag})", context=ctx)
                    pm.enable_verifier(bool(verify))
                    pm.run(module.operation)
                    stage_asm = module.operation.get_asm(enable_debug_info=True)
                    out = _dump_ir(stage_name, dump_dir=dump_dir, asm=stage_asm)
                    print(f"[flir.compile] dump {stage_name} -> {out}")

                    # Dump ISA from the *post-LLVM* module (right before fatbin emission).
                    # This mirrors `tests/utils.py:compile_to_hsaco` and yields readable assembly.
                    if frag.strip() == "reconcile-unrealized-casts":
                        asm_for_isa = stage_asm

                if asm_for_isa is not None:
                    isa_out = _dump_isa_from_rocdl_module_asm(
                        dump_dir=dump_dir,
                        ctx=ctx,
                        asm=asm_for_isa,
                        verify=verify,
                    )
                    if isa_out is not None:
                        isa_stage = f"{stage_num_base + len(stage_frags):02d}_final_isa"
                        print(f"[flir.compile] dump {isa_stage} -> {isa_out}")
            else:
                pm = PassManager.parse(pipeline, context=ctx)
                pm.enable_verifier(bool(verify))
                pm.run(module.operation)
            if print_final_module:
                print(module)

            # Cache store (post-pipeline, with binary embedded). Reuse the same lock.
            if cache is not None:
                try:
                    cache.put_module_asm(
                        module.operation.get_asm(enable_debug_info=False),
                        meta={
                            "chip": str(chip),
                            "pipeline": str(pipeline),
                        },
                        lock_fd=lock_fd,
                    )
                    if dump_enabled:
                        print(f"[flir.compile] cache put key={cache_key}")
                except Exception:
                    pass

    from .executor import ExecutionEngineExecutor as Executor

    if shared_libs is None:
        shared_libs = default_shared_libs().as_list()
    return Executor(module, opt_level=opt_level, shared_libs=shared_libs)


