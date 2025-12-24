"""High-level compilation entrypoint for FLIR modules."""

from __future__ import annotations

import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Sequence, Union

from _mlir import ir
from _mlir.passmanager import PassManager

from pyflir.compiler.context import ensure_flir_python_extensions
from pyflir.runtime.device import get_rocm_arch

from .executor import default_shared_libs

if TYPE_CHECKING:
    from .executor import ExecutionEngineExecutor as Executor


@dataclass(frozen=True)
class CompileOptions:
    verify: bool = True
    print_final_module: bool = False
    opt_level: int = 3
    shared_libs: Optional[Sequence[str]] = None
    backend: Literal["execution_engine"] = "execution_engine"


def _pipeline_stages(*, chip: str) -> list[tuple[str, str]]:
    """Single source of truth for the FLIR compilation pipeline.

    Each entry is (dump_stage_name, pipeline_fragment) where pipeline_fragment is the
    content inside `builtin.module(...)` for PassManager.parse.

    Keeping stages and the full pipeline derived from the same list prevents drift
    (a common source of hard-to-debug issues when editing pass pipelines).
    """
    return [
        ("01_input_cse", "cse"),
        ("02_input_dce", "trivial-dce"),
        ("03_flir_to_standard", "flir-to-standard"),
        ("04_trivial_dce", "trivial-dce"),
        ("05_canonicalize", "canonicalize"),
        ("06_cse", "cse"),
        ("07_gpu_kernel_outlining", "gpu-kernel-outlining{data-layout-str=}"),
        ("08_gpu_convert_scf_to_cf", "gpu.module(convert-scf-to-cf)"),
        (
            "09_gpu_convert_gpu_to_rocdl",
            "gpu.module(convert-gpu-to-rocdl{chipset=gfx000 index-bitwidth=0 runtime=unknown use-bare-ptr-memref-call-conv=true})",
        ),
        ("10_gpu_reconcile_unrealized_casts", "gpu.module(reconcile-unrealized-casts)"),
        (
            "11_rocdl_attach_target",
            # Note: keep this as a formatted string so the chip is visible in dumps and
            # matches the non-dump compilation pipeline.
            f"rocdl-attach-target{{O=2 abi=600 chip={chip} correct-sqrt=true daz=false fast=false features= finite-only=false module= triple=amdgcn-amd-amdhsa unsafe-math=false wave64=true}}",
        ),
        (
            "12_gpu_to_llvm",
            "gpu-to-llvm{intersperse-sizes-for-kernels=false use-bare-pointers-for-host=true use-bare-pointers-for-kernels=true}",
        ),
        ("13_reconcile_unrealized_casts", "reconcile-unrealized-casts"),
        ("14_gpu_module_to_binary", "gpu-module-to-binary{format=fatbin opts= section= toolkit=}"),
    ]


def _build_pipeline_str(*, chip: str) -> str:
    """Build the full PassManager pipeline string from `_pipeline_stages`."""
    frags = [frag for _name, frag in _pipeline_stages(chip=chip)]
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
        from pyflir.dialects.ext.gpu import get_compile_object_bytes
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
        module = ir.Module.parse(asm, context=ctx)

    chip = get_rocm_arch()

    pipeline = _build_pipeline_str(chip=chip)

    with ctx:
        _override_gpu_module_targets(module, chip=chip)
        if dump_enabled:
            # When dumping is enabled, run the pipeline in stages so each intermediate
            # module state is captured to a file.
            out = _dump_ir(
                "00_target_overridden",
                dump_dir=dump_dir,
                asm=module.operation.get_asm(enable_debug_info=True),
            )
            print(f"[flir.compile] dump 01_target_overridden -> {out}")
            asm_for_isa: Optional[str] = None
            for stage_name, frag in _pipeline_stages(chip=chip):
                pm = PassManager.parse(f"builtin.module({frag})", context=ctx)
                pm.enable_verifier(bool(verify))
                pm.run(module.operation)
                stage_asm = module.operation.get_asm(enable_debug_info=True)
                out = _dump_ir(stage_name, dump_dir=dump_dir, asm=stage_asm)
                print(f"[flir.compile] dump {stage_name} -> {out}")

                # Dump ISA from the *post-LLVM* module (right before fatbin emission).
                # This mirrors `tests/utils.py:compile_to_hsaco` and yields readable assembly.
                if stage_name == "13_reconcile_unrealized_casts":
                    asm_for_isa = stage_asm

            if asm_for_isa is not None:
                isa_out = _dump_isa_from_rocdl_module_asm(
                    dump_dir=dump_dir,
                    ctx=ctx,
                    asm=asm_for_isa,
                    verify=verify,
                )
                if isa_out is not None:
                    print(f"[flir.compile] dump 15_final_isa -> {isa_out}")
        else:
            pm = PassManager.parse(pipeline, context=ctx)
            pm.enable_verifier(bool(verify))
            pm.run(module.operation)
        if print_final_module:
            print(module)

    from .executor import ExecutionEngineExecutor as Executor

    if shared_libs is None:
        shared_libs = default_shared_libs().as_list()
    return Executor(module, opt_level=opt_level, shared_libs=shared_libs)


