"""High-level compilation entrypoint for RocDSL modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

from _mlir import ir
from _mlir.passmanager import PassManager

from rocdsl.compiler.context import ensure_rocir_python_extensions
from rocdsl.runtime.device import get_rocm_arch

from .executor import default_shared_libs


@dataclass(frozen=True)
class CompileOptions:
    verify: bool = True
    print_final_module: bool = False
    opt_level: int = 3
    shared_libs: Optional[Sequence[str]] = None
    backend: Literal["execution_engine"] = "execution_engine"


def _build_pipeline_str(*, chip: str) -> str:
    # Keep this as an explicit string so we can pass GPU-to-LLVM options that include dashes.
    #
    # Notes:
    # - We use `format=fatbin` so the ROCm runtime can launch kernels from an embedded binary blob.
    # - We request bare pointers for host+kernel to keep the executor argument marshaling simple.
    return (
        "builtin.module("
        "rocir-coord-lowering,"
        "rocir-to-standard,"
        "trivial-dce,"
        "canonicalize,"
        "cse,"
        "gpu-kernel-outlining{data-layout-str=},"
        "gpu.module("
        "convert-scf-to-cf,"
        "convert-gpu-to-rocdl{chipset=gfx000 index-bitwidth=0 runtime=unknown use-bare-ptr-memref-call-conv=true},"
        "reconcile-unrealized-casts"
        "),"
        f"rocdl-attach-target{{O=2 abi=600 chip={chip} correct-sqrt=true daz=false fast=false features= finite-only=false module= triple=amdgcn-amd-amdhsa unsafe-math=false wave64=true}},"
        "gpu-to-llvm{intersperse-sizes-for-kernels=false use-bare-pointers-for-host=true use-bare-pointers-for-kernels=true},"
        "reconcile-unrealized-casts,"
        "gpu-module-to-binary{format=fatbin opts= section= toolkit=}"
        ")"
    )


def _override_gpu_module_targets(module: ir.Module, *, chip: str) -> None:
    """Force all `gpu.module` targets to a consistent ROCm target.

    Some tests/modules set `gpu.module [...]` targets (e.g. `abi=500`) which can
    produce code objects that fail to load on newer GPUs. `rocdsl.compile()`
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


def compile(
    rocdsl_module_or_ir: Union[object, ir.Module],
    *,
    verify: bool = True,
    print_final_module: bool = False,
    opt_level: int = 3,
    shared_libs: Optional[Sequence[str]] = None,
    backend: Literal["execution_engine"] = "execution_engine",
) -> Executor:
    """Compile a RocDSL module to an Executor.

    Returns an MLIR ExecutionEngine-backed executor.
    """

    # Accept `rocdsl.lang.MlirModule` instances.
    mlir_module = getattr(rocdsl_module_or_ir, "module", None)
    if mlir_module is None:
        mlir_module = rocdsl_module_or_ir
    if not isinstance(mlir_module, ir.Module):
        raise TypeError(f"Expected an MLIR module or rocdsl.lang.MlirModule; got {type(rocdsl_module_or_ir)}")

    ctx = mlir_module.context
    ensure_rocir_python_extensions(ctx)

    # Parse a fresh module for compilation so callers can print/reuse their builder module.
    with ctx:
        asm = mlir_module.operation.get_asm(enable_debug_info=True)
        module = ir.Module.parse(asm, context=ctx)

    chip = get_rocm_arch()

    pipeline = _build_pipeline_str(chip=chip)

    with ctx:
        _override_gpu_module_targets(module, chip=chip)
        pm = PassManager.parse(pipeline, context=ctx)
        pm.enable_verifier(bool(verify))
        pm.run(module.operation)
        if print_final_module:
            print(module)

    from .executor import ExecutionEngineExecutor as Executor

    if shared_libs is None:
        shared_libs = default_shared_libs().as_list()
    return Executor(module, opt_level=opt_level, shared_libs=shared_libs)


