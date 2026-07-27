# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from typing import List, Tuple

from ...runtime.device import get_rocm_arch, is_rdna_arch
from ...utils import env
from .base import BaseBackend, GPUTarget


class RocmBackend(BaseBackend):
    """ROCm / AMDGPU compile backend (HIP runtime, ROCDL lowering)."""

    @staticmethod
    def supports_target(target: GPUTarget) -> bool:
        return target.backend == "rocm"

    @staticmethod
    def detect_target() -> GPUTarget:
        arch = env.compile.arch or get_rocm_arch()
        warp_size = 32 if is_rdna_arch(arch) else 64
        return GPUTarget(backend="rocm", arch=arch, warp_size=warp_size)

    @classmethod
    def make_target(cls, arch: str) -> GPUTarget:
        warp_size = 32 if is_rdna_arch(arch) else 64
        return GPUTarget(backend="rocm", arch=arch, warp_size=warp_size)

    # -- compile pipeline ------------------------------------------------

    @staticmethod
    def _format_pass_opts(opts: dict) -> str:
        """Format {key: value, ...} as 'key=value key2=value2' for MLIR pass options."""
        return " ".join(f"{k}={v}" for k, v in opts.items())

    def _pipeline_parts(self, *, compile_hints: dict) -> Tuple[List[str], str]:
        chip = self.target.arch
        waves_per_eu = compile_hints.get("waves_per_eu")
        maxnreg = compile_hints.get("maxnreg")

        bin_cli_opts = []
        if env.debug.enable_debug_info:
            bin_cli_opts.append("-g")
        if waves_per_eu:
            bin_cli_opts.append(f"--amdgpu-waves-per-eu={waves_per_eu}")
        if maxnreg:
            bin_cli_opts.append(f"--amdgpu-num-vgpr={maxnreg}")

        rocdl_opts = {
            "O": 2,
            "abi": 600,
            "chip": chip,
            "correct-sqrt": "true",
            "daz": "false",
            "fast": "true" if compile_hints.get("fast_fp_math") else "false",
            "features": "",
            "finite-only": "false",
            "module": "",
            "triple": "amdgcn-amd-amdhsa",
            "unsafe-math": "true" if compile_hints.get("unsafe_fp_math") else "false",
            "wave64": "false" if is_rdna_arch(chip) else "true",
        }

        pre_binary_fragments = [
            "fly-rewrite-func-signature",
            "fly-canonicalize",
            "fly-layout-lowering",
            "fly-int-swizzle-simplify",
            "canonicalize",
            "fly-convert-atom-call-to-ssa-form",
            "fly-promote-regmem-to-vectorssa",
            "convert-fly-to-rocdl",
            "canonicalize",
            f"gpu.module(convert-scf-to-cf,cse,"
            f"convert-gpu-to-rocdl{{chipset={chip} index-bitwidth=0 runtime=HIP use-bare-ptr-memref-call-conv=true}},"
            f"fly-rocdl-cluster-attr)",
        ]
        binary_prep_fragments = [
            f"rocdl-attach-target{{{self._format_pass_opts(rocdl_opts)}}}",
            "convert-scf-to-cf",
            "convert-cf-to-llvm",
            "gpu-to-llvm{use-bare-pointers-for-host=true use-bare-pointers-for-kernels=true}",
            "convert-vector-to-llvm",
            "convert-arith-to-llvm",
            "convert-func-to-llvm",
            "reconcile-unrealized-casts",
            *(
                ["ensure-debug-info-scope-on-llvm-func{emission-kind=LineTablesOnly}"]
                if env.debug.enable_debug_info
                else []
            ),
        ]
        binary_fragment = f'gpu-module-to-binary{{format=fatbin opts="{" ".join(bin_cli_opts)}"}}'
        return [*pre_binary_fragments, *binary_prep_fragments], binary_fragment

    def pipeline_fragments(self, *, compile_hints: dict) -> List[str]:
        pre_binary_fragments, binary_fragment = self._pipeline_parts(compile_hints=compile_hints)
        return [*pre_binary_fragments, binary_fragment]

    def external_binary_pipeline_fragments(self, *, compile_hints: dict) -> Tuple[List[str], str]:
        return self._pipeline_parts(compile_hints=compile_hints)

    def lower_compile_hints(self, module, *, compile_hints: dict) -> None:
        """Materialize a scalar waves-per-EU override on kernel entries."""
        waves_per_eu = compile_hints.get("waves_per_eu")
        if waves_per_eu is None:
            return
        if isinstance(waves_per_eu, bool) or not isinstance(waves_per_eu, int):
            raise TypeError(f"waves_per_eu must be a non-negative int, got {waves_per_eu!r}")
        if waves_per_eu < 0:
            raise ValueError(f"waves_per_eu must be >= 0, got {waves_per_eu}")
        if waves_per_eu == 0:
            return

        with module.context:
            for func_op in _iter_gpu_kernel_funcs(module):
                # rocdl.waves_per_eu expresses a minimum. Replace it with the exact
                # min/max LLVM passthrough for an explicit compile-hint override.
                if "rocdl.waves_per_eu" in func_op.attributes:
                    del func_op.attributes["rocdl.waves_per_eu"]
                _set_passthrough(func_op, "amdgpu-waves-per-eu", f"{waves_per_eu},{waves_per_eu}")

    def gpu_module_targets(self) -> List[str]:
        chip = self.target.arch
        return [f'#rocdl.target<chip = "{chip}">']

    # -- cache / fingerprint ---------------------------------------------

    def native_lib_patterns(self) -> List[str]:
        return [
            "_mlirDialectsFly*.so",
            "libFly*.so",
            "libfly_jit_runtime.so",
            "libmlir_rocm_runtime.so",
            "_mlirRegisterEverything*.so",
        ]

    def jit_runtime_lib_basenames(self) -> List[str]:
        return [
            "libfly_jit_runtime.so",
            "libmlir_c_runner_utils.so",
        ]


def _iter_gpu_kernel_funcs(module):
    """Yield entry ``gpu.func`` ops, excluding device helpers."""
    for top in module.body.operations:
        if top.operation.name != "gpu.module":
            continue
        for op in top.regions[0].blocks[0].operations:
            if op.operation.name == "gpu.func" and "gpu.kernel" in op.attributes:
                yield op


def _set_passthrough(func_op, key: str, value: str) -> None:
    """Replace one LLVM passthrough key while preserving unrelated entries."""
    from ..._mlir import ir

    def _entry_key(entry):
        try:
            pair = ir.ArrayAttr(entry)
            return ir.StringAttr(pair[0]).value if len(pair) else None
        except (TypeError, ValueError):
            return None

    new_entry = ir.ArrayAttr.get([ir.StringAttr.get(key), ir.StringAttr.get(value)])
    existing = func_op.attributes["passthrough"] if "passthrough" in func_op.attributes else None
    kept = [entry for entry in existing if _entry_key(entry) != key] if existing is not None else []
    func_op.attributes["passthrough"] = ir.ArrayAttr.get([*kept, new_entry])
