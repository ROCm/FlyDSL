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

    def _bin_cli_opts(self, *, compile_hints: dict) -> List[str]:
        waves_per_eu = compile_hints.get("waves_per_eu")
        maxnreg = compile_hints.get("maxnreg")
        bin_cli_opts = []
        if env.debug.enable_debug_info:
            bin_cli_opts.append("-g")
        if waves_per_eu:
            bin_cli_opts.append(f"--amdgpu-waves-per-eu={waves_per_eu}")
        if maxnreg:
            bin_cli_opts.append(f"--amdgpu-num-vgpr={maxnreg}")
        return bin_cli_opts

    def _rocdl_opts(self, *, compile_hints: dict, opt_level: int = 2) -> dict:
        chip = self.target.arch
        return {
            "O": opt_level,
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

    def llvm_recodegen_fragments(self, *, compile_hints: dict, opt_level: int = 0) -> Tuple[str, str]:
        """Fragments to re-codegen an already-LLVM-dialect ``gpu.module`` that has
        NO target attached: attach a ROCDL target at ``opt_level`` then emit the
        device binary.  Used by the custom-LLVM-pass path, which has already run
        its own ``opt`` pipeline, so codegen runs at ``O=0`` to avoid re-optimizing.
        """
        rocdl_opts = self._rocdl_opts(compile_hints=compile_hints, opt_level=opt_level)
        bin_cli_opts = self._bin_cli_opts(compile_hints=compile_hints)
        attach_fragment = f"rocdl-attach-target{{{self._format_pass_opts(rocdl_opts)}}}"
        binary_fragment = f'gpu-module-to-binary{{format=fatbin opts="{" ".join(bin_cli_opts)}"}}'
        return attach_fragment, binary_fragment

    def _pipeline_parts(self, *, compile_hints: dict) -> Tuple[List[str], str]:
        chip = self.target.arch

        bin_cli_opts = self._bin_cli_opts(compile_hints=compile_hints)
        rocdl_opts = self._rocdl_opts(compile_hints=compile_hints, opt_level=2)

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
