# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from typing import List

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

    def pipeline_fragments(self, *, compile_hints: dict) -> List[str]:
        chip = self.target.arch
        waves_per_eu = compile_hints.get("waves_per_eu")
        maxnreg = compile_hints.get("maxnreg")

        debug_opt = "-g" if env.debug.enable_debug_info else ""
        wpe_opt = f" --amdgpu-waves-per-eu={waves_per_eu}" if waves_per_eu else ""
        maxnreg_opt = f" --amdgpu-num-vgpr={maxnreg}" if maxnreg else ""
        all_opts = f"{debug_opt}{wpe_opt}{maxnreg_opt}".strip()

        wave64 = "false" if is_rdna_arch(chip) else "true"
        return [
            "fly-rewrite-func-signature",
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
            *(
                ["ensure-debug-info-scope-on-llvm-func{emission-kind=LineTablesOnly}"]
                if env.debug.enable_debug_info
                else []
            ),
            f'gpu-module-to-binary{{format=fatbin opts="{all_opts}"}}',
        ]

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
