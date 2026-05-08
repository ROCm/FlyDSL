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

    @staticmethod
    def _format_pass_opts(opts: dict) -> str:
        """Format {key: value, ...} as 'key=value key2=value2' for MLIR pass options."""
        return " ".join(f"{k}={v}" for k, v in opts.items())

    @staticmethod
    def _resolve_toolkit_path(toolkit_hint: str | None) -> str | None:
        """Normalize ROCm toolkit layout for gpu-module-to-binary.

        The MLIR pass expects a toolkit root that contains:
          - <toolkit>/llvm/bin/ld.lld
          - <toolkit>/amdgcn/bitcode/*.bc

        Some distros install a "flat" layout where both `bin/` and `amdgcn/`
        live directly under `.../lib/llvm` instead. In that case, we create a
        user-space shim root under /tmp and symlink:
          - shim/llvm   -> <flat-root>
          - shim/amdgcn -> <flat-root>/amdgcn
        """
        import hashlib as _hashlib
        import os as _os
        import tempfile as _tempfile

        if not toolkit_hint:
            return None

        toolkit_hint = _os.path.realpath(toolkit_hint)
        expected_lld = _os.path.join(toolkit_hint, "llvm", "bin", "ld.lld")
        expected_bitcode = _os.path.join(toolkit_hint, "amdgcn", "bitcode")
        if _os.path.isfile(expected_lld) and _os.path.isdir(expected_bitcode):
            return toolkit_hint

        flat_lld = _os.path.join(toolkit_hint, "bin", "ld.lld")
        flat_amdgcn = _os.path.join(toolkit_hint, "amdgcn")
        flat_bitcode = _os.path.join(flat_amdgcn, "bitcode")
        if not (_os.path.isfile(flat_lld) and _os.path.isdir(flat_bitcode)):
            return None

        digest = _hashlib.sha1(toolkit_hint.encode("utf-8")).hexdigest()[:12]
        shim_root = _os.path.join(_tempfile.gettempdir(), f"flydsl-rocm-toolkit-{digest}")
        _os.makedirs(shim_root, exist_ok=True)
        for name, target in (("llvm", toolkit_hint), ("amdgcn", flat_amdgcn)):
            link_path = _os.path.join(shim_root, name)
            if _os.path.islink(link_path):
                if _os.path.realpath(link_path) == _os.path.realpath(target):
                    continue
                _os.unlink(link_path)
            elif _os.path.exists(link_path):
                continue
            _os.symlink(target, link_path)

        return shim_root

    def pipeline_fragments(self, *, compile_hints: dict) -> List[str]:
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

        # ROCm toolkit path for `gpu-module-to-binary` (where lld pulls in
        # the amdgcn device-libs bitcodes). Order:
        #   1. FLYDSL_ROCM_TOOLKIT_PATH (explicit override)
        #   2. ROCM_PATH (standard ROCm env var)
        #   3. /opt/rocm (default that MLIR uses if neither is set)
        # On distributions where /opt/rocm only contains the runtime symlink
        # and the bitcodes live under a versioned subdir like
        # /opt/rocm/core-7.12/lib/llvm, set FLYDSL_ROCM_TOOLKIT_PATH or
        # ROCM_PATH to that dir so MLIR finds amdgcn/bitcode/*.bc.
        import os as _os
        toolkit_path = self._resolve_toolkit_path(
            _os.environ.get("FLYDSL_ROCM_TOOLKIT_PATH") or _os.environ.get("ROCM_PATH")
        )
        if not toolkit_path:
            for candidate in (
                "/opt/rocm/lib/llvm",
                "/opt/rocm/core-7.12/lib/llvm",
                "/opt/rocm",
            ):
                normalized = self._resolve_toolkit_path(candidate)
                if normalized:
                    toolkit_path = normalized
                    break

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

        return [
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
            (
                f'gpu-module-to-binary{{format=fatbin opts="{" ".join(bin_cli_opts)}"'
                + (f" toolkit={toolkit_path}" if toolkit_path else "")
                + "}"
            ),
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
