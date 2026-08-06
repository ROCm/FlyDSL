# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import glob
import os
import re
import shutil
from typing import List, Optional, Tuple

from ...runtime.device_runtime.cuda import get_cuda_arch
from ...utils import env
from ...utils.logger import log
from .base import BaseBackend, GPUTarget


def _cuda_version_key(path: str):
    """Sort key extracting the numeric version from a /usr/local/cuda-X.Y path."""
    m = re.search(r"cuda-(\d+)(?:\.(\d+))?", path)
    if not m:
        return (0, 0)
    return (int(m.group(1)), int(m.group(2) or 0))


def _detect_cuda_toolkit() -> Optional[str]:
    """Locate a CUDA toolkit (the directory containing ``bin/ptxas``).

    The default ``/usr/bin/ptxas`` may be too old for newer SM targets (e.g.
    sm_90 needs CUDA 12+). Resolution order:

    1. ``CUDA_HOME`` / ``CUDA_PATH`` / ``CUDAToolkit_ROOT`` env vars.
    2. The newest ``/usr/local/cuda-*`` install, then ``/usr/local/cuda``.
    3. The toolkit root inferred from ``ptxas`` on ``PATH``.

    Returns ``None`` when no toolkit is found; callers must handle that
    (the PTX dump is skipped, and device-binary codegen falls back to the
    serializer's default toolkit, which may be too old).
    """
    for var in ("CUDA_HOME", "CUDA_PATH", "CUDAToolkit_ROOT"):
        p = os.environ.get(var)
        if p and os.path.exists(os.path.join(p, "bin", "ptxas")):
            return p
    candidates = sorted(glob.glob("/usr/local/cuda-*"), key=_cuda_version_key, reverse=True)
    candidates.append("/usr/local/cuda")
    for p in candidates:
        if os.path.exists(os.path.join(p, "bin", "ptxas")):
            return p

    ptxas = shutil.which("ptxas")
    if ptxas:
        toolkit = os.path.dirname(os.path.dirname(os.path.realpath(ptxas)))
        if os.path.exists(os.path.join(toolkit, "bin", "ptxas")):
            return toolkit

    log().warning(
        "No CUDA toolkit found (checked CUDA_HOME, CUDA_PATH, CUDAToolkit_ROOT, "
        "/usr/local/cuda-*, /usr/local/cuda, and PATH). "
        "Device-binary codegen will fall back to the default toolkit, which may be "
        "too old for the target arch; set CUDA_HOME to a CUDA 12+ install."
    )
    return None


class CudaBackend(BaseBackend):
    """CUDA / NVIDIA compile backend (CUDA driver runtime, NVVM lowering).

    Stage-one backend: lowers Fly to NVVM/PTX via target-neutral Universal
    atoms plus initial SM80 mma.sync.aligned / cp.async / ldmatrix atoms.
    """

    @staticmethod
    def supports_target(target: GPUTarget) -> bool:
        return target.backend == "cuda"

    @staticmethod
    def detect_target() -> GPUTarget:
        arch = env.compile.arch or get_cuda_arch()
        if not arch:
            raise RuntimeError(
                "No CUDA device found, so the target architecture could not be detected. "
                "Set FLYDSL_GPU_ARCH to an sm_* target to compile without a device."
            )
        return GPUTarget(backend="cuda", arch=arch, warp_size=32)

    @classmethod
    def make_target(cls, arch: str) -> GPUTarget:
        return GPUTarget(backend="cuda", arch=arch, warp_size=32)

    @classmethod
    def llvm_address_space(cls, address_space) -> int:
        """Map an address space to its NVPTX LLVM representation.

        Must stay in sync with ``mapToLLVMAddressSpace`` in
        ``lib/Conversion/FlyToNVVM/FlyToNVVM.cpp``. NVVM has no dedicated
        register address space; register-backed scratch is an alloca in the
        generic (0) address space, unlike AMDGPU's private (5).
        """
        from ..._mlir.dialects.fly import AddressSpace

        mapping = {
            AddressSpace.Generic: 0,
            AddressSpace.Global: 1,
            AddressSpace.Shared: 3,
            AddressSpace.Register: 0,
        }
        try:
            return mapping[address_space]
        except KeyError:
            raise ValueError(f"CUDA address space {address_space} does not lower to a bare LLVM pointer") from None

    # -- compile pipeline ------------------------------------------------

    @staticmethod
    def _format_pass_opts(opts: dict) -> str:
        return " ".join(f"{k}={v}" for k, v in opts.items())

    def _pipeline_parts(self, *, compile_hints: dict) -> Tuple[List[str], str]:
        chip = self.target.arch  # e.g. "sm_90"

        bin_cli_opts = []
        if env.debug.enable_debug_info:
            bin_cli_opts.append("-g")

        nvvm_target_opts = {
            "O": 3,
            "chip": chip,
            "fast": "true" if compile_hints.get("fast_fp_math") else "false",
            "triple": "nvptx64-nvidia-cuda",
        }

        pre_binary_fragments = [
            "fly-rewrite-func-signature",
            "fly-canonicalize",
            "fly-layout-lowering",
            "fly-int-swizzle-simplify",
            "canonicalize",
            "fly-convert-atom-call-to-ssa-form",
            "fly-promote-regmem-to-vectorssa",
            "convert-fly-to-nvvm",
            "canonicalize",
            "gpu.module(convert-scf-to-cf,cse,"
            "convert-gpu-to-nvvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=true})",
        ]
        binary_prep_fragments = [
            f"nvvm-attach-target{{{self._format_pass_opts(nvvm_target_opts)}}}",
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
        toolkit = _detect_cuda_toolkit()
        toolkit_opt = f"toolkit={toolkit} " if toolkit else ""
        binary_fragment = f'gpu-module-to-binary{{{toolkit_opt}format=fatbin opts="{" ".join(bin_cli_opts)}"}}'
        return [*pre_binary_fragments, *binary_prep_fragments], binary_fragment

    def pipeline_fragments(self, *, compile_hints: dict) -> List[str]:
        pre_binary_fragments, binary_fragment = self._pipeline_parts(compile_hints=compile_hints)
        return [*pre_binary_fragments, binary_fragment]

    def external_binary_pipeline_fragments(self, *, compile_hints: dict) -> Tuple[List[str], str]:
        return self._pipeline_parts(compile_hints=compile_hints)

    def gpu_module_targets(self) -> List[str]:
        # The NVVM target attribute is injected by the `nvvm-attach-target` pass
        # in the pipeline. Returning it here too would attach the target twice
        # (the gpu.module would carry two `#nvvm.target` attrs), making
        # gpu-module-to-binary serialize the module once per target. Leave it to
        # the pass and return empty.
        return []

    # -- cache / fingerprint ---------------------------------------------

    def native_lib_patterns(self) -> List[str]:
        return [
            "_mlirDialectsFly*.so",
            "libFly*.so",
            "libfly_cuda_runtime.so",
            "_mlirRegisterEverything*.so",
        ]

    def jit_runtime_lib_basenames(self) -> List[str]:
        return [
            "libfly_cuda_runtime.so",
            "libmlir_c_runner_utils.so",
        ]
