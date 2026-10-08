# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

from ...runtime.device import get_rocm_arch, get_warp_size
from ...utils import env
from .base import AOTRuntimeConfig, BaseBackend, GPUTarget


def _is_junction(path: Path) -> bool:
    if hasattr(os.path, "isjunction"):
        return os.path.isjunction(path)
    try:
        return path.lstat().st_reparse_tag == 0xA0000003
    except (AttributeError, OSError):
        return False


def _windows_rocm_toolkit() -> str:
    """Resolve the split TheRock SDK layout to the toolkit layout used by ROCDL."""
    explicit = os.environ.get("FLYDSL_ROCM_TOOLKIT")
    if explicit:
        toolkit = Path(explicit).resolve()
        if not (toolkit / "llvm" / "bin" / "ld.lld.exe").is_file() or not (
            toolkit / "amdgcn" / "bitcode"
        ).is_dir():
            raise RuntimeError(f"Invalid FLYDSL_ROCM_TOOLKIT: {toolkit}")
        return toolkit.as_posix()

    candidates = []
    for variable in ("ROCM_PATH", "HIP_PATH"):
        if value := os.environ.get(variable):
            candidates.append(Path(value))
    for entry in sys.path:
        for package in ("_rocm_sdk_devel", "_rocm_sdk_core"):
            candidates.append(Path(entry) / package)

    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "llvm" / "bin" / "ld.lld.exe").is_file() and (
            candidate / "amdgcn" / "bitcode"
        ).is_dir():
            return candidate.as_posix()

        llvm = candidate / "lib" / "llvm"
        if not (llvm / "bin" / "ld.lld.exe").is_file() or not (llvm / "amdgcn" / "bitcode").is_dir():
            continue

        stage_root = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "FlyDSL" / "rocm-toolkit"
        stage = stage_root / hashlib.sha256(str(llvm).encode()).hexdigest()[:12]
        stage.mkdir(parents=True, exist_ok=True)
        links = ((stage / "llvm", llvm), (stage / "amdgcn", llvm / "amdgcn"))
        for link, target in links:
            if link.exists() or link.is_symlink() or _is_junction(link):
                if not _is_junction(link) or link.resolve() != target.resolve():
                    raise RuntimeError(f"Refusing to use unexpected ROCm toolkit junction: {link}")
                continue
            result = subprocess.run(
                ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode and not _is_junction(link):
                raise RuntimeError(f"Could not create ROCm toolkit junction {link}: {result.stderr.strip()}")
            if not _is_junction(link) or link.resolve() != target.resolve():
                raise RuntimeError(f"ROCm toolkit junction does not target the active SDK: {link}")

        if not (stage / "llvm" / "bin" / "ld.lld.exe").is_file() or not (
            stage / "amdgcn" / "bitcode"
        ).is_dir():
            raise RuntimeError(f"Incomplete staged ROCm toolkit: {stage}")
        return stage.as_posix()

    raise RuntimeError("Could not find ld.lld.exe and AMDGPU bitcode in the active ROCm SDK")


class RocmBackend(BaseBackend):
    """ROCm / AMDGPU compile backend (HIP runtime, ROCDL lowering)."""

    @staticmethod
    def supports_target(target: GPUTarget) -> bool:
        return target.backend == "rocm"

    @staticmethod
    def detect_target() -> GPUTarget:
        arch = env.compile.arch or get_rocm_arch()
        return GPUTarget(backend="rocm", arch=arch, warp_size=get_warp_size(arch))

    @classmethod
    def make_target(cls, arch: str) -> GPUTarget:
        return GPUTarget(backend="rocm", arch=arch, warp_size=get_warp_size(arch))

    @classmethod
    def llvm_address_space(cls, address_space) -> int:
        """Map an address space to its AMDGPU LLVM representation."""
        from ..._mlir.dialects.fly import AddressSpace

        mapping = {
            AddressSpace.Generic: 0,
            AddressSpace.Global: 1,
            AddressSpace.Shared: 3,
            AddressSpace.Register: 5,
        }
        try:
            return mapping[address_space]
        except KeyError:
            raise ValueError(f"ROCm address space {address_space} does not lower to a bare LLVM pointer") from None

    # -- compile pipeline ------------------------------------------------

    @staticmethod
    def _format_pass_opts(opts: dict) -> str:
        """Format {key: value, ...} as 'key=value key2=value2' for MLIR pass options."""
        return " ".join(f"{k}={v}" for k, v in opts.items())

    def _pipeline_parts(self, *, compile_hints: dict) -> Tuple[List[str], str]:
        chip = self.target.arch

        # ROCDL never reads gpu-module-to-binary's opts=, so nothing may be
        # routed through it.
        bin_cli_opts = []

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
            "wave64": "true" if get_warp_size(chip) == 64 else "false",
        }

        pre_binary_fragments = [
            "fly-rewrite-func-signature",
            "fly-canonicalize",
            "fly-rocdl-expand-ops",
            "fly-layout-lowering",
            "fly-int-swizzle-simplify",
            "canonicalize",
            "fly-convert-atom-call-to-ssa-form",
            "fly-promote-regmem-to-vectorssa",
            "convert-fly-to-rocdl",
            "canonicalize",
            f"gpu.module(convert-scf-to-cf,cse,"
            f"convert-rocdl-fastmath-ops,"
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
        toolkit_opt = f' toolkit="{_windows_rocm_toolkit()}"' if os.name == "nt" else ""
        binary_fragment = f'gpu-module-to-binary{{format=fatbin{toolkit_opt} opts="{" ".join(bin_cli_opts)}"}}'
        return [*pre_binary_fragments, *binary_prep_fragments], binary_fragment

    def pipeline_fragments(self, *, compile_hints: dict) -> List[str]:
        pre_binary_fragments, binary_fragment = self._pipeline_parts(compile_hints=compile_hints)
        return [*pre_binary_fragments, binary_fragment]

    def external_binary_pipeline_fragments(self, *, compile_hints: dict) -> Tuple[List[str], str]:
        return self._pipeline_parts(compile_hints=compile_hints)

    def lower_compile_hints(self, module, *, compile_hints: dict) -> None:
        """Materialize a scalar waves-per-EU override on kernel entries."""
        if compile_hints.get("maxnreg") is not None:
            raise ValueError(
                "maxnreg is not supported. It only ever reached LLVM through "
                "gpu-module-to-binary opts=, which ROCDL never reads, so it has "
                "been silently inert. The underlying amdgpu-num-vgpr attribute is "
                "deprecated in LLVM ('use amdgpu-waves-per-eu instead') and is "
                "silently doubled on gfx90a/gfx942/gfx950, where it is a combined "
                "VGPR+AGPR budget rather than a VGPR cap. Use waves_per_eu to "
                "target occupancy; see the `llvm` skill to verify it applied."
            )

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
            from ..._mlir import ir as _ir

            wpe_attr = _ir.IntegerAttr.get(_ir.IntegerType.get_signless(32), waves_per_eu)
            for func_op in _iter_gpu_kernel_funcs(module):
                func_op.attributes["rocdl.waves_per_eu"] = wpe_attr

    def gpu_module_targets(self) -> List[str]:
        # Intentionally empty: the ROCm pipeline attaches the authoritative
        # #rocdl.target via `rocdl-attach-target` (binary_prep_fragments),
        # which carries the compile options (O, abi, fast/unsafe-math,
        # wave64, link_libs). Attaching a bare target here as well made
        # gpu-module-to-binary serialize one object per target and select the
        # *bare* one (first object, no offloading handler) — doubling backend
        # work and silently discarding compile options and link_libs
        # (ROCm/FlyDSL#1054).
        return []

    # -- cache / fingerprint ---------------------------------------------

    def native_lib_patterns(self) -> List[str]:
        if sys.platform == "win32":
            return [
                "_mlirDialectsFly*.pyd",
                "_mlirDialectsFly*.dll",
                "Fly*.dll",
                "fly_jit_runtime.dll",
                "mlir_rocm_runtime.dll",
                "_mlirRegisterEverything*.pyd",
            ]
        return [
            "_mlirDialectsFly*.so",
            "libFly*.so",
            "libfly_rocm_aot_runtime.a",
            "libfly_jit_runtime.so",
            "libmlir_rocm_runtime.so",
            "_mlirRegisterEverything*.so",
        ]

    def jit_runtime_lib_basenames(self) -> List[str]:
        if sys.platform == "win32":
            return ["fly_jit_runtime.dll", "mlir_c_runner_utils.dll"]
        return [
            "libfly_jit_runtime.so",
            "libmlir_c_runner_utils.so",
        ]

    @classmethod
    def aot_runtime_config(cls) -> AOTRuntimeConfig:
        if sys.platform == "win32":
            raise NotImplementedError("ROCm AOT export is supported on Linux ELF hosts only")
        return AOTRuntimeConfig(
            archive_basename="libfly_rocm_aot_runtime.a",
            runtime_libraries=("libamdhip64.so",),
            linker_flags=("-lamdhip64", "-pthread", "-ldl"),
        )


def _iter_gpu_kernel_funcs(module):
    """Yield entry ``gpu.func`` ops, excluding device helpers."""
    for top in module.body.operations:
        if top.operation.name != "gpu.module":
            continue
        for op in top.regions[0].blocks[0].operations:
            if op.operation.name == "gpu.func" and ("kernel" in op.attributes or "gpu.kernel" in op.attributes):
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
