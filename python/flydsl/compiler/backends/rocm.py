# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

from ...runtime.device import get_rocm_arch, is_rdna_arch
from ...utils import env
from .base import BaseBackend, GPUTarget

#: FlyDSL wrapper around ``gpu-module-to-binary`` that links the HSA code object
#: with the in-process LLD library instead of spawning ``ld.lld`` from the ROCm
#: toolkit path.  See ``lib/Conversion/FlyToROCDL/FlyEmitGPUBinary.cpp``.
BINARY_PASS_NAME = "fly-emit-gpu-binary"

#: ROCm toolkit root bundled with the package; holds ``amdgcn/bitcode/*.bc``
#: copied in at build time.  Lives under ``_mlir`` because that subtree is the
#: packaged build output.  Absent when CMake could not locate a ROCm install.
BUNDLED_ROCM_PATH = Path(__file__).resolve().parents[2] / "_mlir" / "_rocm"

#: ``True`` when FlyDSL was built with in-process LLD (``FLYDSL_HAS_LLD_LIBRARY``).
#: CMake writes a ``.has_inprocess_lld`` marker next to the bundled bitcode.
#: When False, ``toolkit=`` must not point at the bundled directory (which has
#: no ``llvm/bin/ld.lld``), because upstream would use it for the lld lookup.
HAS_INPROCESS_LLD = (BUNDLED_ROCM_PATH / ".has_inprocess_lld").is_file()


def _has_device_bitcode(root: Path) -> bool:
    return (root / "amdgcn" / "bitcode" / "ocml.bc").is_file()


@lru_cache(maxsize=1)
def rocm_toolkit_path() -> str:
    """Resolve the ROCm root that supplies AMDGCN device bitcode.

    Only ``<root>/amdgcn/bitcode`` is read: ``fly-emit-gpu-binary`` links the HSA
    code object in process, so no ``ld.lld`` lookup is involved.  The bundled
    tree is preferred over the environment so that a container that installs
    ROCm somewhere unexpected still compiles kernels that call ``__ocml_*``.

    When in-process LLD is **not** available (``HAS_INPROCESS_LLD`` is False),
    the bundled directory is skipped because it has no ``llvm/bin/ld.lld`` and
    setting ``toolkit=`` to it would break upstream's linker lookup.

    Returns an empty string when nothing is found, which leaves the upstream
    ``ROCM_PATH`` lookup in place rather than forcing a bad path on it.
    """
    candidates: List[Tuple[str, Path]] = []
    if env.compile.rocm_path:
        candidates.append(("FLYDSL_COMPILE_ROCM_PATH", Path(env.compile.rocm_path)))
    if HAS_INPROCESS_LLD:
        candidates.append(("bundled with flydsl", BUNDLED_ROCM_PATH))
    for var in ("ROCM_PATH", "ROCM_ROOT", "ROCM_HOME"):
        value = os.environ.get(var)
        if value:
            candidates.append((var, Path(value)))

    for _, root in candidates:
        if not _has_device_bitcode(root):
            continue
        path = str(root)
        # MLIR's pass-pipeline parser treats whitespace, commas and braces as
        # structural syntax, so such a path cannot be spelled as an option.
        bad = sorted({ch for ch in path if ch.isspace() or ch in ",{}\"'"})
        if bad:
            raise ValueError(
                f"ROCm toolkit path {path!r} contains unsupported character(s) {bad!r} and cannot be "
                "passed to an MLIR pass option. Point FLYDSL_COMPILE_ROCM_PATH at a path without "
                "whitespace, commas, braces, or quotes."
            )
        return path
    return ""


#: Device libraries ``appendStandardLibs()`` can pull out of a toolkit path.
_DEVICE_BITCODE_FILES = ("ocml.bc", "ockl.bc", "hip.bc", "opencl.bc")


@lru_cache(maxsize=4)
def _device_bitcode_fingerprint(toolkit: str) -> str:
    """Digest the device bitcode reachable from *toolkit*."""
    if not toolkit:
        return "none"
    bitcode_dir = Path(toolkit) / "amdgcn" / "bitcode"
    digest = hashlib.sha256()
    for name in _DEVICE_BITCODE_FILES:
        digest.update(name.encode())
        path = bitcode_dir / name
        if path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


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

    def _pipeline_parts(self, *, compile_hints: dict, external: bool = False) -> Tuple[List[str], str]:
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
        opts = f'opts="{" ".join(bin_cli_opts)}"'
        toolkit = rocm_toolkit_path()
        if toolkit:
            opts = f"toolkit={toolkit} {opts}"
        # The external toolchain drives an upstream mlir-opt that does not know
        # about FlyDSL passes, so that path keeps using gpu-module-to-binary.
        binary_fragment = (
            f"gpu-module-to-binary{{format=fatbin {opts}}}" if external else f"{BINARY_PASS_NAME}{{{opts}}}"
        )
        return [*pre_binary_fragments, *binary_prep_fragments], binary_fragment

    def pipeline_fragments(self, *, compile_hints: dict) -> List[str]:
        pre_binary_fragments, binary_fragment = self._pipeline_parts(compile_hints=compile_hints)
        return [*pre_binary_fragments, binary_fragment]

    def external_binary_pipeline_fragments(self, *, compile_hints: dict) -> Tuple[List[str], str]:
        return self._pipeline_parts(compile_hints=compile_hints, external=True)

    def hash(self) -> str:
        """Fold the device bitcode into the JIT cache key.

        Linked-in ocml/ockl changes the generated code but no FlyDSL shared
        library, so the native-library hashes alone would not invalidate a
        stale cache after a rebuild against a different ROCm.
        """
        toolkit = rocm_toolkit_path()
        return f"{self.target}:{toolkit}:{_device_bitcode_fingerprint(toolkit)}"

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
            from ..._mlir import ir as _ir

            wpe_attr = _ir.IntegerAttr.get(_ir.IntegerType.get_signless(32), waves_per_eu)
            for func_op in _iter_gpu_kernel_funcs(module):
                func_op.attributes["rocdl.waves_per_eu"] = wpe_attr

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
