# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from collections.abc import Mapping
from typing import List, Tuple

from ...runtime.device import get_rocm_arch, is_rdna_arch
from ...utils import env, log
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

        # Keep occupancy out of opts=: AMDGPU ignores those flags here.  They are
        # lowered onto gpu.func attributes before this pipeline runs.
        bin_cli_opts = []
        if env.debug.enable_debug_info:
            bin_cli_opts.append("-g")

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

    def lower_occupancy_compile_hints(self, module, *, compile_hints: dict) -> None:
        """Materialize ROCm occupancy compile hints before ROCDL lowering.

        ``waves_per_eu`` and ``maxnreg`` only affect AMDGPU codegen as kernel
        function attrs, so this must run before ``convert-gpu-to-rocdl``.
        """
        _lower_occupancy_compile_hints(module, compile_hints=compile_hints)

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


# -- occupancy compile-hint lowering ------------------------------------------
#
# Config/compiler hints enter here explicitly from MlirCompiler.compile.  Keep
# the AMDGPU knob lowering in one place, and import ``ir`` lazily so backend
# discovery works without compiled MLIR bindings.


def _iter_gpu_kernel_funcs(module):
    """Yield entry-point ``gpu.func`` ops and skip device helpers."""
    for top in module.body.operations:
        if top.operation.name != "gpu.module":
            continue
        for op in top.regions[0].blocks[0].operations:
            if op.operation.name == "gpu.func" and "gpu.kernel" in op.attributes:
                yield op


def _set_passthrough(func_op, key: str, value: str) -> None:
    """Set or replace one LLVM passthrough function attribute."""
    from ..._mlir import ir

    def _entry_key(e):
        # Key/value entries are 2-element arrays [key, value]; unit attributes
        # (e.g. "nounwind") are bare strings and have no key to match.
        try:
            arr = ir.ArrayAttr(e)
            return ir.StringAttr(arr[0]).value if len(arr) else None
        except (ValueError, TypeError):
            return None

    entry = ir.ArrayAttr.get([ir.StringAttr.get(key), ir.StringAttr.get(value)])
    existing = func_op.attributes["passthrough"] if "passthrough" in func_op.attributes else None
    kept = [e for e in existing if _entry_key(e) != key] if existing is not None else []
    func_op.attributes["passthrough"] = ir.ArrayAttr.get(kept + [entry])


def _set_occupancy_attrs(func_op, *, waves_per_eu=None, maxnreg=None) -> None:
    """Write ROCm occupancy knobs onto one kernel ``gpu.func``.

      - ``waves_per_eu`` -> ``rocdl.waves_per_eu`` (translated by convert-gpu-to-rocdl)
      - ``maxnreg``      -> ``amdgpu-num-vgpr`` LLVM passthrough (no native ROCDL attr)

    ``None`` / ``0`` means "leave it to the compiler".
    """
    from ..._mlir import ir

    if waves_per_eu:
        i32 = ir.IntegerType.get_signless(32)
        func_op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(i32, int(waves_per_eu))
    if maxnreg:
        _set_passthrough(func_op, "amdgpu-num-vgpr", str(int(maxnreg)))


def _resolve_occupancy_hint(hint, sym_name: str):
    """Resolve a scalar or ``{sym_name: value}`` hint for one kernel."""
    if isinstance(hint, Mapping):
        return hint.get(sym_name)
    return hint


def _unmatched_occupancy_hint_keys(hints: dict, present: set) -> dict:
    """Return stale per-kernel hint keys by knob name."""
    out = {}
    for knob, hint in hints.items():
        if isinstance(hint, Mapping):
            missing = sorted(k for k in hint if k not in present)
            if missing:
                out[knob] = missing
    return out


def _lower_occupancy_compile_hints(module, *, compile_hints: dict) -> None:
    """Lower scalar or per-kernel occupancy hints onto entry kernels."""
    from ..._mlir import ir

    waves_per_eu = compile_hints.get("waves_per_eu")
    maxnreg = compile_hints.get("maxnreg")
    if not waves_per_eu and not maxnreg:
        return
    seen = set()
    with module.context:
        for func_op in _iter_gpu_kernel_funcs(module):
            sym_name = ir.StringAttr(func_op.attributes["sym_name"]).value
            seen.add(sym_name)
            _set_occupancy_attrs(
                func_op,
                waves_per_eu=_resolve_occupancy_hint(waves_per_eu, sym_name),
                maxnreg=_resolve_occupancy_hint(maxnreg, sym_name),
            )
    for knob, missing in _unmatched_occupancy_hint_keys(
        {"waves_per_eu": waves_per_eu, "maxnreg": maxnreg}, seen
    ).items():
        log().warning(
            "occupancy hint %r targets kernel(s) %s not present in the module (kernels: %s); "
            "those entries were ignored -- check for a typo or stale sym_name.",
            knob,
            missing,
            sorted(seen),
        )
