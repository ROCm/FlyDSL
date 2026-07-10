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

        # Occupancy knobs (waves_per_eu, maxnreg) are handled by this backend's
        # apply_occupancy_hints() (see _apply_occupancy_compile_hints below), which
        # lowers them onto the kernel gpu.func as attributes -- the one mechanism
        # the AMDGPU backend honors. They are deliberately NOT also forwarded to
        # gpu-module-to-binary opts= here: --amdgpu-waves-per-eu / --amdgpu-num-vgpr
        # passed that way are silently ignored, so routing them here too would just
        # consume the same hint twice on a dead path.
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

    def apply_occupancy_hints(self, module) -> None:
        _apply_occupancy_compile_hints(module)

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
# waves_per_eu / maxnreg are ROCm/AMDGPU occupancy knobs that only take effect as
# kernel gpu.func attributes (the one mechanism the AMDGPU backend honors);
# passing them via gpu-module-to-binary opts= is silently dropped. This is the
# single home for that knob -> attribute lowering, invoked from
# ``RocmBackend.apply_occupancy_hints`` (which MlirCompiler.compile calls on the
# parsed module before the lowering pipeline). ``ir`` is imported lazily inside
# each helper so this backend module stays importable for backend discovery even
# without the compiled ``_mlir`` bindings.


def _iter_gpu_kernel_funcs(module):
    """Yield the entry-point kernel ``gpu.func`` ops (those carrying the
    ``gpu.kernel`` attribute) inside every ``gpu.module`` of *module*.

    Non-kernel device helpers are skipped -- occupancy hints only apply to entry
    points, and a module may hold both."""
    for top in module.body.operations:
        if top.operation.name != "gpu.module":
            continue
        for op in top.regions[0].blocks[0].operations:
            if op.operation.name == "gpu.func" and "gpu.kernel" in op.attributes:
                yield op


def _set_passthrough(func_op, key: str, value: str) -> None:
    """Set an LLVM ``passthrough`` ``[key, value]`` entry on a kernel func,
    replacing any existing entry with the same key.

    ``convert-gpu-to-rocdl`` copies the ``passthrough`` discardable attr from the
    gpu.func onto the lowered llvm.func, where the LLVM emitter turns each entry
    into a function attribute -- bridging AMDGPU function attributes the ROCDL
    dialect does not translate natively (e.g. ``amdgpu-num-vgpr``). This is
    related to, but distinct from, ``FlyROCDLClusterAttrPass``: that pass copies
    a ``rocdl.cluster_dims`` attr and *synthesizes* the passthrough on the
    llvm.func in a post-lowering pass, whereas here the ``passthrough`` is set
    directly on the gpu.func and carried through as-is. Preserves unrelated
    entries but must not leave a duplicate key (duplicate LLVM function
    attributes are ill-defined).
    """
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
    """Single source of truth for writing occupancy knobs onto a kernel
    ``gpu.func`` as the attributes the ROCDL / LLVM lowering actually honors:

      - ``waves_per_eu`` -> ``rocdl.waves_per_eu`` (translated by convert-gpu-to-rocdl)
      - ``maxnreg``      -> ``amdgpu-num-vgpr`` LLVM passthrough (no native ROCDL attr)

    The autotune compile-hint path (:func:`_apply_occupancy_compile_hints`)
    routes here, and hand-authored kernels that set ``rocdl.waves_per_eu`` via
    ``value_attrs`` land on the same attribute. Passing these knobs through
    ``gpu-module-to-binary opts=`` does NOT work -- the AMDGPU backend drops them
    silently. A value of 0 / None means "leave it to the compiler". Must be
    called with *func_op*'s MLIR context active.
    """
    from ..._mlir import ir

    if waves_per_eu:
        i32 = ir.IntegerType.get_signless(32)
        func_op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(i32, int(waves_per_eu))
    if maxnreg:
        _set_passthrough(func_op, "amdgpu-num-vgpr", str(int(maxnreg)))


def _resolve_occupancy_hint(hint, sym_name: str):
    """Resolve one occupancy compile-hint for a single kernel entry point.

    A hint is either a scalar (applied uniformly to every ``gpu.kernel``) or a
    ``{sym_name: value}`` mapping (per-kernel; kernels absent from the map are
    left to the compiler -- i.e. resolve to ``None``).
    """
    if isinstance(hint, Mapping):
        return hint.get(sym_name)
    return hint


def _unmatched_occupancy_hint_keys(hints: dict, present: set) -> dict:
    """Map ``knob -> [keys]`` for per-kernel occupancy hints whose keys name no
    kernel in *present* (the ``sym_name``s of the module's entry kernels).

    A ``{sym_name: value}`` hint with a stale/typo'd key would otherwise be a
    silent no-op -- the exact "silently ignored" failure this occupancy rework
    set out to eliminate -- so the caller warns on any leftovers.
    """
    out = {}
    for knob, hint in hints.items():
        if isinstance(hint, Mapping):
            missing = sorted(k for k in hint if k not in present)
            if missing:
                out[knob] = missing
    return out


def _apply_occupancy_compile_hints(module) -> None:
    """Lower the autotuner's occupancy compile-hints onto each kernel gpu.func.

    ``Config.compiler_opts()`` surfaces occupancy knobs (``waves_per_eu``,
    ``maxnreg``) as thread-local ``compile_hints``. They only take effect as
    kernel function attributes, so this walks the entry-point kernels and writes
    them via :func:`_set_occupancy_attrs` -- the single occupancy mechanism. The
    dead ``gpu-module-to-binary opts=`` route (silently ignored by the AMDGPU
    backend) is intentionally not used.

    Per-kernel vs uniform: each hint may be a scalar ``int`` -- applied to
    *every* ``gpu.kernel`` entry point (the common case, e.g. single-kernel
    launchers like rmsnorm) -- or a ``{sym_name: int}`` mapping resolved per
    kernel func against its ``sym_name`` (kernels absent from the map are left
    to the compiler; a map key naming no kernel is warned about, not silently
    dropped). The mapping form lets a multi-kernel ``@jit`` launcher scope
    occupancy per entry kernel via ``CompilationContext.compile_hints``. Note:
    ``Config`` can carry such a mapping, but the autotune *search* still
    enumerates scalar knobs only, so autotune-driven *independent* per-kernel
    tuning is not wired end-to-end yet -- the search would need to explore the
    mappings (deferred; see PR #785).
    """
    from ..._mlir import ir
    from ..kernel_function import CompilationContext

    hints = CompilationContext.get_compile_hints()
    waves_per_eu = hints.get("waves_per_eu")
    maxnreg = hints.get("maxnreg")
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
