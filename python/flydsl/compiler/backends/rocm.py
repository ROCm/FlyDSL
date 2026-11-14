# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import hashlib
import os
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence, Tuple

from ...runtime.device import get_rocm_arch, get_warp_size
from ...utils import env
from .base import AOTRuntimeConfig, BaseBackend, GPUTarget

#: FlyDSL wrapper around ``gpu-module-to-binary`` that links the HSA code object
#: with the in-process LLD library instead of spawning ``ld.lld`` from the ROCm
#: toolkit path.  See ``lib/Conversion/FlyToROCDL/FlyEmitGPUBinary.cpp``.
FLY_GPU_BINARY_PASS = "fly-emit-gpu-binary"

#: ROCm toolkit root bundled with the package; holds ``amdgcn/bitcode/*.bc``
#: copied in at build time.  Lives under ``_mlir`` because that subtree is the
#: packaged build output.  Absent when CMake could not locate a ROCm install.
#: Never contains ``llvm/bin/ld.lld``, so it is only safe for in-process linking.
BUNDLED_ROCM_PATH = Path(__file__).resolve().parents[2] / "_mlir" / "_rocm"

#: Device libraries ``appendStandardLibs()`` can pull out of a toolkit path.
_DEVICE_BITCODE_FILES = ("ocml.bc", "ockl.bc", "hip.bc", "opencl.bc")


def _mlir_capi_lib_name() -> str:
    return "FlyPythonCAPI.dll" if os.name == "nt" else "libFlyPythonCAPI.so"


@lru_cache(maxsize=1)
def load_capi_library():
    """ctypes handle to the FlyPythonCAPI library the Python bindings link against.

    Wheels ship only the versioned ``libFlyPythonCAPI.so.<ver>`` (setup.py drops
    the unversioned copy), so either spelling is accepted.
    """
    import ctypes

    libs_dir = BUNDLED_ROCM_PATH.parent / "_mlir_libs"
    name = _mlir_capi_lib_name()
    for path in (libs_dir / name, *sorted(libs_dir.glob(f"{name}.*"))):
        if path.is_file():
            return ctypes.CDLL(str(path))
    raise OSError(f"{name} not found in {libs_dir}")


@lru_cache(maxsize=1)
def _has_inprocess_lld() -> bool:
    """Query the C library for in-process LLD support (``FLYDSL_HAS_LLD_LIBRARY``)."""
    try:
        import ctypes

        lib = load_capi_library()
        lib.flydsl_has_inprocess_lld.restype = ctypes.c_bool
        return lib.flydsl_has_inprocess_lld()
    except Exception as exc:
        # Not silent: falling back means every link spawns the toolkit ld.lld.
        warnings.warn(f"cannot query in-process LLD support ({exc}); GPU binaries link with the ROCm ld.lld")
        return False


#: ``True`` when FlyDSL was built with in-process LLD (``FLYDSL_HAS_LLD_LIBRARY``).
#: When False, ``toolkit=`` must not point at the bundled directory (which has
#: no ``llvm/bin/ld.lld``), because upstream would use it for the lld lookup.
HAS_INPROCESS_LLD = _has_inprocess_lld()


def _lld_basename() -> str:
    return "ld.lld.exe" if os.name == "nt" else "ld.lld"


def _has_device_bitcode(root: Path) -> bool:
    return (root / "amdgcn" / "bitcode" / "ocml.bc").is_file()


def _has_toolkit_lld(root: Path) -> bool:
    """True when *root* has the ``ld.lld`` layout upstream ``gpu-module-to-binary`` expects."""
    return (root / "llvm" / "bin" / _lld_basename()).is_file()


def _validate_toolkit_option_path(path: str) -> str:
    """Reject characters that cannot appear inside a quoted MLIR pass option value."""
    # Quoted ``toolkit="..."`` tolerates whitespace; commas/braces/quotes still break
    # the pipeline parser, and we refuse embedded quotes that would end the value.
    bad = sorted({ch for ch in path if ch in ",{}\"'"})
    if bad:
        raise ValueError(
            f"ROCm toolkit path {path!r} contains unsupported character(s) {bad!r} and cannot be "
            "passed to an MLIR pass option. Point FLYDSL_COMPILE_ROCM_PATH at a path without "
            "commas, braces, or quotes."
        )
    return path


def _stage_lib_llvm_as_canonical(llvm: Path) -> Path:
    """Stage TheRock ``lib/llvm`` into the ``llvm/bin`` + ``amdgcn/bitcode`` layout ROCDL expects.

    Raises ``OSError`` (including ``PermissionError``) when ``~/.flydsl`` is not writable.
    """
    import shutil

    key = hashlib.sha256(os.path.normcase(os.path.abspath(llvm)).encode()).hexdigest()[:16]
    dest = Path.home() / ".flydsl" / "rocm-toolkit" / key
    lld = _lld_basename()
    pairs = [(llvm / "bin" / lld, dest / "llvm" / "bin" / lld)]
    bitcode_dir = llvm / "amdgcn" / "bitcode"
    if bitcode_dir.is_dir():
        pairs.extend((src, dest / "amdgcn" / "bitcode" / src.name) for src in sorted(bitcode_dir.glob("*.bc")))
    for src, dst in pairs:
        if not src.is_file():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            st = src.stat()
            have = dst.stat()
            if (have.st_size, have.st_mtime_ns) == (st.st_size, st.st_mtime_ns):
                continue
        except FileNotFoundError:
            pass
        tmp = dst.with_name(f"{dst.name}.{os.getpid()}.tmp")
        try:
            try:
                os.link(src, tmp)
            except OSError:
                shutil.copy2(src, tmp)
            os.replace(tmp, dst)
        finally:
            tmp.unlink(missing_ok=True)
    return dest


def _normalize_toolkit_root(root: Path, *, require_lld: bool) -> Path | None:
    """Map an install prefix to a toolkit root ROCDL can consume, or None.

    Accepts the canonical ``<root>/{amdgcn/bitcode, llvm/bin/ld.lld}`` layout and
    TheRock's ``<root>/lib/llvm/{amdgcn/bitcode, bin/ld.lld}`` layout.  When
    in-process LLD is used only bitcode matters, so ``lib/llvm`` can be returned
    directly; when upstream must spawn ``ld.lld``, TheRock trees are staged into
    the canonical layout under ``~/.flydsl/rocm-toolkit/``.  A read-only home
    directory skips staging and returns None so later candidates can still win.
    """
    root = root.expanduser()
    if _has_device_bitcode(root) and (not require_lld or _has_toolkit_lld(root)):
        return root

    for llvm in (root / "lib" / "llvm", root):
        if not _has_device_bitcode(llvm):
            continue
        if not require_lld:
            # appendStandardLibs only reads <toolkit>/amdgcn/bitcode.
            return llvm
        if _has_toolkit_lld(llvm):
            return llvm
        if (llvm / "bin" / _lld_basename()).is_file():
            try:
                return _stage_lib_llvm_as_canonical(llvm)
            except OSError:
                continue
    return None


def _mlir_baked_rocm_path() -> Path | None:
    """ROCm root baked into the MLIR/ROCDL build, if the bindings expose one."""
    try:
        from ..._mlir._mlir_libs import _mlirDialectsFlyROCDL

        path = _mlirDialectsFlyROCDL.rocm_path()
    except Exception:
        return None
    return Path(path) if path else None


def _windows_therock_toolkit_root() -> Path | None:
    """TheRock / HIP SDK root staged into the layout ROCDL expects, or None."""
    if os.name != "nt":
        return None
    try:
        from ...runtime.device_runtime.rocm import _rocm_toolkit_root

        return _rocm_toolkit_root()
    except Exception:
        return None


@lru_cache(maxsize=2)
def rocm_toolkit_path(*, for_external: bool = False) -> str:
    """Resolve the ROCm root that supplies AMDGCN device bitcode (and maybe ``ld.lld``).

    With in-process LLD, both the embedded ``fly-emit-gpu-binary`` path and the
    external ``FLYDSL_COMPILE_LLVM_DIR`` path only need bitcode: external stops
    at ``format=isa`` and FlyDSL finishes the link in process. The bundled tree
    is preferred so a container that installs ROCm somewhere unexpected still
    compiles kernels that call ``__ocml_*``.

    Without in-process LLD, both paths fall back to upstream
    ``gpu-module-to-binary{format=fatbin}`` and therefore need ``llvm/bin/ld.lld``;
    the bitcode-only bundle is skipped.

    ``for_external`` is retained for call-site clarity; with the current link
    split it does not change toolkit requirements.

    Install prefixes may use either the canonical ROCDL layout or TheRock's
    ``lib/llvm`` layout; both are accepted.  On Windows, a TheRock / HIP SDK
    install is also staged through ``_rocm_toolkit_root()``.  When no env or
    bundled root works, fall back to MLIR's baked-in ROCm path and ``/opt/rocm``
    so the JIT cache fingerprint is not stuck at ``none``.

    Returns an empty string when nothing is found, which leaves the upstream
    ``ROCM_PATH`` lookup in place rather than forcing a bad path on it.
    """
    _ = for_external  # public keyword; requirements no longer diverge (see docstring).
    require_lld = not HAS_INPROCESS_LLD

    candidates: List[Tuple[str, Path]] = []
    override = env.compile.rocm_path
    if override:
        candidates.append(("FLYDSL_COMPILE_ROCM_PATH", Path(override)))

    if not require_lld:
        candidates.append(("bundled with flydsl", BUNDLED_ROCM_PATH))

    for var in ("ROCM_PATH", "ROCM_ROOT", "ROCM_HOME"):
        value = os.environ.get(var)
        if value:
            candidates.append((var, Path(value)))

    therock = _windows_therock_toolkit_root()
    if therock is not None:
        candidates.append(("Windows TheRock/HIP SDK", therock))

    mlir_rocm = _mlir_baked_rocm_path()
    if mlir_rocm is not None:
        candidates.append(("MLIR baked-in ROCM_PATH", mlir_rocm))
    candidates.append(("/opt/rocm", Path("/opt/rocm")))

    for label, root in candidates:
        normalized = _normalize_toolkit_root(root, require_lld=require_lld)
        if normalized is None:
            if label == "FLYDSL_COMPILE_ROCM_PATH":
                warnings.warn(
                    f"FLYDSL_COMPILE_ROCM_PATH={str(root)!r} is ignored "
                    "(missing amdgcn/bitcode/ocml.bc or llvm/bin/ld.lld); "
                    "falling back to bundled bitcode or ROCM_PATH/ROCM_ROOT/ROCM_HOME.",
                    stacklevel=2,
                )
            continue
        return _validate_toolkit_option_path(str(normalized))
    return ""


@lru_cache(maxsize=4)
def _device_bitcode_fingerprint(toolkit: str) -> str:
    """Digest the device bitcode reachable from *toolkit*.

    Only the four libraries ``appendStandardLibs()`` may pull in are hashed:
    linked-in ocml/ockl changes the ISA; ld.lld itself does not.
    """
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


def _ktrace_tracing_enabled(compile_hints: dict) -> bool:
    if "ktrace" in compile_hints:
        return bool(compile_hints["ktrace"])
    return bool(env.ktrace.enable)


class RocmBackend(BaseBackend):
    """ROCm / AMDGPU compile backend (HIP runtime, ROCDL lowering)."""

    @staticmethod
    def supports_target(target: GPUTarget) -> bool:
        return target.backend == "rocm"

    @classmethod
    def ktrace_backend(cls):
        from ._rocm_ktrace import ROCM_KTRACE_BACKEND

        return ROCM_KTRACE_BACKEND

    @staticmethod
    def detect_target() -> GPUTarget:
        arch = get_rocm_arch()
        if not arch:
            raise RuntimeError("No ROCm GPU architecture detected; set FLYDSL_GPU_ARCH (e.g. gfx1201)")
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

    def _pipeline_parts(self, *, compile_hints: dict, external: bool = False) -> Tuple[List[str], str]:
        # MLIR Chipset::parse only accepts bare gfxNNNN. The gpu.module target keeps
        # the full arch (e.g. gfx1250-strict) so LLVM codegen sees it.
        chipset = self.target.arch.split("-", 1)[0]

        # ROCDL never reads gpu-module-to-binary's opts=, so nothing may be
        # routed through it.
        bin_cli_opts = []

        ktrace_fragments = []
        if _ktrace_tracing_enabled(compile_hints):
            ktrace_fragments = ["convert-fly-ktrace-to-rocdl"]

        pre_binary_fragments = [
            "fly-canonicalize",
            "fly-rocdl-expand-ops",
            "fly-layout-lowering",
            "fly-int-swizzle-simplify",
            "canonicalize",
            "fly-convert-atom-call-to-ssa-form",
            "fly-promote-regmem-to-vectorssa",
            *ktrace_fragments,
            "convert-fly-to-rocdl",
            "canonicalize",
            f"gpu.module(convert-scf-to-cf,cse,"
            f"convert-rocdl-fastmath-ops,"
            f"convert-gpu-to-rocdl{{chipset={chipset} index-bitwidth=0 runtime=HIP use-bare-ptr-memref-call-conv=true}},"
            f"fly-rocdl-cluster-attr)",
        ]
        binary_prep_fragments = [
            # gpu.module bodies already went through convert-scf-to-cf above; only host functions remain.
            "func.func(convert-scf-to-cf)",
            # gpu-to-llvm applies every loaded dialect's ConvertToLLVM patterns.
            "convert-vector-to-llvm",
            "gpu-to-llvm{use-bare-pointers-for-host=true use-bare-pointers-for-kernels=true}",
            "reconcile-unrealized-casts",
            *(
                ["ensure-debug-info-scope-on-llvm-func{emission-kind=LineTablesOnly}"]
                if env.debug.enable_debug_info
                else []
            ),
        ]
        opts = f'opts="{" ".join(bin_cli_opts)}"'
        toolkit = rocm_toolkit_path(for_external=external)
        if toolkit:
            opts = f'toolkit="{toolkit}" {opts}'
        if not external:
            binary_fragment = f"{FLY_GPU_BINARY_PASS}{{{opts}}}"
        elif HAS_INPROCESS_LLD:
            # ROCDL's format=bin/fatbin always spawn ld.lld. Stop at isa so the
            # external tree only does codegen; fly-emit-gpu-binary then
            # assemble+links in process (see run_external_binary_codegen).
            binary_fragment = f"gpu-module-to-binary{{format=isa {opts}}}"
        else:
            binary_fragment = f"gpu-module-to-binary{{format=fatbin {opts}}}"
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

        wpe_str = _normalize_waves_per_eu(waves_per_eu)
        if wpe_str is None:
            return

        with module.context:
            from ..._mlir import ir as _ir

            wpe_attr = _ir.StringAttr.get(wpe_str)
            for func_op in _iter_gpu_kernel_funcs(module):
                func_op.attributes["rocdl.waves_per_eu"] = wpe_attr

    def gpu_module_targets(self, *, compile_hints: dict, link_libs: Sequence[str] = ()) -> List[str]:
        # The one #rocdl.target, carrying every compile option
        from ..._mlir import ir

        flags = [
            flag
            for flag, enabled in (
                ("no_wave64", get_warp_size(self.target.arch) != 64),
                ("fast", compile_hints.get("fast_fp_math")),
                ("unsafe_math", compile_hints.get("unsafe_fp_math")),
            )
            if enabled
        ]
        params = f'O = 2, triple = "amdgcn-amd-amdhsa", chip = "{self.target.arch}", abi = "600"'
        if flags:
            params += f", flags = {{{', '.join(flags)}}}"
        if link_libs:
            params += f", link = {ir.ArrayAttr.get([ir.StringAttr.get(os.fspath(lib)) for lib in link_libs])}"
        return [f"#rocdl.target<{params}>"]

    # -- cache / fingerprint ---------------------------------------------

    def toolkit_fingerprint(self) -> str:
        # Same resolver as the embedded pipeline / backend.hash(), so disk and
        # in-memory cache keys track the bitcode that was actually linked.
        # External codegen is keyed separately via FLYDSL_COMPILE_LLVM_DIR.
        toolkit = rocm_toolkit_path()
        return f"rocm-bitcode:{_device_bitcode_fingerprint(toolkit)}"

    def jit_runtime_lib_basenames(self) -> List[str]:
        if os.name == "nt":
            return ["fly_jit_runtime.dll", "mlir_c_runner_utils.dll"]
        return [
            "libfly_jit_runtime.so",
            "libmlir_c_runner_utils.so",
        ]

    @classmethod
    def aot_runtime_config(cls) -> AOTRuntimeConfig:
        if sys.platform == "win32":
            # The runtime ships as LLVM-dialect IR (fly_rocm_aot_runtime.mlir, not an
            # archive); compiler/_aot.py merges it into each export.
            return AOTRuntimeConfig(
                archive_basename="fly_rocm_aot_runtime.mlir",
                runtime_libraries=("amdhip64.lib",),
                linker_flags=("amdhip64.lib",),
            )
        return AOTRuntimeConfig(
            archive_basename="libfly_rocm_aot_runtime.a",
            runtime_libraries=("libamdhip64.so",),
            linker_flags=("-lamdhip64", "-pthread", "-ldl"),
        )


def _normalize_waves_per_eu(waves_per_eu) -> "str | None":
    """Validate and normalize ``waves_per_eu`` to a ``"min,max"`` string.

    Accepted forms:
      int            – single minimum value, e.g. ``2`` → ``"2"``
      (int, int)     – explicit (min, max), e.g. ``(1, 1)`` → ``"1,1"``
      str            – pre-formatted, e.g. ``"1,1"`` passed through as-is

    Returns *None* when the effective value is zero (no-op).
    """
    if isinstance(waves_per_eu, bool):
        raise TypeError(f"waves_per_eu must be an int, tuple(int, int), or str, got {waves_per_eu!r}")

    if isinstance(waves_per_eu, int):
        if waves_per_eu < 0:
            raise ValueError(f"waves_per_eu must be >= 0, got {waves_per_eu}")
        return None if waves_per_eu == 0 else str(waves_per_eu)

    if isinstance(waves_per_eu, (tuple, list)):
        if len(waves_per_eu) != 2 or not all(isinstance(v, int) and not isinstance(v, bool) for v in waves_per_eu):
            raise TypeError(f"waves_per_eu tuple must be (int, int), got {waves_per_eu!r}")
        lo, hi = waves_per_eu
        if lo < 0 or hi < 0:
            raise ValueError(f"waves_per_eu values must be >= 0, got {waves_per_eu!r}")
        if lo > hi:
            raise ValueError(f"waves_per_eu min ({lo}) must be <= max ({hi})")
        return f"{lo},{hi}"

    if isinstance(waves_per_eu, str):
        # Parse and validate the same way as a tuple so that "2,1" and "abc"
        # are caught here instead of failing deep inside LLVM codegen.
        parts = waves_per_eu.split(",")
        if len(parts) not in (1, 2):
            raise ValueError(f'waves_per_eu string must be "min" or "min,max", got {waves_per_eu!r}')
        try:
            vals = [int(p) for p in parts]
        except ValueError:
            raise ValueError(f"waves_per_eu string must contain integers, got {waves_per_eu!r}") from None
        if any(v < 0 for v in vals):
            raise ValueError(f"waves_per_eu values must be >= 0, got {waves_per_eu!r}")
        if len(vals) == 2 and vals[0] > vals[1]:
            raise ValueError(f"waves_per_eu min ({vals[0]}) must be <= max ({vals[1]})")
        return None if vals == [0] else ",".join(str(v) for v in vals)

    raise TypeError(f"waves_per_eu must be an int, tuple(int, int), or str, got {type(waves_per_eu).__name__}")


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
