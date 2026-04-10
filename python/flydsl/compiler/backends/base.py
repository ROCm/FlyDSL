# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class GPUTarget:
    """Immutable description of a GPU compilation target.

    Modeled after Triton's GPUTarget — carries just enough info for the
    compile pipeline and cache-key logic.
    """

    backend: str  # e.g. "rocm"
    arch: str  # e.g. "gfx942", "gfx950"
    warp_size: int  # 64 for CDNA, 32 for RDNA


class BaseBackend(metaclass=ABCMeta):
    """Abstract compile-backend interface.

    Each backend provides:
    * target detection and arch defaults,
    * MLIR pass-pipeline fragments for lowering Fly IR to device binary,
    * gpu.module target attributes,
    * native-library patterns for toolchain fingerprinting (cache key),
    * runtime shared-library basenames for the JIT ExecutionEngine.
    """

    def __init__(self, target: GPUTarget) -> None:
        assert self.supports_target(target), (
            f"{type(self).__name__} does not support target {target}"
        )
        self.target = target

    # -- target helpers --------------------------------------------------

    @staticmethod
    @abstractmethod
    def supports_target(target: GPUTarget) -> bool:
        """Return True if this backend can compile for *target*."""
        ...

    @staticmethod
    @abstractmethod
    def detect_target() -> GPUTarget:
        """Auto-detect the current device and return a GPUTarget."""
        ...

    @classmethod
    def make_target(cls, arch: str) -> GPUTarget:
        """Build a GPUTarget for an explicit *arch* string.

        Defaults to ``detect_target()`` then overrides the arch.
        Subclasses should override to compute correct ``warp_size`` etc.
        """
        base = cls.detect_target()
        return GPUTarget(backend=base.backend, arch=arch, warp_size=base.warp_size)

    # -- compile pipeline ------------------------------------------------

    @abstractmethod
    def pipeline_fragments(self, *, compile_hints: dict) -> List[str]:
        """Ordered list of MLIR PassManager.parse fragments.

        ``compile_hints`` carries per-kernel knobs such as ``waves_per_eu``
        and ``maxnreg`` (from ``CompilationContext.get_compile_hints()``).
        """
        ...

    @abstractmethod
    def gpu_module_targets(self) -> List[str]:
        """MLIR target attributes for ``create_gpu_module(..., targets=...)``."""
        ...

    # -- cache / fingerprint ---------------------------------------------

    def hash(self) -> str:
        """Return a string uniquely identifying this backend + target.

        Used as part of the JIT cache key (analogous to Triton's
        ``BaseBackend.hash()``).  Subclasses may override to include
        toolchain version info (e.g. ptxas version).
        """
        return f"{self.target}"

    @abstractmethod
    def native_lib_patterns(self) -> List[str]:
        """Glob patterns (relative to ``_mlir/_mlir_libs/``) whose content
        is hashed into the toolchain fingerprint ``_flydsl_key``."""
        ...

    @abstractmethod
    def jit_runtime_lib_basenames(self) -> List[str]:
        """Basenames of shared libraries passed to ``ExecutionEngine``."""
        ...
