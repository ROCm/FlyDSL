# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared libraries an exported FlyDSL host object links against."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..utils.elf import dynamic_info


@dataclass(frozen=True)
class RuntimeLibrary:
    """One FlyDSL-distributed runtime library.

    ``path`` is absolute with symlinks resolved; ``soname`` is the name the
    dynamic linker looks up (``DT_SONAME``, or the file name when absent).
    """

    path: str
    soname: str

    def to_dict(self) -> dict:
        return {"path": self.path, "soname": self.soname}


def _distribution_dirs() -> List[Path]:
    import flydsl

    root = Path(flydsl.__file__).resolve().parent
    # ``flydsl.libs`` holds libraries vendored into a repaired wheel.
    candidates = [root / "_mlir" / "_mlir_libs", root.parent / "flydsl.libs"]
    return [d.resolve() for d in candidates if d.is_dir()]


def find_runtime_libraries(backend: Optional[str] = None) -> Tuple[RuntimeLibrary, ...]:
    """Return the runtime libraries needed to link and load objects exported
    by ``flyc.compile(...).export_to_c(...)``.

    The result contains the backend's AOT runtime libraries plus their
    transitive dependencies that ship inside the FlyDSL distribution. System
    libraries (for example ROCm or libc) are not included. ``backend``
    defaults to ``FLYDSL_COMPILE_BACKEND``. Raises ``FileNotFoundError`` when
    a required library is missing from the installation.
    """
    from ..compiler.backends import get_backend_class

    backend_cls = get_backend_class(backend)
    dirs = _distribution_dirs()

    def locate(basename: str) -> Optional[Path]:
        for d in dirs:
            if (d / basename).exists():
                return d / basename
        return None

    found: Dict[str, RuntimeLibrary] = {}
    pending = list(backend_cls.aot_runtime_lib_basenames())
    while pending:
        basename = pending.pop(0)
        path = locate(basename)
        if path is None:
            searched = ", ".join(str(d) for d in dirs) or "(no FlyDSL library directory)"
            raise FileNotFoundError(f"FlyDSL runtime library {basename} not found in {searched}")
        resolved = path.resolve()
        info = dynamic_info(resolved)
        soname = info.soname or basename
        if soname in found:
            continue
        found[soname] = RuntimeLibrary(str(resolved), soname)
        pending.extend(dep for dep in info.needed if dep not in found and locate(dep) is not None)
    return tuple(found.values())
