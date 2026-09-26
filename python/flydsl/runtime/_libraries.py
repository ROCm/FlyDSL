# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Internal discovery of libraries used by exported host objects."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..utils._elf import _dynamic_info


@dataclass(frozen=True)
class _RuntimeLibrary:
    """A distributed runtime library and its dynamic-linker name."""

    path: str
    soname: str


def _distribution_dirs() -> List[Path]:
    import flydsl

    root = Path(flydsl.__file__).resolve().parent
    # ``flydsl.libs`` holds libraries vendored into a repaired wheel.
    candidates = [root / "_mlir" / "_mlir_libs", root.parent / "flydsl.libs"]
    return [d.resolve() for d in candidates if d.is_dir()]


def _find_runtime_libraries(backend: Optional[str] = None) -> Tuple[_RuntimeLibrary, ...]:
    """Find packaged runtime libraries and packaged transitive dependencies."""
    from ..compiler.backends import _get_backend_class

    backend_cls = _get_backend_class(backend)
    dirs = _distribution_dirs()

    def locate(basename: str) -> Optional[Path]:
        for d in dirs:
            if (d / basename).exists():
                return d / basename
        return None

    found: Dict[str, _RuntimeLibrary] = {}
    pending = list(backend_cls._aot_runtime_lib_basenames())
    while pending:
        basename = pending.pop(0)
        path = locate(basename)
        if path is None:
            searched = ", ".join(str(d) for d in dirs) or "(no FlyDSL library directory)"
            raise FileNotFoundError(f"FlyDSL runtime library {basename} not found in {searched}")
        resolved = path.resolve()
        info = _dynamic_info(resolved)
        soname = info.soname or basename
        if soname in found:
            continue
        found[soname] = _RuntimeLibrary(str(resolved), soname)
        pending.extend(dep for dep in info.needed if dep not in found and locate(dep) is not None)
    return tuple(found.values())
