# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Internal discovery of native artifacts used while exporting host objects."""

from pathlib import Path
from typing import List, Optional


def _distribution_dirs() -> List[Path]:
    import flydsl

    root = Path(flydsl.__file__).resolve().parent
    # ``flydsl.libs`` holds libraries vendored into a repaired wheel.
    candidates = [root / "_mlir" / "_mlir_libs", root.parent / "flydsl.libs"]
    return [d.resolve() for d in candidates if d.is_dir()]


def _find_aot_runtime_archive(backend: Optional[str] = None) -> Path:
    """Find the PIC runtime archive embedded into exported host objects."""
    from ..compiler.backends import _get_backend_class

    backend_cls = _get_backend_class(backend)
    dirs = _distribution_dirs()
    basename = backend_cls.aot_runtime_config().archive_basename
    for directory in dirs:
        path = directory / basename
        if path.is_file():
            return path.resolve()
    searched = ", ".join(str(d) for d in dirs) or "(no FlyDSL library directory)"
    raise FileNotFoundError(f"FlyDSL AOT runtime archive {basename} not found in {searched}")
