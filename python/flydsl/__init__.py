# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# ruff: noqa: I001

import os
import sys
from pathlib import Path

__version__ = "0.4.0"

if os.name == "nt":
    _dll_directory_handles = []
    _dll_directories = [Path(__file__).parent / "_mlir" / "_mlir_libs"]
    for _root in (os.environ.get("ROCM_PATH"), os.environ.get("HIP_PATH")):
        if _root:
            _dll_directories.append(Path(_root) / "bin")
    for _entry in sys.path:
        _dll_directories.extend(Path(_entry) / sdk / "bin" for sdk in ("_rocm_sdk_core", "_rocm_sdk_devel"))
    for _directory in dict.fromkeys(_dll_directories):
        if _directory.is_dir():
            _dll_directory_handles.append(os.add_dll_directory(str(_directory)))

from .autotune import Config as Config, autotune as autotune

__all__ = [
    "__version__",
]
