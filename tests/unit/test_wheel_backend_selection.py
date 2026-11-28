# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""A wheel follows the configured install rules when the build directory is reused."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]

_REPO = Path(__file__).resolve().parents[2]


def test_wheel_after_disabling_backend_matches_clean_build(tmp_path):
    # Exercise setup.py and CMake's real component installer without building LLVM.
    shutil.copy(_REPO / "setup.py", tmp_path)
    shutil.copy(_REPO / "MANIFEST.in", tmp_path)
    (tmp_path / "cmake").mkdir()
    shutil.copy(_REPO / "cmake/WriteNativeDigest.cmake", tmp_path / "cmake")
    package = tmp_path / "python/flydsl"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "1.0"\n')
    (tmp_path / "CMakeLists.txt").write_text("""cmake_minimum_required(VERSION 3.20)
project(wheel_fixture NONE)
set(dest python_packages/flydsl/_mlir/_mlir_libs)
set(files __init__.py libcommon.so)
if(ENABLE_EXTRA)
  list(APPEND files libextra.so extra.py extra.pyi device.bc)
endif()
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/${dest}")
foreach(name IN LISTS files)
  set(path "${CMAKE_BINARY_DIR}/${dest}/${name}")
  file(WRITE "${path}" "")
  install(FILES "${path}" DESTINATION "${dest}" COMPONENT FlyPythonModules)
endforeach()
""")

    def wheel(build_dir, enabled):
        subprocess.run(
            ["cmake", "-S", ".", "-B", build_dir, f"-DENABLE_EXTRA={'ON' if enabled else 'OFF'}"],
            cwd=tmp_path,
            check=True,
        )
        subprocess.run(
            [sys.executable, "setup.py", "bdist_wheel"],
            cwd=tmp_path,
            env={**os.environ, "FLY_BUILD_DIR": build_dir, "FLY_REBUILD": "0", "FLY_BUILD_WHEEL": "0"},
            check=True,
        )
        with ZipFile(next((tmp_path / "dist").glob("*.whl"))) as archive:
            return {name: archive.read(name) for name in archive.namelist() if name.startswith("flydsl/")}

    libs = "flydsl/_mlir/_mlir_libs/"
    mixed = wheel("reused", True)
    assert {libs + name for name in ("libextra.so", "extra.py", "extra.pyi", "device.bc")} <= mixed.keys()

    if os.name != "nt":
        (package / "_mlir").symlink_to(tmp_path / "reused/python_packages/flydsl/_mlir", target_is_directory=True)
    switched = wheel("reused", False)
    assert (tmp_path / "reused/python_packages" / libs / "libextra.so").is_file()
    assert switched == wheel("clean", False)
    assert libs + "libcommon.so" in switched
    assert libs + "libextra.so" not in switched
    assert b"libcommon.so" in switched[libs + "_native_digest.py"]
    assert b"libextra.so" not in switched[libs + "_native_digest.py"]

    # python -m build constructs the wheel from an unpacked sdist by default.
    subprocess.run(
        [sys.executable, "setup.py", "sdist"],
        cwd=tmp_path,
        env={**os.environ, "FLY_BUILD_DIR": "reused"},
        check=True,
    )
    unpacked = tmp_path / "unpacked"
    shutil.unpack_archive(next((tmp_path / "dist").glob("*.tar.gz")), unpacked)
    source = next(unpacked.iterdir())
    subprocess.run(
        [sys.executable, "setup.py", "bdist_wheel"],
        cwd=source,
        env={**os.environ, "FLY_BUILD_DIR": "build-fly", "FLY_REBUILD": "0", "FLY_BUILD_WHEEL": "0"},
        check=True,
    )
    with ZipFile(next((source / "dist").glob("*.whl"))) as archive:
        assert {n: archive.read(n) for n in archive.namelist() if n.startswith("flydsl/")} == switched
