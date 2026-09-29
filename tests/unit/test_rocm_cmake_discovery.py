# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Check HIP package selection without requiring a ROCm installation."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]

_ROCM_CMAKE_DIR = Path(__file__).resolve().parents[2] / "lib" / "Runtime" / "ROCm"


@pytest.mark.parametrize(
    ("rocm_path", "sdk_fails", "expected"),
    [
        (None, False, "sdk"),
        ("explicit", False, "explicit"),
        ("missing", False, "sdk"),
        ("", False, "sdk"),
        (None, True, "system"),
    ],
)
def test_hip_search_order(tmp_path, rocm_path, sdk_fails, expected):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake not available")

    for name in ("system", "sdk", "explicit"):
        config_dir = tmp_path / name / "lib" / "cmake" / "hip"
        config_dir.mkdir(parents=True)
        (config_dir / "hip-config.cmake").write_text("""add_library(hip::host INTERFACE IMPORTED)
add_library(hip::amdhip64 INTERFACE IMPORTED)
""")

    tool_dir = tmp_path / "bin"
    tool_dir.mkdir()
    sdk_tool = tool_dir / "rocm-sdk"
    sdk_tool.write_text("""#!/bin/sh
if [ "$FAKE_SDK_FAIL" = "1" ]; then exit 1; fi
printf "%s\\n" "$FAKE_SDK_ROOT"
""")
    sdk_tool.chmod(0o755)

    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(HipSearchProbe LANGUAGES CXX)\n"
        f'add_subdirectory("{_ROCM_CMAKE_DIR}" runtime)\n'
    )

    env = os.environ.copy()
    env["PATH"] = f"{tool_dir}{os.pathsep}{env.get('PATH', '')}"
    env["FAKE_SDK_ROOT"] = str(tmp_path / "sdk")
    env["FAKE_SDK_FAIL"] = "1" if sdk_fails else "0"
    if rocm_path is None:
        env.pop("ROCM_PATH", None)
    elif rocm_path:
        env["ROCM_PATH"] = str(tmp_path / rocm_path)
    else:
        env["ROCM_PATH"] = ""

    build_dir = tmp_path / "build"
    result = subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(build_dir), f"-DCMAKE_PREFIX_PATH={tmp_path / 'system'}"],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"hip_DIR:PATH={tmp_path / expected / 'lib' / 'cmake' / 'hip'}" in (build_dir / "CMakeCache.txt").read_text()
