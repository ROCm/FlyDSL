# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""CMake backend default and dependency guardrails."""

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cmake_default_backend_stays_rocdl():
    text = (_REPO_ROOT / "cmake" / "FlyDSLBackends.cmake").read_text()

    assert 'set(FLYDSL_BACKENDS "rocdl"' in text
    assert "set_property(CACHE FLYDSL_BACKENDS PROPERTY STRINGS rocdl)" in text
    assert "set(_FLYDSL_BACKENDS_ALLOWED rocdl)" in text


def test_rocm_runtime_is_only_added_for_rocdl_backend():
    text = (_REPO_ROOT / "lib" / "Runtime" / "CMakeLists.txt").read_text()

    assert 'if("rocdl" IN_LIST FLYDSL_BACKENDS)' in text
    assert "add_subdirectory(ROCm)" in text


def test_rocm_core_wheel_hip_fallback(tmp_path):
    """The core wheel can supply FlyDSL's host-side HIP build dependencies."""
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake not available")

    core = tmp_path / "site-packages" / "_rocm_sdk_core"
    header = core / "include" / "hip" / "hip_runtime.h"
    runtime = core / "lib" / "libamdhip64.so.7"
    header.parent.mkdir(parents=True)
    runtime.parent.mkdir(parents=True)
    header.touch()
    runtime.touch()

    fake_python = tmp_path / "python"
    fake_python.write_text(f'#!/bin/sh\nprintf "%s\\n" "{core}"\n')
    fake_python.chmod(0o755)

    source = tmp_path / "source"
    source.mkdir()
    module = _REPO_ROOT / "cmake" / "FlyDSLFindHip.cmake"
    (source / "CMakeLists.txt").write_text(
        "\n".join(
            [
                "cmake_minimum_required(VERSION 3.20)",
                "project(FlyDSLHipCoreFallback NONE)",
                f'set(Python3_EXECUTABLE "{fake_python}")',
                f'include("{module}")',
                "_flydsl_find_hip_core(found host_target runtime_target)",
                'if(NOT found OR NOT host_target STREQUAL "FlyDSLHipCoreHeaders")',
                '  message(FATAL_ERROR "core HIP fallback was not selected")',
                "endif()",
                'get_target_property(include_dirs "${host_target}" INTERFACE_INCLUDE_DIRECTORIES)',
                f'if(NOT include_dirs STREQUAL "{core}/include")',
                '  message(FATAL_ERROR "wrong HIP include directory: ${include_dirs}")',
                "endif()",
                'get_target_property(runtime_location "${runtime_target}" IMPORTED_LOCATION)',
                f'if(NOT runtime_location STREQUAL "{runtime}")',
                '  message(FATAL_ERROR "wrong HIP runtime: ${runtime_location}")',
                "endif()",
                "",
            ]
        )
    )

    subprocess.run(
        [cmake, "-S", str(source), "-B", str(tmp_path / "build")],
        check=True,
        text=True,
        capture_output=True,
    )


def test_standard_hip_package_precedes_core_wheel_fallback(tmp_path):
    """Keep traditional ROCm installations on their exported HIP targets."""
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake not available")

    rocm_root = tmp_path / "rocm-7.2"
    config_dir = rocm_root / "lib" / "cmake" / "hip"
    config_dir.mkdir(parents=True)
    (config_dir / "hip-config.cmake").write_text(
        "add_library(hip::host INTERFACE IMPORTED)\n"
        "add_library(hip::amdhip64 INTERFACE IMPORTED)\n"
    )

    source = tmp_path / "source"
    source.mkdir()
    module = _REPO_ROOT / "cmake" / "FlyDSLFindHip.cmake"
    (source / "CMakeLists.txt").write_text(
        "\n".join(
            [
                "cmake_minimum_required(VERSION 3.20)",
                "project(FlyDSLStandardHip NONE)",
                f'set(ENV{{ROCM_PATH}} "{rocm_root}")',
                f'include("{module}")',
                "flydsl_find_hip(host_target runtime_target)",
                'if(NOT host_target STREQUAL "hip::host")',
                '  message(FATAL_ERROR "standard HIP host target was not selected")',
                "endif()",
                'if(NOT runtime_target STREQUAL "hip::amdhip64")',
                '  message(FATAL_ERROR "standard HIP runtime target was not selected")',
                "endif()",
                "",
            ]
        )
    )

    subprocess.run(
        [cmake, "-S", str(source), "-B", str(tmp_path / "build")],
        check=True,
        text=True,
        capture_output=True,
    )


def test_backend_descriptors_are_loaded_from_selected_backend_list():
    text = (_REPO_ROOT / "cmake" / "FlyDSLBackends.cmake").read_text()

    assert "foreach(_backend ${FLYDSL_BACKENDS})" in text
    assert 'include("${CMAKE_CURRENT_LIST_DIR}/backends/${_backend}.cmake")' in text
    assert "add_compile_definitions(FLYDSL_BACKEND_COUNT=${_n_backends})" in text
    assert "add_compile_definitions(FLYDSL_BACKEND_${_backend_index}=${_backend})" in text


def test_future_backend_descriptor_is_opt_in(tmp_path):
    """A future backend should be legal only when explicitly selected."""
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake not available")

    cmake_dir = tmp_path / "cmake"
    backend_dir = cmake_dir / "backends"
    backend_dir.mkdir(parents=True)

    text = (_REPO_ROOT / "cmake" / "FlyDSLBackends.cmake").read_text()
    text = text.replace(
        "set_property(CACHE FLYDSL_BACKENDS PROPERTY STRINGS rocdl)",
        "set_property(CACHE FLYDSL_BACKENDS PROPERTY STRINGS rocdl dummy)",
    )
    text = text.replace(
        "set(_FLYDSL_BACKENDS_ALLOWED rocdl)",
        "set(_FLYDSL_BACKENDS_ALLOWED rocdl dummy)",
    )
    (cmake_dir / "FlyDSLBackends.cmake").write_text(text)

    (backend_dir / "rocdl.cmake").write_text('set(GUARDRAIL_SELECTED_ROCDL ON CACHE BOOL "" FORCE)\n')
    (backend_dir / "dummy.cmake").write_text('set(GUARDRAIL_SELECTED_DUMMY ON CACHE BOOL "" FORCE)\n')
    (tmp_path / "CMakeLists.txt").write_text(
        "\n".join(
            [
                "cmake_minimum_required(VERSION 3.20)",
                "project(FlyDSLBackendSelectionGuardrail NONE)",
                'include("${CMAKE_CURRENT_LIST_DIR}/cmake/FlyDSLBackends.cmake")',
                'if(FLYDSL_BACKENDS STREQUAL "dummy" AND GUARDRAIL_SELECTED_ROCDL)',
                '  message(FATAL_ERROR "rocdl descriptor was included for dummy-only build")',
                "endif()",
                'if(FLYDSL_BACKENDS STREQUAL "rocdl" AND GUARDRAIL_SELECTED_DUMMY)',
                '  message(FATAL_ERROR "dummy descriptor was included for default build")',
                "endif()",
                "",
            ]
        )
    )

    default_build = tmp_path / "build-default"
    subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(default_build)],
        check=True,
        text=True,
        capture_output=True,
    )
    default_cache = (default_build / "CMakeCache.txt").read_text()
    assert "FLYDSL_BACKENDS:STRING=rocdl" in default_cache
    assert "GUARDRAIL_SELECTED_ROCDL:BOOL=ON" in default_cache
    assert "GUARDRAIL_SELECTED_DUMMY" not in default_cache

    dummy_build = tmp_path / "build-dummy"
    subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(dummy_build), "-DFLYDSL_BACKENDS=dummy"],
        check=True,
        text=True,
        capture_output=True,
    )
    dummy_cache = (dummy_build / "CMakeCache.txt").read_text()
    assert "FLYDSL_BACKENDS:STRING=dummy" in dummy_cache
    assert "GUARDRAIL_SELECTED_DUMMY:BOOL=ON" in dummy_cache
    assert "GUARDRAIL_SELECTED_ROCDL" not in dummy_cache
