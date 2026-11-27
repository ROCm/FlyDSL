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


def test_rocm_runtime_is_only_added_for_rocdl_backend():
    runtime = (_REPO_ROOT / "lib" / "Runtime" / "CMakeLists.txt").read_text()
    rocdl = (_REPO_ROOT / "cmake" / "backends" / "rocdl.cmake").read_text()

    assert "get_property(_dirs GLOBAL PROPERTY FLYDSL_BACKEND_RUNTIME_SUBDIRS)" in runtime
    assert 'set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_RUNTIME_SUBDIRS "ROCm")' in rocdl


def test_backend_descriptors_are_loaded_from_selected_backend_list():
    text = (_REPO_ROOT / "cmake" / "FlyDSLBackends.cmake").read_text()

    assert "foreach(_backend ${FLYDSL_BACKENDS})" in text
    assert 'include("${CMAKE_CURRENT_LIST_DIR}/backends/${_backend}.cmake")' in text
    assert "add_compile_definitions(FLYDSL_BACKEND_COUNT=${_n_backends})" in text
    assert "add_compile_definitions(FLYDSL_BACKEND_${_backend_index}=${_backend})" in text


def test_future_backend_descriptor_is_opt_in(tmp_path):
    """A descriptor alone makes a backend selectable, and only selection includes it."""
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake not available")

    cmake_dir = tmp_path / "cmake"
    backend_dir = cmake_dir / "backends"
    backend_dir.mkdir(parents=True)

    shutil.copy(_REPO_ROOT / "cmake" / "FlyDSLBackends.cmake", cmake_dir / "FlyDSLBackends.cmake")

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

    missing = subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(tmp_path / "build-missing"), "-DFLYDSL_BACKENDS=missing"],
        text=True,
        capture_output=True,
    )
    assert missing.returncode != 0
    assert "Unknown FLYDSL_BACKENDS entry: 'missing'" in missing.stderr
