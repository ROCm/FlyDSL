# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Both MLIR Python detections must resolve to one interpreter.

MLIRDetectPythonEnv.cmake calls ``find_package(Python ...)`` separately from the
project's ``find_package(Python3 ...)`` because nanobind's CMake package reads
the ``Python_`` variables. Pinning only one lets the bindings be compiled for
one ABI and installed under another, which surfaces as

    ImportError: cannot import name 'ir' from ..._mlir (unknown location)

These tests cover the shell helper that pins both, and the CMake guard that
fails the configure step when they still disagree. No MLIR, network or CMake
build is needed: the helper is sourced directly, and the guard is exercised
with a small standalone CMake project.
"""

import ast
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HELPER = _REPO_ROOT / "scripts" / "python_interpreter.sh"
_GUARD = _REPO_ROOT / "cmake" / "FlyDSLPythonCheck.cmake"

pytestmark = [
    pytest.mark.l0_backend_agnostic,
    pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required"),
]


def _run_helper(env_overrides):
    """Source the helper in a clean shell and print the resulting CMake args."""
    script = 'source "$1" && printf "%s\\n" "${FLY_PYTHON_CMAKE_ARGS[@]}"'
    env = {k: v for k, v in os.environ.items() if k != "PYTHON_EXECUTABLE"}
    env.update(env_overrides)
    return subprocess.run(
        ["bash", "-c", script, "helper", str(_HELPER)],
        capture_output=True,
        text=True,
        env=env,
    )


def test_pins_both_variable_families_to_the_same_interpreter():
    """The bug was pinning Python3_ only; both families must be emitted."""
    result = _run_helper({"PYTHON_EXECUTABLE": sys.executable})

    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    python3_pins = [a for a in args if a.startswith("-DPython3_EXECUTABLE=")]
    python_pins = [a for a in args if a.startswith("-DPython_EXECUTABLE=")]

    assert len(python3_pins) == 1, args
    assert len(python_pins) == 1, args
    # -DPython_EXECUTABLE= must not be satisfied by -DPython3_EXECUTABLE=.
    assert python3_pins[0].split("=", 1)[1] == python_pins[0].split("=", 1)[1]
    assert Path(python_pins[0].split("=", 1)[1]).resolve() == Path(sys.executable).resolve()


def test_nanobind_dir_comes_from_the_pinned_interpreter(tmp_path):
    """nanobind's CMake dir used to be resolved from whatever python3 was on PATH."""
    package = tmp_path / "nanobind"
    (package / "cmake").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    path_python = tmp_path / "python3"
    path_python.write_text("#!/bin/sh\nexit 1\n")
    path_python.chmod(0o755)
    result = _run_helper(
        {
            "PYTHON_EXECUTABLE": sys.executable,
            "PYTHONPATH": str(tmp_path),
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        }
    )

    assert result.returncode == 0, result.stderr
    nanobind_pins = [a for a in result.stdout.splitlines() if a.startswith("-Dnanobind_DIR=")]
    assert len(nanobind_pins) == 1, result.stdout

    pinned = Path(nanobind_pins[0].split("=", 1)[1]).resolve()
    assert pinned == package / "cmake"


def test_reports_version_and_soabi():
    """A free-threaded interpreter's ABI tag must be visible in the log."""
    result = _run_helper({"PYTHON_EXECUTABLE": sys.executable})

    assert result.returncode == 0, result.stderr
    combined = result.stdout + result.stderr
    assert "Python bindings interpreter:" in combined
    assert ".".join(map(str, sys.version_info[:3])) in combined
    assert f"SOABI: {sysconfig.get_config_var('SOABI')}" in combined


def test_missing_interpreter_fails_without_killing_a_set_e_caller():
    """build.sh runs under `set -e`; a bad pin must report, not abort mid-script."""
    result = _run_helper({"PYTHON_EXECUTABLE": "/nonexistent/python3"})

    assert result.returncode != 0
    # Either diagnostic is acceptable: the helper rejects a missing path before
    # it ever runs the interpreter.
    assert "not a usable Python" in result.stderr or "not a file" in result.stderr
    assert "/nonexistent/python3" in result.stderr

    # The helper must return, not exit, so the caller decides what happens.
    survives = subprocess.run(
        [
            "bash",
            "-c",
            'set -e; source "$1" || true; echo CALLER-ALIVE',
            "helper",
            str(_HELPER),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHON_EXECUTABLE": "/nonexistent/python3"},
    )
    assert "CALLER-ALIVE" in survives.stdout


@pytest.mark.parametrize("body", ["exit 0", 'printf "%s\\n" "$@"', "echo 3.14.10; exit 1"])
def test_non_python_executable_is_rejected(tmp_path, body):
    """An executable that is not Python must not be accepted as the pin."""
    fake = tmp_path / "not-python"
    fake.write_text(f"#!/bin/sh\n{body}\n")
    fake.chmod(0o755)

    result = _run_helper({"PYTHON_EXECUTABLE": str(fake)})

    assert result.returncode != 0
    assert "not a usable Python" in result.stderr


def test_venv_launcher_is_preserved_not_collapsed_to_base(tmp_path):
    """build_wheels.sh builds from a venv; resolving its launcher loses the venv."""
    venv = tmp_path / "venv with spaces"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /\n")
    launcher = venv / "bin" / "python"
    launcher.symlink_to(sys.executable)

    result = _run_helper({"PYTHON_EXECUTABLE": str(launcher)})

    assert result.returncode == 0, result.stderr
    pins = [a for a in result.stdout.splitlines() if a.startswith("-DPython3_EXECUTABLE=")]
    assert pins and pins[0].split("=", 1)[1] == str(launcher)


@pytest.mark.parametrize("relative", [True, False])
def test_llvm_build_uses_selected_python_before_changing_directory(tmp_path, relative):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    for name in ("build_llvm.sh", "python_interpreter.sh"):
        shutil.copy(_REPO_ROOT / "scripts" / name, scripts / name)
    (repo / "thirdparty" / "llvm-extensions").mkdir(parents=True)
    (repo / "thirdparty" / "llvm-build-info.json").write_text('{"upstream": {"llvm_hash": "test-ref"}}')
    (tmp_path / "llvm-project").mkdir()
    commands = tmp_path / "commands"
    commands.mkdir()
    log = tmp_path / "commands.log"
    for name, body in {
        "git": "exit 0",
        "pip": 'echo WRONG-PIP >> "$BUILD_LOG"',
        "cmake": 'printf "%s\\n" "$@" >> "$BUILD_LOG"; exit 73',
    }.items():
        command = commands / name
        command.write_text(f"#!/bin/sh\n{body}\n")
        command.chmod(0o755)
    selected = repo / "selected python"
    selected.write_text(
        '#!/bin/sh\nif [ "$1" = "-m" ] && [ "$2" = "pip" ]; then\n'
        '  echo SELECTED-PIP >> "$BUILD_LOG"\n  exit 0\nfi\n'
        f'exec "{sys.executable}" "$@"\n'
    )
    selected.chmod(0o755)

    result = subprocess.run(
        ["bash", str(scripts / "build_llvm.sh")],
        cwd=repo,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHON_EXECUTABLE": "./selected python" if relative else str(selected),
            "PATH": f"{commands}{os.pathsep}{os.environ['PATH']}",
            "BUILD_LOG": str(log),
        },
    )

    assert result.returncode == 73, result.stdout + result.stderr
    commands_run = log.read_text().splitlines()
    assert "SELECTED-PIP" in commands_run
    assert "WRONG-PIP" not in commands_run
    assert f"-DPython_EXECUTABLE={selected}" in commands_run
    assert f"-DPython3_EXECUTABLE={selected}" in commands_run


@pytest.mark.parametrize("override", [None, "/another/python"])
def test_setup_build_uses_packaging_interpreter(tmp_path, monkeypatch, override):
    tree = ast.parse((_REPO_ROOT / "setup.py").read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_assert_embedded_mlir_exists"
    )
    embedded = tmp_path / "_mlir"
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs["env"])
        embedded.mkdir()

    for name in ("FLY_IN_BUILD_SH", "FLY_REBUILD", "PYTHON_EXECUTABLE"):
        monkeypatch.delenv(name, raising=False)
    if override:
        monkeypatch.setenv("PYTHON_EXECUTABLE", override)
    namespace = {
        "os": os,
        "sys": sys,
        "subprocess": SimpleNamespace(run=run),
        "EMBEDDED__MLIR": embedded,
        "REPO_ROOT": _REPO_ROOT,
        "BUILD_DIR_REL": "build-fly",
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "setup.py", "exec"), namespace)
    namespace["_assert_embedded_mlir_exists"]()

    assert len(calls) == 1
    assert calls[0].get("PYTHON_EXECUTABLE") == sys.executable


def test_packaging_metadata_matches_the_helpers_floor():
    """Three sources say 3.10+; the wheel metadata said 3.8 and drifted."""
    tree = ast.parse((_REPO_ROOT / "setup.py").read_text())
    requires = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "python_requires" and isinstance(kw.value, ast.Constant):
                    requires = kw.value.value
    assert requires is not None, "setup.py no longer sets python_requires"

    floor = tuple(int(part) for part in requires.lstrip(">=").split(".")[:2])
    # MLIR's own MLIR_MINIMUM_PYTHON_VERSION is 3.10; the helper enforces it.
    assert floor >= (3, 10), f"python_requires={requires!r} admits interpreters the build rejects"

    readme = (_REPO_ROOT / "README.md").read_text()
    assert "Python 3.10+" in readme or "Python 3.10 or newer" in readme


# ---- cmake/FlyDSLPythonCheck.cmake -------------------------------------------


def _run_cmake_guard(tmp_path, python3_executable, python_executable, allow=False):
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake is required")

    project = tmp_path / "proj"
    (project / "cmake").mkdir(parents=True)
    # Mirror the real tree: the module lives in cmake/ and is included by
    # relative path, exactly as CMakeLists.txt does. Putting the directory on
    # CMAKE_MODULE_PATH instead would not catch a broken include path.
    shutil.copy(_GUARD, project / "cmake" / "FlyDSLPythonCheck.cmake")
    (project / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(guardcheck NONE)\n"
        f'set(Python3_EXECUTABLE "{python3_executable}")\n'
        f'set(Python_EXECUTABLE "{python_executable}")\n'
        f"option(FLY_ALLOW_PYTHON_MISMATCH \"\" {'ON' if allow else 'OFF'})\n"
        "include(cmake/FlyDSLPythonCheck.cmake)\n"
        "fly_check_python_consistency()\n"
        'message(STATUS "guard passed")\n'
    )
    return subprocess.run(
        [cmake, "-S", str(project), "-B", str(tmp_path / "b")],
        capture_output=True,
        text=True,
    )


def test_guard_accepts_one_interpreter_spelled_two_ways(tmp_path):
    """The two passes may name the same interpreter via different paths."""
    resolved = Path(sys.executable).resolve()
    symlink = tmp_path / "python-link"
    symlink.symlink_to(resolved)

    result = _run_cmake_guard(tmp_path, str(resolved), str(symlink))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "guard passed" in result.stdout


def test_guard_fails_when_the_two_detections_disagree(tmp_path):
    """This is the regression: mismatched passes used to configure happily."""
    other = tmp_path / "other-python"
    shutil.copy(sys.executable, other)

    result = _run_cmake_guard(tmp_path, sys.executable, other)

    assert result.returncode != 0
    assert "different interpreters" in result.stdout + result.stderr
    # The error must explain the import failure the user would otherwise hit.
    assert "cannot import name 'ir'" in result.stdout + result.stderr


def test_guard_can_be_downgraded_to_a_warning(tmp_path):
    """Escape hatch for environments that knowingly mix interpreters."""
    other = tmp_path / "other-python"
    shutil.copy(sys.executable, other)

    result = _run_cmake_guard(tmp_path, sys.executable, other, allow=True)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "different interpreters" in result.stdout + result.stderr
    assert "guard passed" in result.stdout
