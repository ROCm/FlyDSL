#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Resolve the Python interpreter used for MLIR/FlyDSL Python bindings and emit
# the CMake arguments that pin it.
#
# Why this exists: MLIR's MLIRDetectPythonEnv.cmake calls `find_package(Python
# ...)` independently of the project's `find_package(Python3 ...)`, because
# nanobind's CMake package looks for the `Python_` variables rather than the
# `Python3_` ones. That split is intentional. Setting only
# `-DPython3_EXECUTABLE` therefore leaves the bindings free to build against a
# *different* interpreter than the one driving the build. The bindings then land
# under that interpreter's ABI tag (for example `_mlir.cpython-312-*.so` while
# the build runs on 3.14), and every import fails with:
#
#   ImportError: cannot import name 'ir' from ..._mlir (unknown location)
#
# Pinning `Python_EXECUTABLE` alongside `Python3_EXECUTABLE` keeps both
# detection passes on one interpreter. `PYTHON_EXECUTABLE` selects it; when
# unset it defaults to the `python3` first on PATH.
#
# Usage:
#   source scripts/python_interpreter.sh
#   cmake "${FLY_PYTHON_CMAKE_ARGS[@]}"
#
# Sets FLY_PYTHON_EXECUTABLE, FLY_PYTHON_VERSION, FLY_PYTHON_SOABI,
# FLY_NANOBIND_DIR and
# FLY_PYTHON_CMAKE_ARGS. Returns non-zero (without exiting the caller's shell)
# if no usable interpreter can be resolved.

FLY_PYTHON_REQUESTED="${PYTHON_EXECUTABLE:-python3}"
if [[ "${FLY_PYTHON_REQUESTED}" != */* ]]; then
  FLY_PYTHON_REQUESTED="$(command -v "${FLY_PYTHON_REQUESTED}" || true)"
fi
if [[ -z "${FLY_PYTHON_REQUESTED}" ]]; then
  echo "Error: Python interpreter not found on PATH: ${PYTHON_EXECUTABLE:-python3}" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ "${FLY_PYTHON_REQUESTED}" != /* ]]; then
  FLY_PYTHON_REQUESTED="${PWD}/${FLY_PYTHON_REQUESTED}"
fi

# Resolve symlinks so the pin is a stable absolute path rather than a wrapper
# that could be repointed later (/opt symlinks, etc.). Keep a venv launcher
# unresolved: following it would select the base interpreter and lose the venv's
# nanobind installation, which is how scripts/build_wheels.sh builds wheels.
FLY_PYTHON_EXECUTABLE="${FLY_PYTHON_REQUESTED}"
if [[ ! -f "$(dirname "${FLY_PYTHON_EXECUTABLE}")/../pyvenv.cfg" ]]; then
  FLY_PYTHON_EXECUTABLE="$(readlink -f "${FLY_PYTHON_REQUESTED}" 2>/dev/null || true)"
fi
if [[ -z "${FLY_PYTHON_EXECUTABLE}" || ! -f "${FLY_PYTHON_EXECUTABLE}" || ! -x "${FLY_PYTHON_EXECUTABLE}" ]]; then
  echo "Error: Python interpreter is missing, not a file, or not executable: ${FLY_PYTHON_REQUESTED}" >&2
  return 1 2>/dev/null || exit 1
fi

# Running the interpreter proves it is a usable Python, not just an executable
# file (coreutils multicall binaries also accept `-c`). MLIR bindings need 3.10+.
if ! FLY_PYTHON_VERSION="$("${FLY_PYTHON_EXECUTABLE}" -c \
  'import sys; print(".".join(map(str, sys.version_info[:3])) if sys.version_info >= (3, 10) else "")' \
  2>/dev/null)" || [[ ! "${FLY_PYTHON_VERSION}" =~ ^3\.[0-9]+\.[0-9]+$ ]]; then
  echo "Error: not a usable Python 3.10+ interpreter: ${FLY_PYTHON_EXECUTABLE}" >&2
  return 1 2>/dev/null || exit 1
fi

# A free-threaded interpreter reports an ABI suffix such as cpython-314t. That
# is fine, but it must be consistent between both detection passes, so report
# it here instead of letting each CMake module guess.
FLY_PYTHON_SOABI="$("${FLY_PYTHON_EXECUTABLE}" -c \
  'import sysconfig; print(sysconfig.get_config_var("SOABI") or "")' 2>/dev/null || true)"

FLY_PYTHON_CMAKE_ARGS=(
  -DPython_EXECUTABLE="${FLY_PYTHON_EXECUTABLE}"
  -DPython3_EXECUTABLE="${FLY_PYTHON_EXECUTABLE}"
)

# nanobind's CMake directory must come from the same interpreter, otherwise the
# bindings compile against one nanobind and load against another.
FLY_NANOBIND_DIR="$("${FLY_PYTHON_EXECUTABLE}" -c \
  'import nanobind, os; print(os.path.dirname(nanobind.__file__) + "/cmake")' 2>/dev/null || true)"
if [[ -n "${FLY_NANOBIND_DIR}" && -d "${FLY_NANOBIND_DIR}" ]]; then
  FLY_PYTHON_CMAKE_ARGS+=(-Dnanobind_DIR="${FLY_NANOBIND_DIR}")
fi

echo "Python bindings interpreter: ${FLY_PYTHON_EXECUTABLE} (${FLY_PYTHON_VERSION}, SOABI: ${FLY_PYTHON_SOABI:-unknown})"
