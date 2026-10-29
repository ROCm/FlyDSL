# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Fail the configure step when MLIR's two Python detections disagree.
#
# MLIRDetectPythonEnv.cmake calls `find_package(Python ...)` separately from the
# project's `find_package(Python3 ...)`, because nanobind's CMake package reads
# the `Python_` variables rather than the `Python3_` ones. Both passes are the
# same bindings detected twice, so they must resolve to one interpreter: when
# they differ, the extension modules are compiled for one ABI and installed
# under another, and every import fails with
#
#   ImportError: cannot import name 'ir' from ..._mlir (unknown location)
#
# Pinning both -DPython_EXECUTABLE and -DPython3_EXECUTABLE (see
# scripts/python_interpreter.sh) keeps them equal. This check catches the case
# where only one is pinned, or where a cache from an older run still holds the
# other, at configure time instead of at import time.
#
# Set FLY_ALLOW_PYTHON_MISMATCH=ON to downgrade the failure to a warning.

function(fly_check_python_consistency)
  if(NOT Python3_EXECUTABLE OR NOT Python_EXECUTABLE)
    message(FATAL_ERROR
      "FlyDSL: MLIR detected Python incompletely (Python3_EXECUTABLE='${Python3_EXECUTABLE}', "
      "Python_EXECUTABLE='${Python_EXECUTABLE}'). Pass both -DPython3_EXECUTABLE and "
      "-DPython_EXECUTABLE, or build through scripts/build.sh, which pins them together.")
  endif()

  # Compare the resolved files rather than the strings: the two passes may spell
  # the same interpreter differently (symlink vs target, /opt link vs real path).
  get_filename_component(_fly_py3_real "${Python3_EXECUTABLE}" REALPATH)
  get_filename_component(_fly_py_real "${Python_EXECUTABLE}" REALPATH)

  if(_fly_py3_real STREQUAL _fly_py_real)
    return()
  endif()

  # A free-threaded interpreter has its own ABI tag (cpython-314t), so a
  # disagreement is not cosmetic: the binding filenames change.
  # Python3_SOABI is the ABI tag MLIR prints and names the extension modules
  # after, so it is the value that actually differs. It is set by FindPython,
  # not by project code, so ask the interpreter directly as a fallback instead
  # of reporting an empty value.
  set(_fly_py3_soabi "${Python3_SOABI}")
  if(NOT _fly_py3_soabi)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c
              "import sysconfig; print(sysconfig.get_config_var('SOABI') or '')"
      OUTPUT_VARIABLE _fly_py3_soabi
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
  endif()

  # string(CONCAT ...) rather than adjacent literals: CMake does not
  # concatenate neighbouring quoted arguments, it makes them a list that would
  # print with ; separators.
  string(CONCAT _detail
    "  Python3_EXECUTABLE=${Python3_EXECUTABLE}\n"
    "  Python_EXECUTABLE=${Python_EXECUTABLE}\n"
    "  Python3_SOABI=${_fly_py3_soabi}")

  if(FLY_ALLOW_PYTHON_MISMATCH)
    message(WARNING
      "FlyDSL: MLIR's two Python detections resolved different interpreters. "
      "Extensions will be built for one ABI and may be unimportable from the other.\n"
      "${_detail}")
  else()
    message(FATAL_ERROR
      "FlyDSL: MLIR's two Python detections resolved different interpreters.\n"
      "${_detail}\n"
      "The bindings would be compiled for one ABI and installed under another, so imports "
      "fail with \"cannot import name 'ir'\". MLIR detects Python twice because nanobind "
      "reads the Python_ variables and the project uses Python3_. Pin both to one "
      "interpreter (-DPython3_EXECUTABLE and -DPython_EXECUTABLE), or build through "
      "scripts/build.sh. Pass -DFLY_ALLOW_PYTHON_MISMATCH=ON to downgrade this to a warning.")
  endif()
endfunction()
