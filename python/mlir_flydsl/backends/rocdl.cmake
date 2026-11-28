# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Python bindings of the ROCDL backend, included by python/mlir_flydsl/CMakeLists.txt.

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT FlyPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/"
  TD_FILE dialects/FlyROCDL.td
  SOURCES
    dialects/fly_rocdl.py
  DIALECT_NAME fly_rocdl
  GEN_ENUM_BINDINGS
)

declare_mlir_python_extension(FlyPythonSources.Core.fly_rocdl
  MODULE_NAME _mlirDialectsFlyROCDL
  ADD_TO_PARENT FlyPythonSources
  ROOT_DIR "${PROJECT_SOURCE_DIR}/lib/Bindings/Python"
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    FlyROCDLExtension.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCPIFlyROCDL
)

list(APPEND _fly_backend_upstream_sources MLIRPythonSources.Dialects.rocdl)
list(APPEND _fly_backend_stubgen_modules flydsl._mlir._mlir_libs._mlirDialectsFlyROCDL)
list(APPEND _fly_backend_dialect_libs MLIRFlyROCDLDialect)
list(APPEND _fly_backend_copy_commands
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${MLIR_BINARY_DIR}/python/mlir_flydsl/dialects/_fly_rocdl_ops_gen.py"
    "${MLIR_BINARY_DIR}/python_packages/flydsl/_mlir/dialects/_fly_rocdl_ops_gen.py"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${MLIR_BINARY_DIR}/python/mlir_flydsl/dialects/_fly_rocdl_enum_gen.py"
    "${MLIR_BINARY_DIR}/python_packages/flydsl/_mlir/dialects/_fly_rocdl_enum_gen.py"
)

# AMDGCN device bitcode (ocml/ockl/...) is loaded from <toolkit>/amdgcn/bitcode
# whenever a kernel calls __ocml_* / __ockl_*.  Bundle the files that
# appendStandardLibs() can ask for into the Python package, so `toolkit=` can
# point inside the package instead of at a ROCm install whose location varies
# per container.  addControlVariables() synthesizes the oclc_* control variables
# in-module, so only these four files are ever read.
#
# They land under `_mlir/` because that subtree is already the build-output
# payload of the package: editable installs symlink it and wheels map it
# through package_dir["flydsl._mlir"], so no extra packaging plumbing is needed.
# Override the source with -DFLYDSL_ROCM_BITCODE_DIR=<dir>.
if(NOT FLYDSL_ROCM_BITCODE_DIR)
  find_path(FLYDSL_ROCM_BITCODE_DIR ocml.bc
    HINTS
      "$ENV{ROCM_PATH}/amdgcn/bitcode"
      "$ENV{ROCM_PATH}/lib/llvm/amdgcn/bitcode"
      "$ENV{ROCM_ROOT}/amdgcn/bitcode"
      "$ENV{ROCM_ROOT}/lib/llvm/amdgcn/bitcode"
      "$ENV{ROCM_HOME}/amdgcn/bitcode"
      "$ENV{ROCM_HOME}/lib/llvm/amdgcn/bitcode"
      "/opt/rocm/amdgcn/bitcode"
      "/opt/rocm/lib/llvm/amdgcn/bitcode"
    NO_DEFAULT_PATH)
endif()
if(FLYDSL_ROCM_BITCODE_DIR)
  message(STATUS "Found AMDGCN device bitcode: ${FLYDSL_ROCM_BITCODE_DIR}")
  # ROCm-Device-Libs is NCSA-licensed, which requires the notice to ship with
  # binary redistributions of the bitcode; the notice is vendored because minimal
  # ROCm installs (e.g. the manylinux wheel image) do not ship share/doc.
  set(FLYDSL_ROCM_BITCODE_LICENSE "${PROJECT_SOURCE_DIR}/thirdparty/rocm-device-libs/LICENSE.TXT"
    CACHE FILEPATH "License notice bundled with the device bitcode")
  set(_bitcode_install_dir "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_rocm/amdgcn/bitcode")
  foreach(_bc ocml ockl hip opencl)
    if(EXISTS "${FLYDSL_ROCM_BITCODE_DIR}/${_bc}.bc")
      set(_output "${MLIR_BINARY_DIR}/${_bitcode_install_dir}/${_bc}.bc")
      install(FILES "${_output}" DESTINATION "${_bitcode_install_dir}" COMPONENT FlyPythonModules)
      list(APPEND _fly_backend_copy_commands
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${FLYDSL_ROCM_BITCODE_DIR}/${_bc}.bc"
          "${_output}")
    else()
      message(WARNING "Device bitcode ${_bc}.bc missing from ${FLYDSL_ROCM_BITCODE_DIR}")
    endif()
  endforeach()
  list(APPEND _fly_backend_copy_commands
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${FLYDSL_ROCM_BITCODE_LICENSE}"
      "${MLIR_BINARY_DIR}/${_bitcode_install_dir}/LICENSE.TXT")
  install(FILES "${MLIR_BINARY_DIR}/${_bitcode_install_dir}/LICENSE.TXT"
    DESTINATION "${_bitcode_install_dir}" COMPONENT FlyPythonModules)
else()
  message(WARNING
    "AMDGCN device bitcode (ocml.bc) not found; the Python package will not bundle it. "
    "Kernels calling __ocml_*/__ockl_* will then need FLYDSL_COMPILE_ROCM_PATH or "
    "ROCM_PATH to point at a ROCm install. Set -DFLYDSL_ROCM_BITCODE_DIR=<dir> to bundle it.")
endif()
