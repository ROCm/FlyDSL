# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Python bindings of the NVVM backend, included by python/mlir_flydsl/CMakeLists.txt.

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT FlyPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/"
  TD_FILE dialects/FlyNVVM.td
  SOURCES
    dialects/fly_nvvm.py
  DIALECT_NAME fly_nvvm
)

declare_mlir_python_extension(FlyPythonSources.Core.fly_nvvm
  MODULE_NAME _mlirDialectsFlyNVVM
  ADD_TO_PARENT FlyPythonSources
  ROOT_DIR "${PROJECT_SOURCE_DIR}/lib/Bindings/Python"
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    FlyNVVMExtension.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCPIFlyRegisterEverything  # links the FlyNVVM dialect this extension calls into
)

list(APPEND _fly_backend_upstream_sources MLIRPythonSources.Dialects.nvvm)
list(APPEND _fly_backend_stubgen_modules flydsl._mlir._mlir_libs._mlirDialectsFlyNVVM)
list(APPEND _fly_backend_dialect_libs MLIRFlyNVVMDialect)
list(APPEND _fly_backend_copy_commands
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${MLIR_BINARY_DIR}/python/mlir_flydsl/dialects/_fly_nvvm_ops_gen.py"
    "${MLIR_BINARY_DIR}/python_packages/flydsl/_mlir/dialects/_fly_nvvm_ops_gen.py"
)
