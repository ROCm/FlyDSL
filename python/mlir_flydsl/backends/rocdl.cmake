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
