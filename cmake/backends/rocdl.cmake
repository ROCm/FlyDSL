# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# ROCDL backend descriptor.
# Self-registers into global properties consumed by downstream CMakeLists.txt,
# and finds the dependencies only this backend uses.
#
# Adding a new backend: copy this file to cmake/backends/<name>.cmake and change
# the property values, and copy python/mlir_flydsl/backends/rocdl.cmake (its
# Python bindings) to python/mlir_flydsl/backends/<name>.cmake;
# -DFLYDSL_BACKENDS can then select <name>.

# TableGen / header subdirectories under include/flydsl/
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_INCLUDE_DIALECT_SUBDIRS "FlyROCDL")
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_INCLUDE_CONVERSION_SUBDIRS "FlyToROCDL")
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_INCLUDE_CONVERSION_SUBDIRS "FlyKtraceToROCDL")

# C++ library subdirectories under lib/
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_LIB_DIALECT_SUBDIRS "FlyROCDL")
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_LIB_CONVERSION_SUBDIRS "FlyToROCDL")
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_LIB_CONVERSION_SUBDIRS "FlyKtraceToROCDL")

# CAPI wrapper subdirectory under lib/CAPI/Dialect/
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_CAPI_SUBDIRS "FlyROCDL")

# Dialect and pass registration hooks under lib/RegisterAll/, and the libraries they use
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_REGISTRATION_SOURCES "ROCDL.cpp")
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_REGISTRATION_LIBS
  MLIRFlyROCDLDialect MLIRFlyToROCDL MLIRFlyKtraceToROCDL)

# JIT and AOT runtime subdirectory under lib/Runtime/
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_RUNTIME_SUBDIRS "ROCm")

# LLD is linked as a library so GPU binary emission never has to locate an
# `ld.lld` executable at runtime (see lib/Conversion/FlyToROCDL/FlyEmitGPUBinary.cpp).
# Always resolve it from the LLVM install that provides MLIR: NO_DEFAULT_PATH
# keeps CMake from picking up a system LLD whose version would not match the
# code generator.  Optional, so an LLVM built without the lld project still works.
if(NOT LLD_DIR)
  set(LLD_DIR ${LLVM_LIBRARY_DIR}/cmake/lld)
endif()
find_package(LLD CONFIG PATHS "${LLD_DIR}" NO_DEFAULT_PATH)
if(LLD_FOUND)
  message(STATUS "Found LLD: ${LLD_DIR} (in-process GPU binary linking)")
  # Separate from LLVM_INCLUDE_DIRS when LLVM is used from its build tree.
  include_directories(SYSTEM ${LLD_INCLUDE_DIRS})
else()
  message(STATUS "LLD not found under ${LLD_DIR}; GPU binary emission falls back to "
                 "spawning ld.lld from the ROCm toolkit path")
endif()
