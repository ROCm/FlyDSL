# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# ROCDL backend descriptor.
# Self-registers into global properties consumed by downstream CMakeLists.txt.
#
# Adding a new backend: copy this file, change the property values, and add the
# new name to _FLYDSL_BACKENDS_ALLOWED in cmake/FlyDSLBackends.cmake.

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

# Upstream MLIR dialect sources needed by this backend's Python bindings
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_UPSTREAM_DIALECT_SOURCES
  "MLIRPythonSources.Dialects.rocdl")

# Stubgen modules for this backend
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_STUBGEN_MODULES
  "flydsl._mlir._mlir_libs._mlirDialectsFlyROCDL")

# Convenience boolean for Python CMakeLists gating of ROCDL-specific bindings
# (dialect bindings, nanobind extension, tablegen copy are still ROCDL-specific)
set(FLYDSL_HAS_ROCDL ON)
