# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# NVVM backend descriptor.
# Self-registers into global properties consumed by downstream CMakeLists.txt.
#
# Stage one ships FlyNVVM SM80 atom types, FlyToNVVM conversion, Python
# bindings, and CUDA runtime support. The Python-side properties below keep the
# generated dialect bindings and stubs in sync with enabled backends.

# TableGen / header subdirectories under include/flydsl/
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_INCLUDE_DIALECT_SUBDIRS "FlyNVVM")
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_INCLUDE_CONVERSION_SUBDIRS "FlyToNVVM")

# C++ library subdirectories under lib/
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_LIB_DIALECT_SUBDIRS "FlyNVVM")
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_LIB_CONVERSION_SUBDIRS "FlyToNVVM")

# CAPI wrapper subdirectory under lib/CAPI/Dialect/
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_CAPI_SUBDIRS "FlyNVVM")

# CAPI link targets for _mlirRegisterEverything (EMBED_CAPI_LINK_LIBS)
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_EMBED_CAPI_LIBS "MLIRCPIFlyNVVM")

# Link targets for fly-opt
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_FLYOPT_LINK_LIBS "MLIRCPIFlyNVVM")

# Upstream MLIR dialect sources needed by this backend's Python bindings
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_UPSTREAM_DIALECT_SOURCES
  "MLIRPythonSources.Dialects.nvvm")

# Stubgen modules for this backend
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_STUBGEN_MODULES
  "flydsl._mlir._mlir_libs._mlirDialectsFlyNVVM")

# Convenience boolean for Python CMakeLists gating of NVVM-specific bindings.
set(FLYDSL_HAS_NVVM ON)
