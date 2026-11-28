# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# NVVM backend descriptor (experimental CUDA backend).
# Self-registers into global properties consumed by downstream CMakeLists.txt.

# Dialect and conversion subdirectories, each under include/flydsl/ (TableGen,
# headers) and lib/ (libraries)
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_DIALECT_SUBDIRS "FlyNVVM")
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_CONVERSION_SUBDIRS "FlyToNVVM")

# Dialect and pass registration hooks under lib/RegisterAll/, and the libraries they use
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_REGISTRATION_SOURCES "NVVM.cpp")
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_REGISTRATION_LIBS
  MLIRFlyNVVMDialect MLIRFlyToNVVM)

# JIT runtime subdirectory under lib/Runtime/
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_RUNTIME_SUBDIRS "CUDA")
