# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Backend plugin infrastructure for FlyDSL.
# Inspired by Triton's add_triton_plugin / TRITON_BACKENDS_TUPLE pattern.
#
# Usage:
#   cmake -DFLYDSL_BACKENDS="rocdl" ..
#
# Each backend descriptor (cmake/backends/<name>.cmake) self-registers into
# global properties that downstream CMakeLists.txt files iterate over.

set(FLYDSL_BACKENDS "rocdl"
    CACHE STRING "Enabled FlyDSL backend stacks (semicolon-separated)")
set_property(CACHE FLYDSL_BACKENDS PROPERTY STRINGS rocdl)

# ---- Validate ----
list(LENGTH FLYDSL_BACKENDS _n_backends)
if(_n_backends EQUAL 0)
  message(FATAL_ERROR "FLYDSL_BACKENDS is empty — at least one backend is required.")
endif()
set(_FLYDSL_BACKENDS_ALLOWED rocdl)
foreach(_b ${FLYDSL_BACKENDS})
  if(NOT _b IN_LIST _FLYDSL_BACKENDS_ALLOWED)
    message(FATAL_ERROR
      "Unknown FLYDSL_BACKENDS entry: '${_b}'. "
      "Allowed values: ${_FLYDSL_BACKENDS_ALLOWED}")
  endif()
endforeach()

# ---- Global properties for backend self-registration ----
set_property(GLOBAL PROPERTY FLYDSL_BACKEND_INCLUDE_DIALECT_SUBDIRS "")
set_property(GLOBAL PROPERTY FLYDSL_BACKEND_INCLUDE_CONVERSION_SUBDIRS "")
set_property(GLOBAL PROPERTY FLYDSL_BACKEND_LIB_DIALECT_SUBDIRS "")
set_property(GLOBAL PROPERTY FLYDSL_BACKEND_LIB_CONVERSION_SUBDIRS "")
set_property(GLOBAL PROPERTY FLYDSL_BACKEND_CAPI_SUBDIRS "")
set_property(GLOBAL PROPERTY FLYDSL_BACKEND_EMBED_CAPI_LIBS "")
set_property(GLOBAL PROPERTY FLYDSL_BACKEND_FLYOPT_LINK_LIBS "")
# Python-side properties
set_property(GLOBAL PROPERTY FLYDSL_BACKEND_UPSTREAM_DIALECT_SOURCES "")
set_property(GLOBAL PROPERTY FLYDSL_BACKEND_STUBGEN_MODULES "")

# ---- Include per-backend descriptors ----
foreach(_backend ${FLYDSL_BACKENDS})
  include("${CMAKE_CURRENT_LIST_DIR}/backends/${_backend}.cmake")
endforeach()

# ---- Assemble FLYDSL_BACKENDS_TUPLE preprocessor define ----
# Produces e.g. FLYDSL_BACKENDS_TUPLE=(rocdl) or (rocdl,myvendor)
# C++ code uses FOR_EACH_BACKEND(MACRO, FLYDSL_BACKENDS_TUPLE) to iterate.
string(JOIN "," _backends_joined ${FLYDSL_BACKENDS})
set(FLYDSL_BACKENDS_TUPLE "(${_backends_joined})")
add_compile_definitions(FLYDSL_BACKENDS_TUPLE=${FLYDSL_BACKENDS_TUPLE})
