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
if(_n_backends GREATER 5)
  message(FATAL_ERROR "FLYDSL_FOR_EACH_BACKEND supports at most 5 backends.")
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

# ---- Assemble backend preprocessor defines ----
# C++ code uses FLYDSL_FOR_EACH_BACKEND(MACRO) to iterate over these.
add_compile_definitions(FLYDSL_BACKEND_COUNT=${_n_backends})
set(_backend_index 0)
foreach(_backend ${FLYDSL_BACKENDS})
  add_compile_definitions(FLYDSL_BACKEND_${_backend_index}=${_backend})
  math(EXPR _backend_index "${_backend_index} + 1")
endforeach()
