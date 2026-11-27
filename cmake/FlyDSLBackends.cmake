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
# global properties that downstream CMakeLists.txt files iterate over. A backend
# can be selected as soon as its descriptor exists.

file(GLOB _FLYDSL_BACKENDS_AVAILABLE RELATIVE "${CMAKE_CURRENT_LIST_DIR}/backends"
     "${CMAKE_CURRENT_LIST_DIR}/backends/*.cmake")
list(TRANSFORM _FLYDSL_BACKENDS_AVAILABLE REPLACE "\\.cmake$" "")

set(FLYDSL_BACKENDS "rocdl"
    CACHE STRING "Enabled FlyDSL backend stacks (semicolon-separated)")
set_property(CACHE FLYDSL_BACKENDS PROPERTY STRINGS ${_FLYDSL_BACKENDS_AVAILABLE})

# ---- Validate ----
list(LENGTH FLYDSL_BACKENDS _n_backends)
if(_n_backends EQUAL 0)
  message(FATAL_ERROR "FLYDSL_BACKENDS is empty — at least one backend is required.")
endif()
if(_n_backends GREATER 5)
  message(FATAL_ERROR "FLYDSL_FOR_EACH_BACKEND supports at most 5 backends.")
endif()

foreach(_b ${FLYDSL_BACKENDS})
  if(NOT _b IN_LIST _FLYDSL_BACKENDS_AVAILABLE)
    message(FATAL_ERROR
      "Unknown FLYDSL_BACKENDS entry: '${_b}'. "
      "Backends with a descriptor in cmake/backends/: ${_FLYDSL_BACKENDS_AVAILABLE}")
  endif()
endforeach()

# ---- Global properties for backend self-registration ----
set_property(GLOBAL PROPERTY FLYDSL_BACKEND_INCLUDE_DIALECT_SUBDIRS "")
set_property(GLOBAL PROPERTY FLYDSL_BACKEND_INCLUDE_CONVERSION_SUBDIRS "")
set_property(GLOBAL PROPERTY FLYDSL_BACKEND_LIB_DIALECT_SUBDIRS "")
set_property(GLOBAL PROPERTY FLYDSL_BACKEND_LIB_CONVERSION_SUBDIRS "")
set_property(GLOBAL PROPERTY FLYDSL_BACKEND_CAPI_SUBDIRS "")
set_property(GLOBAL PROPERTY FLYDSL_BACKEND_REGISTRATION_SOURCES "")
set_property(GLOBAL PROPERTY FLYDSL_BACKEND_REGISTRATION_LIBS "")
set_property(GLOBAL PROPERTY FLYDSL_BACKEND_RUNTIME_SUBDIRS "")
# Filled by the runtime subdirectories: targets whose output belongs in _mlir_libs
set_property(GLOBAL PROPERTY FLYDSL_BACKEND_RUNTIME_TARGETS "")

# ---- Include per-backend descriptors ----
foreach(_backend ${FLYDSL_BACKENDS})
  include("${CMAKE_CURRENT_LIST_DIR}/backends/${_backend}.cmake")
endforeach()

# ---- Assemble backend preprocessor defines ----
# lib/RegisterAll iterates over these with FLYDSL_FOR_EACH_BACKEND(MACRO); only
# that library is compiled with them, so changing the backends rebuilds only it.
set(FLYDSL_BACKEND_DEFINITIONS FLYDSL_BACKEND_COUNT=${_n_backends})
set(_backend_index 0)
foreach(_backend ${FLYDSL_BACKENDS})
  list(APPEND FLYDSL_BACKEND_DEFINITIONS FLYDSL_BACKEND_${_backend_index}=${_backend})
  math(EXPR _backend_index "${_backend_index} + 1")
endforeach()
