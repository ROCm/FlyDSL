# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

# Locate the HIP targets needed by FlyDSL's host-side ROCm runtime.
#
# A complete ROCm installation exports hip::host and hip::amdhip64 through
# hip-config.cmake.  The lightweight rocm-sdk core Python wheel deliberately
# does not ship that CMake package, but it does contain the headers and
# libamdhip64 needed by the runtime wrapper.  Keep the normal HIP package
# discovery authoritative and use the core wheel only as a last-resort build
# fallback.

function(_flydsl_find_hip_core out_found out_host_target out_runtime_target)
  set(_flydsl_core_root "")
  if(Python3_EXECUTABLE)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c
        "from pathlib import Path; import _rocm_sdk_core; print(Path(_rocm_sdk_core.__file__).resolve().parent)"
      RESULT_VARIABLE _flydsl_core_result
      OUTPUT_VARIABLE _flydsl_core_root
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
  endif()

  if(NOT _flydsl_core_result STREQUAL "0" OR NOT _flydsl_core_root)
    set(${out_found} FALSE PARENT_SCOPE)
    return()
  endif()

  set(_flydsl_core_header "${_flydsl_core_root}/include/hip/hip_runtime.h")
  set(_flydsl_core_runtime_candidates "${_flydsl_core_root}/lib/libamdhip64.so")
  file(GLOB _flydsl_core_versioned_runtimes
    LIST_DIRECTORIES FALSE
    "${_flydsl_core_root}/lib/libamdhip64.so.*"
  )
  list(APPEND _flydsl_core_runtime_candidates ${_flydsl_core_versioned_runtimes})

  set(_flydsl_core_runtime "")
  foreach(_flydsl_candidate IN LISTS _flydsl_core_runtime_candidates)
    if(EXISTS "${_flydsl_candidate}")
      set(_flydsl_core_runtime "${_flydsl_candidate}")
      break()
    endif()
  endforeach()

  if(NOT EXISTS "${_flydsl_core_header}" OR NOT _flydsl_core_runtime)
    set(${out_found} FALSE PARENT_SCOPE)
    return()
  endif()

  add_library(FlyDSLHipCoreHeaders INTERFACE)
  target_include_directories(FlyDSLHipCoreHeaders INTERFACE "${_flydsl_core_root}/include")
  target_compile_definitions(FlyDSLHipCoreHeaders INTERFACE __HIP_PLATFORM_AMD__)

  add_library(FlyDSLHipCoreRuntime SHARED IMPORTED)
  set_target_properties(FlyDSLHipCoreRuntime PROPERTIES
    IMPORTED_LOCATION "${_flydsl_core_runtime}"
  )

  message(STATUS "Using HIP headers and runtime from rocm-sdk core: ${_flydsl_core_root}")
  set(${out_found} TRUE PARENT_SCOPE)
  set(${out_host_target} FlyDSLHipCoreHeaders PARENT_SCOPE)
  set(${out_runtime_target} FlyDSLHipCoreRuntime PARENT_SCOPE)
endfunction()

function(flydsl_find_hip out_host_target out_runtime_target)
  # An explicitly selected ROCm installation takes precedence.  A cached
  # hip_DIR remains authoritative through find_package's normal config-mode
  # lookup rules.
  if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
    find_package(hip QUIET CONFIG PATHS "$ENV{ROCM_PATH}" NO_DEFAULT_PATH)
  endif()

  # A complete pip ROCm installation reports the root containing
  # lib/cmake/hip/hip-config.cmake.  The core-only package returns an error,
  # in which case discovery continues to the legacy installation paths and
  # finally to the core-wheel fallback below.
  if(NOT hip_FOUND)
    execute_process(
      COMMAND rocm-sdk path --root
      OUTPUT_VARIABLE _rocm_sdk_root
      RESULT_VARIABLE _rocm_sdk_result
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(_rocm_sdk_result STREQUAL "0" AND _rocm_sdk_root)
      # HIP's config uses ROCM_PATH to find its dependencies.  A stale or empty
      # value must not redirect those lookups away from the selected SDK.
      set(_rocm_path_was_set FALSE)
      if(DEFINED ENV{ROCM_PATH})
        set(_rocm_path_was_set TRUE)
        set(_rocm_original_path "$ENV{ROCM_PATH}")
      endif()
      set(ENV{ROCM_PATH} "${_rocm_sdk_root}")
      find_package(hip QUIET CONFIG PATHS "${_rocm_sdk_root}" NO_DEFAULT_PATH)
      if(_rocm_path_was_set)
        set(ENV{ROCM_PATH} "${_rocm_original_path}")
      else()
        unset(ENV{ROCM_PATH})
      endif()
    endif()
  endif()

  # Preserve support for traditional ROCm installations such as ROCm 7.2,
  # whose HIP package lives under an /opt/rocm* prefix.
  if(NOT hip_FOUND)
    file(GLOB _rocm_paths LIST_DIRECTORIES true "/opt/rocm*")
    list(SORT _rocm_paths ORDER DESCENDING)
    find_package(hip QUIET CONFIG PATHS ${_rocm_paths})
  endif()

  if(hip_FOUND)
    if(NOT TARGET hip::host OR NOT TARGET hip::amdhip64)
      message(FATAL_ERROR "The selected HIP package does not define hip::host and hip::amdhip64.")
    endif()
    set(${out_host_target} hip::host PARENT_SCOPE)
    set(${out_runtime_target} hip::amdhip64 PARENT_SCOPE)
    return()
  endif()

  _flydsl_find_hip_core(_flydsl_core_hip_found _flydsl_core_host _flydsl_core_runtime)
  if(_flydsl_core_hip_found)
    set(${out_host_target} "${_flydsl_core_host}" PARENT_SCOPE)
    set(${out_runtime_target} "${_flydsl_core_runtime}" PARENT_SCOPE)
    return()
  endif()

  message(FATAL_ERROR
    "Could not find HIP. Set hip_DIR or ROCM_PATH to a complete ROCm installation, "
    "install rocm[devel], or use a Python environment containing _rocm_sdk_core "
    "with include/hip/hip_runtime.h and lib/libamdhip64.so.*."
  )
endfunction()
