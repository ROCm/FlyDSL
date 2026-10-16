# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# FlyJitRuntime — thin ROCm runtime with GPU module caching.

# Prefer an explicitly selected ROCm, then the SDK in the active environment.
# NO_DEFAULT_PATH excludes default search locations; a cached hip_DIR still wins.
if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
  find_package(hip QUIET CONFIG PATHS "$ENV{ROCM_PATH}" NO_DEFAULT_PATH)
endif()

if(NOT hip_FOUND)
  execute_process(COMMAND rocm-sdk path --root
                  OUTPUT_VARIABLE _rocm_sdk_root
                  RESULT_VARIABLE _rocm_sdk_result
                  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(_rocm_sdk_result STREQUAL "0" AND _rocm_sdk_root)
    # HIP's config uses ROCM_PATH to find its dependencies. A stale or empty
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

if(NOT hip_FOUND)
  file(GLOB _rocm_paths LIST_DIRECTORIES true "/opt/rocm*")
  list(SORT _rocm_paths ORDER DESCENDING)
  find_package(hip QUIET CONFIG PATHS ${_rocm_paths})
endif()

# rocm-sdk core wheels contain the headers and runtime required by FlyDSL's
# host-side wrappers, but deliberately do not ship hip-config.cmake. Use them
# only after all standard HIP package locations have been exhausted.
if(NOT hip_FOUND AND Python3_EXECUTABLE)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
      "from pathlib import Path; import _rocm_sdk_core; print(Path(_rocm_sdk_core.__file__).resolve().parent)"
    RESULT_VARIABLE _flydsl_hip_core_result
    OUTPUT_VARIABLE _flydsl_hip_core_root
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(_flydsl_hip_core_result STREQUAL "0" AND _flydsl_hip_core_root)
    set(_flydsl_hip_header "${_flydsl_hip_core_root}/include/hip/hip_runtime.h")
    set(_flydsl_hip_runtime_candidates "${_flydsl_hip_core_root}/lib/libamdhip64.so")
    file(GLOB _flydsl_hip_versioned_runtimes
      LIST_DIRECTORIES FALSE
      "${_flydsl_hip_core_root}/lib/libamdhip64.so.*"
    )
    list(APPEND _flydsl_hip_runtime_candidates ${_flydsl_hip_versioned_runtimes})
    foreach(_flydsl_hip_candidate IN LISTS _flydsl_hip_runtime_candidates)
      if(EXISTS "${_flydsl_hip_candidate}")
        set(_flydsl_hip_runtime "${_flydsl_hip_candidate}")
        break()
      endif()
    endforeach()

    if(EXISTS "${_flydsl_hip_header}" AND _flydsl_hip_runtime)
      add_library(hip::host INTERFACE IMPORTED)
      set_target_properties(hip::host PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_flydsl_hip_core_root}/include"
        INTERFACE_COMPILE_DEFINITIONS "__HIP_PLATFORM_AMD__"
      )
      add_library(hip::amdhip64 SHARED IMPORTED)
      set_target_properties(hip::amdhip64 PROPERTIES
        IMPORTED_LOCATION "${_flydsl_hip_runtime}"
      )
      set(hip_FOUND TRUE)
      message(STATUS "Using HIP headers and runtime from rocm-sdk core: ${_flydsl_hip_core_root}")
    endif()
  endif()
endif()

if(NOT hip_FOUND)
  message(FATAL_ERROR
    "Could not find HIP. Set hip_DIR or ROCM_PATH to a complete ROCm installation, "
    "install rocm[devel], or use a Python environment containing _rocm_sdk_core "
    "with include/hip/hip_runtime.h and lib/libamdhip64.so.*."
  )
endif()

set(_fly_rocm_runtime_common
  FlyRocmAotRuntime.cpp
  FlyRocmRuntimeError.cpp
)

add_library(FlyJitRuntime SHARED
  FlyRocmRuntimeWrappers.cpp
  ${_fly_rocm_runtime_common}
)
target_include_directories(FlyJitRuntime PRIVATE ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_features(FlyJitRuntime PRIVATE cxx_std_17)
target_link_libraries(FlyJitRuntime PRIVATE hip::host hip::amdhip64 ${CMAKE_DL_LIBS})
set_target_properties(FlyJitRuntime PROPERTIES OUTPUT_NAME "fly_jit_runtime")

# ROCm AOT exports partially link this PIC archive into the generated host object.
# Its hidden symbols are localized by the relocatable link, leaving HIP and
# standard system libraries as the only deployment dependencies.
add_library(FlyRocmAotRuntime STATIC ${_fly_rocm_runtime_common})
target_compile_definitions(FlyRocmAotRuntime PRIVATE FLYDSL_AOT_RUNTIME_EMBEDDED=1)
target_compile_features(FlyRocmAotRuntime PRIVATE cxx_std_17)
target_link_libraries(FlyRocmAotRuntime PRIVATE hip::host hip::amdhip64 ${CMAKE_DL_LIBS})
set_target_properties(FlyRocmAotRuntime PROPERTIES
  OUTPUT_NAME "fly_rocm_aot_runtime"
  POSITION_INDEPENDENT_CODE ON
)
