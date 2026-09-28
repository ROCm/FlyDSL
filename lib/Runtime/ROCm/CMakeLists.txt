# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# FlyJitRuntime — thin ROCm runtime with GPU module caching.

# Prefer an explicitly selected ROCm, then the SDK in the active environment.
# NO_DEFAULT_PATH keeps a system HIP from winning over either one.
if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
  find_package(hip QUIET CONFIG PATHS "$ENV{ROCM_PATH}" NO_DEFAULT_PATH)
endif()

if(NOT hip_FOUND)
  execute_process(COMMAND rocm-sdk path --root
                  OUTPUT_VARIABLE _rocm_sdk_root
                  RESULT_VARIABLE _rocm_sdk_result
                  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(_rocm_sdk_result STREQUAL "0" AND _rocm_sdk_root)
    find_package(hip QUIET CONFIG PATHS "${_rocm_sdk_root}" NO_DEFAULT_PATH)
  endif()
endif()

if(NOT hip_FOUND)
  file(GLOB _rocm_paths LIST_DIRECTORIES true "/opt/rocm*")
  list(SORT _rocm_paths ORDER DESCENDING)
  find_package(hip REQUIRED CONFIG PATHS ${_rocm_paths})
endif()

add_library(FlyJitRuntime SHARED FlyRocmRuntimeWrappers.cpp)
target_include_directories(FlyJitRuntime PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
target_compile_features(FlyJitRuntime PRIVATE cxx_std_17)
target_link_libraries(FlyJitRuntime PRIVATE hip::host hip::amdhip64 ${CMAKE_DL_LIBS})
set_target_properties(FlyJitRuntime PROPERTIES OUTPUT_NAME "fly_jit_runtime")
