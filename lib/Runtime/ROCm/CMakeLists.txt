# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# FlyJitRuntime — thin ROCm runtime with GPU module caching.

# ROCM_PATH, else the SDK's own answer, else whatever is in /opt.
if(DEFINED ENV{ROCM_PATH})
  set(_rocm_paths "$ENV{ROCM_PATH}")
else()
  execute_process(COMMAND rocm-sdk path --root
                  OUTPUT_VARIABLE _rocm_paths OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()
if(NOT _rocm_paths)
  file(GLOB _rocm_paths LIST_DIRECTORIES true "/opt/rocm*")
  list(SORT _rocm_paths ORDER DESCENDING)
endif()
find_package(hip REQUIRED CONFIG PATHS ${_rocm_paths})

add_library(FlyJitRuntime SHARED FlyRocmRuntimeWrappers.cpp)
target_include_directories(FlyJitRuntime PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
target_compile_features(FlyJitRuntime PRIVATE cxx_std_17)
target_link_libraries(FlyJitRuntime PRIVATE hip::host hip::amdhip64 ${CMAKE_DL_LIBS})
set_target_properties(FlyJitRuntime PROPERTIES OUTPUT_NAME "fly_jit_runtime")
