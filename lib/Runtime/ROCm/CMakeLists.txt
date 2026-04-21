# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# FlyJitRuntime — thin ROCm runtime with GPU module caching.

file(GLOB _rocm_paths LIST_DIRECTORIES true "/opt/rocm*")
list(SORT _rocm_paths ORDER DESCENDING)
find_package(hip REQUIRED CONFIG PATHS ${_rocm_paths})

include(CheckCXXSourceCompiles)
set(_FLY_SAVED_CMAKE_REQUIRED_INCLUDES "${CMAKE_REQUIRED_INCLUDES}")
set(_FLY_SAVED_CMAKE_REQUIRED_DEFINITIONS "${CMAKE_REQUIRED_DEFINITIONS}")
set(_FLY_SAVED_CMAKE_TRY_COMPILE_TARGET_TYPE "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
set(CMAKE_REQUIRED_INCLUDES "${hip_INCLUDE_DIRS};${HIP_INCLUDE_DIRS}")
set(CMAKE_REQUIRED_DEFINITIONS "-D__HIP_PLATFORM_AMD__")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
check_cxx_source_compiles("
  #include <hip/hip_runtime_api.h>
  int main() {
    auto x = hipLaunchAttributeClusterDimension;
    (void)x;
    return 0;
  }
" FLY_HIP_HAS_CLUSTER_ATTR)
set(CMAKE_REQUIRED_INCLUDES "${_FLY_SAVED_CMAKE_REQUIRED_INCLUDES}")
set(CMAKE_REQUIRED_DEFINITIONS "${_FLY_SAVED_CMAKE_REQUIRED_DEFINITIONS}")
set(CMAKE_TRY_COMPILE_TARGET_TYPE "${_FLY_SAVED_CMAKE_TRY_COMPILE_TARGET_TYPE}")

add_library(FlyJitRuntime SHARED FlyRocmRuntimeWrappers.cpp)
target_include_directories(FlyJitRuntime PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
target_compile_features(FlyJitRuntime PRIVATE cxx_std_17)
target_link_libraries(FlyJitRuntime PRIVATE hip::host hip::amdhip64)
target_compile_definitions(FlyJitRuntime PRIVATE
  FLY_HIP_HAS_CLUSTER_ATTR=$<BOOL:${FLY_HIP_HAS_CLUSTER_ATTR}>)
set_target_properties(FlyJitRuntime PROPERTIES OUTPUT_NAME "fly_jit_runtime")
