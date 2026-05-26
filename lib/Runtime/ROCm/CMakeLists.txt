# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# FlyJitRuntime — thin ROCm runtime with GPU module caching.

file(GLOB _rocm_paths LIST_DIRECTORIES true "/opt/rocm*")
list(SORT _rocm_paths ORDER DESCENDING)
find_package(hip REQUIRED CONFIG PATHS ${_rocm_paths})

# Detect cluster-launch enum at configure time. The symbol is an enum value,
# not a macro, so a C++ #ifdef in the wrapper cannot see it. Probe by trying
# to compile a TU that references it, then propagate a real macro.
include(CheckCXXSourceCompiles)
get_target_property(_HIP_HOST_INCLUDES hip::host INTERFACE_INCLUDE_DIRECTORIES)
set(CMAKE_REQUIRED_INCLUDES ${_HIP_HOST_INCLUDES})
set(CMAKE_REQUIRED_DEFINITIONS "-D__HIP_PLATFORM_AMD__")
check_cxx_source_compiles("
  #include <hip/hip_runtime_api.h>
  int main() {
    auto x = hipLaunchAttributeClusterDimension;
    (void)x;
    return 0;
  }
" FLY_HIP_HAS_CLUSTER_ATTR)
unset(CMAKE_REQUIRED_INCLUDES)
unset(CMAKE_REQUIRED_DEFINITIONS)

add_library(FlyJitRuntime SHARED FlyRocmRuntimeWrappers.cpp)
target_include_directories(FlyJitRuntime PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
target_compile_features(FlyJitRuntime PRIVATE cxx_std_17)
if(FLY_HIP_HAS_CLUSTER_ATTR)
  target_compile_definitions(FlyJitRuntime PRIVATE FLY_HIP_HAS_CLUSTER_ATTR=1)
endif()
target_link_libraries(FlyJitRuntime PRIVATE hip::host hip::amdhip64)
set_target_properties(FlyJitRuntime PROPERTIES OUTPUT_NAME "fly_jit_runtime")
