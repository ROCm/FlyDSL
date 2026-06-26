# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# FlyJitRuntime — thin ROCm runtime with GPU module caching.

file(GLOB _rocm_paths LIST_DIRECTORIES true "/opt/rocm*")
list(SORT _rocm_paths ORDER DESCENDING)
find_package(hip REQUIRED CONFIG PATHS ${_rocm_paths})

include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES hip::host)
check_cxx_source_compiles("
    #include <hip/hip_runtime.h>
    int main() {
        hipLaunchAttributeID id = hipLaunchAttributeClusterDimension;
        return 0;
    }
" HIP_HAS_CLUSTER_LAUNCH)
unset(CMAKE_REQUIRED_LIBRARIES)

add_library(FlyJitRuntime SHARED FlyRocmRuntimeWrappers.cpp)
target_include_directories(FlyJitRuntime PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
target_compile_features(FlyJitRuntime PRIVATE cxx_std_17)
target_link_libraries(FlyJitRuntime PRIVATE hip::host hip::amdhip64)
if(HIP_HAS_CLUSTER_LAUNCH)
  target_compile_definitions(FlyJitRuntime PRIVATE HIP_HAS_CLUSTER_LAUNCH)
endif()
set_target_properties(FlyJitRuntime PROPERTIES OUTPUT_NAME "fly_jit_runtime")
