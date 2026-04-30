# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

file(GLOB _flydsl_rocm_paths LIST_DIRECTORIES true "/opt/rocm*")
list(SORT _flydsl_rocm_paths ORDER DESCENDING)
find_package(hip REQUIRED CONFIG PATHS ${_flydsl_rocm_paths})

add_library(FlyJitRuntime SHARED FlyRocmRuntimeWrappers.cpp)
target_include_directories(FlyJitRuntime PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
target_compile_features(FlyJitRuntime PRIVATE cxx_std_17)
target_link_libraries(FlyJitRuntime PRIVATE hip::host hip::amdhip64)
set_target_properties(FlyJitRuntime PROPERTIES OUTPUT_NAME "fly_jit_runtime")
