# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# FlyJitRuntime — thin ROCm runtime with GPU module caching.

file(GLOB _rocm_paths LIST_DIRECTORIES true "/opt/rocm*")
list(SORT _rocm_paths ORDER DESCENDING)
find_package(hip REQUIRED CONFIG PATHS ${_rocm_paths})

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
