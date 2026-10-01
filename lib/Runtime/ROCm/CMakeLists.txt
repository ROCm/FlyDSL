# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# FlyJitRuntime — thin ROCm runtime with GPU module caching.

include("${PROJECT_SOURCE_DIR}/cmake/FlyDSLFindHip.cmake")
flydsl_find_hip(_flydsl_hip_host_target _flydsl_hip_runtime_target)

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
target_link_libraries(FlyJitRuntime PRIVATE ${_flydsl_hip_host_target} ${_flydsl_hip_runtime_target} ${CMAKE_DL_LIBS})
set_target_properties(FlyJitRuntime PROPERTIES OUTPUT_NAME "fly_jit_runtime")

# ROCm AOT exports partially link this PIC archive into the generated host object.
# Its hidden symbols are localized by the relocatable link, leaving HIP and
# standard system libraries as the only deployment dependencies.
add_library(FlyRocmAotRuntime STATIC ${_fly_rocm_runtime_common})
target_compile_definitions(FlyRocmAotRuntime PRIVATE FLYDSL_AOT_RUNTIME_EMBEDDED=1)
target_compile_features(FlyRocmAotRuntime PRIVATE cxx_std_17)
target_link_libraries(FlyRocmAotRuntime PRIVATE ${_flydsl_hip_host_target} ${_flydsl_hip_runtime_target} ${CMAKE_DL_LIBS})
set_target_properties(FlyRocmAotRuntime PROPERTIES
  OUTPUT_NAME "fly_rocm_aot_runtime"
  POSITION_INDEPENDENT_CODE ON
)
