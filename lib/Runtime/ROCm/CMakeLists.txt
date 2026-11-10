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
  find_package(hip REQUIRED CONFIG PATHS ${_rocm_paths})
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

if(WIN32)
  # JIT-compiled host code resolves the mgpu* wrappers in this DLL by name, and
  # a DLL exports nothing that is not marked.
  set_target_properties(FlyJitRuntime PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
  # windows/ stands in for the pthread.h and dlfcn.h the AOT runtime includes.
  target_include_directories(FlyJitRuntime PRIVATE windows)

  # COFF has no relocatable link to embed an archive into an exported object, so
  # the AOT runtime is compiled to LLVM bitcode (clang-cl -flto), which
  # python/mlir_flydsl imports into the LLVM dialect: each export merges in its
  # own internal copy. The static CRT keeps CRT calls free of dllimport, so the
  # code links into /MD and /MT programs alike.
  add_library(FlyRocmAotRuntimeBitcode OBJECT ${_fly_rocm_runtime_common})
  target_include_directories(FlyRocmAotRuntimeBitcode PRIVATE windows)
  target_compile_definitions(FlyRocmAotRuntimeBitcode PRIVATE FLYDSL_AOT_RUNTIME_EMBEDDED=1)
  target_compile_features(FlyRocmAotRuntimeBitcode PRIVATE cxx_std_17)
  target_compile_options(FlyRocmAotRuntimeBitcode PRIVATE -flto)
  target_link_libraries(FlyRocmAotRuntimeBitcode PRIVATE hip::host)
  set_target_properties(FlyRocmAotRuntimeBitcode PROPERTIES MSVC_RUNTIME_LIBRARY MultiThreaded)
  return()
endif()

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
