# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

find_package(CUDAToolkit QUIET)
if(NOT CUDAToolkit_FOUND)
  message(FATAL_ERROR
    "The 'nvvm' backend requires a CUDA toolkit to build the CUDA JIT runtime, "
    "but CUDAToolkit was not found. Install CUDA 12+ or set "
    "CUDAToolkit_ROOT, or drop 'nvvm' from FLYDSL_BACKENDS.")
endif()

# Pick the cuda.h to build against explicitly instead of taking whatever
# FindCUDAToolkit resolved. Distros ship an old header in /usr/include (e.g.
# 11.5) that CMake prefers, which would both disagree with the ptxas the Python
# backend selects (_detect_cuda_toolkit() takes the newest /usr/local/cuda-*)
# and hide the CUDA 12 library-management API this runtime needs. Mirror that
# same "newest /usr/local/cuda-* first" order here.
function(_flydsl_pick_cuda_include out_var)
  set(_candidates "")
  if(DEFINED CUDAToolkit_ROOT)
    list(APPEND _candidates "${CUDAToolkit_ROOT}")
  endif()
  file(GLOB _globbed "/usr/local/cuda-*")
  list(SORT _globbed COMPARE NATURAL ORDER DESCENDING)
  list(APPEND _candidates ${_globbed} "/usr/local/cuda" ${CUDAToolkit_INCLUDE_DIRS})
  foreach(_root ${_candidates})
    foreach(_inc "${_root}/include" "${_root}")
      if(EXISTS "${_inc}/cuda.h")
        file(STRINGS "${_inc}/cuda.h" _line REGEX "^#define CUDA_VERSION ")
        string(REGEX MATCH "[0-9]+" _ver "${_line}")
        if(_ver AND _ver GREATER_EQUAL 12000)
          set(${out_var} "${_inc}" PARENT_SCOPE)
          return()
        endif()
      endif()
    endforeach()
  endforeach()
  set(${out_var} "" PARENT_SCOPE)
endfunction()

_flydsl_pick_cuda_include(FLYDSL_CUDA_INCLUDE_DIR)
if(NOT FLYDSL_CUDA_INCLUDE_DIR)
  message(FATAL_ERROR
    "The 'nvvm' backend needs CUDA 12+ headers: the runtime uses the CUDA 12 "
    "library-management API (cuLibraryLoadData) so one JIT artifact can run on "
    "several devices. No cuda.h with CUDA_VERSION >= 12000 was found under "
    "CUDAToolkit_ROOT or /usr/local/cuda*. Set CUDAToolkit_ROOT to a CUDA 12+ install.")
endif()
message(STATUS "FlyCudaJitRuntime cuda.h: ${FLYDSL_CUDA_INCLUDE_DIR}")

add_library(FlyCudaJitRuntime SHARED FlyCudaRuntimeWrappers.cpp)
# BEFORE so the chosen cuda.h wins over any older one on the default path.
target_include_directories(FlyCudaJitRuntime BEFORE PRIVATE "${FLYDSL_CUDA_INCLUDE_DIR}")
target_include_directories(FlyCudaJitRuntime PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
target_compile_features(FlyCudaJitRuntime PRIVATE cxx_std_17)
target_link_libraries(FlyCudaJitRuntime PRIVATE CUDA::cuda_driver ${CMAKE_DL_LIBS})
set_target_properties(FlyCudaJitRuntime PROPERTIES OUTPUT_NAME "fly_cuda_runtime")
