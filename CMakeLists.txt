cmake_minimum_required(VERSION 3.20)
project(FLYDSL LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FLY_UNITTEST "Build Fly Unit Tests" OFF)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found MLIR: ${MLIR_DIR}")

# LLD is linked as a library so GPU binary emission never has to locate an
# `ld.lld` executable at runtime (see lib/Conversion/FlyToROCDL/FlyEmitGPUBinary.cpp).
# Always resolve it from the LLVM install that provides MLIR: NO_DEFAULT_PATH
# keeps CMake from picking up a system LLD whose version would not match the
# code generator.  Optional, so an LLVM built without the lld project still works.
if(NOT LLD_DIR)
  set(LLD_DIR ${LLVM_LIBRARY_DIR}/cmake/lld)
endif()
find_package(LLD CONFIG PATHS "${LLD_DIR}" NO_DEFAULT_PATH)
if(LLD_FOUND)
  message(STATUS "Found LLD: ${LLD_DIR} (in-process GPU binary linking)")
else()
  message(STATUS "LLD not found under ${LLD_DIR}; GPU binary emission falls back to "
                 "spawning ld.lld from the ROCm toolkit path")
endif()

# AMDGCN device bitcode (ocml/ockl/...) is loaded from <toolkit>/amdgcn/bitcode
# whenever a kernel calls __ocml_* / __ockl_*.  Bundle it into the Python
# package so that lookup does not depend on where the container installs ROCm.
# Override the source with -DFLYDSL_ROCM_BITCODE_DIR=<dir>.
if(NOT FLYDSL_ROCM_BITCODE_DIR)
  find_path(FLYDSL_ROCM_BITCODE_DIR ocml.bc
    HINTS
      "$ENV{ROCM_PATH}/amdgcn/bitcode"
      "$ENV{ROCM_ROOT}/amdgcn/bitcode"
      "$ENV{ROCM_HOME}/amdgcn/bitcode"
      "/opt/rocm/amdgcn/bitcode"
    NO_DEFAULT_PATH)
endif()
if(FLYDSL_ROCM_BITCODE_DIR)
  message(STATUS "Found AMDGCN device bitcode: ${FLYDSL_ROCM_BITCODE_DIR}")
else()
  message(WARNING
    "AMDGCN device bitcode (ocml.bc) not found; the Python package will not bundle it. "
    "Kernels calling __ocml_*/__ockl_* will then need FLYDSL_COMPILE_ROCM_PATH or "
    "ROCM_PATH to point at a ROCm install. Set -DFLYDSL_ROCM_BITCODE_DIR=<dir> to bundle it.")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
  message(STATUS "CMAKE_BUILD_TYPE not set; defaulting to ${CMAKE_BUILD_TYPE}")
endif()

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(LLVM_BUILD_TOOLS ON)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
string(REPLACE "-Wl,-z,defs" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/lib/CAPI)

include_directories(${PROJECT_SOURCE_DIR}/thirdparty/dlpack/include)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/tvm-ffi/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})


set(MLIR_PYTHON_PACKAGE_PREFIX "_mlir" CACHE STRING "" FORCE)
set(MLIR_BINDINGS_PYTHON_INSTALL_PREFIX "python_packages/flydsl/${MLIR_PYTHON_PACKAGE_PREFIX}" CACHE STRING "" FORCE)

include(cmake/FlyDSLBackends.cmake)

add_subdirectory(include/flydsl)
add_subdirectory(lib)
add_subdirectory(tools)

add_subdirectory(python/mlir_flydsl)
