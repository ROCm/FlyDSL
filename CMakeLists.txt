cmake_minimum_required(VERSION 3.20)
project(FLYDSL LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FLY_UNITTEST "Build Fly Unit Tests" OFF)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found MLIR: ${MLIR_DIR}")

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

if(UNIX)
  string(REPLACE "-Wl,-z,defs" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
endif()
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

if(WIN32)
  add_compile_definitions(MLIR_USE_FALLBACK_TYPE_IDS=1)
endif()


set(MLIR_PYTHON_PACKAGE_PREFIX "_mlir" CACHE STRING "" FORCE)
set(MLIR_BINDINGS_PYTHON_INSTALL_PREFIX "python_packages/flydsl/${MLIR_PYTHON_PACKAGE_PREFIX}" CACHE STRING "" FORCE)

include(cmake/FlyDSLBackends.cmake)

add_subdirectory(include/flydsl)
add_subdirectory(lib)
add_subdirectory(tools)

add_subdirectory(python/mlir_flydsl)

if(WIN32)
  set(_fly_link_targets ${MLIR_EXPORTED_TARGETS} MLIRFlyDialect
    MLIRFlyROCDLDialect MLIRFlyToROCDL)
  foreach(_target IN LISTS _fly_link_targets)
    if(TARGET ${_target})
      get_target_property(_links ${_target} INTERFACE_LINK_LIBRARIES)
      if(_links MATCHES "MLIRIR|MLIRSupport")
        get_mlir_filtered_link_libraries(_links_filtered ${_links})
        set_property(TARGET ${_target} PROPERTY INTERFACE_LINK_LIBRARIES "${_links_filtered}")
      endif()
    endif()
  endforeach()
endif()
