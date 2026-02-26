cmake_minimum_required(VERSION 3.20)
project(FLYDSL LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


option(FLY_UNITTEST "Build Fly Unit Tests" OFF)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found MLIR: ${MLIR_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
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

include_directories(${PROJECT_SOURCE_DIR}/thirdparty/dlpack/include)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/tvm-ffi/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})


set(MLIR_PYTHON_PACKAGE_PREFIX "_mlir" CACHE STRING "" FORCE)
set(MLIR_BINDINGS_PYTHON_INSTALL_PREFIX "python_packages/flydsl/${MLIR_PYTHON_PACKAGE_PREFIX}" CACHE STRING "" FORCE)


add_subdirectory(include/flydsl)
add_subdirectory(lib)
add_subdirectory(tools)

add_subdirectory(python/mlir_flydsl)
