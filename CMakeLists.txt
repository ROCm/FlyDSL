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
set(LLVM_BUILD_TOOLS ON)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
if(NOT WIN32)
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


set(MLIR_PYTHON_PACKAGE_PREFIX "_mlir" CACHE STRING "" FORCE)
set(MLIR_BINDINGS_PYTHON_INSTALL_PREFIX "python_packages/flydsl/${MLIR_PYTHON_PACKAGE_PREFIX}" CACHE STRING "" FORCE)


# On Windows, MLIR's default SelfOwningTypeID scheme does not work across DLL
# boundaries: the TypeID's `static SelfOwningTypeID id` member cannot be auto-imported
# from a DLL without explicit __declspec(dllimport). So FlyDSL's Fly-defined types
# (defined in obj.MLIRFlyDialect embedded in FlyPythonCAPI.dll) are unreachable from
# .pyd extensions that reference them, causing link errors or runtime TypeID mismatches.
#
# Switch our compilation to string-based FallbackTypeIDs globally. Our own types all
# route through the process-shared `registerImplicitTypeID` exported from
# FlyPythonCAPI.dll, giving consistent TypeIDs across every DLL. Upstream MLIR's
# prebuilt static libs (MLIRIR.lib, etc.) were compiled with SelfOwning TypeIDs,
# but those are now linked into FlyPythonCAPI.dll as a single copy — internal usage
# within that DLL still agrees with itself, so no mismatch there.
if(WIN32)
  add_compile_definitions(MLIR_USE_FALLBACK_TYPE_IDS=1)
endif()

add_subdirectory(include/flydsl)
add_subdirectory(lib)
add_subdirectory(tools)

add_subdirectory(python/mlir_flydsl)
