cmake_minimum_required(VERSION 3.18)
project(FLIR VERSION 0.1.0 LANGUAGES C CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_RUNTIME "Build C++ runtime library" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_TESTS "Build tests" OFF)
option(ENABLE_ROCM "Enable AMD ROCm support (GFX942/GFX950)" OFF)  # Disabled by default - CUDA only

# Find MLIR (optional)
find_package(MLIR CONFIG)
if(MLIR_FOUND)
    message(STATUS "Found MLIR: ${MLIR_DIR}")
    
    # Include MLIR CMake modules
    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
    list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
    
    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)
    include(HandleLLVMOptions)
    include(MLIRDetectPythonEnv)
    mlir_configure_python_dev_packages()
    
    include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
    include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
    include_directories(${PROJECT_SOURCE_DIR}/flir/include)
    # Generated TableGen headers land under the subdir build tree:
    #   ${PROJECT_BINARY_DIR}/flir/include/flir/*.inc
    # and are included as "flir/*.inc".
    include_directories(${PROJECT_BINARY_DIR}/flir/include)
    include_directories(${PROJECT_BINARY_DIR}/include)
    
    add_definitions(${LLVM_DEFINITIONS})
    
    # TableGen targets
    add_subdirectory(flir/include/flir)
    
    message(STATUS "TableGen targets configured")
else()
    message(WARNING "MLIR not found. TableGen targets will not be built.")
    message(STATUS "Set MLIR_DIR to enable MLIR features.")
endif()

# Find ROCm (optional, for runtime support)
if(ENABLE_ROCM)
    message(STATUS "ROCm support is DISABLED by default (CUDA-only mode)")
    message(STATUS "To enable ROCm, set -DENABLE_ROCM=ON during CMake configuration")
    find_package(hip QUIET CONFIG PATHS /opt/rocm)
    if(hip_FOUND)
        message(STATUS "Found HIP: ${HIP_VERSION}")
        add_definitions(-DHAVE_HIP)
    else()
        message(STATUS "HIP not found. ROCm runtime support will be limited.")
    endif()
else()
    message(STATUS "Building in CUDA-only mode (ROCm disabled)")
endif()

# Build runtime library
if(BUILD_RUNTIME)
#    add_subdirectory(runtime)
endif()

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# Configure Python package prefix (embed upstream MLIR under `_mlir` to avoid
# colliding with MLIR's own exported CMake/Python targets named `mlir.*`).
set(MLIR_PYTHON_PACKAGE_PREFIX "_mlir" CACHE STRING "" FORCE)
set(MLIR_BINDINGS_PYTHON_INSTALL_PREFIX "python_packages/pyflir/${MLIR_PYTHON_PACKAGE_PREFIX}" CACHE STRING "" FORCE)

add_subdirectory(flir/lib)
add_subdirectory(flir/tools)
if(BUILD_PYTHON_BINDINGS AND MLIR_FOUND)
    add_subdirectory(python_bindings)
endif()


# Installation
install(DIRECTORY flir/include/flir
    DESTINATION include
    FILES_MATCHING PATTERN "*.td"
)

install(DIRECTORY docs
    DESTINATION share/flir
)

# Summary
message(STATUS "")
message(STATUS "Flir Configuration Summary")
message(STATUS "==============================")
message(STATUS "  Version:           ${PROJECT_VERSION}")
message(STATUS "  Build Runtime:     ${BUILD_RUNTIME}")
message(STATUS "  Build Python:      ${BUILD_PYTHON_BINDINGS}")
message(STATUS "  Build Tests:       ${BUILD_TESTS}")
message(STATUS "  MLIR Found:        ${MLIR_FOUND}")
message(STATUS "  ROCm Support:      ${ENABLE_ROCM}")
if(ENABLE_ROCM AND hip_FOUND)
message(STATUS "  HIP Version:       ${HIP_VERSION}")
endif()
message(STATUS "")
