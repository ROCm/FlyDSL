cmake_minimum_required(VERSION 3.18)
project(CuteIR VERSION 0.1.0 LANGUAGES C CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(BUILD_RUNTIME "Build C++ runtime library" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_TESTS "Build tests" OFF)
option(ENABLE_ROCM "Enable AMD ROCm support for GFX942" OFF)  # Disabled by default - CUDA only

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
    
    include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
    include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
    include_directories(${PROJECT_SOURCE_DIR}/include)
    include_directories(${PROJECT_BINARY_DIR}/include)
    
    add_definitions(${LLVM_DEFINITIONS})
    
    # TableGen targets
    add_subdirectory(include/cute)
    
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

add_subdirectory(lib)
add_subdirectory(tools)

# Installation
install(DIRECTORY include/cute
    DESTINATION include
    FILES_MATCHING PATTERN "*.td"
)

install(DIRECTORY docs
    DESTINATION share/cute_ir
)

# Summary
message(STATUS "")
message(STATUS "CuTe IR Configuration Summary")
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
