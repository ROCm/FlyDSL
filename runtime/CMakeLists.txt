cmake_minimum_required(VERSION 3.18)
project(CuteRuntime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG QUIET)

# CUDA-only build by default
option(USE_CUDA "Enable CUDA support" ON)
option(USE_ROCM "Enable ROCm/HIP support" OFF)  # Disabled by default

set(CUTE_RUNTIME_SOURCES src/cute_runtime.cpp)
set(CUTE_RUNTIME_LIBS "")
set(CUTE_RUNTIME_DEFS "")

# CUDA Support (Primary backend)
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDA QUIET)
    find_package(CUDAToolkit QUIET)
    
    if(CUDA_FOUND OR CUDAToolkit_FOUND)
        message(STATUS "CUDA support enabled")
        list(APPEND CUTE_RUNTIME_LIBS CUDA::cudart CUDA::cuda_driver)
        list(APPEND CUTE_RUNTIME_DEFS -DHAVE_CUDA)
        
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "80;90")
        endif()
        message(STATUS "Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
        
        set_source_files_properties(${CUTE_RUNTIME_SOURCES} PROPERTIES LANGUAGE CUDA)
    else()
        message(WARNING "CUDA not found, runtime will be limited")
    endif()
else()
    message(STATUS "CUDA support disabled")
endif()

# ROCm/HIP Support (Optional - disabled by default)
if(USE_ROCM)
    find_package(hip QUIET CONFIG PATHS /opt/rocm)
    find_package(hiprtc QUIET CONFIG PATHS /opt/rocm)
    
    if(hip_FOUND)
        enable_language(HIP)
        message(STATUS "ROCm/HIP support enabled")
        message(STATUS "HIP version: ${HIP_VERSION}")
        
        list(APPEND CUTE_RUNTIME_LIBS hip::device hip::host)
        if(hiprtc_FOUND)
            list(APPEND CUTE_RUNTIME_LIBS hiprtc::hiprtc)
        endif()
        
        list(APPEND CUTE_RUNTIME_DEFS -DHAVE_HIP -DENABLE_ROCM_SUPPORT)
        
        if(NOT DEFINED HIP_ARCHITECTURES)
            set(HIP_ARCHITECTURES "gfx942")
        endif()
        message(STATUS "Building for HIP architectures: ${HIP_ARCHITECTURES}")
        
        # Add HIP source file only if ROCm is enabled
        list(APPEND CUTE_RUNTIME_SOURCES src/cute_runtime_hip.cpp)
        set_source_files_properties(${CUTE_RUNTIME_SOURCES} PROPERTIES LANGUAGE HIP)
    else()
        message(WARNING "ROCm/HIP requested but not found")
    endif()
else()
    message(STATUS "ROCm/HIP support disabled (CUDA-only mode)")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(hip_FOUND)
        enable_language(HIP)
    include_directories(${hip_INCLUDE_DIRS})
endif()

#===----------------------------------------------------------------------===#
# CuTe Runtime Library (C++)
#===----------------------------------------------------------------------===#

add_library(cute_runtime SHARED
    ${CUTE_RUNTIME_SOURCES}
)

target_link_libraries(cute_runtime
    ${CUTE_RUNTIME_LIBS}
)

target_compile_definitions(cute_runtime PRIVATE
    ${CUTE_RUNTIME_DEFS}
)

set_target_properties(cute_runtime PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

if(CUDA_FOUND OR CUDAToolkit_FOUND)
    set_target_properties(cute_runtime PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

#===----------------------------------------------------------------------===#
# Python Bindings Module
#===----------------------------------------------------------------------===#

if(pybind11_FOUND)
    pybind11_add_module(_cute_bindings
        ../python/cute_runtime/bindings.cpp
    )

    target_link_libraries(_cute_bindings PRIVATE
        cute_runtime
        ${CUTE_RUNTIME_LIBS}
    )

    target_compile_definitions(_cute_bindings PRIVATE
        ${CUTE_RUNTIME_DEFS}
    )

    target_include_directories(_cute_bindings PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Install targets
    install(TARGETS _cute_bindings
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()

# Install runtime library
install(TARGETS cute_runtime
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES include/cute_runtime.h
    DESTINATION include
)
