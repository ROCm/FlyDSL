# Generate Python bindings for the Rocir dialect family

add_custom_target(RocirPythonModules)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/rocir/RocirPython.td)

# Add include path for TableGen
set(MLIR_TABLEGEN_EXE ${MLIR_TABLEGEN_EXE})
list(APPEND LLVM_TABLEGEN_FLAGS -I ${CMAKE_SOURCE_DIR}/include)
list(APPEND LLVM_TABLEGEN_FLAGS -I ${CMAKE_SOURCE_DIR}/include/rocir)

# Generate Python bindings for rocir dialect
mlir_tablegen(_rocir_ops_gen.py -gen-python-op-bindings -bind-dialect=rocir)
add_public_tablegen_target(RocirPythonOpsIncGen)
add_dependencies(RocirPythonModules RocirPythonOpsIncGen)

# Generate Python bindings for the ROCm dialect extension
mlir_tablegen(_rocir_rocm_ops_gen.py -gen-python-op-bindings -bind-dialect=cute_rocm)
add_public_tablegen_target(RocirRocmPythonOpsIncGen)
add_dependencies(RocirPythonModules RocirRocmPythonOpsIncGen)

# Generate Python enum bindings if needed
mlir_tablegen(_rocir_enum_gen.py -gen-python-enum-bindings)
add_public_tablegen_target(RocirPythonEnumIncGen)
add_dependencies(RocirPythonModules RocirPythonEnumIncGen)

# Build Python extension module for pass registration
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find pybind11
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE pybind11_NOTFOUND
)

if(NOT pybind11_NOTFOUND)
  find_package(pybind11 CONFIG)
endif()

if(pybind11_FOUND)
  message(STATUS "Found pybind11: ${pybind11_DIR}")
  
  pybind11_add_module(_rocirPassesExt RocirPassesModule.cpp)
  
  target_include_directories(_rocirPassesExt PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
  )
  
  # Link to MLIRPythonCAPI.so to share the global pass registry with Python MLIR
  get_filename_component(MLIR_BUILD_DIR "${MLIR_DIR}/../../.." ABSOLUTE)
  find_library(MLIR_PYTHON_CAPI_LIB
    NAMES MLIRPythonCAPI
    PATHS ${MLIR_BUILD_DIR}/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs
    NO_DEFAULT_PATH
  )
  
  if(MLIR_PYTHON_CAPI_LIB)
    message(STATUS "Found MLIRPythonCAPI: ${MLIR_PYTHON_CAPI_LIB}")
  else()
    message(FATAL_ERROR "MLIRPythonCAPI library not found!")
  endif()
  
  target_link_libraries(_rocirPassesExt PRIVATE
    RocirDialect
    RocirTransforms
    ${MLIR_PYTHON_CAPI_LIB}
  )
  set_target_properties(_rocirPassesExt PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
  
  # Install to python_bindings directory
  set_target_properties(_rocirPassesExt PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  
  add_dependencies(RocirPythonModules _rocirPassesExt)
  
  message(STATUS "Rocir passes Python extension module will be built")
else()
  message(WARNING "pybind11 not found - Rocir passes will not be available in Python")
endif()

# Fix TableGen-generated relative imports for standalone usage
# TableGen generates "from ._ods_common import ..." which assumes the code is part of mlir.dialects package
# Since we use build/python_bindings directly in PYTHONPATH, we need absolute imports
add_custom_command(
    TARGET RocirPythonOpsIncGen POST_BUILD
    COMMAND sed -i 's/from \._ods_common/from mlir.dialects._ods_common/g' 
        ${CMAKE_CURRENT_BINARY_DIR}/_rocir_ops_gen.py
    COMMENT "Fixing imports in generated Python bindings"
)

add_custom_command(
    TARGET RocirRocmPythonOpsIncGen POST_BUILD
    COMMAND sed -i 's/from \._ods_common/from mlir.dialects._ods_common/g' 
        ${CMAKE_CURRENT_BINARY_DIR}/_rocir_rocm_ops_gen.py
    COMMENT "Fixing imports in generated Python bindings for the ROCm dialect"
)

# Copy Python wrapper files to build directory at configure time
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/rocir.py
    ${CMAKE_CURRENT_BINARY_DIR}/rocir.py
    COPYONLY
)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/__init__.py "")
