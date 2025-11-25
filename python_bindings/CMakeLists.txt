# Generate Python bindings for CuTe dialect

add_custom_target(CutePythonModules)

set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/cute/CuteOps.td)

# Add include path for TableGen
set(MLIR_TABLEGEN_EXE ${MLIR_TABLEGEN_EXE})
list(APPEND LLVM_TABLEGEN_FLAGS -I ${CMAKE_SOURCE_DIR}/include)
list(APPEND LLVM_TABLEGEN_FLAGS -I ${CMAKE_SOURCE_DIR}/include/cute)

# Generate Python bindings
mlir_tablegen(_cute_ops_gen.py -gen-python-op-bindings -bind-dialect=cute)
add_public_tablegen_target(CutePythonOpsIncGen)
add_dependencies(CutePythonModules CutePythonOpsIncGen)

# Generate Python enum bindings if needed
mlir_tablegen(_cute_enum_gen.py -gen-python-enum-bindings)
add_public_tablegen_target(CutePythonEnumIncGen)
add_dependencies(CutePythonModules CutePythonEnumIncGen)

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
  
  pybind11_add_module(_cutePassesExt CutePassesModule.cpp)
  
  target_include_directories(_cutePassesExt PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
  )
  
  target_link_libraries(_cutePassesExt PRIVATE
    CuteTransforms
    MLIRPass
    MLIRIR
  )
  
  # Install to python_bindings directory
  set_target_properties(_cutePassesExt PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  
  add_dependencies(CutePythonModules _cutePassesExt)
  
  message(STATUS "CuTe passes Python extension module will be built")
else()
  message(WARNING "pybind11 not found - CuTe passes will not be available in Python")
endif()

  # Build Python extension module for dialect registration
  pybind11_add_module(_cuteDialect CuteDialectModule.cpp)
  
  target_include_directories(_cuteDialect PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
  )
  
  # No additional libraries needed for simple stub module
  # target_link_libraries(_cuteDialect PRIVATE)
  
  # Install to python_bindings directory
  set_target_properties(_cuteDialect PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  
  add_dependencies(CutePythonModules _cuteDialect)
  
  message(STATUS "CuTe dialect Python extension module will be built")
