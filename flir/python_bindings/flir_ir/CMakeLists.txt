# FLIR IR Python extension with all symbols hidden
#
# This builds a single Python extension module that statically links all
# MLIR/LLVM code, hiding all symbols except PyInit_*.

cmake_minimum_required(VERSION 3.18)

message(STATUS "Building FLIR IR Python extension")

# Find nanobind
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE NB_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${NB_CMAKE_DIR}")
find_package(nanobind CONFIG REQUIRED)

# MLIR Python binding internal sources path
# Try multiple possible locations
if(DEFINED LLVM_MAIN_SRC_DIR AND EXISTS "${LLVM_MAIN_SRC_DIR}/../mlir/lib/Bindings/Python")
  set(MLIR_PYTHON_BINDINGS_DIR "${LLVM_MAIN_SRC_DIR}/../mlir/lib/Bindings/Python")
elseif(EXISTS "/data/felix/llvm-project/mlir/lib/Bindings/Python")
  set(MLIR_PYTHON_BINDINGS_DIR "/data/felix/llvm-project/mlir/lib/Bindings/Python")
else()
  message(FATAL_ERROR "Cannot find MLIR Python bindings source directory")
endif()
message(STATUS "MLIR Python bindings dir: ${MLIR_PYTHON_BINDINGS_DIR}")

# MLIR Python binding source files that we need to compile
# IRModule.cpp contains PyGlobals implementation (constructor/destructor)
# We need it for the populate* functions, but we create PyGlobals in our module
set(MLIR_PYTHON_CORE_SOURCES
  ${MLIR_PYTHON_BINDINGS_DIR}/IRModule.cpp
  ${MLIR_PYTHON_BINDINGS_DIR}/IRAttributes.cpp
  ${MLIR_PYTHON_BINDINGS_DIR}/IRAffine.cpp
  ${MLIR_PYTHON_BINDINGS_DIR}/IRCore.cpp
  ${MLIR_PYTHON_BINDINGS_DIR}/IRInterfaces.cpp
  ${MLIR_PYTHON_BINDINGS_DIR}/IRTypes.cpp
  ${MLIR_PYTHON_BINDINGS_DIR}/Pass.cpp
  ${MLIR_PYTHON_BINDINGS_DIR}/Rewrite.cpp
)

# Define MLIR_PYTHON_PACKAGE_PREFIX to match nanobind's auto-registered paths.
# C++ does nb::module_::import_(PREFIX "ir") — this must resolve to a module
# that nanobind has already put in sys.modules. With "_flir_ir._mlir.", C++ will
# look for "_flir_ir._mlir.ir" which nanobind auto-registers as a submodule.
add_compile_definitions(MLIR_PYTHON_PACKAGE_PREFIX=_flir_ir._mlir.)

# Create the unified module
nanobind_add_module(_flir_ir
  NB_DOMAIN flir
  FREE_THREADED
  FlirIRModule.cpp
  ${MLIR_PYTHON_CORE_SOURCES}
)

# Include paths
target_include_directories(_flir_ir PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  # MLIR Python binding internal headers
  ${MLIR_PYTHON_BINDINGS_DIR}
  # FLIR headers
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_BINARY_DIR}/include
)

# Compile options - hidden visibility
target_compile_options(_flir_ir PRIVATE
  -fvisibility=hidden
  -fvisibility-inlines-hidden
  -frtti
  -fexceptions
)

# Link libraries - static MLIR CAPI libraries
target_link_libraries(_flir_ir PRIVATE
  # Core MLIR CAPI (static)
  MLIRCAPIIR
  MLIRCAPIInterfaces
  MLIRCAPIDebug
  MLIRCAPITransforms
  MLIRCAPIRegisterEverything
  MLIRCAPIExecutionEngine
  MLIRCAPIConversion
  MLIRCAPIAsync
  # Dialect CAPIs
  MLIRCAPIArith
  MLIRCAPIFunc
  MLIRCAPIGPU
  MLIRCAPILLVM
  MLIRCAPIMath
  MLIRCAPIMemRef
  MLIRCAPISCF
  MLIRCAPIVector
  MLIRCAPIROCDL
  MLIRCAPILinalg
  MLIRCAPITensor
  MLIRCAPIQuant
  MLIRCAPISparseTensor
  # LLVM (for ExecutionEngine)
  LLVMSupport
  # FLIR
  FlirDialect
  FlirTransforms
)

# Symbol hiding (Linux only) - same approach as triton-lang/triton
if(UNIX AND NOT APPLE)
  # Version script: only export PyInit__flir_ir, hide everything else
  set(_version_script "${CMAKE_CURRENT_BINARY_DIR}/flir_ir_exports.lds")
  file(WRITE "${_version_script}" "{ global: PyInit__flir_ir; local: *; };\n")
  target_link_options(_flir_ir PRIVATE
    # Hide all symbols from static libraries
    "LINKER:--exclude-libs,ALL"
    "LINKER:--version-script,${_version_script}"
  )
endif()

# Output to python_packages directory (under flydsl/_mlir/_mlir_libs)
set(_flir_ir_output_dir "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs")

set_target_properties(_flir_ir PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${_flir_ir_output_dir}"
  CXX_VISIBILITY_PRESET hidden
  C_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

# Copy __init__.py to output directory
# (Full Python package setup is done by setup_python_package.sh in build.sh)
add_custom_command(
  TARGET _flir_ir POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
    "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py"
    "${_flir_ir_output_dir}/__init__.py"
  COMMENT "Copying _flir_ir __init__.py"
)
