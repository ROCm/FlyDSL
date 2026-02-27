include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=flydsl.${MLIR_PYTHON_PACKAGE_PREFIX}.")

declare_mlir_python_sources(FlyPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT FlyPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/"
  TD_FILE dialects/FlyOps.td
  SOURCES
    dialects/fly.py
    _mlir_libs/_mlirRegisterEverything/py.typed
  DIALECT_NAME fly
  GEN_ENUM_BINDINGS
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT FlyPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/"
  TD_FILE dialects/FlyROCDL.td
  SOURCES
    dialects/fly_rocdl.py
  DIALECT_NAME fly_rocdl
  GEN_ENUM_BINDINGS
)

# NOTE: Do NOT link MLIRFlyDialect/MLIRFlyROCDLDialect here via PRIVATE_LINK_LIBS.
# These symbols are already provided by FlyPythonCAPI.so (via MLIRCPIFly's transitive
# dependencies in EMBED_CAPI_LINK_LIBS). Statically linking them here creates DUPLICATE
# TypeID static variables, causing "storage uniquer isn't initialized" errors at runtime.
# The COMMON_CAPI_LINK_LIBS FlyPythonCAPI in add_mlir_python_modules ensures _fly.so
# links to FlyPythonCAPI.so, which provides all needed symbols.

declare_mlir_python_extension(FlyPythonSources.Core.fly
  MODULE_NAME _fly
  ADD_TO_PARENT FlyPythonSources
  ROOT_DIR "${PROJECT_SOURCE_DIR}/lib/Bindings/Python"
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    FlyExtension.cpp
    DLTensorAdaptor.h
  PRIVATE_LINK_LIBS
    LLVMSupport
)

declare_mlir_python_extension(FlyPythonSources.Core.fly_rocdl
  MODULE_NAME _fly_rocdl
  ADD_TO_PARENT FlyPythonSources
  ROOT_DIR "${PROJECT_SOURCE_DIR}/lib/Bindings/Python"
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    FlyROCDLExtension.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
)

# NOTE: Do NOT link MLIRFlyToROCDL or other C++ libs via PRIVATE_LINK_LIBS.
# Doing so statically embeds a copy of MLIRPass (and its pass registry) into
# _mlirRegisterEverything.so. Passes registered via registerFlyPasses() then
# go into that LOCAL registry, while PassManager.parse() queries the GLOBAL
# registry in FlyPythonCAPI.so → "does not refer to a registered pass".
#
# Instead, pass registration is done through CAPI functions
# (mlirRegisterFlyPasses, mlirRegisterFlyToROCDLConversionPass) defined in
# the CAPI libraries (MLIRCPIFly, MLIRCPIFlyROCDL). These are already in
# FlyPythonCAPI.so, so ::mlir::registerPass() resolves to the correct
# global instance.
declare_mlir_python_extension(FlyPythonSources.RegisterEverything
  MODULE_NAME _mlirRegisterEverything
  ADD_TO_PARENT FlyPythonSources
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    FlyRegisterEverything.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCPIFly
    MLIRCPIFlyROCDL
    MLIRCAPIArith
    MLIRCAPIGPU
    MLIRCAPILLVM
    MLIRCAPIMath
    MLIRCAPIVector
    MLIRCAPIConversion
    MLIRCAPITransforms
    MLIRCAPIRegisterEverything
)


set(MLIRFlyDSLSources
  FlyPythonSources
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects.builtin
  MLIRPythonSources.Dialects.arith
  MLIRPythonSources.Dialects.math
  MLIRPythonSources.Dialects.memref
  MLIRPythonSources.Dialects.gpu
  MLIRPythonSources.Dialects.func
  MLIRPythonSources.Dialects.cf
  MLIRPythonSources.Dialects.scf
  MLIRPythonSources.Dialects.rocdl
  MLIRPythonSources.Dialects.vector
  MLIRPythonSources.Dialects.llvm
  MLIRPythonSources.ExecutionEngine
)

add_mlir_python_common_capi_library(FlyPythonCAPI
  INSTALL_COMPONENT FlyPythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES ${MLIRFlyDSLSources}
)


set(FlyPythonModules_ROOT_PREFIX "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}")

################################################################################
# Python Modules
################################################################################

add_mlir_python_modules(FlyPythonModules
  ROOT_PREFIX "${FlyPythonModules_ROOT_PREFIX}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  # Use "mlir" domain so Fly types are compatible with upstream IR types.
  # The package is self-contained under flydsl._mlir (no upstream mlir loaded).
  MLIR_BINDINGS_PYTHON_NB_DOMAIN "mlir"
  DECLARED_SOURCES "${MLIRFlyDSLSources}"
  COMMON_CAPI_LINK_LIBS
    FlyPythonCAPI
)

################################################################################
# Type Stubs Generation
################################################################################

set(_FLYDSL_PYTHON_PACKAGES_DIR "${MLIR_BINARY_DIR}/python_packages")
set(_MLIR_LIBS_DIR "${FlyPythonModules_ROOT_PREFIX}/_mlir_libs")
set(_STUB_MARKER_FILE "${_MLIR_LIBS_DIR}/.stubs_generated")

add_custom_command(
  OUTPUT "${_STUB_MARKER_FILE}"
  COMMAND ${CMAKE_COMMAND} -E env
    "PYTHONPATH=${_FLYDSL_PYTHON_PACKAGES_DIR}"
    ${Python3_EXECUTABLE} -m nanobind.stubgen
    -q
    -r
    -m flydsl._mlir._mlir_libs._mlir
    -m flydsl._mlir._mlir_libs._fly
    -m flydsl._mlir._mlir_libs._fly_rocdl
    -m flydsl._mlir._mlir_libs._mlirDialectsGPU
    -m flydsl._mlir._mlir_libs._mlirDialectsLLVM
    -O "${_MLIR_LIBS_DIR}"
  COMMAND ${CMAKE_COMMAND} -E touch "${_STUB_MARKER_FILE}"
  DEPENDS CopyFlyPythonSources
  COMMENT "Generating Python stub files for all extension modules"
  VERBATIM
)

add_custom_target(FlyPythonStubs ALL
  DEPENDS "${_STUB_MARKER_FILE}"
)

add_custom_target(CopyFlyPythonSources ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${PROJECT_SOURCE_DIR}/python/flydsl"
    "${MLIR_BINARY_DIR}/python_packages/flydsl"
  # Copy auto-generated Python bindings from mlir-tblgen to the correct location
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${MLIR_BINARY_DIR}/python/mlir_flydsl/dialects/_fly_ops_gen.py"
    "${MLIR_BINARY_DIR}/python_packages/flydsl/_mlir/dialects/_fly_ops_gen.py"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${MLIR_BINARY_DIR}/python/mlir_flydsl/dialects/_fly_enum_gen.py"
    "${MLIR_BINARY_DIR}/python_packages/flydsl/_mlir/dialects/_fly_enum_gen.py"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${MLIR_BINARY_DIR}/python/mlir_flydsl/dialects/_fly_rocdl_ops_gen.py"
    "${MLIR_BINARY_DIR}/python_packages/flydsl/_mlir/dialects/_fly_rocdl_ops_gen.py"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${MLIR_BINARY_DIR}/python/mlir_flydsl/dialects/_fly_rocdl_enum_gen.py"
    "${MLIR_BINARY_DIR}/python_packages/flydsl/_mlir/dialects/_fly_rocdl_enum_gen.py"
  COMMENT "Copying python/flydsl sources to build/python_packages/flydsl"
  DEPENDS FlyPythonModules
)

add_dependencies(CopyFlyPythonSources FlyPythonModules)

################################################################################
# Custom JIT runtime with CUDA graph capture support
################################################################################

find_package(hip QUIET CONFIG PATHS /opt/rocm)
if(hip_FOUND)
  add_library(FlyJitRuntime SHARED
    ${PROJECT_SOURCE_DIR}/flir/python_bindings/runtime/FlirRocmRuntimeWrappers.cpp
  )
  target_include_directories(FlyJitRuntime PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
  )
  target_compile_features(FlyJitRuntime PRIVATE cxx_std_17)
  target_link_libraries(FlyJitRuntime PRIVATE hip::host hip::amdhip64)
  set_target_properties(FlyJitRuntime PROPERTIES
    OUTPUT_NAME "fly_jit_runtime"
    LIBRARY_OUTPUT_DIRECTORY "${_MLIR_LIBS_DIR}"
  )
  add_dependencies(FlyPythonCAPI FlyJitRuntime)
endif()

################################################################################
# Create symlinks to MLIR runtime libraries
################################################################################

set(_MLIR_RUNTIME_LIBS
  mlir_runner_utils
  mlir_c_runner_utils
)

get_filename_component(_LLVM_LIB_DIR "${MLIR_DIR}/../../../lib" ABSOLUTE)

foreach(_lib ${_MLIR_RUNTIME_LIBS})
  set(_src_lib "${_LLVM_LIB_DIR}/lib${_lib}.so")
  set(_dst_link "${_MLIR_LIBS_DIR}/lib${_lib}.so")
  add_custom_command(
    OUTPUT "${_dst_link}"
    COMMAND ${CMAKE_COMMAND} -E create_symlink "${_src_lib}" "${_dst_link}"
    DEPENDS FlyPythonModules
    COMMENT "Creating symlink to lib${_lib}.so"
  )
  list(APPEND _RUNTIME_SYMLINKS "${_dst_link}")
endforeach()

# Symlink upstream libmlir_rocm_runtime.so as fallback (the custom FlyJitRuntime
# takes precedence at load time via jit_executor.py).
set(_src_rocm "${_LLVM_LIB_DIR}/libmlir_rocm_runtime.so")
set(_dst_rocm "${_MLIR_LIBS_DIR}/libmlir_rocm_runtime.so")
add_custom_command(
  OUTPUT "${_dst_rocm}"
  COMMAND ${CMAKE_COMMAND} -E create_symlink "${_src_rocm}" "${_dst_rocm}"
  DEPENDS FlyPythonModules
  COMMENT "Creating symlink to libmlir_rocm_runtime.so"
)
list(APPEND _RUNTIME_SYMLINKS "${_dst_rocm}")

add_custom_target(MlirRuntimeSymlinks ALL DEPENDS ${_RUNTIME_SYMLINKS})
