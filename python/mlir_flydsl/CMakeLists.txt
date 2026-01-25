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


declare_mlir_python_extension(FlyPythonSources.Core
  MODULE_NAME _fly
  ADD_TO_PARENT FlyPythonSources
  ROOT_DIR "${PROJECT_SOURCE_DIR}/lib/Bindings/Python"
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    MainModules.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
    MLIRFlyDialect
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
)



declare_mlir_python_extension(FlyPythonSources.RegisterEverything
  MODULE_NAME _mlirRegisterEverything
  ADD_TO_PARENT FlyPythonSources
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    FlyRegisterEverything.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
    MLIRFlyToROCDL
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCPIFly
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

# set(_core_type_stub_sources
#   _mlir/__init__.pyi
#   _mlir/ir.pyi
#   _mlir/passmanager.pyi
#   _mlir/rewrite.pyi
# )
# get_target_property(_core_extension_srcs MLIRPythonExtension.Core INTERFACE_SOURCES)
# mlir_generate_type_stubs(
#   MODULE_NAME _mlir
#   DEPENDS_TARGETS FlyPythonModules.extension._mlir.dso
#   OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs/_mlir_libs"
#   OUTPUTS "${_core_type_stub_sources}"
#   DEPENDS_TARGET_SRC_DEPS "${_core_extension_srcs}"
#   IMPORT_PATHS "${FlyPythonModules_ROOT_PREFIX}/_mlir_libs"
#   VERBOSE
# )
# set(_mlir_typestub_gen_target "${NB_STUBGEN_CUSTOM_TARGET}")

# list(TRANSFORM _core_type_stub_sources PREPEND "_mlir_libs/")
# declare_mlir_python_sources(
#   FlyPythonExtension.Core.type_stub_gen
#   ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs"
#   ADD_TO_PARENT FlyPythonSources
#   SOURCES "${_core_type_stub_sources}"
# )



add_mlir_python_modules(FlyPythonModules
  ROOT_PREFIX "${FlyPythonModules_ROOT_PREFIX}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES "${MLIRFlyDSLSources}"
  COMMON_CAPI_LINK_LIBS
    FlyPythonCAPI
)

add_custom_target(CopyFlyPythonSources ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${PROJECT_SOURCE_DIR}/python/flydsl"
    "${MLIR_BINARY_DIR}/python_packages/flydsl"
  COMMENT "Copying python/flydsl sources to build/python_packages/flydsl"
  DEPENDS FlyPythonModules
)

add_dependencies(CopyFlyPythonSources FlyPythonModules)
