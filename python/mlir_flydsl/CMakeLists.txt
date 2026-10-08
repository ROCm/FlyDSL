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

if(FLYDSL_HAS_ROCDL)
  declare_mlir_dialect_python_bindings(
    ADD_TO_PARENT FlyPythonSources
    ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/"
    TD_FILE dialects/FlyROCDL.td
    SOURCES
      dialects/fly_rocdl.py
    DIALECT_NAME fly_rocdl
    GEN_ENUM_BINDINGS
  )
endif()

# ELF resolves these dialect symbols through FlyPythonCAPI.so. COFF extensions need
# explicit import libraries for their C++ dialect methods.
# COMMON_CAPI_LINK_LIBS keeps FlyPythonCAPI linked into each extension.

set(_fly_extension_link_libs LLVMSupport)

declare_mlir_python_extension(FlyPythonSources.Core.fly
  MODULE_NAME _mlirDialectsFly
  ADD_TO_PARENT FlyPythonSources
  ROOT_DIR "${PROJECT_SOURCE_DIR}/lib/Bindings/Python"
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    TiledOpTraits.cpp
    FlyExtension.cpp
  PRIVATE_LINK_LIBS ${_fly_extension_link_libs}
)

if(FLYDSL_HAS_ROCDL)
  set(_fly_rocdl_extension_link_libs ${_fly_extension_link_libs})
  declare_mlir_python_extension(FlyPythonSources.Core.fly_rocdl
    MODULE_NAME _mlirDialectsFlyROCDL
    ADD_TO_PARENT FlyPythonSources
    ROOT_DIR "${PROJECT_SOURCE_DIR}/lib/Bindings/Python"
    PYTHON_BINDINGS_LIBRARY nanobind
    SOURCES
      FlyROCDLExtension.cpp
    PRIVATE_LINK_LIBS ${_fly_rocdl_extension_link_libs}
  )
endif()

# NOTE: Do NOT link MLIRFlyToROCDL or other C++ libs via PRIVATE_LINK_LIBS.
# Doing so statically embeds a copy of MLIRPass (and its pass registry) into
# _mlirRegisterEverything.so. Passes registered via registerFlyPasses() then
# go into that LOCAL registry, while PassManager.parse() queries the GLOBAL
# registry in FlyPythonCAPI.so → "does not refer to a registered pass".
#
# Instead, pass registration is done through CAPI functions
# (mlirRegisterFlyPasses, mlirRegisterConvertROCDLFastMathOpsPass,
# mlirRegisterFlyToROCDLConversionPass) defined in
# the CAPI libraries (MLIRCPIFly, MLIRCPIFlyROCDL). These are already in
# FlyPythonCAPI.so, so ::mlir::registerPass() resolves to the correct
# global instance.
get_property(_backend_embed_capi_libs GLOBAL PROPERTY FLYDSL_BACKEND_EMBED_CAPI_LIBS)
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
    ${_backend_embed_capi_libs}
    MLIRCPIHostObject
    MLIRCPILLVMOption
    MLIRCAPIArith
    MLIRCAPIGPU
    MLIRCAPILLVM
    MLIRCAPIMath
    MLIRCAPIVector
    MLIRCAPIConversion
    MLIRCAPITransforms
    MLIRCAPIRegisterEverything
)


get_property(_backend_upstream_dialect_sources GLOBAL PROPERTY FLYDSL_BACKEND_UPSTREAM_DIALECT_SOURCES)
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
  MLIRPythonSources.Dialects.vector
  MLIRPythonSources.Dialects.llvm
  MLIRPythonSources.ExecutionEngine
  ${_backend_upstream_dialect_sources}
)

add_mlir_python_common_capi_library(FlyPythonCAPI
  INSTALL_COMPONENT FlyPythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES ${MLIRFlyDSLSources}
)

if(WIN32)
  set_target_properties(FlyPythonCAPI PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# ---------------------------------------------------------------------------
# Symbol visibility: prevent LLVM/MLIR internals from leaking out of
# FlyPythonCAPI.so, which would conflict with ROCm's own LLVM
# (libamd_comgr.so).
#
# Strategy:
#   1. -fvisibility=hidden  — hide symbols compiled by FlyDSL itself
#   2. Version script       — controls the FINAL exported symbol set:
#                             exports mlir C API + mlir:: C++ symbols
#                             (needed by _mlirDialectsFly.so, _mlir.so, etc.),
#                             hides everything else (llvm::, LLVM C API).
#
# NOTE: Triton uses --exclude-libs,ALL because it ships a single
# libtriton.so with no downstream consumers.  FlyDSL cannot use
# --exclude-libs,ALL because GNU ld's --exclude-libs forces symbols
# to LOCAL *before* the version script is evaluated — wildcard patterns
# in the version script cannot re-export them.  The version script
# alone is sufficient: its `local: *;` catch-all hides LLVM internals.
# ---------------------------------------------------------------------------
if(UNIX AND NOT APPLE)
  target_compile_options(FlyPythonCAPI PRIVATE -fvisibility=hidden)

  set(FLYDSL_VERSION_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/flydsl_exports.ld")
  target_link_options(FlyPythonCAPI PRIVATE
    -Wl,--version-script=${FLYDSL_VERSION_SCRIPT})
  set_property(TARGET FlyPythonCAPI APPEND PROPERTY
    LINK_DEPENDS "${FLYDSL_VERSION_SCRIPT}")
endif()


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

if(WIN32)
  # Multi-config generators append the configuration name to generic output
  # directories. Keep extensions and their runtime DLLs beside Python packages.
  add_dependencies(FlyPythonCAPI MLIRFlyDialect MLIRFlyROCDLDialect)
  target_sources(FlyPythonCAPI PRIVATE
    $<TARGET_OBJECTS:obj.MLIRFlyDialect>
    $<TARGET_OBJECTS:obj.MLIRFlyROCDLDialect>)

  # Export dialect C++ APIs from this single DLL so all Python extensions share
  # the same type IDs and storage uniquers.
  set_target_properties(FlyPythonCAPI PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
  set_property(TARGET FlyPythonCAPI APPEND PROPERTY MLIR_AGGREGATE_EXCLUDE_LIBS
    MLIRIR MLIRSupport MLIRFlyDialect MLIRFlyROCDLDialect)
  get_target_property(_flypython_links FlyPythonCAPI LINK_LIBRARIES)
  list(FILTER _flypython_links EXCLUDE REGEX "MLIRIR|MLIRSupport")
  set_property(TARGET FlyPythonCAPI PROPERTY LINK_LIBRARIES "${_flypython_links}")

  set(_fly_export_archives MLIRIR MLIRSupport)
  set(_fly_export_objects)
  foreach(_archive IN LISTS _fly_export_archives)
    set(_archive_path "${MLIR_DIR}/../../../lib/${_archive}.lib")
    set(_extract_dir "${CMAKE_BINARY_DIR}/extracted_objs/${_archive}")
    if(NOT EXISTS "${_archive_path}")
      message(FATAL_ERROR "Required MLIR archive not found: ${_archive_path}")
    endif()
    file(MAKE_DIRECTORY "${_extract_dir}")
    execute_process(
      COMMAND "${LLVM_TOOLS_BINARY_DIR}/llvm-ar.exe" t "${_archive_path}"
      OUTPUT_VARIABLE _archive_members
      RESULT_VARIABLE _list_result
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_VARIABLE _list_error)
    if(NOT _list_result EQUAL 0)
      message(FATAL_ERROR "Could not list ${_archive_path}: ${_list_error}")
    endif()
    execute_process(
      COMMAND "${LLVM_TOOLS_BINARY_DIR}/llvm-ar.exe" x "${_archive_path}"
      WORKING_DIRECTORY "${_extract_dir}"
      RESULT_VARIABLE _extract_result
      ERROR_VARIABLE _extract_error)
    if(NOT _extract_result EQUAL 0)
      message(FATAL_ERROR "Could not extract ${_archive_path}: ${_extract_error}")
    endif()
    string(REPLACE "\n" ";" _archive_members "${_archive_members}")
    list(FILTER _archive_members EXCLUDE REGEX "^$")
    set(_archive_objects)
    foreach(_member IN LISTS _archive_members)
      get_filename_component(_member_name "${_member}" NAME)
      list(APPEND _archive_objects "${_extract_dir}/${_member_name}")
    endforeach()
    list(APPEND _fly_export_objects ${_archive_objects})
  endforeach()
  target_sources(FlyPythonCAPI PRIVATE ${_fly_export_objects})

  get_property(_fly_python_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
  foreach(_target IN LISTS _fly_python_targets)
    if(_target MATCHES "^FlyPythonModules\.extension\." OR
       _target STREQUAL "FlyPythonCAPI" OR _target STREQUAL "FlyJitRuntime" OR
       _target STREQUAL "nanobind-mlir")
      foreach(_config IN LISTS CMAKE_CONFIGURATION_TYPES)
        string(TOUPPER "${_config}" _config_upper)
        set_target_properties(${_target} PROPERTIES
          RUNTIME_OUTPUT_DIRECTORY_${_config_upper} "${FlyPythonModules_ROOT_PREFIX}/_mlir_libs"
          LIBRARY_OUTPUT_DIRECTORY_${_config_upper} "${FlyPythonModules_ROOT_PREFIX}/_mlir_libs")
      endforeach()
    endif()
  endforeach()
endif()

################################################################################
# Type Stubs Generation
################################################################################

set(_FLYDSL_PYTHON_PACKAGES_DIR "${MLIR_BINARY_DIR}/python_packages")
set(_MLIR_LIBS_DIR "${FlyPythonModules_ROOT_PREFIX}/_mlir_libs")
set(_STUB_MARKER_FILE "${_MLIR_LIBS_DIR}/.stubs_generated")
set(_STUBGEN_PYTHONPATH "${_FLYDSL_PYTHON_PACKAGES_DIR}")
if(DEFINED nanobind_DIR AND NOT "${nanobind_DIR}" STREQUAL "")
  get_filename_component(_NANOBIND_MODULE_DIR "${nanobind_DIR}/.." ABSOLUTE)
  get_filename_component(_NANOBIND_SITE_PACKAGES_DIR "${_NANOBIND_MODULE_DIR}/.." ABSOLUTE)
  if(EXISTS "${_NANOBIND_SITE_PACKAGES_DIR}" AND
     (EXISTS "${_NANOBIND_SITE_PACKAGES_DIR}/nanobind" OR
      EXISTS "${_NANOBIND_SITE_PACKAGES_DIR}/nanobind.py"))
    list(APPEND _STUBGEN_PYTHONPATH "${_NANOBIND_SITE_PACKAGES_DIR}")
  endif()
endif()
if(WIN32)
  list(JOIN _STUBGEN_PYTHONPATH ";" _STUBGEN_PYTHONPATH_VALUE)
else()
  list(JOIN _STUBGEN_PYTHONPATH ":" _STUBGEN_PYTHONPATH_VALUE)
endif()

get_property(_backend_stubgen_modules GLOBAL PROPERTY FLYDSL_BACKEND_STUBGEN_MODULES)
set(_flydsl_stubgen_modules
  flydsl._mlir._mlir_libs._mlir
  flydsl._mlir._mlir_libs._mlirDialectsFly
  ${_backend_stubgen_modules}
)
list(APPEND _flydsl_stubgen_modules
  flydsl._mlir._mlir_libs._mlirDialectsGPU
  flydsl._mlir._mlir_libs._mlirDialectsLLVM
)
set(_flydsl_stubgen_args -q -r)
set(_flydsl_stubgen_m_flags "")
foreach(_m IN LISTS _flydsl_stubgen_modules)
  list(APPEND _flydsl_stubgen_args -m "${_m}")
  string(APPEND _flydsl_stubgen_m_flags " -m ${_m}")
endforeach()
list(APPEND _flydsl_stubgen_args -O "${_MLIR_LIBS_DIR}")

if(UNIX)
  set(_FLY_PATCH_RPATH_COMMANDS
    COMMAND bash -c "patchelf --set-rpath '\\$$ORIGIN' '${_MLIR_LIBS_DIR}/$<TARGET_FILE_NAME:mlir_c_runner_utils>'"
    COMMAND bash -c "patchelf --set-rpath '\\$$ORIGIN' '${_MLIR_LIBS_DIR}/$<TARGET_FILE_NAME:mlir_float16_utils>'"
  )
endif()

if(WIN32)
  add_custom_command(
    OUTPUT "${_STUB_MARKER_FILE}"
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${_STUBGEN_PYTHONPATH_VALUE};$ENV{PYTHONPATH}"
      "${Python3_EXECUTABLE}" -m nanobind.stubgen ${_flydsl_stubgen_args}
    COMMAND ${CMAKE_COMMAND} -E touch "${_STUB_MARKER_FILE}"
    DEPENDS CopyFlyPythonSources
    COMMENT "Generating Python stub files for all extension modules"
    VERBATIM
  )
else()
  add_custom_command(
    OUTPUT "${_STUB_MARKER_FILE}"
    COMMAND /bin/bash -c "\
      export PYTHONPATH='${_STUBGEN_PYTHONPATH_VALUE}'\"\${PYTHONPATH:+:\${PYTHONPATH}}\"; \
      if '${Python3_EXECUTABLE}' -c 'import nanobind.stubgen' >/dev/null 2>&1; then \
        '${Python3_EXECUTABLE}' -m nanobind.stubgen \
          -q -r${_flydsl_stubgen_m_flags} \
          -O '${_MLIR_LIBS_DIR}'; \
      else \
        echo 'Warning: nanobind.stubgen not available -- skipping stub generation'; \
      fi"
    COMMAND ${CMAKE_COMMAND} -E touch "${_STUB_MARKER_FILE}"
    DEPENDS CopyFlyPythonSources
    COMMENT "Generating Python stub files for all extension modules"
    VERBATIM
  )
endif()

add_custom_target(FlyPythonStubs ALL
  DEPENDS "${_STUB_MARKER_FILE}"
)

if(FLYDSL_HAS_ROCDL)
  set(_FLY_COPY_ROCDL_TABLEGEN
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${MLIR_BINARY_DIR}/python/mlir_flydsl/dialects/_fly_rocdl_ops_gen.py"
      "${MLIR_BINARY_DIR}/python_packages/flydsl/_mlir/dialects/_fly_rocdl_ops_gen.py"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${MLIR_BINARY_DIR}/python/mlir_flydsl/dialects/_fly_rocdl_enum_gen.py"
      "${MLIR_BINARY_DIR}/python_packages/flydsl/_mlir/dialects/_fly_rocdl_enum_gen.py"
  )
else()
  set(_FLY_COPY_ROCDL_TABLEGEN "")
endif()

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
  ${_FLY_COPY_ROCDL_TABLEGEN}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE:mlir_c_runner_utils>"
    "${_MLIR_LIBS_DIR}/$<TARGET_FILE_NAME:mlir_c_runner_utils>"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE:mlir_float16_utils>"
    "${_MLIR_LIBS_DIR}/$<TARGET_FILE_NAME:mlir_float16_utils>"
  ${_FLY_PATCH_RPATH_COMMANDS}
  COMMENT "Copying python/flydsl sources to build/python_packages/flydsl"
  DEPENDS FlyPythonModules
)

add_dependencies(CopyFlyPythonSources FlyPythonModules)

################################################################################
# FlyJitRuntime output + dependency (built in lib/Runtime/)
################################################################################
if(TARGET FlyJitRuntime)
  set_target_properties(FlyJitRuntime PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${_MLIR_LIBS_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${_MLIR_LIBS_DIR}")
  add_dependencies(FlyPythonCAPI FlyJitRuntime)
endif()

if(TARGET FlyRocmAotRuntime)
  set_target_properties(FlyRocmAotRuntime PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${_MLIR_LIBS_DIR}")
  add_dependencies(FlyPythonCAPI FlyRocmAotRuntime)
endif()
