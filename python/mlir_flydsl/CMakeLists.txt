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

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT FlyPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/"
  TD_FILE dialects/FlyKtrace.td
  SOURCES
    dialects/fly_ktrace.py
  DIALECT_NAME fly_ktrace
)

# NOTE: Do NOT link MLIRFlyDialect/MLIRFlyROCDLDialect here via PRIVATE_LINK_LIBS.
# These symbols are already provided by FlyPythonCAPI.so (via MLIRCPIFly's transitive
# dependencies in EMBED_CAPI_LINK_LIBS). Statically linking them here creates DUPLICATE
# TypeID static variables, causing "storage uniquer isn't initialized" errors at runtime.
# The COMMON_CAPI_LINK_LIBS FlyPythonCAPI in add_mlir_python_modules ensures _mlirDialectsFly.so
# links to FlyPythonCAPI.so, which provides all needed symbols.

declare_mlir_python_extension(FlyPythonSources.Core.fly
  MODULE_NAME _mlirDialectsFly
  ADD_TO_PARENT FlyPythonSources
  ROOT_DIR "${PROJECT_SOURCE_DIR}/lib/Bindings/Python"
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    TiledOpTraits.cpp
    FlyExtension.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
)

if(FLYDSL_HAS_ROCDL)
  declare_mlir_python_extension(FlyPythonSources.Core.fly_rocdl
    MODULE_NAME _mlirDialectsFlyROCDL
    ADD_TO_PARENT FlyPythonSources
    ROOT_DIR "${PROJECT_SOURCE_DIR}/lib/Bindings/Python"
    PYTHON_BINDINGS_LIBRARY nanobind
    SOURCES
      FlyROCDLExtension.cpp
    PRIVATE_LINK_LIBS
      LLVMSupport
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

# ---------------------------------------------------------------------------
# Windows: FlyPythonCAPI.dll must also be the single home of the C++ code the
# Fly extensions call besides the C API (BindingUtils.h: getTypeID, isa/cast):
# MLIR IR and Support, and the Fly dialects. On Linux the version script below
# exports it from the archives linked in. A DLL exports only what
# WINDOWS_EXPORT_ALL_SYMBOLS finds in the target's own objects, never archive
# members, so those objects become sources. Ninja puts them ahead of the
# archives on the link line, so the archive copies are never pulled in.
# ---------------------------------------------------------------------------
if(WIN32)
  set(_fly_dialect_libs MLIRFlyDialect MLIRFlyKtraceDialect)
  if(FLYDSL_HAS_ROCDL)
    list(APPEND _fly_dialect_libs MLIRFlyROCDLDialect)
  endif()
  foreach(_lib IN LISTS _fly_dialect_libs)
    # add_mlir_library makes obj.<lib> for every generator except Visual Studio's.
    if(NOT TARGET obj.${_lib})
      message(FATAL_ERROR "obj.${_lib} is missing: build FlyDSL with the Ninja generator")
    endif()
    target_sources(FlyPythonCAPI PRIVATE $<TARGET_OBJECTS:obj.${_lib}>)
  endforeach()

  # The MLIR install ships object files only for its C API libraries, so unpack
  # MLIRIR's and MLIRSupport's archives. Configure depends on the archives: an
  # LLVM reinstalled in place re-runs this instead of linking stale objects.
  find_program(FLY_LLVM_AR llvm-ar HINTS "${LLVM_TOOLS_BINARY_DIR}" NO_DEFAULT_PATH REQUIRED)
  foreach(_lib MLIRIR MLIRSupport)
    get_target_property(_archive ${_lib} LOCATION)
    set(_dir "${CMAKE_CURRENT_BINARY_DIR}/archive_objects/${_lib}")
    file(REMOVE_RECURSE "${_dir}")
    file(MAKE_DIRECTORY "${_dir}")
    execute_process(COMMAND "${FLY_LLVM_AR}" x "${_archive}"
                    WORKING_DIRECTORY "${_dir}" RESULT_VARIABLE _rc)
    if(NOT _rc EQUAL 0)
      message(FATAL_ERROR "llvm-ar could not unpack ${_archive}")
    endif()
    file(GLOB _objects "${_dir}/*.obj")
    target_sources(FlyPythonCAPI PRIVATE ${_objects})
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_archive}")
  endforeach()
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
list(JOIN _STUBGEN_PYTHONPATH ":" _STUBGEN_PYTHONPATH_VALUE)

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
set(_flydsl_stubgen_m_flags "")
foreach(_m IN LISTS _flydsl_stubgen_modules)
  string(APPEND _flydsl_stubgen_m_flags " -m ${_m}")
endforeach()

if(WIN32)
  # Same stubs without /bin/bash; PYTHONPATH is ';'-separated here. Whether
  # nanobind.stubgen imports is settled at configure time, with that same path.
  list(JOIN _STUBGEN_PYTHONPATH ";" _fly_stubgen_pythonpath)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${_fly_stubgen_pythonpath}"
            "${Python3_EXECUTABLE}" -c "import nanobind.stubgen"
    RESULT_VARIABLE _fly_no_stubgen OUTPUT_QUIET ERROR_QUIET)
  if(_fly_no_stubgen)
    set(_fly_stubgen ${CMAKE_COMMAND} -E echo
        "Warning: nanobind.stubgen not available -- skipping stub generation")
  else()
    separate_arguments(_fly_stubgen_m_args UNIX_COMMAND "${_flydsl_stubgen_m_flags}")
    set(_fly_stubgen "${Python3_EXECUTABLE}" -m nanobind.stubgen
        -q -r ${_fly_stubgen_m_args} -O "${_MLIR_LIBS_DIR}")
  endif()
  add_custom_command(
    OUTPUT "${_STUB_MARKER_FILE}"
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${_fly_stubgen_pythonpath}" ${_fly_stubgen}
    COMMAND ${CMAKE_COMMAND} -E touch "${_STUB_MARKER_FILE}"
    DEPENDS CopyFlyPythonSources
    COMMENT "Generating Python stub files for all extension modules"
    VERBATIM
  )
else()
# Unindented so the quoted bash script keeps its bytes.
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

# The JIT asks for the C runner utils by an unversioned name. Windows keeps the
# DLL names and has no rpath to patch: Python preloads mlir_c_runner_utils.dll,
# and that load finds mlir_float16_utils.dll in the same folder
# (flydsl/runtime/device_runtime/rocm.py).
set(_FLY_C_RUNNER_UTILS_NAME libmlir_c_runner_utils.so)
set(_FLY_PATCHELF_RUNTIME_LIBS
  COMMAND bash -c "patchelf --set-rpath '\\$$ORIGIN' '${_MLIR_LIBS_DIR}/libmlir_c_runner_utils.so'"
  COMMAND bash -c "patchelf --set-rpath '\\$$ORIGIN' '${_MLIR_LIBS_DIR}/$<TARGET_FILE_NAME:mlir_float16_utils>'"
)
if(WIN32)
  set(_FLY_C_RUNNER_UTILS_NAME mlir_c_runner_utils.dll)
  set(_FLY_PATCHELF_RUNTIME_LIBS "")
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
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${MLIR_BINARY_DIR}/python/mlir_flydsl/dialects/_fly_ktrace_ops_gen.py"
    "${MLIR_BINARY_DIR}/python_packages/flydsl/_mlir/dialects/_fly_ktrace_ops_gen.py"
  ${_FLY_COPY_ROCDL_TABLEGEN}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE:mlir_c_runner_utils>"
    "${_MLIR_LIBS_DIR}/${_FLY_C_RUNNER_UTILS_NAME}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE:mlir_float16_utils>"
    "${_MLIR_LIBS_DIR}/$<TARGET_FILE_NAME:mlir_float16_utils>"
  ${_FLY_PATCHELF_RUNTIME_LIBS}
  COMMENT "Copying python/flydsl sources to build/python_packages/flydsl"
  DEPENDS FlyPythonModules
)

add_dependencies(CopyFlyPythonSources FlyPythonModules)

################################################################################
# FlyJitRuntime output + dependency (built in lib/Runtime/)
################################################################################
if(TARGET FlyJitRuntime)
  set_target_properties(FlyJitRuntime PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${_MLIR_LIBS_DIR}")
  # On Windows the DLL is a RUNTIME output, not a LIBRARY one.
  set_property(TARGET FlyJitRuntime PROPERTY RUNTIME_OUTPUT_DIRECTORY "${_MLIR_LIBS_DIR}")
  add_dependencies(FlyPythonCAPI FlyJitRuntime)
endif()

if(TARGET FlyRocmAotRuntime)
  set_target_properties(FlyRocmAotRuntime PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${_MLIR_LIBS_DIR}")
  add_dependencies(FlyPythonCAPI FlyRocmAotRuntime)
endif()

################################################################################
# Native library digest (JIT disk-cache toolchain fingerprint)
################################################################################
# Runs on every build, after everything that writes native files into
# _mlir_libs (the patchelf'd runtime copies in CopyFlyPythonSources included),
# so the table always describes the libraries next to it. Any new target that
# writes into _mlir_libs must be added as a dependency here.
add_custom_target(FlyNativeDigest ALL
  COMMAND ${CMAKE_COMMAND} -DLIBS_DIR=${_MLIR_LIBS_DIR}
    -P ${PROJECT_SOURCE_DIR}/cmake/WriteNativeDigest.cmake
  COMMENT "Fingerprinting native libraries in ${_MLIR_LIBS_DIR}"
  VERBATIM
)
add_dependencies(FlyNativeDigest CopyFlyPythonSources FlyPythonModules FlyPythonCAPI)
foreach(_runtime_target IN ITEMS FlyJitRuntime FlyRocmAotRuntime)
  if(TARGET ${_runtime_target})
    add_dependencies(FlyNativeDigest ${_runtime_target})
  endif()
endforeach()
