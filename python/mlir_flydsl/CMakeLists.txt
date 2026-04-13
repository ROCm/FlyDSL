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

# NOTE: Do NOT link MLIRFlyDialect/MLIRFlyROCDLDialect here via PRIVATE_LINK_LIBS on Linux.
# These symbols are already provided by FlyPythonCAPI.so (via MLIRCPIFly's transitive
# dependencies in EMBED_CAPI_LINK_LIBS). Statically linking them here creates DUPLICATE
# TypeID static variables, causing "storage uniquer isn't initialized" errors at runtime.
# The COMMON_CAPI_LINK_LIBS FlyPythonCAPI in add_mlir_python_modules ensures _mlirDialectsFly.so
# links to FlyPythonCAPI.so, which provides all needed symbols.
#
# Windows: FlyPythonCAPI.dll is built with WINDOWS_EXPORT_ALL_SYMBOLS (set below), so it
# exports ALL C++ symbols from its statically-linked deps (MLIRFlyDialect, MLIRArithDialect,
# etc.) via an auto-generated .def file. The .pyd extensions then resolve all mlir::fly::
# symbols from FlyPythonCAPI.lib at link time — no second static copy, no duplicate
# storage uniquers. This is the correct Windows equivalent of ELF's global symbol table.
if(WIN32)
  set(_FLY_EXTRA_LINK_LIBS "")
  set(_FLY_ROCDL_EXTRA_LINK_LIBS "")
else()
  set(_FLY_EXTRA_LINK_LIBS "")
  set(_FLY_ROCDL_EXTRA_LINK_LIBS "")
endif()

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
    ${_FLY_EXTRA_LINK_LIBS}
)

declare_mlir_python_extension(FlyPythonSources.Core.fly_rocdl
  MODULE_NAME _mlirDialectsFlyROCDL
  ADD_TO_PARENT FlyPythonSources
  ROOT_DIR "${PROJECT_SOURCE_DIR}/lib/Bindings/Python"
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    FlyROCDLExtension.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
    ${_FLY_ROCDL_EXTRA_LINK_LIBS}
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

# ---------------------------------------------------------------------------
# Windows: cross-DLL TypeID resolution via string fallback
#
# MLIR's MLIR_DECLARE_EXPLICIT_SELF_OWNING_TYPE_ID declares a static DATA
# member (SelfOwningTypeID id) with no __declspec(dllimport/dllexport).
# On Windows, static DATA members defined in a DLL cannot be accessed by
# consumers without dllimport — the linker sees an unresolved symbol because
# the .obj references the symbol directly, not via __imp_ indirection.
#
# The correct fix is MLIR_USE_FALLBACK_TYPE_IDS: when true, TypeID uses
# string-based comparison (FallbackTypeIDResolver) instead of pointer
# identity of a cross-DLL static. This is exactly what the comment in
# TypeID.h recommends for "complex shared library setups". No DATA symbols
# cross DLL boundaries, so no dllimport/dllexport is required.
#
# We define this globally so FlyPythonCAPI and all .pyd extensions use the
# same TypeID resolution strategy. The fallback is slightly slower but
# correct and is the standard approach for Windows MLIR ports.
# ---------------------------------------------------------------------------
if(WIN32)
  # MLIR_USE_FALLBACK_TYPE_IDS is set at the top-level CMakeLists.txt so it applies
  # to every compilation unit (including obj.MLIRFlyDialect). See comment there.

  # Export ALL C++ symbols from FlyPythonCAPI.dll via an auto-generated .def file.
  # WINDOWS_EXPORT_ALL_SYMBOLS scans .obj files directly linked into the DLL; it does
  # NOT scan linked static libraries. To make the mlir::fly::* C++ symbols visible to
  # .pyd extensions (which directly #include & call these APIs in FlyExtension.cpp /
  # TiledOpTraits.cpp), we add the OBJECT library outputs of MLIRFlyDialect and
  # MLIRFlyROCDLDialect as direct sources of FlyPythonCAPI. The scan then includes
  # all Fly dialect symbols in the auto-generated exports.def.
  #
  # NOTE: We disable the transitive MLIRFlyDialect/MLIRFlyROCDLDialect link chain
  # on Windows (see lib/CAPI/Dialect/{Fly,FlyROCDL}/CMakeLists.txt) to avoid double
  # linkage of the same symbols.
  set_target_properties(FlyPythonCAPI PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON)

  target_sources(FlyPythonCAPI PRIVATE
    $<TARGET_OBJECTS:obj.MLIRFlyDialect>
    $<TARGET_OBJECTS:obj.MLIRFlyROCDLDialect>
  )
  # TableGen'd headers must exist before FlyPythonCAPI compiles, since the embedded
  # objects reference them via their .cpp.obj metadata.
  add_dependencies(FlyPythonCAPI MLIRFlyDialect MLIRFlyROCDLDialect)

  # --------------------------------------------------------------------------
  # Extract .obj files from upstream MLIR static archives so WINDOWS_EXPORT_ALL_SYMBOLS
  # can scan them and re-export their C++ symbols from FlyPythonCAPI.dll.
  #
  # The .pyd extensions (FlyExtension.cpp, TiledOpTraits.cpp) directly use C++ APIs
  # from MLIRIR (Float8E5M2Type, MLIRContext::getTypeUniquer, etc.) and MLIRSupport
  # (StorageUniquer::*). These live in .lib archives which WINDOWS_EXPORT_ALL_SYMBOLS
  # cannot scan — so we extract their .obj files at configure time and inject them
  # as direct sources. The MSVC/LLD linker resolves symbols from .obj inputs before
  # searching .lib archives, so no duplicate-definition errors arise.
  # --------------------------------------------------------------------------
  set(_FLY_EXPORT_STATIC_LIBS MLIRIR MLIRSupport)
  set(_FLY_EXTRA_OBJS "")
  foreach(_lib IN LISTS _FLY_EXPORT_STATIC_LIBS)
    set(_lib_path "${MLIR_DIR}/../../../lib/${_lib}.lib")
    get_filename_component(_lib_path "${_lib_path}" ABSOLUTE)
    if(NOT EXISTS "${_lib_path}")
      message(FATAL_ERROR "Cannot locate ${_lib}.lib for Windows exports augmentation at ${_lib_path}")
    endif()
    set(_extract_dir "${CMAKE_BINARY_DIR}/extracted_objs/${_lib}")
    file(MAKE_DIRECTORY "${_extract_dir}")
    execute_process(
      COMMAND "${LLVM_TOOLS_BINARY_DIR}/llvm-ar.exe" x "${_lib_path}"
      WORKING_DIRECTORY "${_extract_dir}"
      RESULT_VARIABLE _extract_res
    )
    if(NOT _extract_res EQUAL 0)
      message(FATAL_ERROR "llvm-ar x ${_lib}.lib failed (result=${_extract_res})")
    endif()
    file(GLOB _lib_objs "${_extract_dir}/*.obj")
    list(APPEND _FLY_EXTRA_OBJS ${_lib_objs})
  endforeach()
  target_sources(FlyPythonCAPI PRIVATE ${_FLY_EXTRA_OBJS})
endif()

################################################################################
# Type Stubs Generation
################################################################################

set(_FLYDSL_PYTHON_PACKAGES_DIR "${MLIR_BINARY_DIR}/python_packages")
set(_MLIR_LIBS_DIR "${FlyPythonModules_ROOT_PREFIX}/_mlir_libs")
set(_STUB_MARKER_FILE "${_MLIR_LIBS_DIR}/.stubs_generated")

if(WIN32)
  # NOTE: omit `-m flydsl._mlir._mlir_libs._mlir` on Windows. That extension has
  # submodules (ir, passmanager, rewrite) and nanobind.stubgen -r generates a
  # sibling directory `_mlir/` containing `__init__.pyi` + submodule stubs.
  # On Windows, Python's import system treats that directory as a PEP 420
  # namespace package and shadows the `_mlir.cp312-win_amd64.pyd`, so
  # `from ._mlir import ir` fails. The stubs are ergonomics-only; skipping
  # upstream MLIR's stubs is acceptable. The Fly-specific extensions don't
  # have Python submodules and generate flat `.pyi` files that don't conflict.
  add_custom_command(
    OUTPUT "${_STUB_MARKER_FILE}"
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${_FLYDSL_PYTHON_PACKAGES_DIR}"
      "${Python3_EXECUTABLE}" -m nanobind.stubgen
        -q -r
        -m flydsl._mlir._mlir_libs._mlirDialectsFly
        -m flydsl._mlir._mlir_libs._mlirDialectsFlyROCDL
        -m flydsl._mlir._mlir_libs._mlirDialectsGPU
        -m flydsl._mlir._mlir_libs._mlirDialectsLLVM
        -O "${_MLIR_LIBS_DIR}"
    COMMAND ${CMAKE_COMMAND} -E touch "${_STUB_MARKER_FILE}"
    DEPENDS CopyFlyPythonSources
    COMMENT "Generating Python stub files for all extension modules"
    VERBATIM
  )
else()
  add_custom_command(
    OUTPUT "${_STUB_MARKER_FILE}"
    COMMAND /bin/bash -c "\
      PYTHONPATH='${_FLYDSL_PYTHON_PACKAGES_DIR}' \
      '${Python3_EXECUTABLE}' -m nanobind.stubgen \
        -q -r \
        -m flydsl._mlir._mlir_libs._mlir \
        -m flydsl._mlir._mlir_libs._mlirDialectsFly \
        -m flydsl._mlir._mlir_libs._mlirDialectsFlyROCDL \
        -m flydsl._mlir._mlir_libs._mlirDialectsGPU \
        -m flydsl._mlir._mlir_libs._mlirDialectsLLVM \
        -O '${_MLIR_LIBS_DIR}' \
      || echo 'Warning: nanobind.stubgen not available -- skipping stub generation'"
    COMMAND ${CMAKE_COMMAND} -E touch "${_STUB_MARKER_FILE}"
    DEPENDS CopyFlyPythonSources
    COMMENT "Generating Python stub files for all extension modules"
    VERBATIM
  )
endif()

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
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE:mlir_c_runner_utils>"
    "${_MLIR_LIBS_DIR}/$<TARGET_FILE_NAME:mlir_c_runner_utils>"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE:mlir_float16_utils>"
    "${_MLIR_LIBS_DIR}/$<TARGET_FILE_NAME:mlir_float16_utils>"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE:mlir_rocm_runtime>"
    "${_MLIR_LIBS_DIR}/$<TARGET_FILE_NAME:mlir_rocm_runtime>"
  COMMENT "Copying python/flydsl sources to build/python_packages/flydsl"
  DEPENDS FlyPythonModules
)

add_dependencies(CopyFlyPythonSources FlyPythonModules)

# Windows: upstream MLIRPythonSources.Core copies `_mlir_libs/_mlir/py.typed` and
# `_mlir_libs/_mlirRegisterEverything/py.typed` from the MLIR source tree. On Linux
# this is harmless — the `.so` extension takes precedence over a same-named
# namespace-package directory. On Windows, Python's import system treats those
# directories as PEP 420 namespace packages and SHADOWS the `_mlir.cp312-win_amd64.pyd`
# / `_mlirRegisterEverything.cp312-win_amd64.pyd` extensions, breaking
# `from ._mlir import ir`. The stubs are ergonomics-only; safe to drop on Windows.
if(WIN32)
  add_custom_command(TARGET CopyFlyPythonSources POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E rm -rf
      "${_MLIR_LIBS_DIR}/_mlir"
      "${_MLIR_LIBS_DIR}/_mlirRegisterEverything"
    COMMENT "Removing py.typed shadow directories that conflict with .pyd extensions"
    VERBATIM
  )
endif()

# patchelf RPATH fixup — Linux only (Windows uses PATH / DLL search order)
if(UNIX AND NOT APPLE)
  add_custom_command(TARGET CopyFlyPythonSources POST_BUILD
    COMMAND bash -c "patchelf --set-rpath '\\$$ORIGIN' '${_MLIR_LIBS_DIR}/$<TARGET_FILE_NAME:mlir_c_runner_utils>'" || true
    COMMAND bash -c "patchelf --set-rpath '\\$$ORIGIN' '${_MLIR_LIBS_DIR}/$<TARGET_FILE_NAME:mlir_float16_utils>'" || true
    COMMENT "Setting RPATH on copied runtime libraries"
    VERBATIM
  )
endif()


################################################################################
# FlyJitRuntime — thin ROCm runtime with GPU module caching
################################################################################

if(WIN32)
  # On Windows, ROCm / HIP is typically found via ROCM_PATH or HIP_PATH env vars,
  # or installed by TheRock into the Python venv.
  set(_ROCM_SEARCH_PATHS)
  if(DEFINED ENV{ROCM_PATH})
    list(APPEND _ROCM_SEARCH_PATHS "$ENV{ROCM_PATH}")
  endif()
  if(DEFINED ENV{HIP_PATH})
    list(APPEND _ROCM_SEARCH_PATHS "$ENV{HIP_PATH}")
  endif()
  # Also check the active Python prefix (TheRock installs HIP into the venv)
  if(DEFINED Python3_EXECUTABLE)
    get_filename_component(_PY_PREFIX "${Python3_EXECUTABLE}" DIRECTORY)
    get_filename_component(_PY_PREFIX "${_PY_PREFIX}" DIRECTORY)
    list(APPEND _ROCM_SEARCH_PATHS "${_PY_PREFIX}")
  endif()
else()
  file(GLOB _ROCM_SEARCH_PATHS LIST_DIRECTORIES true "/opt/rocm*")
  list(SORT _ROCM_SEARCH_PATHS ORDER DESCENDING)
endif()
find_package(hip REQUIRED CONFIG PATHS ${_ROCM_SEARCH_PATHS})

add_library(FlyJitRuntime SHARED
  ${PROJECT_SOURCE_DIR}/lib/Runtime/FlyRocmRuntimeWrappers.cpp
)
target_include_directories(FlyJitRuntime PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
target_compile_features(FlyJitRuntime PRIVATE cxx_std_17)
if(WIN32)
  target_compile_definitions(FlyJitRuntime PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
target_link_libraries(FlyJitRuntime PRIVATE hip::host hip::amdhip64)
set_target_properties(FlyJitRuntime PROPERTIES
  OUTPUT_NAME "fly_jit_runtime"
  LIBRARY_OUTPUT_DIRECTORY "${_MLIR_LIBS_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${_MLIR_LIBS_DIR}"  # Windows puts DLLs here
  ARCHIVE_OUTPUT_DIRECTORY "${_MLIR_LIBS_DIR}"  # Windows import libs
)
add_dependencies(FlyPythonCAPI FlyJitRuntime)


