include(AddMLIRPython)

################################################################################
# FlyDSL Python Bindings - Out-of-tree Dialect Configuration
################################################################################
#
# DESIGN PRINCIPLE:
# - Do NOT rebuild upstream MLIR Python infrastructure  
# - Only build Fly dialect extensions
# - Link to upstream MLIR's Python support libraries at runtime
# - Avoid symbol conflicts by using PRIVATE_LINK_LIBS (no EMBED_CAPI_LINK_LIBS)
#
################################################################################

# Find upstream MLIR Python libraries for RPATH configuration
get_filename_component(UPSTREAM_MLIR_LIB_DIR "${MLIR_DIR}/../../lib" ABSOLUTE)
get_filename_component(UPSTREAM_MLIR_PYTHON_LIBS 
    "${MLIR_DIR}/../../../tools/mlir/python_packages/mlir_core/mlir/_mlir_libs" ABSOLUTE)

message(STATUS "Upstream MLIR lib dir: ${UPSTREAM_MLIR_LIB_DIR}")
message(STATUS "Upstream MLIR Python libs: ${UPSTREAM_MLIR_PYTHON_LIBS}")

# Check for upstream nanobind library
find_library(UPSTREAM_NANOBIND_LIB
  NAMES nanobind-mlir
  PATHS ${UPSTREAM_MLIR_PYTHON_LIBS}
  NO_DEFAULT_PATH
)

if(UPSTREAM_NANOBIND_LIB)
  message(STATUS "✓ Found upstream nanobind-mlir: ${UPSTREAM_NANOBIND_LIB}")
  # Note: We cannot prevent AddMLIRPython.cmake from building a local nanobind-mlir
  # But we can force our extension modules to link to the upstream version via RPATH
  set(USE_UPSTREAM_NANOBIND TRUE)
else()
  message(WARNING "⚠️  Upstream nanobind-mlir not found at ${UPSTREAM_MLIR_PYTHON_LIBS}")
  message(WARNING "    Will use locally built nanobind-mlir")
  set(USE_UPSTREAM_NANOBIND FALSE)
endif()

# Check for upstream MLIRPythonSupport library (CRITICAL for PyGlobals!)
find_library(UPSTREAM_MLIR_PYTHON_SUPPORT_LIB
  NAMES MLIRPythonSupport-mlir
  PATHS ${UPSTREAM_MLIR_PYTHON_LIBS}
  NO_DEFAULT_PATH
)

if(UPSTREAM_MLIR_PYTHON_SUPPORT_LIB)
  message(STATUS "✓ Found upstream MLIRPythonSupport: ${UPSTREAM_MLIR_PYTHON_SUPPORT_LIB}")
else()
  message(FATAL_ERROR "❌ Upstream MLIRPythonSupport-mlir.so not found! Cannot link Fly extensions.")
endif()

# Check for upstream MLIRPythonCAPI library (CRITICAL to avoid symbol conflicts!)
find_library(UPSTREAM_MLIR_PYTHON_CAPI_LIB
  NAMES MLIRPythonCAPI
  PATHS ${UPSTREAM_MLIR_PYTHON_LIBS}
  NO_DEFAULT_PATH
)

if(UPSTREAM_MLIR_PYTHON_CAPI_LIB)
  message(STATUS "✓ Found upstream MLIRPythonCAPI: ${UPSTREAM_MLIR_PYTHON_CAPI_LIB}")
else()
  message(WARNING "⚠️  Upstream MLIRPythonCAPI.so not found at ${UPSTREAM_MLIR_PYTHON_LIBS}")
endif()

################################################################################
# Fly Python Sources Declaration
################################################################################

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

################################################################################
# Custom Support Library for flydsl domain
################################################################################
# CRITICAL: Build this BEFORE declaring extensions that depend on it

# Extract MLIR source directory from MLIR_INCLUDE_DIRS
list(GET MLIR_INCLUDE_DIRS 0 MLIR_FIRST_INCLUDE_DIR)
get_filename_component(MLIR_SRC_ROOT "${MLIR_FIRST_INCLUDE_DIR}" DIRECTORY)
set(MLIR_PYTHON_BINDINGS_SRC_DIR "${MLIR_SRC_ROOT}/lib/Bindings/Python")

message(STATUS "MLIR source root: ${MLIR_SRC_ROOT}")
message(STATUS "MLIR Python bindings source: ${MLIR_PYTHON_BINDINGS_SRC_DIR}")

# Define the core support sources (NO MainModule.cpp!)
set(FLYDSL_PYTHON_SUPPORT_SOURCES
    "${MLIR_PYTHON_BINDINGS_SRC_DIR}/Globals.cpp"
    "${MLIR_PYTHON_BINDINGS_SRC_DIR}/IRAffine.cpp"
    "${MLIR_PYTHON_BINDINGS_SRC_DIR}/IRAttributes.cpp"
    "${MLIR_PYTHON_BINDINGS_SRC_DIR}/IRCore.cpp"
    "${MLIR_PYTHON_BINDINGS_SRC_DIR}/IRInterfaces.cpp"
    "${MLIR_PYTHON_BINDINGS_SRC_DIR}/IRTypes.cpp"
    "${MLIR_PYTHON_BINDINGS_SRC_DIR}/Pass.cpp"
    "${MLIR_PYTHON_BINDINGS_SRC_DIR}/Rewrite.cpp"
    # ❌ DO NOT include MainModule.cpp - it creates _mlir module entry!
)

# Verify files exist
message(STATUS "Checking support source files:")
foreach(src_file ${FLYDSL_PYTHON_SUPPORT_SOURCES})
    if(EXISTS ${src_file})
        message(STATUS "  ✓ Found: ${src_file}")
    else()
        message(FATAL_ERROR "  ❌ Source file not found: ${src_file}")
    endif()
endforeach()

# Build the support library directly (bypass declare_mlir_python_extension which mangles paths)
# This creates libMLIRPythonSupport-flydsl.so
set(FLYDSL_SUPPORT_LIB_NAME "libMLIRPythonSupport-flydsl")
add_library(${FLYDSL_SUPPORT_LIB_NAME} SHARED ${FLYDSL_PYTHON_SUPPORT_SOURCES})

# Configure the library
target_include_directories(${FLYDSL_SUPPORT_LIB_NAME} PRIVATE
    ${MLIR_INCLUDE_DIRS}
    ${MLIR_SRC_ROOT}/include
    ${nanobind_INCLUDE_DIR}  # Add nanobind headers
    ${Python3_INCLUDE_DIRS}  # Add Python headers
)

target_compile_definitions(${FLYDSL_SUPPORT_LIB_NAME} PRIVATE
    NB_DOMAIN=flydsl
    MLIR_CAPI_BUILDING_LIBRARY=1
    MLIR_BINDINGS_PYTHON_DOMAIN=flydsl
)

# Link requirements
target_link_libraries(${FLYDSL_SUPPORT_LIB_NAME} PRIVATE
    ${UPSTREAM_MLIR_PYTHON_CAPI_LIB}
    LLVMSupport
    nanobind-flydsl  # Link our nanobind-flydsl library
)

# nanobind configuration
nanobind_link_options(${FLYDSL_SUPPORT_LIB_NAME})
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(${FLYDSL_SUPPORT_LIB_NAME} PRIVATE "LINKER:-z,undefs")
endif()

# Add dependency on nanobind library
add_dependencies(${FLYDSL_SUPPORT_LIB_NAME} nanobind-flydsl)

# Set output location (same directory as other _mlir_libs)
set_target_properties(${FLYDSL_SUPPORT_LIB_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
    OUTPUT_NAME "MLIRPythonSupport-flydsl"
    PREFIX "lib"
)

message(STATUS "Created support library target: ${FLYDSL_SUPPORT_LIB_NAME}")

################################################################################
# Fly Bindings Extension (_fly.so)
################################################################################

declare_mlir_python_extension(FlyPythonSources.Extension.Fly
  MODULE_NAME _fly
  ADD_TO_PARENT FlyPythonSources
  ROOT_DIR "${PROJECT_SOURCE_DIR}/lib/Bindings/Python"
  PYTHON_BINDINGS_LIBRARY nanobind
  
  SOURCES
    FlyExtension.cpp
    DLTensorAdaptor.h
  
  # ✅ Use PRIVATE_LINK_LIBS to link without embedding
  # This allows runtime linking to upstream libraries
  PRIVATE_LINK_LIBS
    # 🔥 CRITICAL: Link upstream MLIRPythonCAPI FIRST!
    # This ensures all MLIR core symbols (BuiltinDialect, StringAttr, etc.)
    # come from the upstream shared library, NOT from static copies.
    ${UPSTREAM_MLIR_PYTHON_CAPI_LIB}
    $<TARGET_FILE:${FLYDSL_SUPPORT_LIB_NAME}>  # CRITICAL: Link our flydsl-domain support library
    MLIRCPIFly          # Our Fly dialect C API
    # NOTE: Do NOT link MLIRCPIFlyROCDL here! It causes ODR violation with _fly_rocdl.so
    # FlyROCDL dialect should only be loaded via _fly_rocdl.so
    MLIRFlyToROCDL      # Fly to ROCDL conversion pass
    LLVMSupport
    # Note: registerFlyPasses() and registerFlyCanonicalizePass() are inline functions
    # defined in Passes.h.inc, they get linked from MLIRFlyDialect which is already
    # a dependency of MLIRCPIFly.
    # ❌ DO NOT link MLIRArithDialect or MLIRFuncDialect!
    # These will static-link MLIR core (including BuiltinDialect),
    # causing symbol conflicts with upstream MLIR.
    # Instead, we rely on upstream's already-loaded symbols.
    
  # ❌ DO NOT USE EMBED_CAPI_LINK_LIBS!
  # Embedding causes duplicate symbols and "Option already exists" errors
)

################################################################################
# FlyROCDL Bindings Extension (_fly_rocdl.so)
################################################################################

declare_mlir_python_extension(FlyPythonSources.Extension.FlyROCDL
  MODULE_NAME _fly_rocdl
  ADD_TO_PARENT FlyPythonSources
  ROOT_DIR "${PROJECT_SOURCE_DIR}/lib/Bindings/Python"
  PYTHON_BINDINGS_LIBRARY nanobind
  
  SOURCES
    FlyROCDLExtension.cpp
  
  PRIVATE_LINK_LIBS
    MLIRCPIFlyROCDL     # Our FlyROCDL dialect C API
    LLVMSupport
    $<TARGET_FILE:${FLYDSL_SUPPORT_LIB_NAME}>  # CRITICAL: Link our flydsl-domain support library
    
  # ❌ DO NOT EMBED CAPI LIBS
)

################################################################################
# REMOVED: _mlirRegisterEverything Module
################################################################################
#
# This module was causing "Option already exists" errors because it was
# embedding CAPI libraries that duplicate upstream MLIR's libraries.
#
# Dialect registration now happens automatically via:
# 1. C++ static initializers in _fly.so and _fly_rocdl.so
# 2. Python-level explicit registration if needed
#
# The old code (kept for reference):
# declare_mlir_python_extension(FlyPythonSources.RegisterEverything
#   MODULE_NAME _mlirRegisterEverything
#   ...
#   EMBED_CAPI_LINK_LIBS MLIRCPIFly MLIRCPIFlyROCDL MLIRCAPIROCDL
# )
################################################################################


################################################################################
# Python Module Assembly
################################################################################

# Only include our Fly dialect sources  
set(MLIRFlyDSLSources
  FlyPythonSources   # Our Fly dialect bindings
)

# ❌ DO NOT BUILD add_mlir_python_common_capi_library!
# This would rebuild upstream MLIR's Python CAPI, causing conflicts.
# Instead, we rely on upstream's already-built libraries.

# Old code (removed):
# add_mlir_python_common_capi_library(FlyPythonCAPI
#   INSTALL_COMPONENT FlyPythonModules
#   ...
#   DECLARED_SOURCES ${MLIRFlyDSLSources}
# )


set(FlyPythonModules_ROOT_PREFIX "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}")

################################################################################
# Python Modules - Build Configuration
################################################################################

add_mlir_python_modules(FlyPythonModules
  ROOT_PREFIX "${FlyPythonModules_ROOT_PREFIX}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  
  # CRITICAL: Set this to a unique string for your project to avoid symbol conflicts
  # From LLVM commit 29c15ee: "if you are defining such types/attributes in a downstream 
  # project, it is critical you define MLIR_BINDINGS_PYTHON_NB_DOMAIN such that it is 
  # unique to your project"
  MLIR_BINDINGS_PYTHON_NB_DOMAIN "flydsl"
  
  # ✅ Only declare Fly sources AND upstream MLIR sources
  # Including MLIRPythonSources causes CMake to rebuild support libraries with our domain
  DECLARED_SOURCES "${MLIRFlyDSLSources}"
  
  # CRITICAL: Link to upstream MLIR CAPI library for the support library
  # This provides symbols like mlirTypeIsAShaped that libMLIRPythonSupport-flydsl.so needs
  COMMON_CAPI_LINK_LIBS ${UPSTREAM_MLIR_PYTHON_CAPI_LIB}
)

# ✅ CRITICAL: Set RPATH to prioritize upstream MLIR Python libs over local ones
# Place upstream paths BEFORE $ORIGIN to ensure upstream nanobind-mlir.so is used
# The target is created by add_mlir_python_modules above
set_target_properties(FlyPythonModules.extension._fly.dso PROPERTIES
  BUILD_RPATH "${UPSTREAM_MLIR_PYTHON_LIBS}:${UPSTREAM_MLIR_LIB_PATH}:\$ORIGIN"
  INSTALL_RPATH "${UPSTREAM_MLIR_PYTHON_LIBS}:${UPSTREAM_MLIR_LIB_PATH}:\$ORIGIN"
)
set_target_properties(FlyPythonModules.extension._fly_rocdl.dso PROPERTIES
  BUILD_RPATH "${UPSTREAM_MLIR_PYTHON_LIBS}:${UPSTREAM_MLIR_LIB_PATH}:\$ORIGIN"
  INSTALL_RPATH "${UPSTREAM_MLIR_PYTHON_LIBS}:${UPSTREAM_MLIR_LIB_PATH}:\$ORIGIN"
)


################################################################################
# RPATH Configuration - Link to Upstream MLIR at Runtime
################################################################################

# Post-build step to replace local nanobind with upstream version
if(USE_UPSTREAM_NANOBIND)
  add_custom_target(ReplaceWithUpstreamNanobind ALL
    COMMAND ${CMAKE_COMMAND} -E echo "=== Patching extension modules to use upstream nanobind and PythonSupport ==="
    COMMAND ${CMAKE_COMMAND} -E echo "Upstream nanobind: ${UPSTREAM_NANOBIND_LIB}"
    COMMAND ${CMAKE_COMMAND} -E echo "Upstream PythonSupport: ${UPSTREAM_MLIR_PYTHON_SUPPORT_LIB}"
    # Replace libnanobind-flydsl.so with upstream libnanobind-mlir.so
    COMMAND bash -c "find ${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX} -name '_fly*.so' -exec patchelf --replace-needed libnanobind-flydsl.so libnanobind-mlir.so {} \\; 2>/dev/null || true"
    # Also add upstream libs to RPATH if not present
    COMMAND bash -c "find ${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX} -name '_fly*.so' -exec patchelf --set-rpath '${UPSTREAM_MLIR_PYTHON_LIBS}:\$ORIGIN' {} \\; 2>/dev/null || echo 'patchelf not available, relying on existing RPATH'"
    DEPENDS FlyPythonModules
    COMMENT "Patching extension modules to link to upstream MLIR Python libs"
    VERBATIM
  )
  
  # If patchelf is not available, rely on RPATH
  message(STATUS "✓ Will use RPATH to prefer upstream nanobind-mlir")
  message(STATUS "  Note: If 'patchelf' is installed, DT_NEEDED will be patched directly")
endif()

################################################################################
# Type Stubs Generation
################################################################################

set(_FLYDSL_PYTHON_PACKAGES_DIR "${MLIR_BINARY_DIR}/python_packages")
set(_MLIR_LIBS_DIR "${FlyPythonModules_ROOT_PREFIX}/_mlir_libs")
set(_STUB_MARKER_FILE "${_MLIR_LIBS_DIR}/.stubs_generated")

add_custom_command(
  OUTPUT "${_STUB_MARKER_FILE}"
  # TEMPORARILY DISABLED: Stubgen fails due to import issues
  # COMMAND ${CMAKE_COMMAND} -E env
  #   "PYTHONPATH=${_FLYDSL_PYTHON_PACKAGES_DIR}"
  #   ${Python3_EXECUTABLE} -m nanobind.stubgen
  #   -q
  #   -r
  #   -m flydsl._mlir._mlir_libs._mlir
  #   -m flydsl._mlir._mlir_libs._fly
  #   -m flydsl._mlir._mlir_libs._fly_rocdl
  #   -m flydsl._mlir._mlir_libs._mlirDialectsGPU
  #   -m flydsl._mlir._mlir_libs._mlirDialectsLLVM
  #   -O "${_MLIR_LIBS_DIR}"
  COMMAND ${CMAKE_COMMAND} -E touch "${_STUB_MARKER_FILE}"
  DEPENDS CopyFlyPythonSources
  COMMENT "Skipping stub generation (temporarily disabled)"
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
  # These files are generated in build/python/mlir_flydsl/dialects/ by declare_mlir_dialect_python_bindings
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${MLIR_BINARY_DIR}/python/mlir_flydsl/dialects/_fly_ops_gen.py"
    "${MLIR_BINARY_DIR}/python_packages/flydsl/_mlir/dialects/_fly_ops_gen.py"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${MLIR_BINARY_DIR}/python/mlir_flydsl/dialects/_fly_enum_gen.py"
    "${MLIR_BINARY_DIR}/python_packages/flydsl/_mlir/dialects/_fly_enum_gen.py"
  COMMENT "Copying python/flydsl sources to build/python_packages/flydsl"
  DEPENDS FlyPythonModules
)

add_dependencies(CopyFlyPythonSources FlyPythonModules)
