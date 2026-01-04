include(AddMLIRPython)

# Use the embedded MLIR package name prefix (`_mlir.*`), matching the install
# layout under build/python_packages/pyflir/_mlir.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")

# Declare Flir Python sources
declare_mlir_python_sources(FlirPythonSources)

# Declare Flir dialect Python bindings
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT FlirPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/FlirOps.td
  SOURCES
    dialects/flir.py
  DIALECT_NAME flir
  GEN_ENUM_BINDINGS
)

#
# Note: Upstream MLIR already provides Python bindings for ROCDL
# (`MLIRPythonSources.Dialects.rocdl`). We include that declared source group
# below instead of re-declaring it here.

# Declare Python extension for pass registration
declare_mlir_python_extension(FlirPythonSources.Passes
  MODULE_NAME _flirPasses
  ADD_TO_PARENT FlirPythonSources
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    FlirRegisterPasses.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
    FlirDialect
    FlirTransforms
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPITransforms
    MLIRCAPIRegisterEverything
)

# Define the Python sources to include from upstream MLIR
set(FlirMLIRPythonSources
  FlirPythonSources
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects.builtin
  MLIRPythonSources.Dialects.arith
  MLIRPythonSources.Dialects.rocdl
  MLIRPythonSources.Dialects.math
  MLIRPythonSources.Dialects.memref
  MLIRPythonSources.Dialects.gpu
  MLIRPythonSources.Dialects.func
  MLIRPythonSources.Dialects.cf
  MLIRPythonSources.Dialects.scf
  MLIRPythonSources.Dialects.vector
  MLIRPythonSources.Dialects.llvm
  MLIRPythonSources.ExecutionEngine
)

# Create common CAPI library for Python bindings
add_mlir_python_common_capi_library(FlirPythonCAPI
  INSTALL_COMPONENT FlirPythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES ${FlirMLIRPythonSources}
)

# Set the root prefix for Python modules
set(FlirPythonModules_ROOT_PREFIX "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}")

# Add MLIR Python modules
add_mlir_python_modules(FlirPythonModules
  ROOT_PREFIX "${FlirPythonModules_ROOT_PREFIX}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES "${FlirMLIRPythonSources}"
  COMMON_CAPI_LINK_LIBS
    FlirPythonCAPI
)

# -----------------------------------------------------------------------------
set(_flir_embedded_mlir_libs_root "${FlirPythonModules_ROOT_PREFIX}/_mlir_libs")
set(_flir_embedded_mlir_runtime_dir "${_flir_embedded_mlir_libs_root}/lib")

option(FLIR_BUILD_THIN_JIT_RUNTIME
  "Build and embed a thin ROCm JIT runtime shared library (recommended)."
  ON
)

if(FLIR_BUILD_THIN_JIT_RUNTIME)
  # Only build the ROCm runtime if HIP is available.
  find_package(hip QUIET CONFIG PATHS /opt/rocm)
  if(hip_FOUND)
    add_library(FlirJitRuntime SHARED
      runtime/FlirRocmRuntimeWrappers.cpp
    )
    target_include_directories(FlirJitRuntime PRIVATE
      ${LLVM_INCLUDE_DIRS}
      ${MLIR_INCLUDE_DIRS}
    )
    target_compile_features(FlirJitRuntime PRIVATE cxx_std_17)
    target_link_libraries(FlirJitRuntime PRIVATE hip::host hip::amdhip64)

    set_target_properties(FlirJitRuntime PROPERTIES
      OUTPUT_NAME "flir_jit_runtime"
      LIBRARY_OUTPUT_DIRECTORY "${_flir_embedded_mlir_runtime_dir}"
    )

    # Ensure the thin runtime is staged when building FlirPythonCAPI (build.sh).
    add_dependencies(FlirPythonCAPI FlirJitRuntime)
  else()
    message(WARNING
      "hip not found; cannot build thin ROCm JIT runtime. "
      "GPU execution via ExecutionEngine may require external MLIR runtime libs."
    )
  endif()
endif()

option(FLIR_SUPPRESS_THIRD_PARTY_WARNINGS
  "Suppress warnings coming from third-party sources used to build the embedded Python bindings (MLIR python binding sources, nanobind)."
  ON
)

if(FLIR_SUPPRESS_THIRD_PARTY_WARNINGS AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # Suppress noisy third-party warnings when building embedded Python extensions.
  # - nanobind uses some non-ISO constructs that trigger -Wpedantic
  # - MLIR python binding sources may trigger -Waddress / -Wparentheses under GCC
  set(_flir_python_extension_targets
    FlirPythonModules.extension._mlir.dso
    FlirPythonModules.extension._flirPasses.dso
    FlirPythonModules.extension._mlirDialectsGPU.dso
    FlirPythonModules.extension._mlirDialectsLLVM.dso
    FlirPythonModules.extension._mlirExecutionEngine.dso
    FlirPythonModules.extension._mlirGPUPasses.dso
  )

  foreach(_t IN LISTS _flir_python_extension_targets)
    if(TARGET "${_t}")
      target_compile_options("${_t}" PRIVATE
        -Wno-pedantic
        -Wno-address
        -Wno-parentheses
      )
    endif()
  endforeach()

  # nanobind itself triggers -Wpedantic warnings under GCC (zero-size array).
  if(TARGET nanobind-static)
    target_compile_options(nanobind-static PRIVATE -Wno-pedantic)
  endif()
endif()

# Copy the pure Python sources (pyflir package) to the build directory
add_custom_target(CopyFlirPythonSources ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${PROJECT_SOURCE_DIR}/pyflir/src/pyflir"
    "${MLIR_BINARY_DIR}/python_packages/pyflir/pyflir"
  COMMENT "Copying pyflir/src/pyflir sources to build/python_packages/pyflir/pyflir"
  DEPENDS FlirPythonModules
)

add_dependencies(CopyFlirPythonSources FlirPythonModules)
