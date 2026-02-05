include(AddMLIRPython)

# Use the embedded MLIR package name prefix (`_mlir.*`), matching the install
# layout under build/python_packages/flydsl/_mlir.
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

# Build ROCm JIT runtime by compiling MLIR's RocmRuntimeWrappers.cpp directly.
# This avoids needing to modify LLVM and automatically follows upstream updates.
find_package(hip QUIET CONFIG PATHS /opt/rocm)
if(hip_FOUND)
  # Locate MLIR's RocmRuntimeWrappers.cpp source file
  # MLIR_DIR is typically <build>/lib/cmake/mlir, we need <source>/mlir
  # Try to find it via MLIR_INCLUDE_DIRS which points to source includes
  set(MLIR_ROCM_RUNTIME_SRC "")
  foreach(_inc_dir ${MLIR_INCLUDE_DIRS})
    if("${_inc_dir}" MATCHES ".*/mlir/include$")
      get_filename_component(_mlir_src_root "${_inc_dir}/.." ABSOLUTE)
      set(_candidate "${_mlir_src_root}/lib/ExecutionEngine/RocmRuntimeWrappers.cpp")
      if(EXISTS "${_candidate}")
        set(MLIR_ROCM_RUNTIME_SRC "${_candidate}")
        break()
      endif()
    endif()
  endforeach()
  
  if(NOT "${MLIR_ROCM_RUNTIME_SRC}" STREQUAL "" AND EXISTS "${MLIR_ROCM_RUNTIME_SRC}")
    message(STATUS "Using MLIR source: ${MLIR_ROCM_RUNTIME_SRC}")
    
    add_library(FlirJitRuntime SHARED
      ${MLIR_ROCM_RUNTIME_SRC}
    )
    target_include_directories(FlirJitRuntime PRIVATE
      ${LLVM_INCLUDE_DIRS}
      ${MLIR_INCLUDE_DIRS}
    )
    target_compile_features(FlirJitRuntime PRIVATE cxx_std_17)
    
    # Override hidden visibility for this target - mgpu* symbols must be exported
    # for MLIR JIT ExecutionEngine to resolve them at runtime
    set_target_properties(FlirJitRuntime PROPERTIES
      CXX_VISIBILITY_PRESET default
      C_VISIBILITY_PRESET default
    )
    
    # Link LLVM support library (required for llvm::SmallVector used in MLIR source)
    target_link_libraries(FlirJitRuntime PRIVATE 
      hip::host hip::amdhip64
      LLVMSupport
    )

    set_target_properties(FlirJitRuntime PROPERTIES
      OUTPUT_NAME "flir_jit_runtime"
      LIBRARY_OUTPUT_DIRECTORY "${_flir_embedded_mlir_runtime_dir}"
    )

    add_dependencies(FlirPythonCAPI FlirJitRuntime)
    message(STATUS "Building JIT runtime from MLIR source (auto-follows upstream)")
  else()
    message(STATUS "MLIR RocmRuntimeWrappers.cpp not found at ${MLIR_ROCM_RUNTIME_SRC}, using local copy")
    # Fallback to our local copy
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

    add_dependencies(FlirPythonCAPI FlirJitRuntime)
  endif()
else()
  message(WARNING
    "hip not found; cannot build ROCm JIT runtime. "
    "GPU execution via ExecutionEngine may require external MLIR runtime libs."
  )
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

# NOTE: Pure Python sources (flydsl package) are NOT copied to the build directory.
# Use `pip install -e .` which reads flydsl directly from flydsl/src/ and
# _mlir from the build output (python_packages/flydsl/_mlir).
