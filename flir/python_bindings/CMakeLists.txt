# FLIR Python Bindings
#
# Uses a unified Python extension (_flir_ir) with all symbols hidden.
# Only PyInit__flir_ir is exported.

include(AddMLIRPython)

# Declare Flir Python sources (for dialect .py files only)
declare_mlir_python_sources(FlirPythonSources)

# Declare Flir dialect Python bindings (generates _flir_ops_gen.py)
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT FlirPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/FlirOps.td
  SOURCES
    dialects/flir.py
  DIALECT_NAME flir
  GEN_ENUM_BINDINGS
)

# -----------------------------------------------------------------------------
# Unified FLIR IR Python extension
# -----------------------------------------------------------------------------
add_subdirectory(flir_ir)

# -----------------------------------------------------------------------------
# JIT Runtime (required for GPU execution)
# -----------------------------------------------------------------------------
set(_flir_ir_output_dir "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs")
set(_flir_embedded_mlir_runtime_dir "${_flir_ir_output_dir}/lib")

option(FLIR_BUILD_THIN_JIT_RUNTIME
  "Build and embed a thin ROCm JIT runtime shared library (recommended)."
  ON
)

if(FLIR_BUILD_THIN_JIT_RUNTIME)
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
      CXX_VISIBILITY_PRESET hidden
      C_VISIBILITY_PRESET hidden
      VISIBILITY_INLINES_HIDDEN ON
    )

    # Hide symbols
    if(UNIX AND NOT APPLE)
      target_link_options(FlirJitRuntime PRIVATE
        "LINKER:--exclude-libs,ALL"
      )
    endif()

    add_dependencies(_flir_ir FlirJitRuntime)
  else()
    message(WARNING
      "hip not found; cannot build thin ROCm JIT runtime. "
      "GPU execution via ExecutionEngine may require external MLIR runtime libs."
    )
  endif()
endif()
