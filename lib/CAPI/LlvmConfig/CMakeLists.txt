# LLVM cl::opt runtime control — independent of Fly dialect.
# Compiled into libFlyPythonCAPI.so to share the cl::opt registry
# with the MLIR compilation pipeline.
add_mlir_public_c_api_library(MLIRCPILLVMOption
  llvm.cc
  LINK_LIBS PUBLIC
  LLVMSupport
)
