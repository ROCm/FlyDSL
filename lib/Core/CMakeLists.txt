# The Core categories the Fly dialect uses. OBJECT creates obj.FlyCore, which
# python/mlir_flydsl/CMakeLists.txt exports from FlyPythonCAPI.dll on Windows.
add_mlir_library(FlyCore OBJECT
  EmitOp.cpp
  Trait.cpp

  DISABLE_INSTALL
  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRSCFDialect
)
