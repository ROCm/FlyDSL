# The Core categories the Fly dialect uses.
add_mlir_library(FlyCore
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
