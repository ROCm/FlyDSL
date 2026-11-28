add_mlir_conversion_library(MLIRFlyToNVVM
  FlyToNVVM.cpp

  DEPENDS
  MLIRFlyIncGen
  MLIRFlyNVVMIncGen
  FlyToNVVMPassIncGen

  LINK_LIBS PUBLIC
  MLIRFlyDialect
  MLIRFlyNVVMDialect

  MLIRArithDialect
  MLIRControlFlowTransforms
  MLIRFuncTransforms
  MLIRGPUDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTransforms
  MLIRUBDialect
  MLIRVectorDialect
)
