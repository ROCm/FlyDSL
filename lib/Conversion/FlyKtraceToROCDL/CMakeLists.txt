add_mlir_conversion_library(MLIRFlyKtraceToROCDL
  FlyKtraceToROCDL.cpp

  DEPENDS
  FlyKtraceToROCDLPassIncGen

  LINK_LIBS PUBLIC
  MLIRFlyKtraceDialect
  MLIRFlyDialect
  MLIRLLVMDialect
  MLIRSCFDialect
  MLIRArithDialect
  MLIRROCDLDialect
  MLIRGPUDialect
  MLIRPass
  MLIRTransforms
)
