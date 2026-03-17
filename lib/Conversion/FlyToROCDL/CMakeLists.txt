add_mlir_conversion_library(MLIRFlyToROCDL
  FlyToROCDL.cpp

  DEPENDS
  MLIRFlyIncGen
  MLIRFlyROCDLIncGen
  FlyConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRFlyDialect
  MLIRFlyROCDLDialect
  
  MLIRArithDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransforms
  MLIRVectorDialect
)
