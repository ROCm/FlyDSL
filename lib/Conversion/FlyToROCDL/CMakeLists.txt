add_mlir_conversion_library(MLIRFlyToROCDL
  FlyToROCDL.cpp

  DEPENDS
  MLIRFlyIncGen
  FlyConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRFlyDialect
  
  MLIRAffineDialect
  MLIRAffineTransforms
  MLIRAffineUtils
  MLIRArithDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransforms
  MLIRVectorDialect
)
