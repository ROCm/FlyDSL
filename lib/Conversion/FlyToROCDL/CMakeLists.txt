add_mlir_conversion_library(MLIRFlyToROCDL
  FastMathOps.cpp
  FlyToROCDL.cpp
  FlyRocmModuleToBinary.cpp

  DEPENDS
  MLIRFlyIncGen
  MLIRFlyROCDLIncGen
  FlyToROCDLPassIncGen

  LINK_LIBS PUBLIC
  MLIRFlyDialect
  MLIRFlyROCDLDialect
  
  MLIRArithDialect
  MLIRControlFlowTransforms
  MLIRFuncTransforms
  MLIRGPUDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRPass
  MLIRROCDLTarget
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTransforms
  MLIRVectorDialect
)
