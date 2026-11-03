add_mlir_conversion_library(MLIRFlyToROCDL
  FastMathOps.cpp
  FlyToROCDL.cpp
  RegisterPlacement.cpp
  SerializeRegisterKernels.cpp

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
  MLIRIR
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTransforms
  MLIRUBDialect
  MLIRVectorDialect
  LLVMCodeGen
  LLVMTarget
  LLVMAMDGPUAsmParser
  LLVMMCParser
  MLIRGPUTransforms
  MLIRROCDLTarget
)
