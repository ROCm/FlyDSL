add_mlir_conversion_library(MLIRFlyGpuStreamInject
  FlyGpuStreamInject.cpp

  DEPENDS
  FlyConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithToLLVM
  MLIRAsyncToLLVM
  MLIRControlFlowToLLVM
  MLIRConvertToLLVMInterface
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRGPUDialect
  MLIRGPUToLLVMIRTranslation
  MLIRGPUTransforms
  MLIRIndexToLLVM
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRMemRefToLLVM
  MLIRPass
  MLIRUBToLLVM
  MLIRVectorToLLVM
)
