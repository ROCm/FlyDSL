add_mlir_conversion_library(MLIRFlyGpuStreamInject
  FlyGpuStreamInject.cpp

  DEPENDS
  FlyConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
)
