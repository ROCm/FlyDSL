add_mlir_public_c_api_library(MLIRCPIFlyROCDL
  FlyROCDLDialect.cpp
  LINK_LIBS PUBLIC
  MLIRFlyROCDLDialect
  MLIRFlyToROCDL
  MLIRFlyGpuStreamInject
)
