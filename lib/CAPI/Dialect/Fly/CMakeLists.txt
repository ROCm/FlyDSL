add_mlir_public_c_api_library(MLIRCPIFly
  FlyDialect.cpp
  LINK_LIBS PUBLIC
  MLIRFlyDialect
)
