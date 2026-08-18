add_mlir_public_c_api_library(MLIRCPIFlyNVVM
  FlyNVVMDialect.cpp

  DEPENDS
  FlyToNVVMPassIncGen

  LINK_LIBS PUBLIC
  MLIRFlyNVVMDialect
  MLIRFlyToNVVM
)
