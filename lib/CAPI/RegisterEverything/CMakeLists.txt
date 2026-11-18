add_mlir_public_c_api_library(MLIRCPIFlyRegisterEverything
  RegisterEverything.cpp
  LINK_LIBS PUBLIC
  MLIRFlyRegisterAll
)
