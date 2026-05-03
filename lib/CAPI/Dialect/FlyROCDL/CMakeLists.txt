add_mlir_public_c_api_library(MLIRCPIFlyROCDL
  FlyROCDLDialect.cpp
  LINK_LIBS PUBLIC
  MLIRFlyROCDLDialect
  MLIRFlyToROCDL
)

target_compile_definitions(MLIRCPIFlyROCDL PRIVATE FLYDSL_HAS_ROCDL_TARGET_STACK)
