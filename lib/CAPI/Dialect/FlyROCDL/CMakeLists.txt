add_mlir_public_c_api_library(MLIRCPIFlyROCDL
  FlyROCDLDialect.cpp
  LINK_LIBS PUBLIC
  MLIRFlyROCDLDialect
  MLIRFlyToROCDL
  MLIRFlyKtraceToROCDL
  MLIRROCDLTarget
)

if(LLD_FOUND)
  target_compile_definitions(obj.MLIRCPIFlyROCDL PRIVATE FLYDSL_HAS_LLD_LIBRARY)
endif()
