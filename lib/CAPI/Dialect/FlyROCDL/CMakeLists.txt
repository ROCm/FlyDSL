if(WIN32)
  add_mlir_public_c_api_library(MLIRCPIFlyROCDL
    FlyROCDLDialect.cpp
    LINK_LIBS PUBLIC
    MLIRFlyToROCDL)
  add_dependencies(obj.MLIRCPIFlyROCDL MLIRFlyROCDLDialect MLIRFlyROCDLIncGen)
  target_include_directories(obj.MLIRCPIFlyROCDL PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_BINARY_DIR}/include")
else()
  add_mlir_public_c_api_library(MLIRCPIFlyROCDL
    FlyROCDLDialect.cpp
    LINK_LIBS PUBLIC
    MLIRFlyROCDLDialect
    MLIRFlyToROCDL)
endif()
