if(WIN32)
  add_mlir_public_c_api_library(MLIRCPIFly FlyDialect.cpp)
  add_dependencies(obj.MLIRCPIFly MLIRFlyDialect MLIRFlyIncGen)
  target_include_directories(obj.MLIRCPIFly PRIVATE
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_BINARY_DIR}/include")
else()
  add_mlir_public_c_api_library(MLIRCPIFly
    FlyDialect.cpp
    LINK_LIBS PUBLIC
    MLIRFlyDialect)
endif()
