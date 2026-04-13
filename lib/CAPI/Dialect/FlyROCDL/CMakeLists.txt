# On Windows, MLIRFlyROCDLDialect's OBJECT files are added directly to FlyPythonCAPI
# so WINDOWS_EXPORT_ALL_SYMBOLS exports the C++ symbols. Avoid duplicate linkage here.
# MLIRFlyToROCDL is still linked as a static lib — its symbols are only used through
# the CAPI registerFlyToROCDLConversionPass wrapper, which IS in an embedded obj.
if(WIN32)
  add_mlir_public_c_api_library(MLIRCPIFlyROCDL
    FlyROCDLDialect.cpp
    LINK_LIBS PUBLIC
    MLIRFlyToROCDL
  )
  add_dependencies(obj.MLIRCPIFlyROCDL MLIRFlyROCDLIncGen)
  target_include_directories(obj.MLIRCPIFlyROCDL PRIVATE
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
  )
else()
  add_mlir_public_c_api_library(MLIRCPIFlyROCDL
    FlyROCDLDialect.cpp
    LINK_LIBS PUBLIC
    MLIRFlyROCDLDialect
    MLIRFlyToROCDL
  )
endif()
