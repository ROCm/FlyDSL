# On Windows, MLIRFlyDialect's OBJECT files are added directly to FlyPythonCAPI
# (see python/mlir_flydsl/CMakeLists.txt) so that WINDOWS_EXPORT_ALL_SYMBOLS exports
# the mlir::fly::* C++ symbols. Linking MLIRFlyDialect here transitively would cause
# duplicate-symbol errors at FlyPythonCAPI.dll link time.
if(WIN32)
  add_mlir_public_c_api_library(MLIRCPIFly
    FlyDialect.cpp
  )
  # Still need to compile against the dialect's headers (TableGen'd types).
  add_dependencies(obj.MLIRCPIFly MLIRFlyIncGen)
  target_include_directories(obj.MLIRCPIFly PRIVATE
    ${CMAKE_BINARY_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
  )
else()
  add_mlir_public_c_api_library(MLIRCPIFly
    FlyDialect.cpp
    LINK_LIBS PUBLIC
    MLIRFlyDialect
  )
endif()
