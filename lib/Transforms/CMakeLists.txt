add_mlir_library(CuteTransforms
  CuteToStandard.cpp
  
  PARTIAL_SOURCES_INTENDED
  
  DEPENDS
  CutePassesIncGen
  
  LINK_LIBS
  CuteDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRGPUDialect
  MLIRVectorDialect
)
