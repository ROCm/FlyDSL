add_mlir_library(FlirTransforms
  FlirToStandard.cpp
  FlirDCE.cpp
  
  PARTIAL_SOURCES_INTENDED
  
  DEPENDS
  FlirPassesIncGen
  
  LINK_LIBS
  FlirDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRControlFlowDialect
  MLIRGPUDialect
  MLIRVectorDialect
)
