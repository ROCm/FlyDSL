add_mlir_dialect_library(CuteDialect
  CuteDialect.cpp
  
  DEPENDS
  CuteDialectIncGen
  CuteOpsIncGen
  
  LINK_LIBS
  MLIRIR
)
