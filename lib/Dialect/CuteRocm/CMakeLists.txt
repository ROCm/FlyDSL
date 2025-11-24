add_mlir_dialect_library(CuteRocmDialect
  CuteRocmDialect.cpp
  
  DEPENDS
  CuteRocmDialectIncGen
  CuteRocmOpsIncGen
  
  LINK_LIBS
  MLIRIR
)
