add_mlir_dialect_library(FlirRocmDialect
  FlirRocmDialect.cpp
  FlirRocmOps.cpp
  
  DEPENDS
  FlirRocmDialectIncGen
  FlirRocmOpsIncGen
  
  LINK_LIBS
  MLIRIR
)
