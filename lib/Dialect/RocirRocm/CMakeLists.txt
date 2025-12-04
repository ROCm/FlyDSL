add_mlir_dialect_library(RocirRocmDialect
  RocirRocmDialect.cpp
  RocirRocmOps.cpp
  
  DEPENDS
  RocirRocmDialectIncGen
  RocirRocmOpsIncGen
  
  LINK_LIBS
  MLIRIR
)
