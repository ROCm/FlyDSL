add_mlir_dialect_library(CuteNvgpuDialect
  CuteNvgpuDialect.cpp
  
  DEPENDS
  CuteNvgpuDialectIncGen
  CuteNvgpuOpsIncGen
  
  LINK_LIBS
  MLIRIR
)
