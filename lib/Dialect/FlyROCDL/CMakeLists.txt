add_mlir_dialect_library(MLIRFlyROCDLDialect
  Dialect.cpp
  CDNA3/MmaAtom.cpp
  CDNA3/CopyAtom.cpp

  DEPENDS
  MLIRFlyROCDLIncGen
)
