add_mlir_dialect_library(MLIRFlyNVVMDialect
  Dialect.cpp
  SM80/MmaAtom.cpp
  SM80/CopyAtom.cpp

  DEPENDS
  MLIRFlyNVVMIncGen
)
