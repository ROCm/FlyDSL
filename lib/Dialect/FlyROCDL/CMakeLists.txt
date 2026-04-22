# Standard static library - the ODR TypeID issue is worked around in FlyToROCDL.cpp
# by using string-based type matching instead of dyn_cast when needed.
add_mlir_dialect_library(MLIRFlyROCDLDialect
  Dialect.cpp
  CDNA3/MmaAtom.cpp
  CDNA3/CopyAtom.cpp
  CDNA4/MmaAtom.cpp
  CDNA4/CopyAtom.cpp
  GFX1250/MmaAtom.cpp

  DEPENDS
  MLIRFlyROCDLIncGen
)
