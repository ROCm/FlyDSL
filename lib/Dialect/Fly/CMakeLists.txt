add_mlir_dialect_library(MLIRFlyDialect
  IR/FlyDialect.cpp
  IR/FlyOps.cpp
  IR/FlyTypeDefs.cpp
  IR/FlyUniversalOps.cpp
  IR/FlyAttrDefs.cpp
  Utils/IntUtils.cpp
  Utils/IntTupleUtils.cpp
  Utils/NormalForm.cpp
  Utils/PointerUtils.cpp
  Transforms/LayoutLowering.cpp
  Transforms/Canonicalize.cpp
  Transforms/RewriteFuncSignature.cpp
  Transforms/ConvertAtomCallToSSAForm.cpp
  
  DEPENDS
  MLIRFlyIncGen
  FlyTransformPassIncGen
  FlyRewritePatternsIncGen
  
  LINK_LIBS
  MLIRIR
)
