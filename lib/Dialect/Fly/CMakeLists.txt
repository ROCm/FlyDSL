add_mlir_dialect_library(MLIRFlyDialect
  IR/FlyDialect.cpp
  IR/FlyOps.cpp
  IR/FlyTypeDefs.cpp
  IR/FlyUniversalOps.cpp
  IR/FlyAttrDefs.cpp
  IR/FlyLLVMTranslation.cpp
  Utils/IntUtils.cpp
  Utils/IntTupleUtils.cpp
  Utils/NormalForm.cpp
  Utils/PointerUtils.cpp
  Transforms/LayoutLowering.cpp
  Transforms/Canonicalize.cpp
  Transforms/RewriteFuncSignature.cpp
  Transforms/ConvertAtomCallToSSAForm.cpp
  Transforms/PromoteRegMemToVectorSSA.cpp

  DEPENDS
  MLIRFlyIncGen
  FlyTransformPassIncGen
  FlyRewritePatternsIncGen

  LINK_LIBS
  MLIRGPUDialect
  MLIRIR
  MLIRTargetLLVMIRExport
)
