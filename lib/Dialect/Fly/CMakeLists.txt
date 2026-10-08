if(WIN32)
  set(_fly_dialect_object_arg OBJECT)
endif()

add_mlir_dialect_library(MLIRFlyDialect
  ${_fly_dialect_object_arg}
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
  Transforms/IntSwizzleSimplify.cpp

  DEPENDS
  MLIRFlyIncGen
  FlyTransformPassIncGen
  FlyRewritePatternsIncGen

  LINK_LIBS
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRGPUDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRSCFTransforms
  MLIRTargetLLVMIRExport
  MLIRTransforms
)
