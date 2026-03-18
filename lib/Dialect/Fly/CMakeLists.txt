add_mlir_dialect_library(MLIRFlyDialect
  IR/FlyDialect.cpp
  IR/FlyOps.cpp
  IR/FlyTypeDefs.cpp
  IR/FlyAttrDefs.cpp
  Utils/IntUtils.cpp
  Utils/IntTupleUtils.cpp
  Utils/NormalForm.cpp
  Transforms/LayoutLowering.cpp
  Transforms/FlyCanonicalize.cpp
  
  DEPENDS
  MLIRFlyIncGen
  FlyTransformPassIncGen
  
  LINK_LIBS
  MLIRIR
)
