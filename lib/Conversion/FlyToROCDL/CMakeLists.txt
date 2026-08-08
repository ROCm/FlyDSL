# Directory scope so the definition also reaches obj.MLIRFlyToROCDL, which is
# where add_mlir_conversion_library actually compiles the sources.
if(LLD_FOUND)
  add_compile_definitions(FLYDSL_HAS_LLD_LIBRARY)
endif()

add_mlir_conversion_library(MLIRFlyToROCDL
  FlyToROCDL.cpp
  FlyEmitGPUBinary.cpp

  DEPENDS
  MLIRFlyIncGen
  MLIRFlyROCDLIncGen
  FlyToROCDLPassIncGen

  LINK_LIBS PUBLIC
  MLIRFlyDialect
  MLIRFlyROCDLDialect

  MLIRArithDialect
  MLIRGPUDialect
  MLIRGPUTransforms
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRROCDLDialect
  MLIRROCDLTarget
  MLIRSCFDialect
  MLIRTransforms
  MLIRVectorDialect
)

if(LLD_FOUND)
  target_link_libraries(MLIRFlyToROCDL PUBLIC lldCommon lldELF)
endif()
