set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Fly)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header -name Fly)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl -name Fly)

add_mlir_generic_tablegen_target(FlyTransformPassIncGen)

set(LLVM_TARGET_DEFINITIONS LayoutLowering.td)
mlir_tablegen(LayoutLowering.cpp.inc -gen-rewriters)

add_mlir_generic_tablegen_target(FlyTransformPatternsIncGen)

