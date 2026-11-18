set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Fly)

add_mlir_generic_tablegen_target(FlyTransformPassIncGen)
