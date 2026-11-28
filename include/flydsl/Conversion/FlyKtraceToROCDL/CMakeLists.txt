set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name FlyKtraceToROCDL)

add_mlir_generic_tablegen_target(FlyKtraceToROCDLPassIncGen)
