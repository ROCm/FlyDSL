set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name FlyKtraceToROCDL)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix FlyKtraceToROCDL)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix FlyKtraceToROCDL)

add_mlir_generic_tablegen_target(FlyKtraceToROCDLPassIncGen)
