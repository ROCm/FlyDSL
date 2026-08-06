set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name FlyToNVVM)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix FlyToNVVM)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix FlyToNVVM)

add_mlir_generic_tablegen_target(FlyToNVVMPassIncGen)
