set(LLVM_TARGET_DEFINITIONS Dialect.td)

mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS TypeDefs.td)
mlir_tablegen(TypeDefs.h.inc -gen-typedef-decls -typedefs-dialect=fly_ktrace)
mlir_tablegen(TypeDefs.cpp.inc -gen-typedef-defs -typedefs-dialect=fly_ktrace)

set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)

add_public_tablegen_target(MLIRFlyKtraceIncGen)
