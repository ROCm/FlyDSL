set(LLVM_TARGET_DEFINITIONS Dialect.td)

mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS Atom.td)
mlir_tablegen(Atom.h.inc -gen-typedef-decls -typedefs-dialect=fly_nvvm)
mlir_tablegen(Atom.cpp.inc -gen-typedef-defs -typedefs-dialect=fly_nvvm)

add_public_tablegen_target(MLIRFlyNVVMIncGen)
