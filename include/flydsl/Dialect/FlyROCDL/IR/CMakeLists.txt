set(LLVM_TARGET_DEFINITIONS Dialect.td)

mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)

mlir_tablegen(Enums.h.inc -gen-enum-decls -typedefs-dialect=fly_rocdl)
mlir_tablegen(Enums.cpp.inc -gen-enum-defs -typedefs-dialect=fly_rocdl)

set(LLVM_TARGET_DEFINITIONS Atom.td)
mlir_tablegen(Atom.h.inc -gen-typedef-decls -typedefs-dialect=fly_rocdl)
mlir_tablegen(Atom.cpp.inc -gen-typedef-defs -typedefs-dialect=fly_rocdl)

add_public_tablegen_target(MLIRFlyROCDLIncGen)
