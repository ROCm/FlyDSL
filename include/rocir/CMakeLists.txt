# Simplified CMakeLists for modern MLIR

set(LLVM_TARGET_DEFINITIONS RocirDialect.td)
mlir_tablegen(RocirDialect.h.inc -gen-dialect-decls)
mlir_tablegen(RocirDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(RocirDialectIncGen)

set(LLVM_TARGET_DEFINITIONS RocirOps.td)
mlir_tablegen(RocirOps.h.inc -gen-op-decls)
mlir_tablegen(RocirOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(RocirOpsIncGen)

set(LLVM_TARGET_DEFINITIONS RocirRocmDialect.td)
mlir_tablegen(RocirRocmDialect.h.inc -gen-dialect-decls -dialect=rocir_rocm)
mlir_tablegen(RocirRocmDialect.cpp.inc -gen-dialect-defs -dialect=rocir_rocm)
mlir_tablegen(RocirRocmTypes.h.inc -gen-typedef-decls -dialect=rocir_rocm)
mlir_tablegen(RocirRocmTypes.cpp.inc -gen-typedef-defs -dialect=rocir_rocm)
add_public_tablegen_target(RocirRocmDialectIncGen)

set(LLVM_TARGET_DEFINITIONS RocirRocmOps.td)
mlir_tablegen(RocirRocmOps.h.inc -gen-op-decls)
mlir_tablegen(RocirRocmOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(RocirRocmOpsIncGen)

set(LLVM_TARGET_DEFINITIONS RocirPasses.td)
mlir_tablegen(RocirPasses.h.inc -gen-pass-decls -name Rocir)
add_public_tablegen_target(RocirPassesIncGen)
