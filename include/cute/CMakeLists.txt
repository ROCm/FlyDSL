# Simplified CMakeLists for modern MLIR

set(LLVM_TARGET_DEFINITIONS CuteDialect.td)
mlir_tablegen(CuteDialect.h.inc -gen-dialect-decls)
mlir_tablegen(CuteDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(CuteDialectIncGen)

set(LLVM_TARGET_DEFINITIONS CuteOps.td)
mlir_tablegen(CuteOps.h.inc -gen-op-decls)
mlir_tablegen(CuteOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(CuteOpsIncGen)

set(LLVM_TARGET_DEFINITIONS CutePasses.td)
mlir_tablegen(CutePasses.h.inc -gen-pass-decls -name Cute)
add_public_tablegen_target(CutePassesIncGen)
