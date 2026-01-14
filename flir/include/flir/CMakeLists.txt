# Simplified CMakeLists for modern MLIR

set(LLVM_TARGET_DEFINITIONS FlirDialect.td)
mlir_tablegen(FlirDialect.h.inc -gen-dialect-decls)
mlir_tablegen(FlirDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(FlirDialectIncGen)

set(LLVM_TARGET_DEFINITIONS FlirAttrDefs.td)
mlir_tablegen(FlirAttrDefs.h.inc -gen-attrdef-decls -attrdefs-dialect=flir)
mlir_tablegen(FlirAttrDefs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=flir)
add_public_tablegen_target(FlirAttrDefsIncGen)

set(LLVM_TARGET_DEFINITIONS FlirTypeDefs.td)
mlir_tablegen(FlirTypeDefs.h.inc -gen-typedef-decls -typedefs-dialect=flir)
mlir_tablegen(FlirTypeDefs.cpp.inc -gen-typedef-defs -typedefs-dialect=flir)
add_public_tablegen_target(FlirTypeDefsIncGen)

set(LLVM_TARGET_DEFINITIONS FlirOps.td)
mlir_tablegen(FlirOps.h.inc -gen-op-decls)
mlir_tablegen(FlirOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(FlirOpsIncGen)

set(LLVM_TARGET_DEFINITIONS FlirTypeDefs.td)
mlir_tablegen(FlirTypes.h.inc -gen-typedef-decls)
mlir_tablegen(FlirTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(FlirTypesIncGen)

set(LLVM_TARGET_DEFINITIONS FlirAttrDefs.td)
mlir_tablegen(FlirAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(FlirAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(FlirAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS FlirRocmDialect.td)
mlir_tablegen(FlirRocmDialect.h.inc -gen-dialect-decls -dialect=flir_rocm)
mlir_tablegen(FlirRocmDialect.cpp.inc -gen-dialect-defs -dialect=flir_rocm)
mlir_tablegen(FlirRocmTypes.h.inc -gen-typedef-decls -dialect=flir_rocm)
mlir_tablegen(FlirRocmTypes.cpp.inc -gen-typedef-defs -dialect=flir_rocm)
add_public_tablegen_target(FlirRocmDialectIncGen)

set(LLVM_TARGET_DEFINITIONS FlirRocmOps.td)
mlir_tablegen(FlirRocmOps.h.inc -gen-op-decls)
mlir_tablegen(FlirRocmOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(FlirRocmOpsIncGen)

set(LLVM_TARGET_DEFINITIONS FlirPasses.td)
mlir_tablegen(FlirPasses.h.inc -gen-pass-decls -name Flir)
add_public_tablegen_target(FlirPassesIncGen)
