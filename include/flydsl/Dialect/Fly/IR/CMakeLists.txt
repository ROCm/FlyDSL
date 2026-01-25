set(LLVM_TARGET_DEFINITIONS FlyDialect.td)
mlir_tablegen(FlyDialect.h.inc -gen-dialect-decls)
mlir_tablegen(FlyDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS FlyInterfaces.td)
mlir_tablegen(FlyAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(FlyAttrInterfaces.cpp.inc -gen-attr-interface-defs)
mlir_tablegen(FlyTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(FlyTypeInterfaces.cpp.inc -gen-type-interface-defs)
mlir_tablegen(FlyOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(FlyOpInterfaces.cpp.inc -gen-op-interface-defs)

set(LLVM_TARGET_DEFINITIONS FlyTypeDefs.td)
mlir_tablegen(FlyTypeDefs.h.inc -gen-typedef-decls -typedefs-dialect=fly)
mlir_tablegen(FlyTypeDefs.cpp.inc -gen-typedef-defs -typedefs-dialect=fly)
mlir_tablegen(FlyTypeConstraints.h.inc -gen-type-constraint-decls -typedefs-dialect=fly)
mlir_tablegen(FlyTypeConstraints.cpp.inc -gen-type-constraint-defs -typedefs-dialect=fly)

set(LLVM_TARGET_DEFINITIONS FlyAttrDefs.td)
mlir_tablegen(FlyEnums.h.inc -gen-enum-decls)
mlir_tablegen(FlyEnums.cpp.inc -gen-enum-defs)

set(LLVM_TARGET_DEFINITIONS FlyAttrDefs.td)
mlir_tablegen(FlyAttrDefs.h.inc -gen-attrdef-decls -attrdefs-dialect=fly)
mlir_tablegen(FlyAttrDefs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=fly)
mlir_tablegen(FlyAttrConstraints.h.inc -gen-attr-constraint-decls -attrdefs-dialect=fly)
mlir_tablegen(FlyAttrConstraints.cpp.inc -gen-attr-constraint-defs -attrdefs-dialect=fly)

set(LLVM_TARGET_DEFINITIONS FlyOps.td)
mlir_tablegen(FlyOps.h.inc -gen-op-decls)
mlir_tablegen(FlyOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(MLIRFlyIncGen)
