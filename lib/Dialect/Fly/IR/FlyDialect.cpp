#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

using namespace mlir;
using namespace mlir::fly;

#include "flydsl/Dialect/Fly/IR/FlyDialect.cpp.inc"

#include "flydsl/Dialect/Fly/IR/FlyEnums.cpp.inc"

namespace mlir::fly {
#include "flydsl/Dialect/Fly/IR/FlyAttrInterfaces.cpp.inc"
#include "flydsl/Dialect/Fly/IR/FlyTypeInterfaces.cpp.inc"

#include "flydsl/Dialect/Fly/IR/FlyAttrConstraints.cpp.inc"
#include "flydsl/Dialect/Fly/IR/FlyTypeConstraints.cpp.inc"
} // namespace mlir::fly

#define GET_TYPEDEF_CLASSES
#include "flydsl/Dialect/Fly/IR/FlyTypeDefs.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "flydsl/Dialect/Fly/IR/FlyAttrDefs.cpp.inc"

void FlyDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "flydsl/Dialect/Fly/IR/FlyTypeDefs.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "flydsl/Dialect/Fly/IR/FlyAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "flydsl/Dialect/Fly/IR/FlyOps.cpp.inc"
      >();
}
