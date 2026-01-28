#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

using namespace mlir;
using namespace mlir::fly;

#include "flydsl/Dialect/Fly/IR/FlyDialect.cpp.inc"

#include "flydsl/Dialect/Fly/IR/FlyEnums.cpp.inc"

#include "flydsl/Dialect/Fly/IR/FlyAttrInterfaces.cpp.inc"
#include "flydsl/Dialect/Fly/IR/FlyTypeInterfaces.cpp.inc"

namespace mlir::fly {
#include "flydsl/Dialect/Fly/IR/FlyAttrConstraints.cpp.inc"
#include "flydsl/Dialect/Fly/IR/FlyTypeConstraints.cpp.inc"

ParseResult parseMNKDimensionList(AsmParser &parser, int32_t &m, int32_t &n, int32_t &k) {
  SmallVector<int64_t, 3> dimensions;
  if (parser.parseDimensionList(dimensions, false, false))
    return failure();
  if (dimensions.size() != 3)
    return parser.emitError(parser.getCurrentLocation())
           << "expected 3 dimensions in MNK dimension list";
  m = dimensions[0];
  n = dimensions[1];
  k = dimensions[2];
  return success();
}

void printMNKDimensionList(AsmPrinter &printer, int32_t m, int32_t n, int32_t k) {
  printer.printDimensionList(ArrayRef<int64_t>{m, n, k});
}

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
