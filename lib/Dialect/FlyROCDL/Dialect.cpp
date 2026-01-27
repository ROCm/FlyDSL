#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"

using namespace mlir;
using namespace mlir::fly_rocdl;

#include "flydsl/Dialect/FlyROCDL/IR/Dialect.cpp.inc"
#include "flydsl/Dialect/FlyROCDL/IR/Enums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "flydsl/Dialect/FlyROCDL/IR/Atom.cpp.inc"

namespace mlir::fly_rocdl {

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

} // namespace mlir::fly_rocdl

void FlyROCDLDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "flydsl/Dialect/FlyROCDL/IR/Atom.cpp.inc"
      >();
}
