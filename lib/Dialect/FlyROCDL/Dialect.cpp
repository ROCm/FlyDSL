#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"

using namespace mlir;
using namespace mlir::fly;
using namespace mlir::fly_rocdl;

#include "flydsl/Dialect/FlyROCDL/IR/Dialect.cpp.inc"
#include "flydsl/Dialect/FlyROCDL/IR/Enums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "flydsl/Dialect/FlyROCDL/IR/Atom.cpp.inc"

void FlyROCDLDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "flydsl/Dialect/FlyROCDL/IR/Atom.cpp.inc"
      >();
}
