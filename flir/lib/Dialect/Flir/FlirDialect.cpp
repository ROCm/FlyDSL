//===- FlirDialect.cpp - Flir Dialect Implementation --------------------===//

#include "flir/FlirDialect.h"
#include "flir/FlirOps.h" // Required for generated FlirOps.cpp.inc op class references.
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::flir;

#define GET_ATTRDEF_CLASSES
#include "flir/FlirAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "flir/FlirTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect initialization
//===----------------------------------------------------------------------===//

#include "flir/FlirDialect.cpp.inc"

void FlirDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "flir/FlirTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "flir/FlirAttrs.cpp.inc"
      >();
  
  addOperations<
#define GET_OP_LIST
#include "flir/FlirOps.cpp.inc"
  >();
}

bool mlir::flir::isValidDyncIntAttr(::mlir::Attribute attr) {
  return llvm::isa<mlir::flir::DyncI32Attr, mlir::flir::DyncI64Attr>(attr);
}
