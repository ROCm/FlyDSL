//===- RocirOps.cpp - Rocir Operations Implementation --------------------===//
//
// Implementation of Rocir operation verification and methods
//
//===----------------------------------------------------------------------===//

#include "rocir/RocirOps.h"
#include "rocir/RocirDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::rocir;

//===----------------------------------------------------------------------===//
// TableGen generated code
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "rocir/RocirOps.cpp.inc"
