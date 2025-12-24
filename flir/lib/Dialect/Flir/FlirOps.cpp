//===- FlirOps.cpp - Flir Operations Implementation --------------------===//
//
// Implementation of Flir operation verification and methods
//
//===----------------------------------------------------------------------===//

#include "flir/FlirOps.h"
#include "flir/FlirDialect.h"
#include "flir/FlirLayoutAlgebra.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::flir;

//===----------------------------------------------------------------------===//
// TableGen generated code
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "flir/FlirOps.cpp.inc"
