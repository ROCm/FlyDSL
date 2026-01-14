//===- FlirOps.cpp - Flir Operations Implementation --------------------===//
//
// Implementation of Flir operation verification and methods
//
//===----------------------------------------------------------------------===//

#include "flir/FlirOps.h"
#include "flir/FlirDialect.h"
#include "flir/FlirLayoutAlgebra.h"
#include "flir/FlirPatternAttr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace mlir::flir;

namespace {

/// Shared verifier for structured (type-mode) tuple-carrying types used by
/// make_shape/make_stride/make_coord.
template <typename StructuredTy>
static LogicalResult verifyStructuredTupleType(Operation *op, Type resultTy,
                                               ValueRange operands,
                                               llvm::StringRef prettyName,
                                               bool operandsAreDynOnly) {
  auto st = llvm::dyn_cast<StructuredTy>(resultTy);
  if (!st)
    return op->emitOpError() << "expects " << prettyName << " result type";

  Attribute pattern = st.getPattern();
  if (!pattern)
    return op->emitOpError() << "requires " << prettyName
                             << " type to carry a tuple pattern (e.g. "
                             << prettyName << "<(...)>)";

  int64_t leafCount = getPatternRank(pattern);
  int64_t dynCount = countDynLeaves(pattern);

  int64_t expectedOperands = operandsAreDynOnly ? dynCount : leafCount;

  if ((int64_t)operands.size() != expectedOperands) {
    if (operandsAreDynOnly) {
      return op->emitOpError()
             << "expects " << expectedOperands
             << " dynamic leaf operands (for '?' leaves) but got "
             << operands.size();
    }
    return op->emitOpError()
           << "expects " << expectedOperands
           << " index operands but got " << operands.size();
  }
  return success();
}

} // namespace

LogicalResult MakeShapeOp::verify() {
  return verifyStructuredTupleType<ShapeType>(getOperation(), getResult().getType(),
                                             getValues(), "!flir.shape",
                                             /*operandsAreDynOnly=*/true);
}

LogicalResult MakeStrideOp::verify() {
  return verifyStructuredTupleType<StrideType>(getOperation(), getResult().getType(),
                                               getValues(), "!flir.stride",
                                               /*operandsAreDynOnly=*/true);
}

LogicalResult MakeCoordOp::verify() {
  // Coords are runtime values: operands provide all leaf coordinates.
  return verifyStructuredTupleType<CoordType>(getOperation(), getResult().getType(),
                                              getValues(), "!flir.coord",
                                              /*operandsAreDynOnly=*/false);
}

//===----------------------------------------------------------------------===//
// TableGen generated code
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "flir/FlirOps.cpp.inc"
