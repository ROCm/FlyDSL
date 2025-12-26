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

  auto structure = st.getStructure();
  auto dims = st.getDims();
  if (structure.empty())
    return op->emitOpError() << "requires structured " << prettyName
                             << " type (e.g. " << prettyName << "<(...)>)";

  int64_t leafCount = 0;
  for (int32_t tag : structure)
    if (tag == -1)
      ++leafCount;

  if ((int64_t)dims.size() != leafCount)
    return op->emitOpError() << "invalid " << prettyName << " type: dims size ("
                             << dims.size() << ") must equal leaf count ("
                             << leafCount << ")";

  int64_t expectedOperands = leafCount;
  if (operandsAreDynOnly) {
    expectedOperands = 0;
    for (int64_t d : dims)
      if (d == -1)
        ++expectedOperands;
  }

  if ((int64_t)operands.size() != expectedOperands) {
    if (operandsAreDynOnly) {
      return op->emitOpError()
             << "expects " << expectedOperands
             << " dynamic leaf operands but got " << operands.size();
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
