#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_rocdl {

bool CopyOp_CDNA3_BufferLSAType::isStatic() const { return true; }

Attribute CopyOp_CDNA3_BufferLSAType::getThrLayout() const {
  return FxLayout(FxC(1), FxC(1));
}
Attribute CopyOp_CDNA3_BufferLSAType::getThrBitLayoutSrc() const { return {}; }
Attribute CopyOp_CDNA3_BufferLSAType::getThrBitLayoutDst() const { return {}; }
Attribute CopyOp_CDNA3_BufferLSAType::getThrBitLayoutRef() const { return {}; }

} // namespace mlir::fly_rocdl
