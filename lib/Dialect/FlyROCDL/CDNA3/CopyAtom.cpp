#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_rocdl {

bool CopyOpCDNA3BufferLDSTType::isStatic() const { return true; }

Attribute CopyOpCDNA3BufferLDSTType::getThrLayout() const { return FxLayout(FxC(1), FxC(1)); }

Attribute CopyOpCDNA3BufferLDSTType::getThrBitLayoutSrc() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpCDNA3BufferLDSTType::getThrBitLayoutDst() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpCDNA3BufferLDSTType::getThrBitLayoutRef() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}

} // namespace mlir::fly_rocdl
