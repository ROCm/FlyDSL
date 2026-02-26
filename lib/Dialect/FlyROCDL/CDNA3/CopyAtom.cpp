#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_rocdl {

bool CopyAtom_CDNA3_BufferLSAType::isStatic() const { return true; }

Attribute CopyAtom_CDNA3_BufferLSAType::getThrLayout() const {
  return FxLayout(FxC(1), FxC(1));
}
Attribute CopyAtom_CDNA3_BufferLSAType::getThrValLayoutSrc() const { return {}; }
Attribute CopyAtom_CDNA3_BufferLSAType::getThrValLayoutDst() const { return {}; }
Attribute CopyAtom_CDNA3_BufferLSAType::getThrValLayoutRef() const { return {}; }

} // namespace mlir::fly_rocdl
