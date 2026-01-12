#ifndef FLIR_DIALECT_H
#define FLIR_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir::flir {

// AttrDef helper predicates (used by TableGen AnyAttrOf).
bool isValidDyncIntAttr(::mlir::Attribute attr);

} // namespace mlir::flir

// Include generated dialect declarations
#include "flir/FlirDialect.h.inc"

// Include generated attribute declarations
#define GET_ATTRDEF_CLASSES
#include "flir/FlirAttrs.h.inc"

// Include generated type declarations
#define GET_TYPEDEF_CLASSES
#include "flir/FlirTypes.h.inc"

// Include generated operation declarations
#endif // FLIR_DIALECT_H
