#ifndef FLIR_DIALECT_H
#define FLIR_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
// Include generated dialect declarations.
#include "flir/FlirDialect.h.inc"

// Include generated type declarations.
#define GET_TYPEDEF_CLASSES
#include "flir/FlirTypeDefs.h.inc"

// Include generated attribute declarations.
#define GET_ATTRDEF_CLASSES
#include "flir/FlirAttrDefs.h.inc"

#endif // FLIR_DIALECT_H
