#ifndef ROCIR_ROCM_DIALECT_H
#define ROCIR_ROCM_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

#include "flir/FlirRocmDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "flir/FlirRocmTypes.h.inc"

#endif // ROCIR_ROCM_DIALECT_H

