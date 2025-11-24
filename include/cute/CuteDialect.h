#ifndef CUTE_DIALECT_H
#define CUTE_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"

// Include generated dialect declarations
#include "cute/CuteDialect.h.inc"

// Include custom types
#include "cute/CuteTypes.h"

// Include generated operation declarations
#define GET_OP_CLASSES
#include "cute/CuteOps.h.inc"

#endif // CUTE_DIALECT_H
