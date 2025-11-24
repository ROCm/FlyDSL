#ifndef CUTE_ROCM_DIALECT_H
#define CUTE_ROCM_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "cute/CuteRocmDialect.h.inc"

#define GET_OP_CLASSES
#include "cute/CuteRocmOps.h.inc"

#endif // CUTE_ROCM_DIALECT_H
