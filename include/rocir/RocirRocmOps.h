#ifndef ROCIR_ROCM_OPS_H
#define ROCIR_ROCM_OPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "rocir/RocirRocmDialect.h"

#define GET_OP_CLASSES
#include "rocir/RocirRocmOps.h.inc"

#endif // ROCIR_ROCM_OPS_H

