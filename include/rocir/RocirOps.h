#ifndef ROCIR_OPS_H
#define ROCIR_OPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "rocir/RocirDialect.h"

#define GET_OP_CLASSES
#include "rocir/RocirOps.h.inc"

#endif // ROCIR_OPS_H

