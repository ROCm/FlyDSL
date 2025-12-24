#ifndef ROCIR_ROCM_OPS_H
#define ROCIR_ROCM_OPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "flir/FlirRocmDialect.h"

#define GET_OP_CLASSES
#include "flir/FlirRocmOps.h.inc"

#endif // ROCIR_ROCM_OPS_H

