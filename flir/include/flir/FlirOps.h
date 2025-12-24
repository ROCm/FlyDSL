#ifndef ROCIR_OPS_H
#define ROCIR_OPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "flir/FlirDialect.h"

#define GET_OP_CLASSES
#include "flir/FlirOps.h.inc"

#endif // ROCIR_OPS_H

