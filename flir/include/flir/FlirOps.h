#ifndef FLIR_OPS_H
#define FLIR_OPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "flir/FlirDialect.h"
#include "flir/FlirPatternAttr.h"

#define GET_OP_CLASSES
#include "flir/FlirOps.h.inc"

#endif // FLIR_OPS_H

