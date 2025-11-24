#ifndef CUTE_NVGPU_DIALECT_H
#define CUTE_NVGPU_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "cute/CuteNvgpuDialect.h.inc"

#define GET_OP_CLASSES
#include "cute/CuteNvgpuOps.h.inc"

#endif // CUTE_NVGPU_DIALECT_H
