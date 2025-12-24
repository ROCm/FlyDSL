#ifndef ROCIR_C_DIALECT_H
#define ROCIR_C_DIALECT_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Flir, flir);

#ifdef __cplusplus
}
#endif

#endif // ROCIR_C_DIALECT_H

