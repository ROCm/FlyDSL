#ifndef FLYDSL_C_DIALECTS_H
#define FLYDSL_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Fly, fly);

#ifdef __cplusplus
}
#endif

#endif // FLYDSL_C_DIALECTS_H
