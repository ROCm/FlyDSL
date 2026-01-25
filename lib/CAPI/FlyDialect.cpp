#include "flydsl-c/FlyDialect.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Fly, fly, mlir::fly::FlyDialect)
