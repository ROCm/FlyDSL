#include "flydsl-c/FlyROCDLDialect.h"

#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(FlyROCDL, fly_rocdl, mlir::fly_rocdl::FlyROCDLDialect)
