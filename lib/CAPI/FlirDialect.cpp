#include "flir-c/FlirDialect.h"
#include "flir/FlirDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Flir, flir, mlir::flir::FlirDialect)

