#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include "flydsl-c/FlyDialect.h"
#include "flydsl-c/FlyROCDLDialect.h"
#include "flydsl/Conversion/FlyToROCDL/FlyToROCDL.h"
#include "flydsl/Dialect/Fly/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_REGISTRATION
#include "flydsl/Conversion/Passes.h.inc"
} // namespace mlir

NB_MODULE(_mlirRegisterEverything, m) {
  m.doc() = "Fly Dialect Registration";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    // Only register Fly dialects
    // Upstream dialects are already registered by upstream MLIR
    
    MlirDialectHandle flyHandle = mlirGetDialectHandle__fly__();
    mlirDialectHandleInsertDialect(flyHandle, registry);
    MlirDialectHandle flyROCDLHandle = mlirGetDialectHandle__fly_rocdl__();
    mlirDialectHandleInsertDialect(flyROCDLHandle, registry);
  });

  // Register ONLY Fly's passes (not all upstream passes)
  mlir::fly::registerFlyPasses();
  mlir::registerFlyToROCDLConversionPass();
}
