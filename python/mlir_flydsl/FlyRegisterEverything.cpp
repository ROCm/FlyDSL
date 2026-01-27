#include "mlir-c/RegisterEverything.h"
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
  m.doc() = "MLIR All Upstream Dialects, Translations and Passes Registration";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    mlirRegisterAllDialects(registry);

    MlirDialectHandle flyHandle = mlirGetDialectHandle__fly__();
    mlirDialectHandleInsertDialect(flyHandle, registry);
    MlirDialectHandle flyROCDLHandle = mlirGetDialectHandle__fly_rocdl__();
    mlirDialectHandleInsertDialect(flyROCDLHandle, registry);
  });
  m.def("register_llvm_translations",
        [](MlirContext context) { mlirRegisterAllLLVMTranslations(context); });

  // Register all passes on load.
  mlirRegisterAllPasses();

  mlir::fly::registerFlyPasses();
  // Register Fly to ROCDL conversion pass
  mlir::registerFlyToROCDLConversionPass();
}
