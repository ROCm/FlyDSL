#include "mlir-c/RegisterEverything.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include "flydsl-c/FlyDialect.h"
#include "flydsl-c/FlyROCDLDialect.h"

// NOTE: Pass registration uses CAPI functions (mlirRegisterFlyPasses,
// mlirRegisterFlyToROCDLConversionPass) defined in the CAPI libraries
// (MLIRCPIFly, MLIRCPIFlyROCDL) rather than inline C++ functions.
// This ensures ::mlir::registerPass() calls resolve to the SAME pass
// registry instance in FlyPythonCAPI.so, avoiding ODR violations from
// statically-linked duplicate MLIRPass copies.

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
  // Use CAPI functions so pass registration goes into FlyPythonCAPI.so's
  // global pass registry (not a local copy in _mlirRegisterEverything.so).
  mlirRegisterAllPasses();
  mlirRegisterFlyPasses();
  mlirRegisterFlyToROCDLConversionPass();
}
