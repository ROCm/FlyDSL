//===- RocirRegisterPasses.cpp - Rocir Passes Registration ----------------===//

#include "mlir-c/RegisterEverything.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include "rocir-c/RocirDialect.h"
#include "rocir/RocirDialect.h"
#include "rocir/RocirPasses.h"

#include <mutex>

namespace mlir {
#define GEN_PASS_REGISTRATION
#include "rocir/RocirPasses.h.inc"
} // namespace mlir

namespace nb = nanobind;

NB_MODULE(_rocirPasses, m) {
  m.doc() = "Rocir Passes Registration Module";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    mlirRegisterAllDialects(registry);
    
    MlirDialectHandle rocirHandle = mlirGetDialectHandle__rocir__();
    mlirDialectHandleInsertDialect(rocirHandle, registry);
  });

  m.def("register_llvm_translations",
        [](MlirContext context) { 
          mlirRegisterAllLLVMTranslations(context); 
        });

  // Register upstream passes once.
  //
  // Note: In some environments, repeatedly registering can crash with
  // "Option 'basic' already exists!". Guard with std::call_once.
  static std::once_flag register_all_passes_once;
  std::call_once(register_all_passes_once, []() { mlirRegisterAllPasses(); });

  // Register Rocir-specific passes
  // Manual registration for RocirCoordLoweringPass
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return ::mlir::rocir::createRocirCoordLoweringPass();
  });

  // TableGen-generated registrations
  ::mlir::registerRocirToStandardPass();
  ::mlir::registerRocirTrivialDCEPass();
}

