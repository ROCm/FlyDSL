//===- FlirRegisterPasses.cpp - Flir Passes Registration ----------------===//

#include "mlir-c/RegisterEverything.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include "flir/FlirDialect.h"
#include "flir/FlirPasses.h"

#include <mutex>

namespace mlir {
#define GEN_PASS_REGISTRATION
#include "flir/FlirPasses.h.inc"
} // namespace mlir

namespace nb = nanobind;

NB_MODULE(_flirPasses, m) {
  m.doc() = "Flir Passes Registration Module";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    mlirRegisterAllDialects(registry);

    // Register Flir dialect directly via the C++ DialectRegistry (no custom C-API required).
    auto *cppRegistry = unwrap(registry);
    cppRegistry->insert<mlir::flir::FlirDialect>();
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

  // Register Flir-specific passes
  // TableGen-generated registrations
  ::mlir::registerFlirToStandardPass();
  ::mlir::registerFlirTrivialDCEPass();
}

