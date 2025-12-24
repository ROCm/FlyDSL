//===- FlirPassesModule.cpp - Flir Passes Python Module -----------------===//

#include "mlir-c/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "flir/FlirDialect.h"
#include "flir/FlirPasses.h"

#include <pybind11/pybind11.h>

using namespace mlir::python::adaptors;

namespace py = pybind11;

// Provide a C API handle so Python can register the Flir dialect.
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Flir, flir, mlir::flir::FlirDialect)

// Auto-generated pass registration functions
#define GEN_PASS_REGISTRATION
#include "flir/FlirPasses.h.inc"

PYBIND11_MODULE(_flirPassesExt, m) {
  m.doc() = "Flir transformation passes and dialect helpers";

  // Register all passes at module load time
  
  // TableGen-generated registrations
  registerFlirToStandardPass();           // To standard dialect

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__flir__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context"), py::arg("load") = true,
      "Register (and optionally load) the Flir dialect for the given MLIR context");
}
