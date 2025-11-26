//===- RocirPassesModule.cpp - Rocir Passes Python Module -----------------===//

#include "mlir-c/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "rocir/RocirDialect.h"
#include "rocir/RocirPasses.h"

#include <pybind11/pybind11.h>

using namespace mlir::python::adaptors;

namespace py = pybind11;

// Provide a C API handle so Python can register the Rocir dialect.
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Rocir, rocir, mlir::rocir::RocirDialect)

PYBIND11_MODULE(_rocirPassesExt, m) {
  m.doc() = "Rocir transformation passes and dialect helpers";

  // Register Rocir passes with the global registry (shared with Python MLIR).
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::rocir::createRocirCoordLoweringPass();
  });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return mlir::rocir::createRocirToStandardPass();
  });

  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__rocir__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context"), py::arg("load") = true,
      "Register (and optionally load) the Rocir dialect for the given MLIR context");
}
