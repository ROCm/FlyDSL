//===- CutePassesModule.cpp - CuTe Passes Python Module ------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "cute/CutePasses.h"

#include <pybind11/pybind11.h>
#include <iostream>

namespace py = pybind11;

// Manual registration of implemented passes
inline void registerCuteToStandardPassManual() {
  std::cerr << "[CutePassesExt] Registering cute-to-standard pass..." << std::endl;
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return ::mlir::cute::createCuteToStandardPass();
  });
  std::cerr << "[CutePassesExt] Pass registered!" << std::endl;
}

PYBIND11_MODULE(_cutePassesExt, m) {
  std::cerr << "[CutePassesExt] Module initialization started" << std::endl;
  
  m.doc() = "CuTe transformation passes extension module";

  m.def("register_passes", []() {
    registerCuteToStandardPassManual();
  }, "Register all implemented CuTe transformation passes with MLIR");
  
  // Auto-register on module import
  registerCuteToStandardPassManual();
  
  std::cerr << "[CutePassesExt] Module initialization completed" << std::endl;
}
