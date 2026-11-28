// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

// FlyDSL's site initializer. The upstream _mlirRegisterEverything module
// registers the MLIR dialects, passes and LLVM translations; _mlir_libs then
// imports this module and adds the FlyDSL dialects to the same registry.

#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include "flydsl-c/RegisterEverything.h"

NB_MODULE(_site_initialize_0, m) {
  m.doc() = "FlyDSL Dialects and Passes Registration";

  m.def("register_dialects",
        [](MlirDialectRegistry registry) { flydslRegisterAllDialects(registry); });

  // Register all passes on load, as _mlirRegisterEverything does.
  flydslRegisterAllPasses();
}
