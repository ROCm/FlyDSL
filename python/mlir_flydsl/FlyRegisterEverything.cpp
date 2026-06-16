// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir-c/RegisterEverything.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

#include "flydsl-c/FlyDialect.h"
#include "flydsl/Backend/Backend.h"

// Forward-declare per-backend CAPI registration functions.
// FLYDSL_BACKEND_COUNT and FLYDSL_BACKEND_0..N-1 are set by CMake.
#define DECLARE_BACKEND(name)                                                                      \
  extern "C" void flydsl_register_##name##_dialects(MlirDialectRegistry);                          \
  extern "C" void flydsl_register_##name##_passes(void);
FLYDSL_FOR_EACH_BACKEND(DECLARE_BACKEND)

#define REGISTER_BACKEND_DIALECTS(name) flydsl_register_##name##_dialects(registry);
#define REGISTER_BACKEND_PASSES(name) flydsl_register_##name##_passes();

NB_MODULE(_mlirRegisterEverything, m) {
  m.doc() = "MLIR All Upstream Dialects, Translations and Passes Registration";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    mlirRegisterAllDialects(registry);

    MlirDialectHandle flyHandle = mlirGetDialectHandle__fly__();
    mlirDialectHandleInsertDialect(flyHandle, registry);
    FLYDSL_FOR_EACH_BACKEND(REGISTER_BACKEND_DIALECTS)
  });
  m.def("register_llvm_translations", [](MlirContext context) {
    mlirRegisterAllLLVMTranslations(context);
    mlirRegisterFlyExplicitModuleOffloadingLLVMTranslation(context);
  });

  mlirRegisterAllPasses();
  mlirRegisterFlyPasses();
  FLYDSL_FOR_EACH_BACKEND(REGISTER_BACKEND_PASSES)
}
