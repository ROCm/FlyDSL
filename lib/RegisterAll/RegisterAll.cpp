// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#include "flydsl/InitAllDialects.h"
#include "flydsl/InitAllPasses.h"

#include "flydsl/Backend/Backend.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/IR/FlyLLVMTranslation.h"
#include "flydsl/Dialect/Fly/Transforms/Passes.h"
#include "flydsl/Dialect/FlyKtrace/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"

namespace mlir::fly {

// Every backend in FLYDSL_BACKENDS defines these in lib/RegisterAll/<Backend>.cpp.
#define DECLARE_BACKEND(name)                                                                      \
  void registerBackendDialects_##name(DialectRegistry &registry);                                  \
  void registerBackendPasses_##name();
FLYDSL_FOR_EACH_BACKEND(DECLARE_BACKEND)
#undef DECLARE_BACKEND

void registerAllDialects(DialectRegistry &registry) {
  registry.insert<FlyDialect, fly_ktrace::FlyKtraceDialect>();
  registerOffloadingLLVMTranslationInterfaceExternalModels(registry);
#define REGISTER_BACKEND_DIALECTS(name) registerBackendDialects_##name(registry);
  FLYDSL_FOR_EACH_BACKEND(REGISTER_BACKEND_DIALECTS)
#undef REGISTER_BACKEND_DIALECTS
}

void registerAllPasses() {
  registerFlyPasses();
#define REGISTER_BACKEND_PASSES(name) registerBackendPasses_##name();
  FLYDSL_FOR_EACH_BACKEND(REGISTER_BACKEND_PASSES)
#undef REGISTER_BACKEND_PASSES
}

} // namespace mlir::fly
