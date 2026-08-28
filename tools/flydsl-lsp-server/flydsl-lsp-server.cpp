// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// FlyDSL MLIR Language Server entry point.
//
// Provides editor support (diagnostics, hover, go-to-definition, completion)
// for FlyDSL `.mlir` files (e.g. tests/mlir and IR dumps) via upstream
// MlirLspServerMain. This is not a Python DSL / kernel language server.
//
// Point the editor's MLIR LSP client at:
//   build-fly/bin/flydsl-lsp-server

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"

#include "flydsl/Backend/Backend.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

// Forward-declare per-backend CAPI dialect registration.
// FLYDSL_BACKEND_COUNT and FLYDSL_BACKEND_0..N-1 are set by CMake.
#define DECLARE_BACKEND(name)                                                                      \
  extern "C" void flydsl_register_##name##_dialects(MlirDialectRegistry);
FLYDSL_FOR_EACH_BACKEND(DECLARE_BACKEND)

#define REGISTER_BACKEND_DIALECTS(name) flydsl_register_##name##_dialects(wrap(&registry));

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  registry.insert<mlir::fly::FlyDialect>();
  FLYDSL_FOR_EACH_BACKEND(REGISTER_BACKEND_DIALECTS)

  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
