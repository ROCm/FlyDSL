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

#include "flydsl/InitAllDialects.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::fly::registerAllDialects(registry);

  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
