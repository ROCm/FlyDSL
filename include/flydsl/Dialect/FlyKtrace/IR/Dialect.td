// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYKTRACE_DIALECT
#define FLYKTRACE_DIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def FlyKtrace_Dialect : Dialect {
  let name = "fly_ktrace";
  let cppNamespace = "::mlir::fly_ktrace";

  let summary = "In-kernel wave tracing annotations, before they are expanded.";
  let description = [{
    Holds one op per `fx.ktrace` entry point so a trace event survives as a single
    operation until `convert-fly-ktrace-to-rocdl` expands it into the slot claim,
    leader guard, timestamp read and record stores.

    The ops carry the event *name*; the 24-bit id the device writes is assigned
    host-side, so nothing here depends on the name table.
  }];

  // The expansion emits llvm.* and scf.* ops.
  let dependentDialects = [
    "LLVM::LLVMDialect",
    "scf::SCFDialect"
  ];

  let useDefaultTypePrinterParser = 1;
}

#endif // FLYKTRACE_DIALECT
