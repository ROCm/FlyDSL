// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "flydsl/Dialect/FlyNVVM/IR/Dialect.h"

using namespace mlir;
using namespace mlir::fly;
using namespace mlir::fly_nvvm;

#include "flydsl/Dialect/FlyNVVM/IR/Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "flydsl/Dialect/FlyNVVM/IR/Atom.cpp.inc"

void FlyNVVMDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "flydsl/Dialect/FlyNVVM/IR/Atom.cpp.inc"
      >();
}
