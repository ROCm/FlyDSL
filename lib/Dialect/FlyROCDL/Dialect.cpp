// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"

using namespace mlir;
using namespace mlir::fly;
using namespace mlir::fly_rocdl;

#include "flydsl/Dialect/FlyROCDL/IR/AtomStateEnums.cpp.inc"
#include "flydsl/Dialect/FlyROCDL/IR/AttrEnums.cpp.inc"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "flydsl/Dialect/FlyROCDL/IR/Atom.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "flydsl/Dialect/FlyROCDL/IR/AttrDefs.cpp.inc"

void FlyROCDLDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "flydsl/Dialect/FlyROCDL/IR/Atom.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "flydsl/Dialect/FlyROCDL/IR/AttrDefs.cpp.inc"
      >();
}
