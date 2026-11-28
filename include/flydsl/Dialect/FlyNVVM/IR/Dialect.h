// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLYNVVM_IR_DIALECT_H
#define FLYDSL_DIALECT_FLYNVVM_IR_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

#include "flydsl/Dialect/FlyNVVM/IR/Dialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "flydsl/Dialect/FlyNVVM/IR/Atom.h.inc"

namespace mlir::fly_nvvm {} // namespace mlir::fly_nvvm

#endif // FLYDSL_DIALECT_FLYNVVM_IR_DIALECT_H
