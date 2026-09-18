// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLYKTRACE_IR_DIALECT_H
#define FLYDSL_DIALECT_FLYKTRACE_IR_DIALECT_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "flydsl/Dialect/FlyKtrace/IR/Dialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "flydsl/Dialect/FlyKtrace/IR/TypeDefs.h.inc"

#define GET_OP_CLASSES
#include "flydsl/Dialect/FlyKtrace/IR/Ops.h.inc"

#endif // FLYDSL_DIALECT_FLYKTRACE_IR_DIALECT_H
