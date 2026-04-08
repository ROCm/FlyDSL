// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLY_UTILS_POINTERUTILS_H
#define FLYDSL_DIALECT_FLY_UTILS_POINTERUTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir::fly {

TypedValue<LLVM::LLVMPointerType> applySwizzleOnPtr(OpBuilder &b, Location loc, TypedValue<LLVM::LLVMPointerType> ptr, SwizzleAttr swizzle);

} // namespace mlir::fly

#endif // FLYDSL_DIALECT_FLY_UTILS_POINTERUTILS_H
