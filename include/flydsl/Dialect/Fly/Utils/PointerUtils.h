// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLY_UTILS_POINTERUTILS_H
#define FLYDSL_DIALECT_FLY_UTILS_POINTERUTILS_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

namespace mlir::fly {

TypedValue<LLVM::LLVMPointerType> applySwizzleOnPtr(OpBuilder &b, Location loc,
                                                    TypedValue<LLVM::LLVMPointerType> ptr,
                                                    SwizzleAttr swizzle);

/// Project llvm unsupported small-float element types (Float8/Float6/Float4) to integer types of
/// the same bit-width. Non-small-float types are returned unchanged.
Type projectToLLVMCompatibleElemTy(Type elemTy);

/// Compute the SSA-value type corresponding to a register memref.
///
/// \p llvmCompatibleType controls whether llvm unsupported small-float element types (e.g.
/// f8E4M3FNUZ/f8E5M2FNUZ) are projected to their same-width integer counterpart.
Type RegMem2SSAType(fly::MemRefType memRefTy, bool llvmCompatibleType = false);

} // namespace mlir::fly

#endif // FLYDSL_DIALECT_FLY_UTILS_POINTERUTILS_H
