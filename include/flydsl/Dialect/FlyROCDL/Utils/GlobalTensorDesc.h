// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLYROCDL_UTILS_GLOBALTENSORDESC_H
#define FLYDSL_DIALECT_FLYROCDL_UTILS_GLOBALTENSORDESC_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

// Operand layout for a global_tensor_desc pointer: a plain global base pointer
// plus the tensor's 2D runtime extent (globalDim). The base is a normal
// addrspace(1) pointer (PtrToInt-able), unlike the buffer_desc fat resource, so
// descriptor-driven DMA atoms can still compute a raw VA and read tensor_dim0 /
// tensor_dim1 for out-of-bounds handling.
namespace mlir::fly_rocdl {

struct GlobalTensorDesc {
  static constexpr unsigned kBaseAddrSpace = 1; // global

  static LLVM::LLVMStructType getType(MLIRContext *ctx) {
    return LLVM::LLVMStructType::getLiteral(
        ctx, {LLVM::LLVMPointerType::get(ctx, kBaseAddrSpace), IntegerType::get(ctx, 32),
              IntegerType::get(ctx, 32)});
  }

  static Value pack(OpBuilder &b, Location loc, Value base, Value dim0, Value dim1) {
    Value s = LLVM::UndefOp::create(b, loc, getType(b.getContext()));
    s = LLVM::InsertValueOp::create(b, loc, s, base, ArrayRef<int64_t>{0});
    s = LLVM::InsertValueOp::create(b, loc, s, dim0, ArrayRef<int64_t>{1});
    s = LLVM::InsertValueOp::create(b, loc, s, dim1, ArrayRef<int64_t>{2});
    return s;
  }

  static Value base(OpBuilder &b, Location loc, Value v) {
    return LLVM::ExtractValueOp::create(b, loc, v, ArrayRef<int64_t>{0});
  }
  static Value dim0(OpBuilder &b, Location loc, Value v) {
    return LLVM::ExtractValueOp::create(b, loc, v, ArrayRef<int64_t>{1});
  }
  static Value dim1(OpBuilder &b, Location loc, Value v) {
    return LLVM::ExtractValueOp::create(b, loc, v, ArrayRef<int64_t>{2});
  }
};

} // namespace mlir::fly_rocdl

#endif // FLYDSL_DIALECT_FLYROCDL_UTILS_GLOBALTENSORDESC_H
