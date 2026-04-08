// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLYROCDL_UTILS_BUFFERFATPTR_H
#define FLYDSL_DIALECT_FLYROCDL_UTILS_BUFFERFATPTR_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

namespace mlir::fly {

class BufferFatPtr {
  static constexpr unsigned kRsrcAddrSpace = 8;   // BufferDesc
  static constexpr unsigned kOffsetBitWidth = 32; // constrained by BufferCopy instruction

  fly::PointerType ptrTy;
  Value fatPtr;

public:
  BufferFatPtr(fly::PointerType ptrTy, Value v) : ptrTy(ptrTy), fatPtr(v) {
    assert(ptrTy.getAddressSpace().getValue() == AddressSpace::BufferDesc);
  }

  static LLVM::LLVMStructType getType(MLIRContext *ctx) {
    return LLVM::LLVMStructType::getLiteral(ctx, {LLVM::LLVMPointerType::get(ctx, kRsrcAddrSpace),
                                                  IntegerType::get(ctx, kOffsetBitWidth)});
  }
  static Value pack(OpBuilder &b, Location loc, Value bufferRsrc, Value valOffset = nullptr) {
    auto structTy = getType(b.getContext());
    Value undef = LLVM::UndefOp::create(b, loc, structTy);
    if (!valOffset) {
      valOffset = arith::ConstantIntOp::create(b, loc, 0, kOffsetBitWidth);
    }
    Value withRsrc = LLVM::InsertValueOp::create(b, loc, undef, bufferRsrc, ArrayRef<int64_t>{0});
    return LLVM::InsertValueOp::create(b, loc, withRsrc, valOffset, ArrayRef<int64_t>{1});
  }

  Value bufferRsrc(OpBuilder &b, Location loc) const {
    return LLVM::ExtractValueOp::create(b, loc, fatPtr, ArrayRef<int64_t>{0});
  }

  Value valOffset(OpBuilder &b, Location loc) const {
    return LLVM::ExtractValueOp::create(b, loc, fatPtr, ArrayRef<int64_t>{1});
  }

  Value byteOffset(OpBuilder &b, Location loc) const {
    int64_t bits = ptrTy.getElemTy().getIntOrFloatBitWidth();
    Value off = valOffset(b, loc);
    if (bits == 8)
      return off;
    if (bits > 8 && bits % 8 == 0) {
      int64_t elemBytes = bits / 8;
      Value scale = arith::ConstantIntOp::create(b, loc, elemBytes, kOffsetBitWidth);
      return arith::MulIOp::create(b, loc, off, scale);
    }
    Value scale = arith::ConstantIntOp::create(b, loc, bits, kOffsetBitWidth);
    off = arith::MulIOp::create(b, loc, off, scale);
    Value const8 = arith::ConstantIntOp::create(b, loc, 8, kOffsetBitWidth);
    return arith::DivUIOp::create(b, loc, off, const8);
  }

  Value swizzleByteOffset(OpBuilder &b, Location loc) const {
    Value off = byteOffset(b, loc);
    SwizzleAttr swizzle = ptrTy.getSwizzle();
    if (swizzle.isTrivialSwizzle())
      return off;
    auto offsetTy = IntegerType::get(b.getContext(), kOffsetBitWidth);
    int64_t bitMaskValue = ((int64_t{1} << swizzle.getMask()) - 1)
                           << (swizzle.getBase() + swizzle.getShift());
    Value bitMask = arith::ConstantIntOp::create(b, loc, offsetTy, bitMaskValue);
    Value shiftAmt = arith::ConstantIntOp::create(b, loc, offsetTy, swizzle.getShift());
    Value masked = arith::AndIOp::create(b, loc, off, bitMask);
    Value shifted = arith::ShRUIOp::create(b, loc, masked, shiftAmt);
    return arith::XOrIOp::create(b, loc, off, shifted);
  }

  Value addOffset(OpBuilder &b, Location loc, Value delta) const {
    Type offTy = IntegerType::get(b.getContext(), kOffsetBitWidth);
    if (delta.getType() != offTy) {
      if (delta.getType().isIndex())
        delta = arith::IndexCastOp::create(b, loc, offTy, delta);
      else if (delta.getType().getIntOrFloatBitWidth() < kOffsetBitWidth)
        delta = arith::ExtSIOp::create(b, loc, offTy, delta);
      else
        delta = arith::TruncIOp::create(b, loc, offTy, delta);
    }
    Value newOff = arith::AddIOp::create(b, loc, valOffset(b, loc), delta);
    return pack(b, loc, bufferRsrc(b, loc), newOff);
  }
};

} // namespace mlir::fly

#endif // FLYDSL_DIALECT_FLYROCDL_UTILS_BUFFERFATPTR_H
