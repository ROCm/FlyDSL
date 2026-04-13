// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "flydsl/Dialect/FlyROCDL/Utils/BufferFatPtr.h"

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_rocdl {

bool CopyOpCDNA3BufferCopyType::isStatic() const { return true; }

Value CopyOpCDNA3BufferCopyType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                                    Value currentValue) const {
  if (currentValue && isa<MakeCopyAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  return MakeCopyAtomOp::create(builder, loc, CopyAtomType::get(*this, getBitSize()), getBitSize());
}

Attribute CopyOpCDNA3BufferCopyType::getThrLayout() const { return FxLayout(FxC(1), FxC(1)); }

Attribute CopyOpCDNA3BufferCopyType::getThrBitLayoutSrc() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpCDNA3BufferCopyType::getThrBitLayoutDst() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpCDNA3BufferCopyType::getThrBitLayoutRef() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}

LogicalResult CopyOpCDNA3BufferCopyType::emitAtomCall(OpBuilder &builder, Location loc,
                                                      Type copyAtomTyArg, Type srcMemTyArg,
                                                      Type dstMemTyArg, Value atomVal, Value src,
                                                      Value dst) const {
  auto srcMemTy = cast<fly::MemRefType>(srcMemTyArg);
  auto dstMemTy = cast<fly::MemRefType>(dstMemTyArg);

  IntegerType copyTy = builder.getIntegerType(getBitSize());

  AddressSpace srcAS = srcMemTy.getAddressSpace().getValue();
  AddressSpace dstAS = dstMemTy.getAddressSpace().getValue();

  bool srcIsBuffer = (srcAS == AddressSpace::BufferDesc);
  bool dstIsBuffer = (dstAS == AddressSpace::BufferDesc);

  if (!(srcIsBuffer || dstIsBuffer))
    return failure();

  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  ArrayAttr noAttrs;

  auto unpackBuffer = [&](Value val, fly::MemRefType flyTy) -> std::pair<Value, Value> {
    BufferFatPtr bp(flyTy.getPointerType(), val);
    return {bp.bufferRsrc(builder, loc), bp.swizzleByteOffset(builder, loc)};
  };

  // When writing to a register alloca, split wide stores into 64-bit
  // element-typed vector chunks (<4 x half>, <4 x i16>, etc.) that match
  // MFMA operand load types.  SROA can then forward store→load at each
  // offset and eliminate the alloca entirely.
  auto storeToRegSplit = [&](Value loaded, Value dstPtr) {
    Type elemTy = dstMemTy.getElemTy();
    unsigned totalBits = getBitSize();
    if (dstAS == AddressSpace::Register && elemTy && elemTy.isIntOrFloat()) {
      unsigned elemBits = elemTy.getIntOrFloatBitWidth();
      unsigned chunkBits = 64;
      if (elemBits < chunkBits && chunkBits % elemBits == 0 && totalBits > chunkBits) {
        // Split wide integer into 64-bit chunks.
        // For i8/fp8 elements, store as i64 (not <8 x i8>) to match MFMA
        // operand load types — type mismatch blocks LLVM SROA promotion.
        unsigned numChunks = totalBits / chunkBits;
        Type chunkStoreTy;
        if (elemBits <= 8)
          chunkStoreTy = builder.getI64Type();
        else
          chunkStoreTy = VectorType::get({static_cast<int64_t>(chunkBits / elemBits)}, elemTy);
        auto i64Ty = builder.getI64Type();
        auto i8Ty = builder.getI8Type();
        for (unsigned i = 0; i < numChunks; ++i) {
          Value chunk;
          if (i == 0) {
            chunk = LLVM::TruncOp::create(builder, loc, i64Ty, loaded);
          } else {
            Value shiftAmt = LLVM::ConstantOp::create(builder, loc, copyTy,
                builder.getIntegerAttr(copyTy, i * chunkBits));
            Value shifted = LLVM::LShrOp::create(builder, loc, loaded, shiftAmt);
            chunk = LLVM::TruncOp::create(builder, loc, i64Ty, shifted);
          }
          Value storeVal;
          if (isa<IntegerType>(chunkStoreTy))
            storeVal = chunk;  // already i64, no bitcast needed
          else
            storeVal = LLVM::BitcastOp::create(builder, loc, chunkStoreTy, chunk);
          Value dPtr = dstPtr;
          if (i > 0) {
            Value offset = LLVM::ConstantOp::create(builder, loc,
                builder.getIntegerType(32), i * (chunkBits / 8));
            dPtr = LLVM::GEPOp::create(builder, loc, dstPtr.getType(),
                                        i8Ty, dstPtr, ValueRange{offset});
          }
          LLVM::StoreOp::create(builder, loc, storeVal, dPtr);
        }
        return;
      }
      // totalBits <= 64: single store (i64 for i8/fp8, vector for others)
      if (elemBits < totalBits && totalBits % elemBits == 0) {
        if (elemBits <= 8) {
          // Store as integer to match MFMA operand load type
          LLVM::StoreOp::create(builder, loc, loaded, dstPtr);
        } else {
          auto vecTy = VectorType::get({static_cast<int64_t>(totalBits / elemBits)}, elemTy);
          Value vec = LLVM::BitcastOp::create(builder, loc, vecTy, loaded);
          LLVM::StoreOp::create(builder, loc, vec, dstPtr);
        }
        return;
      }
    }
    LLVM::StoreOp::create(builder, loc, loaded, dstPtr);
  };

  auto loadFromRegSplit = [&](Value srcPtr) -> Value {
    Type elemTy = srcMemTy.getElemTy();
    unsigned totalBits = getBitSize();
    if (srcAS == AddressSpace::Register && elemTy && elemTy.isIntOrFloat()) {
      unsigned elemBits = elemTy.getIntOrFloatBitWidth();
      unsigned chunkBits = 64;
      if (elemBits < chunkBits && chunkBits % elemBits == 0 && totalBits > chunkBits) {
        unsigned numChunks = totalBits / chunkBits;
        auto vecTy = VectorType::get({static_cast<int64_t>(chunkBits / elemBits)}, elemTy);
        auto i64Ty = builder.getI64Type();
        auto i8Ty = builder.getI8Type();
        Value result = LLVM::ConstantOp::create(builder, loc, copyTy,
            builder.getIntegerAttr(copyTy, 0));
        for (unsigned i = 0; i < numChunks; ++i) {
          Value sPtr = srcPtr;
          if (i > 0) {
            Value offset = LLVM::ConstantOp::create(builder, loc,
                builder.getIntegerType(32), i * (chunkBits / 8));
            sPtr = LLVM::GEPOp::create(builder, loc, srcPtr.getType(),
                                        i8Ty, srcPtr, ValueRange{offset});
          }
          Value vec = LLVM::LoadOp::create(builder, loc, vecTy, sPtr);
          Value chunk = LLVM::BitcastOp::create(builder, loc, i64Ty, vec);
          Value extended = LLVM::ZExtOp::create(builder, loc, copyTy, chunk);
          if (i > 0) {
            Value shiftAmt = LLVM::ConstantOp::create(builder, loc, copyTy,
                builder.getIntegerAttr(copyTy, i * chunkBits));
            extended = LLVM::ShlOp::create(builder, loc, extended, shiftAmt);
          }
          result = LLVM::OrOp::create(builder, loc, result, extended);
        }
        return result;
      }
      if (elemBits < totalBits && totalBits % elemBits == 0) {
        auto vecTy = VectorType::get({static_cast<int64_t>(totalBits / elemBits)}, elemTy);
        Value vec = LLVM::LoadOp::create(builder, loc, vecTy, srcPtr);
        return LLVM::BitcastOp::create(builder, loc, copyTy, vec);
      }
    }
    return LLVM::LoadOp::create(builder, loc, copyTy, srcPtr);
  };

  if (srcIsBuffer && !dstIsBuffer) {
    auto [srcRsrc, srcOff] = unpackBuffer(src, srcMemTy);
    Value loaded = ROCDL::RawPtrBufferLoadOp::create(builder, loc, copyTy, srcRsrc, srcOff, zero,
                                                     zero, noAttrs, noAttrs, noAttrs);
    storeToRegSplit(loaded, dst);
  } else if (!srcIsBuffer && dstIsBuffer) {
    auto [dstRsrc, dstOff] = unpackBuffer(dst, dstMemTy);
    Value loaded = loadFromRegSplit(src);
    ROCDL::RawPtrBufferStoreOp::create(builder, loc, loaded, dstRsrc, dstOff, zero, zero, noAttrs,
                                       noAttrs, noAttrs);
  } else {
    auto [srcRsrc, srcOff] = unpackBuffer(src, srcMemTy);
    auto [dstRsrc, dstOff] = unpackBuffer(dst, dstMemTy);
    Value loaded = ROCDL::RawPtrBufferLoadOp::create(builder, loc, copyTy, srcRsrc, srcOff, zero,
                                                     zero, noAttrs, noAttrs, noAttrs);
    ROCDL::RawPtrBufferStoreOp::create(builder, loc, loaded, dstRsrc, dstOff, zero, zero, noAttrs,
                                       noAttrs, noAttrs);
  }
  return success();
}

LogicalResult CopyOpCDNA3BufferCopyType::emitAtomCall(OpBuilder &builder, Location loc,
                                                      Type copyAtomTyArg, Type srcMemTyArg,
                                                      Type dstMemTyArg, Type predMemTyArg,
                                                      Value atomVal, Value src, Value dst,
                                                      Value pred) const {
  auto predMemTy = cast<fly::MemRefType>(predMemTyArg);
  Value predVal = LLVM::LoadOp::create(builder, loc, predMemTy.getElemTy(), pred);
  auto ifOp = scf::IfOp::create(builder, loc, TypeRange{}, predVal, /*withElse=*/false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

  return emitAtomCall(builder, loc, copyAtomTyArg, srcMemTyArg, dstMemTyArg, atomVal, src, dst);
}

} // namespace mlir::fly_rocdl
