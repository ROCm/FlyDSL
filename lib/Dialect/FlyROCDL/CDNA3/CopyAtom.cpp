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

  if (srcIsBuffer && !dstIsBuffer) {
    auto [srcRsrc, srcOff] = unpackBuffer(src, srcMemTy);
    Value loaded = ROCDL::RawPtrBufferLoadOp::create(builder, loc, copyTy, srcRsrc, srcOff, zero,
                                                     zero, noAttrs, noAttrs, noAttrs);
    LLVM::StoreOp::create(builder, loc, loaded, dst);
  } else if (!srcIsBuffer && dstIsBuffer) {
    auto [dstRsrc, dstOff] = unpackBuffer(dst, dstMemTy);
    Value loaded = LLVM::LoadOp::create(builder, loc, copyTy, src);
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
