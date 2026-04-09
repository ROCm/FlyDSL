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

std::optional<unsigned> CopyOpCDNA3BufferCopyType::getFieldIndex(AtomStateField field) {
  switch (field) {
  case AtomStateField::Soffset:
    return 0;
  }
  return std::nullopt;
}

Type CopyOpCDNA3BufferCopyType::getConvertedType(MLIRContext *ctx) const {
  return LLVM::LLVMStructType::getLiteral(ctx, {IntegerType::get(ctx, 32)});
}

Value CopyOpCDNA3BufferCopyType::getDefaultState(OpBuilder &builder, Location loc) const {
  auto structTy = cast<LLVM::LLVMStructType>(getConvertedType(builder.getContext()));
  Value state = LLVM::UndefOp::create(builder, loc, structTy);
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  return LLVM::InsertValueOp::create(builder, loc, state, zero,
                                     ArrayRef<int64_t>{*getFieldIndex(AtomStateField::Soffset)});
}

Value CopyOpCDNA3BufferCopyType::setAtomState(OpBuilder &builder, Location loc, Value atomStruct,
                                              Attribute fieldAttr, Value fieldValue) const {
  auto fieldStr = dyn_cast<StringAttr>(fieldAttr);
  if (!fieldStr)
    return nullptr;
  auto field = symbolizeAtomStateField(fieldStr.getValue());
  if (!field)
    return nullptr;
  auto idx = getFieldIndex(*field);
  if (!idx)
    return nullptr;
  return LLVM::InsertValueOp::create(builder, loc, atomStruct, fieldValue, ArrayRef<int64_t>{*idx});
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

  if (srcIsBuffer == dstIsBuffer)
    return failure();

  Value soffsetRaw = LLVM::ExtractValueOp::create(
      builder, loc, atomVal, ArrayRef<int64_t>{*getFieldIndex(AtomStateField::Soffset)});

  fly::MemRefType bufferMemTy = srcIsBuffer ? srcMemTy : dstMemTy;
  int64_t elemBits = bufferMemTy.getElemTy().getIntOrFloatBitWidth();
  Value soffset;
  if (elemBits == 8) {
    soffset = soffsetRaw;
  } else if (elemBits > 8 && elemBits % 8 == 0) {
    Value scale = arith::ConstantIntOp::create(builder, loc, elemBits / 8, 32);
    soffset = arith::MulIOp::create(builder, loc, soffsetRaw, scale);
  } else {
    Value scale = arith::ConstantIntOp::create(builder, loc, elemBits, 32);
    Value bits = arith::MulIOp::create(builder, loc, soffsetRaw, scale);
    Value eight = arith::ConstantIntOp::create(builder, loc, 8, 32);
    soffset = arith::DivUIOp::create(builder, loc, bits, eight);
  }

  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  ArrayAttr noAttrs;

  auto unpackBuffer = [&](Value val, fly::MemRefType flyTy) -> std::pair<Value, Value> {
    BufferFatPtr bp(flyTy.getPointerType(), val);
    return {bp.bufferRsrc(builder, loc), bp.swizzleByteOffset(builder, loc)};
  };

  if (srcIsBuffer && !dstIsBuffer) {
    auto [srcRsrc, srcOff] = unpackBuffer(src, srcMemTy);
    Value loaded = ROCDL::RawPtrBufferLoadOp::create(builder, loc, copyTy, srcRsrc, srcOff, soffset,
                                                     zero, noAttrs, noAttrs, noAttrs);
    LLVM::StoreOp::create(builder, loc, loaded, dst);
  } else if (!srcIsBuffer && dstIsBuffer) {
    auto [dstRsrc, dstOff] = unpackBuffer(dst, dstMemTy);
    Value loaded = LLVM::LoadOp::create(builder, loc, copyTy, src);
    ROCDL::RawPtrBufferStoreOp::create(builder, loc, loaded, dstRsrc, dstOff, soffset, zero,
                                       noAttrs, noAttrs, noAttrs);
  } else {
    return failure();
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
