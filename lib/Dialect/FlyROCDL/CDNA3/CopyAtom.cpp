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
  default:
    return std::nullopt;
  }
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

// --- CopyOpCDNA3BufferCopyLDS ---

LogicalResult CopyOpCDNA3BufferCopyLDSType::verify(function_ref<InFlightDiagnostic()> emitError,
                                                   int32_t bitSize) {
  if (bitSize != 32 && bitSize != 64 && bitSize != 128)
    return emitError() << "unsupported bitSize = " << bitSize << " for BufferCopyLDS";
  return success();
}

std::optional<unsigned> CopyOpCDNA3BufferCopyLDSType::getFieldIndex(AtomStateField field) {
  switch (field) {
  case AtomStateField::Soffset:
    return 0;
  case AtomStateField::ImmOffset:
    return 1;
  }
  return std::nullopt;
}

Type CopyOpCDNA3BufferCopyLDSType::getConvertedType(MLIRContext *ctx) const {
  auto i32Ty = IntegerType::get(ctx, 32);
  return LLVM::LLVMStructType::getLiteral(ctx, {i32Ty, i32Ty});
}

Value CopyOpCDNA3BufferCopyLDSType::getDefaultState(OpBuilder &builder, Location loc) const {
  auto structTy = cast<LLVM::LLVMStructType>(getConvertedType(builder.getContext()));
  Value state = LLVM::UndefOp::create(builder, loc, structTy);
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  state = LLVM::InsertValueOp::create(builder, loc, state, zero,
                                      ArrayRef<int64_t>{*getFieldIndex(AtomStateField::Soffset)});
  state = LLVM::InsertValueOp::create(builder, loc, state, zero,
                                      ArrayRef<int64_t>{*getFieldIndex(AtomStateField::ImmOffset)});
  return state;
}

Value CopyOpCDNA3BufferCopyLDSType::setAtomState(OpBuilder &builder, Location loc, Value atomStruct,
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

Attribute CopyOpCDNA3BufferCopyLDSType::getThrLayout() const { return FxLayout(FxC(1), FxC(1)); }

Attribute CopyOpCDNA3BufferCopyLDSType::getThrBitLayoutSrc() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpCDNA3BufferCopyLDSType::getThrBitLayoutDst() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpCDNA3BufferCopyLDSType::getThrBitLayoutRef() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}

LogicalResult CopyOpCDNA3BufferCopyLDSType::emitAtomCall(OpBuilder &builder, Location loc,
                                                         Type copyAtomTyArg, Type srcMemTyArg,
                                                         Type dstMemTyArg, Value atomVal, Value src,
                                                         Value dst) const {
  auto srcMemTy = cast<fly::MemRefType>(srcMemTyArg);
  auto dstMemTy = cast<fly::MemRefType>(dstMemTyArg);

  AddressSpace srcAS = srcMemTy.getAddressSpace().getValue();
  AddressSpace dstAS = dstMemTy.getAddressSpace().getValue();

  if (srcAS != AddressSpace::BufferDesc || dstAS != AddressSpace::Shared)
    return failure();

  int32_t sizeBytes = getBitSize() / 8;

  Value soffsetRaw = LLVM::ExtractValueOp::create(
      builder, loc, atomVal, ArrayRef<int64_t>{*getFieldIndex(AtomStateField::Soffset)});
  Value immOffset = LLVM::ExtractValueOp::create(
      builder, loc, atomVal, ArrayRef<int64_t>{*getFieldIndex(AtomStateField::ImmOffset)});

  int64_t elemBits = srcMemTy.getElemTy().getIntOrFloatBitWidth();
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
  Value size = arith::ConstantIntOp::create(builder, loc, sizeBytes, 32);

  BufferFatPtr bp(srcMemTy.getPointerType(), src);
  Value srcRsrc = bp.bufferRsrc(builder, loc);
  Value srcOff = bp.swizzleByteOffset(builder, loc);

  ArrayAttr noAttrs;
  ROCDL::RawPtrBufferLoadLdsOp::create(builder, loc, srcRsrc, dst, size, srcOff, soffset, immOffset,
                                       zero, noAttrs, noAttrs, noAttrs);
  return success();
}

LogicalResult CopyOpCDNA3BufferCopyLDSType::emitAtomCall(OpBuilder &builder, Location loc,
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

// --- CopyOpCDNA3BufferAtomic ---

std::optional<unsigned> CopyOpCDNA3BufferAtomicType::getFieldIndex(AtomStateField field) {
  switch (field) {
  case AtomStateField::Soffset:
    return 0;
  default:
    return std::nullopt;
  }
}

Type CopyOpCDNA3BufferAtomicType::getConvertedType(MLIRContext *ctx) const {
  return LLVM::LLVMStructType::getLiteral(ctx, {IntegerType::get(ctx, 32)});
}

Value CopyOpCDNA3BufferAtomicType::getDefaultState(OpBuilder &builder, Location loc) const {
  auto structTy = cast<LLVM::LLVMStructType>(getConvertedType(builder.getContext()));
  Value state = LLVM::UndefOp::create(builder, loc, structTy);
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  return LLVM::InsertValueOp::create(builder, loc, state, zero,
                                     ArrayRef<int64_t>{*getFieldIndex(AtomStateField::Soffset)});
}

Value CopyOpCDNA3BufferAtomicType::setAtomState(OpBuilder &builder, Location loc, Value atomStruct,
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

Attribute CopyOpCDNA3BufferAtomicType::getThrLayout() const { return FxLayout(FxC(1), FxC(1)); }

Attribute CopyOpCDNA3BufferAtomicType::getThrBitLayoutSrc() const {
  int32_t bits = getValType().getIntOrFloatBitWidth();
  return FxLayout(FxShape(FxC(1), FxC(bits)), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpCDNA3BufferAtomicType::getThrBitLayoutDst() const {
  int32_t bits = getValType().getIntOrFloatBitWidth();
  return FxLayout(FxShape(FxC(1), FxC(bits)), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpCDNA3BufferAtomicType::getThrBitLayoutRef() const {
  int32_t bits = getValType().getIntOrFloatBitWidth();
  return FxLayout(FxShape(FxC(1), FxC(bits)), FxStride(FxC(1), FxC(1)));
}

LogicalResult CopyOpCDNA3BufferAtomicType::emitAtomCall(OpBuilder &builder, Location loc,
                                                        Type copyAtomTyArg, Type srcMemTyArg,
                                                        Type dstMemTyArg, Value atomVal, Value src,
                                                        Value dst) const {
  auto srcMemTy = cast<fly::MemRefType>(srcMemTyArg);
  auto dstMemTy = cast<fly::MemRefType>(dstMemTyArg);

  AddressSpace srcAS = srcMemTy.getAddressSpace().getValue();
  AddressSpace dstAS = dstMemTy.getAddressSpace().getValue();

  if (srcAS != AddressSpace::Register || dstAS != AddressSpace::BufferDesc)
    return failure();

  Type valTy = getValType();
  Type scalarTy = valTy;
  if (auto vecTy = dyn_cast<VectorType>(valTy))
    scalarTy = vecTy.getElementType();
  bool isFloat = isa<FloatType>(scalarTy);

  Value loaded = LLVM::LoadOp::create(builder, loc, valTy, src);

  BufferFatPtr bp(dstMemTy.getPointerType(), dst);
  Value dstRsrc = bp.bufferRsrc(builder, loc);
  Value dstOff = bp.swizzleByteOffset(builder, loc);

  Value soffsetRaw = LLVM::ExtractValueOp::create(
      builder, loc, atomVal, ArrayRef<int64_t>{*getFieldIndex(AtomStateField::Soffset)});

  int64_t elemBits = dstMemTy.getElemTy().getIntOrFloatBitWidth();
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

  AtomicOp op = getAtomicOp().getValue();

  switch (op) {
  case AtomicOp::Add:
    if (!isFloat)
      return failure();
    ROCDL::RawPtrBufferAtomicFaddOp::create(builder, loc, loaded, dstRsrc, dstOff, soffset, zero,
                                            noAttrs, noAttrs, noAttrs);
    break;
  case AtomicOp::Max:
    if (isFloat)
      ROCDL::RawPtrBufferAtomicFmaxOp::create(builder, loc, loaded, dstRsrc, dstOff, soffset, zero,
                                              noAttrs, noAttrs, noAttrs);
    else
      ROCDL::RawPtrBufferAtomicSmaxOp::create(builder, loc, loaded, dstRsrc, dstOff, soffset, zero,
                                              noAttrs, noAttrs, noAttrs);
    break;
  case AtomicOp::Min:
    if (isFloat)
      return failure();
    ROCDL::RawPtrBufferAtomicUminOp::create(builder, loc, loaded, dstRsrc, dstOff, soffset, zero,
                                            noAttrs, noAttrs, noAttrs);
    break;
  default:
    return failure();
  }

  return success();
}

LogicalResult CopyOpCDNA3BufferAtomicType::emitAtomCall(OpBuilder &builder, Location loc,
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
