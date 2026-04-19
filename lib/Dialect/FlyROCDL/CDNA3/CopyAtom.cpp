// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/PointerUtils.h"
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

FailureOr<Value> CopyOpCDNA3BufferCopyType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                            Type resultTy, Type copyAtomTyArg,
                                                            Type srcTyArg, Type dstTyArg,
                                                            Value atomVal, Value src,
                                                            Value dst) const {
  IntegerType copyTy = builder.getIntegerType(getBitSize());

  Value soffsetRaw = LLVM::ExtractValueOp::create(
      builder, loc, atomVal, ArrayRef<int64_t>{*getFieldIndex(AtomStateField::Soffset)});

  auto computeSoffset = [&](int64_t elemBits) -> Value {
    if (elemBits == 8)
      return soffsetRaw;
    if (elemBits > 8 && elemBits % 8 == 0) {
      Value scale = arith::ConstantIntOp::create(builder, loc, elemBits / 8, 32);
      return arith::MulIOp::create(builder, loc, soffsetRaw, scale);
    }
    Value scale = arith::ConstantIntOp::create(builder, loc, elemBits, 32);
    Value bits = arith::MulIOp::create(builder, loc, soffsetRaw, scale);
    Value eight = arith::ConstantIntOp::create(builder, loc, 8, 32);
    return arith::DivUIOp::create(builder, loc, bits, eight);
  };

  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  ArrayAttr noAttrs;

  auto srcMemTy = srcTyArg ? dyn_cast<fly::MemRefType>(srcTyArg) : fly::MemRefType();
  auto dstMemTy = dstTyArg ? dyn_cast<fly::MemRefType>(dstTyArg) : fly::MemRefType();

  if (srcMemTy && srcMemTy.getAddressSpace().getValue() == AddressSpace::BufferDesc) {
    // buffer -> reg
    Value soffset = computeSoffset(srcMemTy.getElemTy().getIntOrFloatBitWidth());
    BufferFatPtr bp(srcMemTy.getPointerType(), src);
    Value srcRsrc = bp.bufferRsrc(builder, loc);
    Value srcOff = bp.swizzleByteOffset(builder, loc);

    Value loaded = ROCDL::RawPtrBufferLoadOp::create(builder, loc, copyTy, srcRsrc, srcOff, soffset,
                                                     zero, noAttrs, noAttrs, noAttrs);
    if (resultTy && loaded.getType() != resultTy)
      loaded = LLVM::BitcastOp::create(builder, loc, resultTy, loaded);
    return loaded;
  }

  if (dstMemTy && dstMemTy.getAddressSpace().getValue() == AddressSpace::BufferDesc) {
    // reg -> buffer
    Value soffset = computeSoffset(dstMemTy.getElemTy().getIntOrFloatBitWidth());
    BufferFatPtr bp(dstMemTy.getPointerType(), dst);
    Value dstRsrc = bp.bufferRsrc(builder, loc);
    Value dstOff = bp.swizzleByteOffset(builder, loc);

    Value stored = src;
    if (stored.getType() != copyTy)
      stored = LLVM::BitcastOp::create(builder, loc, copyTy, stored);
    ROCDL::RawPtrBufferStoreOp::create(builder, loc, stored, dstRsrc, dstOff, soffset, zero,
                                       noAttrs, noAttrs, noAttrs);
    return stored;
  }

  return failure();
}

FailureOr<Value> CopyOpCDNA3BufferCopyType::emitAtomCallSSA(
    OpBuilder &builder, Location loc, Type resultTy, Type copyAtomTyArg, Type srcTyArg,
    Type dstTyArg, Type predTyArg, Value atomVal, Value src, Value dst, Value pred) const {
  OpBuilder::InsertionGuard guard(builder);
  if (resultTy) {
    // buffer -> reg
    auto ifOp = scf::IfOp::create(builder, loc, resultTy, pred, /*withElseRegion=*/true);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    auto result = emitAtomCallSSA(builder, loc, resultTy, copyAtomTyArg, srcTyArg, dstTyArg,
                                  atomVal, src, dst);
    if (failed(result))
      return failure();
    scf::YieldOp::create(builder, loc, *result);
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    scf::YieldOp::create(builder, loc, dst);
    return ifOp.getResult(0);
  } else {
    // reg -> buffer
    auto ifOp = scf::IfOp::create(builder, loc, TypeRange{}, pred, /*withElse=*/false);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    auto result = emitAtomCallSSA(builder, loc, resultTy, copyAtomTyArg, srcTyArg, dstTyArg,
                                  atomVal, src, dst);
    if (failed(result))
      return failure();
    return Value();
  }
}

LogicalResult CopyOpCDNA3BufferCopyType::emitAtomCall(OpBuilder &builder, Location loc,
                                                      Type copyAtomTyArg, Type srcMemTyArg,
                                                      Type dstMemTyArg, Value atomVal, Value src,
                                                      Value dst) const {
  auto srcMemTy = cast<fly::MemRefType>(srcMemTyArg);
  auto dstMemTy = cast<fly::MemRefType>(dstMemTyArg);

  bool srcIsBuffer = (srcMemTy.getAddressSpace().getValue() == AddressSpace::BufferDesc);
  bool dstIsBuffer = (dstMemTy.getAddressSpace().getValue() == AddressSpace::BufferDesc);

  if (srcIsBuffer == dstIsBuffer)
    return failure();

  if (srcIsBuffer) {
    auto dstSSATy = fly::RegMem2SSAType(dstMemTy);
    auto res = emitAtomCallSSA(builder, loc, dstSSATy, copyAtomTyArg, srcMemTyArg, Type{}, atomVal,
                               src, Value{});
    if (failed(res))
      return failure();
    LLVM::StoreOp::create(builder, loc, *res, dst);
  } else {
    auto srcSSATy = fly::RegMem2SSAType(srcMemTy);
    Value srcVal = LLVM::LoadOp::create(builder, loc, srcSSATy, src);
    auto res = emitAtomCallSSA(builder, loc, Type{}, copyAtomTyArg, srcSSATy, dstMemTyArg, atomVal,
                               srcVal, dst);
    if (failed(res))
      return failure();
  }
  return success();
}

LogicalResult CopyOpCDNA3BufferCopyType::emitAtomCall(OpBuilder &builder, Location loc,
                                                      Type copyAtomTyArg, Type srcMemTyArg,
                                                      Type dstMemTyArg, Type predMemTyArg,
                                                      Value atomVal, Value src, Value dst,
                                                      Value pred) const {
  OpBuilder::InsertionGuard guard(builder);
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

FailureOr<Value> CopyOpCDNA3BufferCopyLDSType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                               Type resultTy, Type copyAtomTyArg,
                                                               Type srcTyArg, Type dstTyArg,
                                                               Value atomVal, Value src,
                                                               Value dst) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, atomVal, src, dst)))
    return failure();
  return Value{};
}

FailureOr<Value> CopyOpCDNA3BufferCopyLDSType::emitAtomCallSSA(
    OpBuilder &builder, Location loc, Type resultTy, Type copyAtomTyArg, Type srcTyArg,
    Type dstTyArg, Type predTyArg, Value atomVal, Value src, Value dst, Value pred) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, predTyArg, atomVal, src,
                          dst, pred)))
    return failure();
  return Value{};
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
  OpBuilder::InsertionGuard guard(builder);
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

FailureOr<Value> CopyOpCDNA3BufferAtomicType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                              Type resultTy, Type copyAtomTyArg,
                                                              Type srcTyArg, Type dstTyArg,
                                                              Value atomVal, Value src,
                                                              Value dst) const {
  auto dstMemTy = cast<fly::MemRefType>(dstTyArg);
  if (dstMemTy.getAddressSpace().getValue() != AddressSpace::BufferDesc)
    return failure();

  Type valTy = getValType();
  Type scalarTy = valTy;
  if (auto vecTy = dyn_cast<VectorType>(valTy))
    scalarTy = vecTy.getElementType();
  bool isFloat = isa<FloatType>(scalarTy);

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
    ROCDL::RawPtrBufferAtomicFaddOp::create(builder, loc, src, dstRsrc, dstOff, soffset, zero,
                                            noAttrs, noAttrs, noAttrs);
    break;
  case AtomicOp::Max:
    if (isFloat)
      ROCDL::RawPtrBufferAtomicFmaxOp::create(builder, loc, src, dstRsrc, dstOff, soffset, zero,
                                              noAttrs, noAttrs, noAttrs);
    else
      ROCDL::RawPtrBufferAtomicSmaxOp::create(builder, loc, src, dstRsrc, dstOff, soffset, zero,
                                              noAttrs, noAttrs, noAttrs);
    break;
  case AtomicOp::Min:
    if (isFloat)
      return failure();
    ROCDL::RawPtrBufferAtomicUminOp::create(builder, loc, src, dstRsrc, dstOff, soffset, zero,
                                            noAttrs, noAttrs, noAttrs);
    break;
  default:
    return failure();
  }

  return src;
}

FailureOr<Value> CopyOpCDNA3BufferAtomicType::emitAtomCallSSA(
    OpBuilder &builder, Location loc, Type resultTy, Type copyAtomTyArg, Type srcTyArg,
    Type dstTyArg, Type predTyArg, Value atomVal, Value src, Value dst, Value pred) const {
  OpBuilder::InsertionGuard guard(builder);
  if (resultTy) {
    auto ifOp = scf::IfOp::create(builder, loc, resultTy, pred, /*withElseRegion=*/true);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    auto result = emitAtomCallSSA(builder, loc, resultTy, copyAtomTyArg, srcTyArg, dstTyArg,
                                  atomVal, src, dst);
    if (failed(result))
      return failure();
    scf::YieldOp::create(builder, loc, *result);
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    scf::YieldOp::create(builder, loc, dst);
    return ifOp.getResult(0);
  }

  auto ifOp = scf::IfOp::create(builder, loc, TypeRange{}, pred, /*withElse=*/false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  auto result =
      emitAtomCallSSA(builder, loc, resultTy, copyAtomTyArg, srcTyArg, dstTyArg, atomVal, src, dst);
  if (failed(result))
    return failure();
  return Value();
}

LogicalResult CopyOpCDNA3BufferAtomicType::emitAtomCall(OpBuilder &builder, Location loc,
                                                        Type copyAtomTyArg, Type srcMemTyArg,
                                                        Type dstMemTyArg, Value atomVal, Value src,
                                                        Value dst) const {
  auto srcMemTy = cast<fly::MemRefType>(srcMemTyArg);
  auto srcSSATy = fly::RegMem2SSAType(srcMemTy);
  Value srcVal = LLVM::LoadOp::create(builder, loc, srcSSATy, src);
  auto res = emitAtomCallSSA(builder, loc, Type{}, copyAtomTyArg, srcSSATy, dstMemTyArg, atomVal,
                             srcVal, dst);
  if (failed(res))
    return failure();
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
