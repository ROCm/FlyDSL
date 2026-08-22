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

bool CopyOpCDNA4LdsReadTransposeType::isStatic() const { return true; }

Value CopyOpCDNA4LdsReadTransposeType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                                          Value currentValue) const {
  if (currentValue && isa<MakeCopyAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  return MakeCopyAtomOp::create(builder, loc, CopyAtomType::get(*this, getBitSize()), getBitSize());
}

Attribute CopyOpCDNA4LdsReadTransposeType::getThrLayout() const {
  return FxLayout(FxC(16), FxC(1));
}

Attribute CopyOpCDNA4LdsReadTransposeType::getThrBitLayoutSrc() const {
  int32_t bitSize = getBitSize();
  int32_t transGranularity = getTransGranularity();
  if (bitSize == 64 && transGranularity == 4) {
    return FxLayout(FxShape(FxC(16), FxC(64)), FxStride(FxC(64), FxC(1)));
  } else if (bitSize == 64 && transGranularity == 8) {
    return FxLayout(FxShape(FxC(16), FxC(64)), FxStride(FxC(64), FxC(1)));
  } else if (bitSize == 96 && transGranularity == 6) {
    return FxLayout(FxShape(FxC(16), FxC(96)), FxStride(FxC(96), FxC(1)));
  } else if (bitSize == 64 && transGranularity == 16) {
    return FxLayout(FxShape(FxC(16), FxC(64)), FxStride(FxC(64), FxC(1)));
  } else {
    llvm_unreachable("Invalid (bitSize, transGranularity) for LDS read transpose");
  }
}

Attribute CopyOpCDNA4LdsReadTransposeType::getThrBitLayoutDst() const {
  int32_t bitSize = getBitSize();
  int32_t transGranularity = getTransGranularity();
  if (bitSize == 64 && transGranularity == 4) {
    return FxLayout(FxShape(FxC(16), FxVal(4, 16)), FxStride(FxC(4), FxVal(1, 64)));
  } else if (bitSize == 64 && transGranularity == 8) {
    return FxLayout(FxShape(FxC(16), FxVal(8, 8)), FxStride(FxC(8), FxVal(1, 128)));
  } else if (bitSize == 96 && transGranularity == 6) {
    return FxLayout(FxShape(FxC(16), FxVal(6, 16)), FxStride(FxC(6), FxVal(1, 96)));
  } else if (bitSize == 64 && transGranularity == 16) {
    return FxLayout(FxShape(FxC(16), FxVal(16, 4)), FxStride(FxC(16), FxVal(1, 256)));
  } else {
    llvm_unreachable("Invalid (bitSize, transGranularity) for LDS read transpose");
  }
}

Attribute CopyOpCDNA4LdsReadTransposeType::getThrBitLayoutRef() const {
  return getThrBitLayoutDst();
}

FailureOr<Value> CopyOpCDNA4LdsReadTransposeType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                                  Type resultTy, Type copyAtomTyArg,
                                                                  Type srcTyArg, Type dstTyArg,
                                                                  Value atomVal, Value src,
                                                                  Value dst) const {
  int32_t bitSize = getBitSize();
  int32_t transGranularity = getTransGranularity();

  Value loaded;
  if (bitSize == 64 && transGranularity == 4) {
    auto intrTy = VectorType::get({2}, builder.getI32Type());
    loaded = ROCDL::ds_read_tr4_b64::create(builder, loc, intrTy, src);
  } else if (bitSize == 64 && transGranularity == 8) {
    auto intrTy = VectorType::get({2}, builder.getI32Type());
    loaded = ROCDL::ds_read_tr8_b64::create(builder, loc, intrTy, src);
  } else if (bitSize == 96 && transGranularity == 6) {
    auto intrTy = VectorType::get({3}, builder.getI32Type());
    loaded = ROCDL::ds_read_tr6_b96::create(builder, loc, intrTy, src);
  } else if (bitSize == 64 && transGranularity == 16) {
    auto intrTy = VectorType::get({4}, builder.getI16Type());
    loaded = ROCDL::ds_read_tr16_b64::create(builder, loc, intrTy, src);
  } else {
    return failure();
  }

  if (resultTy && loaded.getType() != resultTy)
    loaded = LLVM::BitcastOp::create(builder, loc, resultTy, loaded);

  return loaded;
}

FailureOr<Value> CopyOpCDNA4LdsReadTransposeType::emitAtomCallSSA(
    OpBuilder &builder, Location loc, Type resultTy, Type copyAtomTyArg, Type srcTyArg,
    Type dstTyArg, Type predTyArg, Value atomVal, Value src, Value dst, Value pred) const {
  assert(resultTy && "resultTy must be SSA Type");
  OpBuilder::InsertionGuard guard(builder);
  auto ifOp = scf::IfOp::create(builder, loc, resultTy, pred, /*withElseRegion=*/true);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  auto result =
      emitAtomCallSSA(builder, loc, resultTy, copyAtomTyArg, srcTyArg, dstTyArg, atomVal, src, dst);
  if (failed(result))
    return failure();
  scf::YieldOp::create(builder, loc, *result);

  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  scf::YieldOp::create(builder, loc, dst);
  return ifOp.getResult(0);
}

LogicalResult CopyOpCDNA4LdsReadTransposeType::emitAtomCall(OpBuilder &builder, Location loc,
                                                            Type copyAtomTyArg, Type srcMemTyArg,
                                                            Type dstMemTyArg, Value atomVal,
                                                            Value src, Value dst) const {
  auto dstSSATy = fly::RegMem2SSAType(cast<fly::MemRefType>(dstMemTyArg), true);
  auto res = emitAtomCallSSA(builder, loc, dstSSATy, copyAtomTyArg, srcMemTyArg, Type{}, atomVal,
                             src, Value{});
  if (failed(res))
    return failure();
  LLVM::StoreOp::create(builder, loc, *res, dst);
  return success();
}

LogicalResult CopyOpCDNA4LdsReadTransposeType::emitAtomCall(OpBuilder &builder, Location loc,
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

LogicalResult CopyOpCDNA4LdsReadTransposeType::verify(function_ref<InFlightDiagnostic()> emitError,
                                                      int32_t transGranularity, int32_t bitSize) {
  bool valid =
      (bitSize == 64 && transGranularity == 4) || (bitSize == 64 && transGranularity == 8) ||
      (bitSize == 96 && transGranularity == 6) || (bitSize == 64 && transGranularity == 16);
  if (!valid)
    return emitError() << "unsupported (bitSize, transGranularity) = (" << bitSize << ", "
                       << transGranularity << ") for LDS read transpose";
  return success();
}

// --- CopyOpCDNA4BufferLoadAsyncLDS ---
//
// Buffer resource -> LDS, same shape as CopyOpCDNA3BufferCopyLDS but emitting the
// async intrinsic: the backend stops auto-inserting the vmcnt wait before the LDS
// data is consumed, so the kernel must bracket these calls with rocdl.asyncmark /
// rocdl.wait.asyncmark.

LogicalResult
CopyOpCDNA4BufferLoadAsyncLDSType::verify(function_ref<InFlightDiagnostic()> emitError,
                                          int32_t bitSize) {
  // LDS DMA transfer widths are 1/2/4 bytes everywhere, plus 12/16 bytes on
  // gfx950 (hasLDSLoadB96_B128), which CDNA4 is. There is no 8-byte form.
  if (bitSize != 32 && bitSize != 96 && bitSize != 128)
    return emitError() << "unsupported bitSize = " << bitSize
                       << " for BufferLoadAsyncLDS; expected 32, 96 or 128 (there is no 8-byte "
                          "LDS DMA instruction)";
  return success();
}

std::optional<unsigned> CopyOpCDNA4BufferLoadAsyncLDSType::getFieldIndex(AtomStateField field) {
  switch (field) {
  case AtomStateField::Soffset:
    return 0;
  case AtomStateField::ImmOffset:
    return 1;
  default:
    return std::nullopt;
  }
}

Type CopyOpCDNA4BufferLoadAsyncLDSType::getConvertedType(MLIRContext *ctx) const {
  auto i32Ty = IntegerType::get(ctx, 32);
  return LLVM::LLVMStructType::getLiteral(ctx, {i32Ty, i32Ty});
}

Value CopyOpCDNA4BufferLoadAsyncLDSType::getDefaultState(OpBuilder &builder, Location loc) const {
  auto structTy = cast<LLVM::LLVMStructType>(getConvertedType(builder.getContext()));
  Value state = LLVM::UndefOp::create(builder, loc, structTy);
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  state = LLVM::InsertValueOp::create(builder, loc, state, zero,
                                      ArrayRef<int64_t>{*getFieldIndex(AtomStateField::Soffset)});
  state = LLVM::InsertValueOp::create(builder, loc, state, zero,
                                      ArrayRef<int64_t>{*getFieldIndex(AtomStateField::ImmOffset)});
  return state;
}

Value CopyOpCDNA4BufferLoadAsyncLDSType::setAtomState(OpBuilder &builder, Location loc,
                                                      Value atomStruct, Attribute fieldAttr,
                                                      Value fieldValue) const {
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

Attribute CopyOpCDNA4BufferLoadAsyncLDSType::getThrLayout() const {
  return FxLayout(FxC(1), FxC(1));
}

Attribute CopyOpCDNA4BufferLoadAsyncLDSType::getThrBitLayoutSrc() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpCDNA4BufferLoadAsyncLDSType::getThrBitLayoutDst() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpCDNA4BufferLoadAsyncLDSType::getThrBitLayoutRef() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}

FailureOr<Value>
CopyOpCDNA4BufferLoadAsyncLDSType::emitAtomCallSSA(OpBuilder &builder, Location loc, Type resultTy,
                                                   Type copyAtomTyArg, Type srcTyArg, Type dstTyArg,
                                                   Value atomVal, Value src, Value dst) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, atomVal, src, dst)))
    return failure();
  return Value{};
}

FailureOr<Value> CopyOpCDNA4BufferLoadAsyncLDSType::emitAtomCallSSA(
    OpBuilder &builder, Location loc, Type resultTy, Type copyAtomTyArg, Type srcTyArg,
    Type dstTyArg, Type predTyArg, Value atomVal, Value src, Value dst, Value pred) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, predTyArg, atomVal, src,
                          dst, pred)))
    return failure();
  return Value{};
}

LogicalResult CopyOpCDNA4BufferLoadAsyncLDSType::emitAtomCall(OpBuilder &builder, Location loc,
                                                              Type copyAtomTyArg, Type srcMemTyArg,
                                                              Type dstMemTyArg, Value atomVal,
                                                              Value src, Value dst) const {
  auto srcMemTy = cast<fly::MemRefType>(srcMemTyArg);
  auto dstMemTy = cast<fly::MemRefType>(dstMemTyArg);

  if (!isTargetAddressSpace<BufferDescAddressAttr>(srcMemTy.getAddressSpace()) ||
      !isGenericAddressSpace<fly::AddressSpace::Shared>(dstMemTy.getAddressSpace()))
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

  Value size = arith::ConstantIntOp::create(builder, loc, sizeBytes, 32);

  BufferFatPtr bp(srcMemTy.getPointerType(), src);
  Value srcRsrc = bp.bufferRsrc(builder, loc);
  Value srcOff = bp.swizzleByteOffset(builder, loc);

  ArrayAttr noAttrs;
  auto auxAttr = builder.getI32IntegerAttr(0);
  ROCDL::RawPtrBufferLoadAsyncLdsOp::create(builder, loc, srcRsrc, dst, size, srcOff, soffset,
                                            immOffset, auxAttr, noAttrs, noAttrs, noAttrs);
  return success();
}

LogicalResult CopyOpCDNA4BufferLoadAsyncLDSType::emitAtomCall(OpBuilder &builder, Location loc,
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

// --- CopyOpCDNA4GlobalLoadAsyncLDS ---
//
// Global pointer -> LDS. `rocdl.global.load.async.lds` takes size and offset as
// compile-time attributes rather than SSA operands, so unlike the buffer variant
// there is no runtime soffset / imm_offset state: the address lives entirely in
// the global memref's pointer and `offset` stays 0.

LogicalResult
CopyOpCDNA4GlobalLoadAsyncLDSType::verify(function_ref<InFlightDiagnostic()> emitError,
                                          int32_t bitSize) {
  // Same transfer widths as the buffer variant: 1/2/4 bytes everywhere, 12/16
  // bytes on gfx950. No 8-byte form.
  if (bitSize != 32 && bitSize != 96 && bitSize != 128)
    return emitError() << "unsupported bitSize = " << bitSize
                       << " for GlobalLoadAsyncLDS; expected 32, 96 or 128 (there is no 8-byte "
                          "LDS DMA instruction)";
  return success();
}

bool CopyOpCDNA4GlobalLoadAsyncLDSType::isStatic() const { return true; }

Value CopyOpCDNA4GlobalLoadAsyncLDSType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                                            Value currentValue) const {
  if (currentValue && isa<MakeCopyAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  return MakeCopyAtomOp::create(builder, loc, CopyAtomType::get(*this, getBitSize()), getBitSize());
}

Attribute CopyOpCDNA4GlobalLoadAsyncLDSType::getThrLayout() const {
  return FxLayout(FxC(1), FxC(1));
}

Attribute CopyOpCDNA4GlobalLoadAsyncLDSType::getThrBitLayoutSrc() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpCDNA4GlobalLoadAsyncLDSType::getThrBitLayoutDst() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpCDNA4GlobalLoadAsyncLDSType::getThrBitLayoutRef() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}

FailureOr<Value>
CopyOpCDNA4GlobalLoadAsyncLDSType::emitAtomCallSSA(OpBuilder &builder, Location loc, Type resultTy,
                                                   Type copyAtomTyArg, Type srcTyArg, Type dstTyArg,
                                                   Value atomVal, Value src, Value dst) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, atomVal, src, dst)))
    return failure();
  return Value{};
}

FailureOr<Value> CopyOpCDNA4GlobalLoadAsyncLDSType::emitAtomCallSSA(
    OpBuilder &builder, Location loc, Type resultTy, Type copyAtomTyArg, Type srcTyArg,
    Type dstTyArg, Type predTyArg, Value atomVal, Value src, Value dst, Value pred) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, predTyArg, atomVal, src,
                          dst, pred)))
    return failure();
  return Value{};
}

LogicalResult CopyOpCDNA4GlobalLoadAsyncLDSType::emitAtomCall(OpBuilder &builder, Location loc,
                                                              Type copyAtomTyArg, Type srcMemTyArg,
                                                              Type dstMemTyArg, Value atomVal,
                                                              Value src, Value dst) const {
  auto srcMemTy = cast<fly::MemRefType>(srcMemTyArg);
  auto dstMemTy = cast<fly::MemRefType>(dstMemTyArg);

  if (!isGenericAddressSpace<fly::AddressSpace::Global>(srcMemTy.getAddressSpace()) ||
      !isGenericAddressSpace<fly::AddressSpace::Shared>(dstMemTy.getAddressSpace()))
    return failure();

  ArrayAttr noAttrs;
  auto sizeAttr = builder.getI32IntegerAttr(getBitSize() / 8);
  auto offsetAttr = builder.getI32IntegerAttr(0);
  auto auxAttr = builder.getI32IntegerAttr(0);
  ROCDL::GlobalLoadAsyncLDSOp::create(builder, loc, src, dst, sizeAttr, offsetAttr, auxAttr,
                                      noAttrs, noAttrs, noAttrs);
  return success();
}

LogicalResult CopyOpCDNA4GlobalLoadAsyncLDSType::emitAtomCall(OpBuilder &builder, Location loc,
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

} // namespace mlir::fly_rocdl
