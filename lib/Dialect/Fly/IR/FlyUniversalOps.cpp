// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/PointerUtils.h"
#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"

namespace mlir::fly {

bool CopyOpUniversalCopyType::isStatic() const { return true; }

Value CopyOpUniversalCopyType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                                  Value currentValue) const {
  if (currentValue && isa<MakeCopyAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  return MakeCopyAtomOp::create(builder, loc, CopyAtomType::get(*this, getBitSize()), getBitSize());
}

Attribute CopyOpUniversalCopyType::getThrLayout() const { return FxLayout(FxC(1), FxC(1)); }

Attribute CopyOpUniversalCopyType::getThrBitLayoutSrc() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpUniversalCopyType::getThrBitLayoutDst() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpUniversalCopyType::getThrBitLayoutRef() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}

bool CopyOpUniversalAtomicType::isStatic() const { return true; }

Value CopyOpUniversalAtomicType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                                    Value currentValue) const {
  if (currentValue && isa<MakeCopyAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  int32_t bits = getValType().getIntOrFloatBitWidth();
  return MakeCopyAtomOp::create(builder, loc, CopyAtomType::get(*this, bits), bits);
}

Attribute CopyOpUniversalAtomicType::getThrLayout() const { return FxLayout(FxC(1), FxC(1)); }

Attribute CopyOpUniversalAtomicType::getThrBitLayoutSrc() const {
  int32_t bits = getValType().getIntOrFloatBitWidth();
  return FxLayout(FxShape(FxC(1), FxC(bits)), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpUniversalAtomicType::getThrBitLayoutDst() const {
  int32_t bits = getValType().getIntOrFloatBitWidth();
  return FxLayout(FxShape(FxC(1), FxC(bits)), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpUniversalAtomicType::getThrBitLayoutRef() const {
  int32_t bits = getValType().getIntOrFloatBitWidth();
  return FxLayout(FxShape(FxC(1), FxC(bits)), FxStride(FxC(1), FxC(1)));
}

bool MmaOpUniversalFMAType::isStatic() const { return true; }

Value MmaOpUniversalFMAType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                                Value currentValue) const {
  if (currentValue && isa<MakeMmaAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  return MakeMmaAtomOp::create(builder, loc, MmaAtomType::get(*this));
}

Attribute MmaOpUniversalFMAType::getShapeMNK() const {
  return IntTupleAttr::get(ArrayAttr::get(getContext(), {FxC(1), FxC(1), FxC(1)}));
}

Attribute MmaOpUniversalFMAType::getThrLayout() const { return FxLayout(FxC(1), FxC(1)); }

Type MmaOpUniversalFMAType::getValTypeA() const { return getElemTy(); }
Type MmaOpUniversalFMAType::getValTypeB() const { return getElemTy(); }
Type MmaOpUniversalFMAType::getValTypeC() const { return getElemTy(); }
Type MmaOpUniversalFMAType::getValTypeD() const { return getElemTy(); }

Attribute MmaOpUniversalFMAType::getThrValLayoutA() const {
  return FxLayout(FxShape(FxC(1), FxC(1)), FxStride(FxC(1), FxC(1)));
}
Attribute MmaOpUniversalFMAType::getThrValLayoutB() const {
  return FxLayout(FxShape(FxC(1), FxC(1)), FxStride(FxC(1), FxC(1)));
}
Attribute MmaOpUniversalFMAType::getThrValLayoutC() const {
  return FxLayout(FxShape(FxC(1), FxC(1)), FxStride(FxC(1), FxC(1)));
}

Type MmaOpUniversalFMAType::parse(AsmParser &parser) {
  Type elemTyA, elemTyB, elemTyC;
  if (parser.parseLess())
    return {};
  int32_t m, n, k;
  if (parseMNKDimensionList(parser, m, n, k))
    return {};
  if (m != 1 || n != 1 || k != 1) {
    parser.emitError(parser.getCurrentLocation())
        << "expected 1x1x1 dimensions for universal FMA, got " << m << "x" << n << "x" << k;
    return {};
  }
  // Parse ", (elemTy, elemTy) -> elemTy>"
  if (parser.parseComma() || parser.parseLParen() || parser.parseType(elemTyA) ||
      parser.parseComma() || parser.parseType(elemTyB) || parser.parseRParen() ||
      parser.parseArrow() || parser.parseType(elemTyC) || parser.parseGreater())
    return {};
  // For universal FMA, all element types should be the same
  if (elemTyA != elemTyB || elemTyB != elemTyC) {
    parser.emitError(parser.getCurrentLocation())
        << "expected all element types to be the same for universal FMA";
    return {};
  }
  return get(parser.getContext(), elemTyA);
}

void MmaOpUniversalFMAType::print(AsmPrinter &printer) const {
  printer << "<";
  printMNKDimensionList(printer, 1, 1, 1);
  printer << ", (" << getElemTy() << ", " << getElemTy() << ") -> " << getElemTy() << ">";
}

LogicalResult CopyOpUniversalCopyType::emitAtomCall(OpBuilder &builder, Location loc,
                                                    Type copyAtomTyArg, Type srcMemTyArg,
                                                    Type dstMemTyArg, Value atomVal, Value src,
                                                    Value dst) const {
  auto srcMemTy = cast<fly::MemRefType>(srcMemTyArg);
  auto dstMemTy = cast<fly::MemRefType>(dstMemTyArg);

  if (!isa<LLVM::LLVMPointerType>(src.getType()) || !isa<LLVM::LLVMPointerType>(dst.getType()))
    return failure();

  Value srcPtr = applySwizzleOnPtr(builder, loc, cast<TypedValue<LLVM::LLVMPointerType>>(src),
                                   srcMemTy.getSwizzle());
  Value dstPtr = applySwizzleOnPtr(builder, loc, cast<TypedValue<LLVM::LLVMPointerType>>(dst),
                                   dstMemTy.getSwizzle());

  // Use element-type-aware vector chunks so that store types match MFMA
  // load types.  SROA can then forward register-alloca stores to loads,
  // eliminating allocas that AMDGPUPromoteAllocaToVector would otherwise
  // give full-function lifetime.
  //
  // f16 MFMA reads <4 x half>  (64 b) → store <4 x half>
  // f32 MFMA reads f32          (32 b) → store f32
  // bf16 MFMA reads <4 x i16>  (64 b) → store <4 x i16>
  // f8/i8 MFMA reads i64       (64 b) → store i64
  unsigned totalBits = getBitSize();
  Type chunkTy;
  unsigned chunkBits;

  Type elemTy = dstMemTy.getElemTy();
  if (elemTy && elemTy.isF16()) {
    chunkTy = VectorType::get({4}, elemTy);
    chunkBits = 64;
  } else if (elemTy && elemTy.isF32()) {
    chunkTy = elemTy;
    chunkBits = 32;
  } else if (elemTy && elemTy.isBF16()) {
    auto i16Ty = IntegerType::get(builder.getContext(), 16);
    chunkTy = VectorType::get({4}, i16Ty);
    chunkBits = 64;
  } else {
    // f8/i8/other: use i64 integer chunks (matches MFMA i64 operand)
    chunkBits = std::min(totalBits, 64u);
    chunkTy = IntegerType::get(builder.getContext(), chunkBits);
  }

  unsigned numChunks = totalBits / chunkBits;
  unsigned chunkBytes = chunkBits / 8;
  auto i8Ty = IntegerType::get(builder.getContext(), 8);

  for (unsigned i = 0; i < numChunks; ++i) {
    Value sPtr = srcPtr, dPtr = dstPtr;
    if (i > 0) {
      auto offset = LLVM::ConstantOp::create(builder, loc,
          IntegerType::get(builder.getContext(), 32), i * chunkBytes);
      sPtr = LLVM::GEPOp::create(builder, loc, srcPtr.getType(),
                                  i8Ty, srcPtr, ValueRange{offset});
      dPtr = LLVM::GEPOp::create(builder, loc, dstPtr.getType(),
                                  i8Ty, dstPtr, ValueRange{offset});
    }
    Value val = LLVM::LoadOp::create(builder, loc, chunkTy, sPtr);
    LLVM::StoreOp::create(builder, loc, val, dPtr);
  }

  return success();
}

LogicalResult CopyOpUniversalCopyType::emitAtomCall(OpBuilder &builder, Location loc,
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

static std::optional<LLVM::AtomicBinOp> convertAtomicOp(AtomicOp binOp, bool isFloat) {
  switch (binOp) {
  case AtomicOp::Add:
    return isFloat ? LLVM::AtomicBinOp::fadd : LLVM::AtomicBinOp::add;
  case AtomicOp::Max:
    return isFloat ? LLVM::AtomicBinOp::fmax : LLVM::AtomicBinOp::max;
  case AtomicOp::Min:
    return isFloat ? LLVM::AtomicBinOp::fmin : LLVM::AtomicBinOp::min;
  case AtomicOp::And:
    return isFloat ? std::nullopt : std::optional(LLVM::AtomicBinOp::_and);
  case AtomicOp::Or:
    return isFloat ? std::nullopt : std::optional(LLVM::AtomicBinOp::_or);
  case AtomicOp::Inc:
    return isFloat ? std::nullopt : std::optional(LLVM::AtomicBinOp::uinc_wrap);
  case AtomicOp::Dec:
    return isFloat ? std::nullopt : std::optional(LLVM::AtomicBinOp::udec_wrap);
  default:
    return std::nullopt;
  }
}

LogicalResult CopyOpUniversalAtomicType::emitAtomCall(OpBuilder &builder, Location loc,
                                                      Type copyAtomTyArg, Type srcMemTyArg,
                                                      Type dstMemTyArg, Value atomVal, Value src,
                                                      Value dst) const {
  if (!isa<LLVM::LLVMPointerType>(src.getType()) || !isa<LLVM::LLVMPointerType>(dst.getType()))
    return failure();

  auto srcMemTy = cast<fly::MemRefType>(srcMemTyArg);
  auto dstMemTy = cast<fly::MemRefType>(dstMemTyArg);

  if (srcMemTy.getAddressSpace().getValue() != AddressSpace::Register)
    return failure();

  Type elemTy = getValType();
  bool isFloat = isa<FloatType>(elemTy);

  Value dstPtr = applySwizzleOnPtr(builder, loc, cast<TypedValue<LLVM::LLVMPointerType>>(dst),
                                   dstMemTy.getSwizzle());

  Value loaded = LLVM::LoadOp::create(builder, loc, elemTy, src);

  auto binOp = convertAtomicOp(getAtomicOp().getValue(), isFloat);
  if (!binOp)
    return failure();
  LLVM::AtomicRMWOp::create(builder, loc, *binOp, dstPtr, loaded, LLVM::AtomicOrdering::monotonic);
  return success();
}

LogicalResult CopyOpUniversalAtomicType::emitAtomCall(OpBuilder &builder, Location loc,
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

LogicalResult MmaOpUniversalFMAType::emitAtomCall(OpBuilder &builder, Location loc, Type mmaAtomTy,
                                                  Type dMemTy, Type aMemTy, Type bMemTy,
                                                  Type cMemTy, Value atomVal, Value dPtr,
                                                  Value aPtr, Value bPtr, Value cPtr) const {
  Type elemTy = getElemTy();

  Value a = LLVM::LoadOp::create(builder, loc, elemTy, aPtr);
  Value b = LLVM::LoadOp::create(builder, loc, elemTy, bPtr);
  Value c = LLVM::LoadOp::create(builder, loc, elemTy, cPtr);

  Value mul = LLVM::FMulOp::create(builder, loc, elemTy, a, b);
  Value res = LLVM::FAddOp::create(builder, loc, elemTy, mul, c);

  LLVM::StoreOp::create(builder, loc, res, dPtr);
  return success();
}

LogicalResult MmaOpUniversalFMAType::emitAtomCallSsa(OpBuilder &builder, Location loc,
                                                     Type mmaAtomTy, Type aMemTy, Type bMemTy,
                                                     Value atomVal, Value aPtr, Value bPtr,
                                                     Value cVal, Value &result) const {
  Type elemTy = getElemTy();

  Value a = LLVM::LoadOp::create(builder, loc, elemTy, aPtr);
  Value b = LLVM::LoadOp::create(builder, loc, elemTy, bPtr);

  Value mul = LLVM::FMulOp::create(builder, loc, elemTy, a, b);
  result = LLVM::FAddOp::create(builder, loc, elemTy, mul, cVal);
  return success();
}

} // namespace mlir::fly
