// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

  int32_t copyBytes = getBitSize() / 8;
  Value srcPtr = applySwizzleOnPtr(builder, loc, cast<TypedValue<LLVM::LLVMPointerType>>(src),
                                   srcMemTy.getSwizzle());
  Value dstPtr = applySwizzleOnPtr(builder, loc, cast<TypedValue<LLVM::LLVMPointerType>>(dst),
                                   dstMemTy.getSwizzle());
  Value len = arith::ConstantIntOp::create(builder, loc, copyBytes, /*width=*/32);
  LLVM::MemcpyOp::create(builder, loc, dstPtr, srcPtr, len, /*isVolatile=*/false);

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

} // namespace mlir::fly
