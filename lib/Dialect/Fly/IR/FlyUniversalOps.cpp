// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectImplementation.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
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

} // namespace mlir::fly
