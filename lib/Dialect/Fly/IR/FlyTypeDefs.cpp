#include "mlir/IR/DialectImplementation.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

namespace mlir::fly {

bool BasisType::isStatic() const { return getAttr().isStatic(); }
bool IntTupleType::isStatic() const { return getAttr().isStatic(); }
bool LayoutType::isStatic() const { return getAttr().isStatic(); }
bool ComposedLayoutType::isStatic() const { return getAttr().isStatic(); }
bool CoordTensorType::isStatic() const { return getBase().isStatic() && getLayout().isStatic(); }

int32_t BasisType::depth() { return getAttr().depth(); }

bool IntTupleType::isLeaf() const { return getAttr().isLeaf(); }
int32_t IntTupleType::rank() const { return getAttr().rank(); }
int32_t IntTupleType::rank(int32_t idx) const { return getAttr().rank(idx); }
int32_t IntTupleType::rank(ArrayRef<int32_t> idxs) const { return getAttr().rank(idxs); }
int32_t IntTupleType::depth() const { return getAttr().depth(); }
int32_t IntTupleType::depth(int32_t idx) const { return getAttr().depth(idx); }
int32_t IntTupleType::depth(ArrayRef<int32_t> idxs) const { return getAttr().depth(idxs); }

bool LayoutType::isLeaf() const { return getAttr().isLeaf(); }
int32_t LayoutType::rank() const { return getAttr().rank(); }
int32_t LayoutType::rank(int32_t idx) const { return getAttr().rank(idx); }
int32_t LayoutType::rank(ArrayRef<int32_t> idxs) const { return getAttr().rank(idxs); }
int32_t LayoutType::depth() const { return getAttr().depth(); }
int32_t LayoutType::depth(int32_t idx) const { return getAttr().depth(idx); }
int32_t LayoutType::depth(ArrayRef<int32_t> idxs) const { return getAttr().depth(idxs); }
bool LayoutType::isStaticShape() const { return getAttr().isStaticShape(); }
bool LayoutType::isStaticStride() const { return getAttr().isStaticStride(); }

bool ComposedLayoutType::isLeaf() const { return getAttr().isLeaf(); }
int32_t ComposedLayoutType::rank() const { return getAttr().rank(); }
int32_t ComposedLayoutType::rank(int32_t idx) const { return getAttr().rank(idx); }
int32_t ComposedLayoutType::rank(ArrayRef<int32_t> idxs) const { return getAttr().rank(idxs); }
int32_t ComposedLayoutType::depth() const { return getAttr().depth(); }
int32_t ComposedLayoutType::depth(int32_t idx) const { return getAttr().depth(idx); }
int32_t ComposedLayoutType::depth(ArrayRef<int32_t> idxs) const { return getAttr().depth(idxs); }
bool ComposedLayoutType::isStaticOuter() const { return getAttr().isStaticOuter(); }
bool ComposedLayoutType::isStaticInner() const { return getAttr().isStaticInner(); }
bool ComposedLayoutType::isStaticOffset() const { return getAttr().isStaticOffset(); }

int32_t TileType::rank() const { return getAttr().rank(); }

bool CoordTensorType::isLeaf() const { return getLayout().isLeaf(); }
int32_t CoordTensorType::rank() const { return getLayout().rank(); }
int32_t CoordTensorType::rank(int32_t idx) const { return getLayout().rank(idx); }
int32_t CoordTensorType::rank(ArrayRef<int32_t> idxs) const { return getLayout().rank(idxs); }
int32_t CoordTensorType::depth() const { return getLayout().depth(); }
int32_t CoordTensorType::depth(int32_t idx) const { return getLayout().depth(idx); }
int32_t CoordTensorType::depth(ArrayRef<int32_t> idxs) const { return getLayout().depth(idxs); }

IntTupleType IntTupleType::at(int32_t idx) const {
  return IntTupleType::get(getContext(), getAttr().at(idx));
}
IntTupleType IntTupleType::at(ArrayRef<int32_t> idxs) const {
  return IntTupleType::get(getContext(), getAttr().at(idxs));
}
LayoutType LayoutType::at(int32_t idx) const {
  return LayoutType::get(getContext(), getAttr().at(idx));
}
LayoutType LayoutType::at(ArrayRef<int32_t> idxs) const {
  return LayoutType::get(getContext(), getAttr().at(idxs));
}
ComposedLayoutType ComposedLayoutType::at(int32_t idx) const {
  return ComposedLayoutType::get(getContext(), getAttr().at(idx));
}
ComposedLayoutType ComposedLayoutType::at(ArrayRef<int32_t> idxs) const {
  return ComposedLayoutType::get(getContext(), getAttr().at(idxs));
}

CoordTensorType CoordTensorType::at(int32_t idx) const {
  return CoordTensorType::get(getContext(), getBase().at(idx), getLayout().at(idx));
}
CoordTensorType CoordTensorType::at(ArrayRef<int32_t> idxs) const {
  return CoordTensorType::get(getContext(), getBase().at(idxs), getLayout().at(idxs));
}

#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"

bool CopyAtomUniversalCopyType::isStatic() const { return true; }

Attribute CopyAtomUniversalCopyType::getThrLayout() const {
  return FxLayout(FxC(1), FxC(1));
}

// TODO: Back to getBitSize() when recast_layout is ready.
Attribute CopyAtomUniversalCopyType::getThrValLayoutSrc() const {
  return FxLayout(FxShape(FxC(1), FxC(1)), FxStride(FxC(1), FxC(1)));
}
Attribute CopyAtomUniversalCopyType::getThrValLayoutDst() const {
  return FxLayout(FxShape(FxC(1), FxC(1)), FxStride(FxC(1), FxC(1)));
}
Attribute CopyAtomUniversalCopyType::getThrValLayoutRef() const {
  return FxLayout(FxShape(FxC(1), FxC(1)), FxStride(FxC(1), FxC(1)));
}

bool MmaAtomUniversalFMAType::isStatic() const { return true; }

Attribute MmaAtomUniversalFMAType::getShapeMNK() const {
  return IntTupleAttr::get(ArrayAttr::get(getContext(), {FxC(1), FxC(1), FxC(1)}));
}

Attribute MmaAtomUniversalFMAType::getThrLayout() const {
  return FxLayout(FxC(1), FxC(1));
}

Attribute MmaAtomUniversalFMAType::getThrValLayoutA() const {
  return FxLayout(FxShape(FxC(1), FxC(1)), FxStride(FxC(1), FxC(1)));
}
Attribute MmaAtomUniversalFMAType::getThrValLayoutB() const {
  return FxLayout(FxShape(FxC(1), FxC(1)), FxStride(FxC(1), FxC(1)));
}
Attribute MmaAtomUniversalFMAType::getThrValLayoutC() const {
  return FxLayout(FxShape(FxC(1), FxC(1)), FxStride(FxC(1), FxC(1)));
}

Type MmaAtomUniversalFMAType::parse(AsmParser &parser) {
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

void MmaAtomUniversalFMAType::print(AsmPrinter &printer) const {
  printer << "<";
  printMNKDimensionList(printer, 1, 1, 1);
  printer << ", (" << getElemTy() << ", " << getElemTy() << ") -> " << getElemTy() << ">";
}

} // namespace mlir::fly
