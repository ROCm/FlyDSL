// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/IR/DialectImplementation.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/LayoutUtils.h"
#include "flydsl/Dialect/Fly/Utils/NormalForm.h"

namespace mlir::fly {

bool BasisType::isStatic() const { return getAttr().isStatic(); }
bool IntTupleType::isStatic() const { return getAttr().isStatic(); }
bool SwizzleType::isStatic() const { return true; }
bool LayoutType::isStatic() const { return getAttr().isStatic(); }
bool ComposedLayoutType::isStatic() const { return getAttr().isStatic(); }
bool TileType::isStatic() const { return true; }
bool CoordTensorType::isStatic() const {
  return getBase().isStatic() && cast<MayStaticAttrInterface>(getLayout()).isStatic();
}

bool TiledCopyType::isStatic() const {
  return cast<MayStaticTypeInterface>(getCopyAtom()).isStatic();
}
bool TiledMmaType::isStatic() const {
  return cast<MayStaticTypeInterface>(getMmaAtom()).isStatic();
}

Value BasisType::rebuildStaticValue(OpBuilder &, Location, Value) const { return nullptr; }

Value IntTupleType::rebuildStaticValue(OpBuilder &builder, Location loc, Value currentValue) const {
  if (currentValue && isNormalForm(cast<TypedValue<IntTupleType>>(currentValue)))
    return nullptr;
  return MakeIntTupleOp::create(builder, loc, *this, ValueRange{});
}

Value LayoutType::rebuildStaticValue(OpBuilder &builder, Location loc, Value currentValue) const {
  if (currentValue && isNormalForm(cast<TypedValue<LayoutType>>(currentValue)))
    return nullptr;
  Value shape = IntTupleType::get(getAttr().getShape()).rebuildStaticValue(builder, loc, nullptr);
  Value stride = IntTupleType::get(getAttr().getStride()).rebuildStaticValue(builder, loc, nullptr);
  return MakeLayoutOp::create(builder, loc, *this, shape, stride);
}

Value SwizzleType::rebuildStaticValue(OpBuilder &builder, Location loc, Value currentValue) const {
  if (currentValue)
    return nullptr;
  return StaticOp::create(builder, loc, *this);
}

Value ComposedLayoutType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                             Value currentValue) const {
  if (currentValue && isNormalForm(cast<TypedValue<ComposedLayoutType>>(currentValue)))
    return nullptr;
  ComposedLayoutAttr attr = getAttr();
  Attribute innerAttr = attr.getInner();
  Value inner;
  if (auto layoutAttr = dyn_cast<LayoutAttr>(innerAttr))
    inner = LayoutType::get(layoutAttr).rebuildStaticValue(builder, loc, nullptr);
  else if (auto composedAttr = dyn_cast<ComposedLayoutAttr>(innerAttr))
    inner = ComposedLayoutType::get(composedAttr).rebuildStaticValue(builder, loc, nullptr);
  else if (auto swizzleAttr = dyn_cast<SwizzleAttr>(innerAttr))
    inner = SwizzleType::get(swizzleAttr).rebuildStaticValue(builder, loc, nullptr);
  if (!inner)
    return nullptr;
  Value offset = IntTupleType::get(attr.getOffset()).rebuildStaticValue(builder, loc, nullptr);
  Value outer = LayoutType::get(attr.getOuter()).rebuildStaticValue(builder, loc, nullptr);
  return MakeComposedLayoutOp::create(builder, loc, *this, inner, offset, outer);
}

Value TileType::rebuildStaticValue(OpBuilder &builder, Location loc, Value currentValue) const {
  if (currentValue)
    return nullptr;
  return StaticOp::create(builder, loc, *this);
}

Value CoordTensorType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                          Value currentValue) const {
  if (currentValue && isWeaklyNormalForm(cast<TypedValue<CoordTensorType>>(currentValue)))
    return nullptr;
  Value base =
      IntTupleType::get(cast<IntTupleAttr>(getBase())).rebuildStaticValue(builder, loc, nullptr);
  Value layout;
  if (auto layoutAttr = dyn_cast<LayoutAttr>(getLayout()))
    layout = LayoutType::get(layoutAttr).rebuildStaticValue(builder, loc, nullptr);
  else if (auto composedAttr = dyn_cast<ComposedLayoutAttr>(getLayout()))
    layout = ComposedLayoutType::get(composedAttr).rebuildStaticValue(builder, loc, nullptr);
  else
    return nullptr;
  return MakeViewOp::create(builder, loc, base, layout);
}

Value TiledCopyType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                        Value currentValue) const {
  if (currentValue && isa<MakeTiledCopyOp>(currentValue.getDefiningOp()))
    return nullptr;
  Value copyAtom =
      cast<MayStaticTypeInterface>(getCopyAtom()).rebuildStaticValue(builder, loc, nullptr);
  if (!copyAtom)
    return nullptr;
  Value layoutThrVal = getLayoutThrVal().rebuildStaticValue(builder, loc, nullptr);
  Value tileMN = TileType::get(getTileMN().getAttr()).rebuildStaticValue(builder, loc, nullptr);
  if (!tileMN)
    return nullptr;
  return MakeTiledCopyOp::create(builder, loc, *this, copyAtom, layoutThrVal, tileMN);
}

Value TiledMmaType::rebuildStaticValue(OpBuilder &builder, Location loc, Value currentValue) const {
  if (currentValue && isa<MakeTiledMmaOp>(currentValue.getDefiningOp()))
    return nullptr;
  Value mmaAtom =
      cast<MayStaticTypeInterface>(getMmaAtom()).rebuildStaticValue(builder, loc, nullptr);
  if (!mmaAtom)
    return nullptr;
  Value atomLayout = getAtomLayout().rebuildStaticValue(builder, loc, nullptr);
  Value permutation = getPermutation().rebuildStaticValue(builder, loc, nullptr);
  return MakeTiledMmaOp::create(builder, loc, *this, mmaAtom, atomLayout, permutation);
}

Value CopyAtomType::rebuildStaticValue(OpBuilder &builder, Location loc, Value currentValue) const {
  if (currentValue && isa<MakeCopyAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  return MakeCopyAtomOp::create(builder, loc, *this, getValBits());
}

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

bool CoordTensorType::isLeaf() const { return cast<NestedAttrInterface>(getLayout()).isLeaf(); }
int32_t CoordTensorType::rank() const { return cast<NestedAttrInterface>(getLayout()).rank(); }
int32_t CoordTensorType::rank(int32_t idx) const {
  return cast<NestedAttrInterface>(getLayout()).rank(idx);
}
int32_t CoordTensorType::rank(ArrayRef<int32_t> idxs) const {
  return cast<NestedAttrInterface>(getLayout()).rank(idxs);
}
int32_t CoordTensorType::depth() const { return cast<NestedAttrInterface>(getLayout()).depth(); }
int32_t CoordTensorType::depth(int32_t idx) const {
  return cast<NestedAttrInterface>(getLayout()).depth(idx);
}
int32_t CoordTensorType::depth(ArrayRef<int32_t> idxs) const {
  return cast<NestedAttrInterface>(getLayout()).depth(idxs);
}

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

int32_t PointerType::getValueDivisibility() const {
  int32_t bitWidth = getElemTy().getIntOrFloatBitWidth();
  int32_t alignmentBytes = getAlignment().getAlignment();
  assert(alignmentBytes * 8 % bitWidth == 0);
  return alignmentBytes * 8 / bitWidth;
}

int32_t MemRefType::getValueDivisibility() const {
  int32_t bitWidth = getElemTy().getIntOrFloatBitWidth();
  int32_t alignmentBytes = getAlignment().getAlignment();
  assert(alignmentBytes * 8 % bitWidth == 0);
  return alignmentBytes * 8 / bitWidth;
}

MemRefType MemRefType::at(int32_t idx) const {
  Attribute layoutAttr = getLayout();
  if (auto layout = dyn_cast<LayoutAttr>(layoutAttr))
    return MemRefType::get(getElemTy(), getAddressSpace(), layout.at(idx), getAlignment(),
                           getSwizzle());
  auto composed = cast<ComposedLayoutAttr>(layoutAttr);
  return MemRefType::get(getElemTy(), getAddressSpace(), composed.at(idx), getAlignment(),
                         getSwizzle());
}
MemRefType MemRefType::at(ArrayRef<int32_t> idxs) const {
  Attribute layoutAttr = getLayout();
  if (auto layout = dyn_cast<LayoutAttr>(layoutAttr))
    return MemRefType::get(getElemTy(), getAddressSpace(), layout.at(idxs), getAlignment(),
                           getSwizzle());
  auto composed = cast<ComposedLayoutAttr>(layoutAttr);
  return MemRefType::get(getElemTy(), getAddressSpace(), composed.at(idxs), getAlignment(),
                         getSwizzle());
}

PointerType MemRefType::getPointerType() const {
  return PointerType::get(getElemTy(), getAddressSpace(), getAlignment(), getSwizzle());
}

CoordTensorType CoordTensorType::at(int32_t idx) const {
  Attribute layoutAttr = getLayout();
  if (auto layout = dyn_cast<LayoutAttr>(layoutAttr))
    return CoordTensorType::get(getContext(), getBase().at(idx), layout.at(idx));
  auto composed = cast<ComposedLayoutAttr>(layoutAttr);
  return CoordTensorType::get(getContext(), getBase().at(idx), composed.at(idx));
}
CoordTensorType CoordTensorType::at(ArrayRef<int32_t> idxs) const {
  Attribute layoutAttr = getLayout();
  if (auto layout = dyn_cast<LayoutAttr>(layoutAttr))
    return CoordTensorType::get(getContext(), getBase().at(idxs), layout.at(idxs));
  auto composed = cast<ComposedLayoutAttr>(layoutAttr);
  return CoordTensorType::get(getContext(), getBase().at(idxs), composed.at(idxs));
}

Type CoordTensorType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return {};
  auto base = FieldParser<IntTupleAttr>::parse(parser);
  if (failed(base))
    return {};
  if (parser.parseComma())
    return {};
  Attribute layout = ComposedLayoutAttr::parse(parser, {});
  if (!layout)
    return {};
  if (parser.parseGreater())
    return {};
  return get((*base).getContext(), *base, layout);
}

void CoordTensorType::print(AsmPrinter &printer) const {
  printer << "<";
  printer.printStrippedAttrOrType(getBase());
  printer << ",";
  Attribute layoutAttr = getLayout();
  if (auto layout = dyn_cast<LayoutAttr>(layoutAttr))
    printer.printStrippedAttrOrType(layout);
  else
    printer.printStrippedAttrOrType(cast<ComposedLayoutAttr>(layoutAttr));
  printer << ">";
}

static LogicalResult parseAlignAndSwizzle(AsmParser &parser, Type elemTy, AlignAttr &alignment,
                                          SwizzleAttr &swizzle) {
  alignment = AlignAttr::getTrivialAlignment(elemTy);
  swizzle = SwizzleAttr::getTrivialSwizzle(elemTy.getContext());
  if (succeeded(parser.parseOptionalComma())) {
    if (succeeded(parser.parseOptionalKeyword("align"))) {
      int32_t val;
      if (parser.parseLess() || parser.parseInteger(val) || parser.parseGreater())
        return failure();
      int32_t elemByte = (elemTy.getIntOrFloatBitWidth() + 7) / 8;
      if (val <= 0 || val % elemByte != 0)
        return parser.emitError(parser.getCurrentLocation(),
                                "alignment must be a positive multiple of "
                                "element byte size (")
               << elemByte << "), got " << val;
      alignment = AlignAttr::get(elemTy.getContext(), val);
      if (succeeded(parser.parseOptionalComma())) {
        auto sw = FieldParser<SwizzleAttr>::parse(parser);
        if (failed(sw))
          return failure();
        swizzle = *sw;
      }
    } else {
      auto sw = FieldParser<SwizzleAttr>::parse(parser);
      if (failed(sw))
        return failure();
      swizzle = *sw;
    }
  }
  return success();
}

static void printAlignAndSwizzle(AsmPrinter &printer, Type elemTy, AlignAttr alignment,
                                 SwizzleAttr swizzle, MLIRContext *ctx) {
  if (alignment != AlignAttr::getTrivialAlignment(elemTy)) {
    printer << ",";
    printer.printStrippedAttrOrType(alignment);
  }
  if (swizzle != SwizzleAttr::getTrivialSwizzle(ctx)) {
    printer << ",";
    printer.printStrippedAttrOrType(swizzle);
  }
}

Type PointerType::parse(AsmParser &parser) {
  parser.getContext()->getOrLoadDialect<FlyDialect>();
  Type elemTy;
  FailureOr<AddressSpaceAttr> addressSpace;
  if (parser.parseLess() || parser.parseType(elemTy) || parser.parseComma())
    return {};
  addressSpace = FieldParser<AddressSpaceAttr>::parse(parser);
  if (failed(addressSpace))
    return {};
  AlignAttr alignment;
  SwizzleAttr swizzle;
  if (failed(parseAlignAndSwizzle(parser, elemTy, alignment, swizzle)) || parser.parseGreater())
    return {};
  return get(elemTy.getContext(), elemTy, *addressSpace, alignment, swizzle);
}

void PointerType::print(AsmPrinter &printer) const {
  printer << "<" << getElemTy() << ",";
  printer.printStrippedAttrOrType(getAddressSpace());
  printAlignAndSwizzle(printer, getElemTy(), getAlignment(), getSwizzle(), getContext());
  printer << ">";
}

Type MemRefType::parse(AsmParser &parser) {
  parser.getContext()->getOrLoadDialect<FlyDialect>();
  Type elemTy;
  FailureOr<AddressSpaceAttr> addressSpace;
  if (parser.parseLess() || parser.parseType(elemTy) || parser.parseComma())
    return {};
  addressSpace = FieldParser<AddressSpaceAttr>::parse(parser);
  if (failed(addressSpace))
    return {};
  if (parser.parseComma())
    return {};
  Attribute layout = ComposedLayoutAttr::parse(parser, {});
  if (!layout)
    return {};
  AlignAttr alignment;
  SwizzleAttr swizzle;
  if (failed(parseAlignAndSwizzle(parser, elemTy, alignment, swizzle)) || parser.parseGreater())
    return {};
  return get(elemTy.getContext(), elemTy, *addressSpace, layout, alignment, swizzle);
}

void MemRefType::print(AsmPrinter &printer) const {
  printer << "<" << getElemTy() << ",";
  printer.printStrippedAttrOrType(getAddressSpace());
  printer << ", ";
  Attribute layoutAttr = getLayout();
  if (auto layout = dyn_cast<LayoutAttr>(layoutAttr))
    printer.printStrippedAttrOrType(layout);
  else
    printer.printStrippedAttrOrType(cast<ComposedLayoutAttr>(layoutAttr));
  printAlignAndSwizzle(printer, getElemTy(), getAlignment(), getSwizzle(), getContext());
  printer << ">";
}

#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"

TileType TiledMmaType::getDefaultPermutationMNK(MLIRContext *ctx) {
  Attribute noneVal = IntAttr::getNone(ctx);
  SmallVector<Attribute> elems(3, noneVal);
  return TileType::get(ctx, TileAttr::get(ArrayAttr::get(ctx, elems)));
}

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

bool CopyAtomType::isStatic() const {
  auto copyOp = dyn_cast<CopyOpTypeInterface>(getCopyOp());
  if (!copyOp)
    return false;
  return copyOp.isStatic();
}

Attribute CopyAtomType::getThrLayout() {
  auto copyOp = cast<CopyOpTypeInterface>(getCopyOp());
  return copyOp.getThrLayout();
}

Attribute CopyAtomType::getThrValLayoutSrc() {
  auto copyOp = cast<CopyOpTypeInterface>(getCopyOp());
  LayoutBuilder<LayoutAttr> builder(getContext());
  return layoutRecast(builder, cast<LayoutAttr>(copyOp.getThrBitLayoutSrc()), 1, getValBits());
}
Attribute CopyAtomType::getThrValLayoutDst() {
  auto copyOp = cast<CopyOpTypeInterface>(getCopyOp());
  LayoutBuilder<LayoutAttr> builder(getContext());
  return layoutRecast(builder, cast<LayoutAttr>(copyOp.getThrBitLayoutDst()), 1, getValBits());
}
Attribute CopyAtomType::getThrValLayoutRef() {
  auto copyOp = cast<CopyOpTypeInterface>(getCopyOp());
  LayoutBuilder<LayoutAttr> builder(getContext());
  return layoutRecast(builder, cast<LayoutAttr>(copyOp.getThrBitLayoutRef()), 1, getValBits());
}

bool MmaAtomUniversalFMAType::isStatic() const { return true; }

Value MmaAtomUniversalFMAType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                                  Value currentValue) const {
  if (currentValue && isa<MakeMmaAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  return MakeMmaAtomOp::create(builder, loc, Type(*this));
}

Attribute MmaAtomUniversalFMAType::getShapeMNK() const {
  return IntTupleAttr::get(ArrayAttr::get(getContext(), {FxC(1), FxC(1), FxC(1)}));
}

Attribute MmaAtomUniversalFMAType::getThrLayout() const { return FxLayout(FxC(1), FxC(1)); }

Type MmaAtomUniversalFMAType::getValTypeA() const { return getElemTy(); }
Type MmaAtomUniversalFMAType::getValTypeB() const { return getElemTy(); }
Type MmaAtomUniversalFMAType::getValTypeC() const { return getElemTy(); }
Type MmaAtomUniversalFMAType::getValTypeD() const { return getElemTy(); }

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
