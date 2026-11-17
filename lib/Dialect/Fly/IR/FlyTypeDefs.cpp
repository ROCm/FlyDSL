// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

namespace mlir::fly {

LogicalResult BasisType::verify(function_ref<InFlightDiagnostic()> emitError, LeafAttr attr) {
  if (!attr || !attr.isBasis())
    return emitError() << "expected a basis leaf";
  return success();
}

bool BasisType::isStatic() const { return getAttr().isStatic(); }
bool IntTupleType::isStatic() const { return getAttr().isStatic(); }
bool SwizzleType::isStatic() const { return true; }
bool CoordSwizzleType::isStatic() const { return true; }
bool LayoutType::isStatic() const { return getAttr().isStatic(); }
bool ComposedLayoutType::isStatic() const { return getAttr().isStatic(); }
bool TileType::isStatic() const { return getRef().isStatic(); }
bool CoordTensorType::isStatic() const {
  return getBase().isStatic() && cast<MayStaticAttrInterface>(getLayout()).isStatic();
}

bool TiledCopyType::isStatic() const {
  return cast<MayStaticTypeInterface>(getCopyAtom()).isStatic();
}
bool TiledMmaType::isStatic() const {
  return cast<MayStaticTypeInterface>(getMmaAtom()).isStatic();
}

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
  printer << ", ";
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
    printer << ", ";
    printer.printStrippedAttrOrType(alignment);
  }
  if (swizzle != SwizzleAttr::getTrivialSwizzle(ctx)) {
    printer << ", ";
    printer.printStrippedAttrOrType(swizzle);
  }
}

// Parses the address-space. Accepts either:
//   * a fly address space enum keyword (`global`, `shared`, `register`) parsed as
//     `AddressSpaceAttr`, or
//   * a dialect-qualified attribute (e.g. `#fly_rocdl.buffer_desc`) for
//     target-specific spaces.
static FailureOr<Attribute> parseAddressSpaceAttribute(AsmParser &parser) {
  Attribute attr;
  OptionalParseResult opt = parser.parseOptionalAttribute(attr);
  if (opt.has_value()) {
    if (failed(*opt))
      return failure();
    return attr;
  }
  auto enumAttr = FieldParser<AddressSpaceAttr>::parse(parser);
  if (failed(enumAttr))
    return failure();
  return Attribute(*enumAttr);
}

static void printAddressSpaceAttribute(AsmPrinter &printer, Attribute attr) {
  if (auto e = dyn_cast<AddressSpaceAttr>(attr))
    printer.printStrippedAttrOrType(e);
  else
    printer.printAttribute(attr);
}

Type PointerType::parse(AsmParser &parser) {
  parser.getContext()->getOrLoadDialect<FlyDialect>();
  Type elemTy;
  if (parser.parseLess() || parser.parseType(elemTy) || parser.parseComma())
    return {};
  auto addressSpace = parseAddressSpaceAttribute(parser);
  if (failed(addressSpace))
    return {};
  AlignAttr alignment;
  SwizzleAttr swizzle;
  if (failed(parseAlignAndSwizzle(parser, elemTy, alignment, swizzle)) || parser.parseGreater())
    return {};
  return get(elemTy.getContext(), elemTy, *addressSpace, alignment, swizzle);
}

void PointerType::print(AsmPrinter &printer) const {
  printer << "<" << getElemTy() << ", ";
  printAddressSpaceAttribute(printer, getAddressSpace());
  printAlignAndSwizzle(printer, getElemTy(), getAlignment(), getSwizzle(), getContext());
  printer << ">";
}

Type MemRefType::parse(AsmParser &parser) {
  parser.getContext()->getOrLoadDialect<FlyDialect>();
  Type elemTy;
  if (parser.parseLess() || parser.parseType(elemTy) || parser.parseComma())
    return {};
  auto addressSpace = parseAddressSpaceAttribute(parser);
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
  printer << "<" << getElemTy() << ", ";
  printAddressSpaceAttribute(printer, getAddressSpace());
  printer << ", ";
  Attribute layoutAttr = getLayout();
  if (auto layout = dyn_cast<LayoutAttr>(layoutAttr))
    printer.printStrippedAttrOrType(layout);
  else
    printer.printStrippedAttrOrType(cast<ComposedLayoutAttr>(layoutAttr));
  printAlignAndSwizzle(printer, getElemTy(), getAlignment(), getSwizzle(), getContext());
  printer << ">";
}

TileType TiledMmaType::getDefaultPermutationMNK(MLIRContext *ctx) {
  auto none = core::Tile<core::Trait>::getNone();
  return TileType::get(ctx, core::Tile<core::Trait>::getTuple({none, none, none}));
}

bool CopyAtomType::isStatic() const {
  auto mayStatic = dyn_cast<MayStaticTypeInterface>(getCopyOp());
  if (!mayStatic)
    return false;
  return mayStatic.isStatic();
}

Attribute CopyAtomType::getThrLayout() {
  auto copyOp = cast<CopyOpTypeInterface>(getCopyOp());
  return copyOp.getThrLayout();
}

Attribute CopyAtomType::getThrValLayoutSrc() {
  auto copyOp = cast<CopyOpTypeInterface>(getCopyOp());
  auto layout = cast<LayoutAttr>(copyOp.getThrBitLayoutSrc());
  return getValBits() <= 0
             ? layout
             : LayoutAttr::get(getContext(), core::recast_layout(getValBits(), 1, layout.getRef()));
}
Attribute CopyAtomType::getThrValLayoutDst() {
  auto copyOp = cast<CopyOpTypeInterface>(getCopyOp());
  auto layout = cast<LayoutAttr>(copyOp.getThrBitLayoutDst());
  return getValBits() <= 0
             ? layout
             : LayoutAttr::get(getContext(), core::recast_layout(getValBits(), 1, layout.getRef()));
}
Attribute CopyAtomType::getThrValLayoutRef() {
  auto copyOp = cast<CopyOpTypeInterface>(getCopyOp());
  auto layout = cast<LayoutAttr>(copyOp.getThrBitLayoutRef());
  return getValBits() <= 0
             ? layout
             : LayoutAttr::get(getContext(), core::recast_layout(getValBits(), 1, layout.getRef()));
}

LogicalResult CopyAtomType::emitAtomCall(OpBuilder &builder, Location loc, Type copyAtomTy,
                                         Type srcMemTy, Type dstMemTy, Value atomVal, Value src,
                                         Value dst) const {
  return cast<CopyOpTypeInterface>(getCopyOp())
      .emitAtomCall(builder, loc, copyAtomTy, srcMemTy, dstMemTy, atomVal, src, dst);
}

LogicalResult CopyAtomType::emitAtomCall(OpBuilder &builder, Location loc, Type copyAtomTy,
                                         Type srcMemTy, Type dstMemTy, Type predMemTy,
                                         Value atomVal, Value src, Value dst, Value pred) const {
  return cast<CopyOpTypeInterface>(getCopyOp())
      .emitAtomCall(builder, loc, copyAtomTy, srcMemTy, dstMemTy, predMemTy, atomVal, src, dst,
                    pred);
}

FailureOr<Value> CopyAtomType::emitAtomCallSSA(OpBuilder &builder, Location loc, Type resultTy,
                                               Type copyAtomTy, Type srcTy, Type dstTy,
                                               Value atomVal, Value src, Value dst) const {
  return cast<CopyOpTypeInterface>(getCopyOp())
      .emitAtomCallSSA(builder, loc, resultTy, copyAtomTy, srcTy, dstTy, atomVal, src, dst);
}

FailureOr<Value> CopyAtomType::emitAtomCallSSA(OpBuilder &builder, Location loc, Type resultTy,
                                               Type copyAtomTy, Type srcTy, Type dstTy, Type predTy,
                                               Value atomVal, Value src, Value dst,
                                               Value pred) const {
  return cast<CopyOpTypeInterface>(getCopyOp())
      .emitAtomCallSSA(builder, loc, resultTy, copyAtomTy, srcTy, dstTy, predTy, atomVal, src, dst,
                       pred);
}

bool CopyAtomType::isStateful() const { return isa<StatefulOpTypeInterface>(getCopyOp()); }

Type CopyAtomType::getConvertedType(MLIRContext *ctx) const {
  if (auto stateful = dyn_cast<StatefulOpTypeInterface>(getCopyOp()))
    return stateful.getConvertedType(ctx);
  return Type();
}

Value CopyAtomType::setAtomState(OpBuilder &builder, Location loc, Value atomStruct,
                                 Attribute fieldAttr, Value fieldValue) const {
  assert(this->isStateful() && "CopyAtom is not stateful");
  return cast<StatefulOpTypeInterface>(getCopyOp())
      .setAtomState(builder, loc, atomStruct, fieldAttr, fieldValue);
}

bool MmaAtomType::isStatic() const {
  auto mayStatic = dyn_cast<MayStaticTypeInterface>(getMmaOp());
  if (!mayStatic)
    return false;
  return mayStatic.isStatic();
}

Attribute MmaAtomType::getThrLayout() const {
  return cast<MmaOpTypeInterface>(getMmaOp()).getThrLayout();
}
Attribute MmaAtomType::getShapeMNK() const {
  return cast<MmaOpTypeInterface>(getMmaOp()).getShapeMNK();
}
Type MmaAtomType::getValTypeA() const { return cast<MmaOpTypeInterface>(getMmaOp()).getValTypeA(); }
Type MmaAtomType::getValTypeB() const { return cast<MmaOpTypeInterface>(getMmaOp()).getValTypeB(); }
Type MmaAtomType::getValTypeC() const { return cast<MmaOpTypeInterface>(getMmaOp()).getValTypeC(); }
Type MmaAtomType::getValTypeD() const { return cast<MmaOpTypeInterface>(getMmaOp()).getValTypeD(); }
Attribute MmaAtomType::getThrValLayoutA() const {
  return cast<MmaOpTypeInterface>(getMmaOp()).getThrValLayoutA();
}
Attribute MmaAtomType::getThrValLayoutB() const {
  return cast<MmaOpTypeInterface>(getMmaOp()).getThrValLayoutB();
}
Attribute MmaAtomType::getThrValLayoutC() const {
  return cast<MmaOpTypeInterface>(getMmaOp()).getThrValLayoutC();
}

LogicalResult MmaAtomType::emitAtomCall(OpBuilder &builder, Location loc, Type mmaAtomTy,
                                        Type dMemTy, TypeRange aMemTy, TypeRange bMemTy,
                                        Type cMemTy, Value atomVal, Value d, ValueRange a,
                                        ValueRange b, Value c) const {
  return cast<MmaOpTypeInterface>(getMmaOp())
      .emitAtomCall(builder, loc, mmaAtomTy, dMemTy, aMemTy, bMemTy, cMemTy, atomVal, d, a, b, c);
}
FailureOr<Value> MmaAtomType::emitAtomCallSSA(OpBuilder &builder, Location loc, Type resultTy,
                                              Type mmaAtomTy, Type dTy, TypeRange aTy,
                                              TypeRange bTy, Type cTy, Value atomVal, Value d,
                                              ValueRange a, ValueRange b, Value c) const {
  return cast<MmaOpTypeInterface>(getMmaOp())
      .emitAtomCallSSA(builder, loc, resultTy, mmaAtomTy, dTy, aTy, bTy, cTy, atomVal, d, a, b, c);
}

bool MmaAtomType::isStateful() const { return isa<StatefulOpTypeInterface>(getMmaOp()); }

Type MmaAtomType::getConvertedType(MLIRContext *ctx) const {
  if (auto stateful = dyn_cast<StatefulOpTypeInterface>(getMmaOp()))
    return stateful.getConvertedType(ctx);
  return Type();
}

Value MmaAtomType::setAtomState(OpBuilder &builder, Location loc, Value atomStruct,
                                Attribute fieldAttr, Value fieldValue) const {
  assert(this->isStateful() && "MmaAtom is not stateful");
  return cast<StatefulOpTypeInterface>(getMmaOp())
      .setAtomState(builder, loc, atomStruct, fieldAttr, fieldValue);
}

//===----------------------------------------------------------------------===//
// Core descriptions
//===----------------------------------------------------------------------===//

static FailureOr<core::AnyLayout<core::Trait>> getLayoutRef(Type type) {
  Attribute attr = llvm::TypeSwitch<Type, Attribute>(type)
                       .Case<LayoutType, ComposedLayoutType>([](auto t) { return t.getAttr(); })
                       .Case<MemRefType, CoordTensorType>([](auto t) { return t.getLayout(); })
                       .Default([](Type) { return Attribute(); });
  if (auto layout = dyn_cast_or_null<LayoutAttr>(attr))
    return core::AnyLayout<core::Trait>(layout.getRef());
  if (auto layout = dyn_cast_or_null<ComposedLayoutAttr>(attr))
    return core::AnyLayout<core::Trait>(layout.getRef());
  return failure();
}

/// i1 is stored in a whole byte; every other element is stored at its width.
static int32_t storageBitWidth(Type elem) {
  return elem.isInteger(1) ? 8 : elem.getIntOrFloatBitWidth();
}

static LogicalResult verifyElement(function_ref<InFlightDiagnostic()> emitError, Type elemTy,
                                   Attribute addressSpace, AlignAttr alignment) {
  if (!isa<IntegerType, FloatType>(elemTy))
    return emitError() << "expected an integer or float element, got " << elemTy;
  if (!addressSpace)
    return emitError() << "expected an address space";
  if (!alignment || alignment.getAlignment() <= 0 || alignment.getBitAlignment() <= 0)
    return emitError() << "expected a positive alignment";
  return success();
}

static LogicalResult verifyLayout(function_ref<InFlightDiagnostic()> emitError, Attribute layout) {
  if (!isa<LayoutAttr, ComposedLayoutAttr>(layout))
    return emitError() << "expected a layout or composed layout, got " << layout;
  return success();
}

LogicalResult PointerType::verify(function_ref<InFlightDiagnostic()> emitError, Type elemTy,
                                  Attribute addressSpace, AlignAttr alignment, SwizzleAttr) {
  return verifyElement(emitError, elemTy, addressSpace, alignment);
}

LogicalResult MemRefType::verify(function_ref<InFlightDiagnostic()> emitError, Type elemTy,
                                 Attribute addressSpace, Attribute layout, AlignAttr alignment,
                                 SwizzleAttr) {
  if (failed(verifyElement(emitError, elemTy, addressSpace, alignment)))
    return failure();
  return verifyLayout(emitError, layout);
}

LogicalResult CoordTensorType::verify(function_ref<InFlightDiagnostic()> emitError, IntTupleAttr,
                                      Attribute layout) {
  return verifyLayout(emitError, layout);
}

LogicalResult FastDivModType::verify(function_ref<InFlightDiagnostic()> emitError,
                                     IntegerType divisorType) {
  if (!divisorType.isSignless() || (divisorType.getWidth() != 32 && divisorType.getWidth() != 64))
    return emitError() << "expected an i32 or i64 divisor, got " << divisorType;
  return success();
}

core::Pointer<core::Trait> PointerType::getRef() const {
  Type elem = getElemTy();
  AlignAttr alignment = getAlignment();
  return core::Pointer<core::Trait>(
      core::OpaqueValue::get(elem), elem.getIntOrFloatBitWidth(), alignment.getAlignment(),
      alignment.getBitAlignment(), getSwizzle().getRef(), core::OpaqueValue::get(getAddressSpace()),
      storageBitWidth(elem));
}

namespace {

/// Fly wraps MLIR handles as their base class: OpaqueValue keeps the exact C++
/// type, so an element type is recovered from a Type, or from a TypeAttr held
/// as an Attribute.
Type getOpaqueElementType(const core::OpaqueValue &value) {
  if (value.is<Type>())
    return value.get<Type>();
  if (value.is<Attribute>())
    if (auto attr = dyn_cast_or_null<TypeAttr>(value.get<Attribute>()))
      return attr.getValue();
  return {};
}

/// An address space is a Fly AddressSpace or an attribute handle, which keeps
/// target-specific spaces opaque to Fly.
Attribute getOpaqueAddressSpace(MLIRContext *ctx, const core::OpaqueValue &value) {
  if (value.is<AddressSpace>()) {
    auto space = value.get<AddressSpace>();
    if (!symbolizeAddressSpace(static_cast<uint32_t>(space)))
      return {};
    return AddressSpaceAttr::get(ctx, space);
  }
  return value.is<Attribute>() ? value.get<Attribute>() : Attribute();
}

} // namespace

std::optional<PointerType::CoreParameters>
PointerType::getCoreParameters(MLIRContext *ctx, const core::Pointer<core::Trait> &value) {
  Type elem = getOpaqueElementType(value.elementType());
  if (!elem || !isa<IntegerType, FloatType>(elem) ||
      static_cast<int32_t>(elem.getIntOrFloatBitWidth()) != value.elementBitWidth() ||
      value.storageElementBitWidth() != storageBitWidth(elem))
    return std::nullopt;
  ctx = elem.getContext();
  Attribute addressSpace = getOpaqueAddressSpace(ctx, value.addressSpace());
  if (!addressSpace || addressSpace.getContext() != ctx)
    return std::nullopt;
  auto alignment = AlignAttr::get(ctx, value.byteAlignment());
  if (!alignment)
    return std::nullopt;
  return CoreParameters{elem, addressSpace, alignment, SwizzleAttr::get(ctx, value.swizzle())};
}

core::MemRef<core::Trait> MemRefType::getRef() const {
  return core::MemRef<core::Trait>(getPointerType().getRef(), *getLayoutRef(*this));
}

core::CoordTensor<core::Trait> CoordTensorType::getRef() const {
  return core::CoordTensor<core::Trait>(getBase().getRef(), *getLayoutRef(*this));
}

core::CopyAtom<core::Trait> CopyAtomType::getRef() const {
  auto operation = cast<CopyOpTypeInterface>(getCopyOp());
  return core::CopyAtom<core::Trait>(core::OpaqueValue::get(getCopyOp()),
                                     core::OpaqueValue::get(Type(*this)), getValBits(),
                                     cast<LayoutAttr>(operation.getThrLayout()).getRef(),
                                     cast<LayoutAttr>(operation.getThrBitLayoutSrc()).getRef(),
                                     cast<LayoutAttr>(operation.getThrBitLayoutDst()).getRef(),
                                     cast<LayoutAttr>(operation.getThrBitLayoutRef()).getRef());
}

core::MMAAtom<core::Trait> MmaAtomType::getRef() const {
  return core::MMAAtom<core::Trait>(
      core::OpaqueValue::get(getMmaOp()), cast<LayoutAttr>(getThrLayout()).getRef(),
      cast<IntTupleAttr>(getShapeMNK()).getRef(),
      core::MMAValueTypeIdentities{
          core::OpaqueValue::get(getValTypeA()), core::OpaqueValue::get(getValTypeB()),
          core::OpaqueValue::get(getValTypeC()), core::OpaqueValue::get(getValTypeD())},
      cast<LayoutAttr>(getThrValLayoutA()).getRef(), cast<LayoutAttr>(getThrValLayoutB()).getRef(),
      cast<LayoutAttr>(getThrValLayoutC()).getRef());
}

bool TiledCopyType::isValidRef() const {
  auto atomTy = dyn_cast<CopyAtomType>(getCopyAtom());
  auto operation = atomTy ? dyn_cast<CopyOpTypeInterface>(atomTy.getCopyOp()) : nullptr;
  if (!operation)
    return false;
  auto thr = dyn_cast_or_null<LayoutAttr>(operation.getThrLayout());
  auto src = dyn_cast_or_null<LayoutAttr>(operation.getThrBitLayoutSrc());
  auto dst = dyn_cast_or_null<LayoutAttr>(operation.getThrBitLayoutDst());
  auto ref = dyn_cast_or_null<LayoutAttr>(operation.getThrBitLayoutRef());
  if (!thr || !src || !dst || !ref ||
      !core::CopyAtom<core::Trait>::isValid(atomTy.getValBits(), thr.getRef(), src.getRef(),
                                            dst.getRef(), ref.getRef()))
    return false;
  return core::TiledCopy<core::Trait>::isValid(atomTy.getRef(), getLayoutThrVal().getRef());
}

core::TiledCopy<core::Trait> TiledCopyType::getRef() const {
  return core::TiledCopy<core::Trait>(cast<CopyAtomType>(getCopyAtom()).getRef(),
                                      getLayoutThrVal().getRef(), getTileMN().getRef());
}

core::TiledMMA<core::Trait> TiledMmaType::getRef() const {
  return core::TiledMMA<core::Trait>(cast<MmaAtomType>(getMmaAtom()).getRef(),
                                     getAtomLayout().getRef(), getPermutation().getRef());
}

} // namespace mlir::fly
