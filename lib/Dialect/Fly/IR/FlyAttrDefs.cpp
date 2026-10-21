// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

namespace mlir::fly {

//===----------------------------------------------------------------------===//
// AotModuleAttr
//===----------------------------------------------------------------------===//

LogicalResult AotModuleAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                    StringRef symbolPrefix, Attribute target) {
  if (symbolPrefix.empty())
    return emitError() << "the exported symbol prefix must not be empty";
  if (!target)
    return success();
  if (auto index = dyn_cast<IntegerAttr>(target)) {
    if (index.getInt() < 0)
      return emitError() << "the object index must be non-negative";
    return success();
  }
  if (!target.hasPromiseOrImplementsInterface<gpu::TargetAttrInterface>())
    return emitError() << "the target attribute must implement gpu::TargetAttrInterface";
  return success();
}

//===----------------------------------------------------------------------===//
// AlignAttr
//===----------------------------------------------------------------------===//

AlignAttr AlignAttr::getFromBitAlignment(MLIRContext *ctx, int64_t bits) {
  if (bits <= 0 || bits % 8 != 0)
    return {};
  return get(ctx, static_cast<int32_t>(bits / 8));
}

Attribute AlignAttr::parse(AsmParser &parser, Type) {
  int32_t bytes;
  if (parser.parseKeyword("align") || parser.parseLess() || parser.parseInteger(bytes) ||
      parser.parseGreater())
    return {};
  auto result = get(parser.getContext(), bytes);
  if (!result)
    parser.emitError(parser.getCurrentLocation(), "expected a positive byte alignment");
  return result;
}

void AlignAttr::print(AsmPrinter &printer) const { printer << "align<" << getAlignment() << ">"; }

AlignAttr AlignAttr::getTrivialAlignment(Type elemTy) {
  assert((isa<IntegerType>(elemTy) || isa<FloatType>(elemTy)) &&
         "Trivial alignment is only supported for integer and float types");
  return get(elemTy.getContext(), (elemTy.getIntOrFloatBitWidth() + 7) / 8);
}

//===----------------------------------------------------------------------===//
// LeafAttr
//===----------------------------------------------------------------------===//

int64_t LeafAttr::getValue() const {
  assert(!isBasis() && !isRatio() && "expected an integer scalar");
  return getRef().isDInt() ? 0 : getRef().staticValue();
}
int32_t LeafAttr::getWidth() const { return isNone() ? 0 : getRef().bitWidth(); }
int64_t LeafAttr::getDivisibility() const {
  if (getRef().isDInt())
    return getRef().divisibility();
  assert(!isRatio() && "ratio has no integer divisibility");
  return isNone() ? 0 : (getValue() == 0 ? 1 : getValue());
}
bool LeafAttr::isRatio() const { return getRef().isRatio(); }
int32_t LeafAttr::getNumerator() const { return getRef().num(); }
int32_t LeafAttr::getDenominator() const { return getRef().den(); }
LeafAttr LeafAttr::getRatio(MLIRContext *ctx, int32_t numerator, int32_t denominator) {
  return get(ctx, core::Leaf<core::Trait>::getRatio(numerator, denominator));
}
bool LeafAttr::isNone() const { return getRef().isNone(); }
bool LeafAttr::isStaticValue(int64_t value) const {
  return !isBasis() && !isRatio() && isStatic() && getValue() == value;
}
LeafAttr LeafAttr::getNone(MLIRContext *ctx) {
  return get(ctx, core::Leaf<core::Trait>::getNone());
}
LeafAttr LeafAttr::getStatic(MLIRContext *ctx, int64_t value) {
  return get(ctx, core::Leaf<core::Trait>::getSInt(value));
}
LeafAttr LeafAttr::getDynamic(MLIRContext *ctx, int32_t width, int32_t divisibility) {
  if ((width != 32 && width != 64) || divisibility <= 0)
    return {};
  return get(ctx, core::Leaf<core::Trait>::getDInt(
                      width == 64 ? core::LogWidth::I64 : core::LogWidth::I32, divisibility));
}
bool LeafAttr::isStatic() const { return getRef().isStatic(); }

//===----------------------------------------------------------------------===//
// SwizzleAttr
//===----------------------------------------------------------------------===//

Attribute SwizzleAttr::parse(AsmParser &parser, Type) {
  int32_t mask, base, shift;
  if (parser.parseKeyword("S") || parser.parseLess() || parser.parseInteger(mask) ||
      parser.parseComma() || parser.parseInteger(base) || parser.parseComma() ||
      parser.parseInteger(shift) || parser.parseGreater())
    return {};
  if (!core::Swizzle<core::Trait>::isValid(mask, base, shift)) {
    parser.emitError(parser.getCurrentLocation(), "invalid swizzle parameters");
    return {};
  }
  return get(parser.getContext(), mask, base, shift);
}
void SwizzleAttr::print(AsmPrinter &printer) const {
  printer << "S<" << getMask() << "," << getBase() << "," << getShift() << ">";
}

bool SwizzleAttr::isTrivialSwizzle() const { return getMask() == 0; }
SwizzleAttr SwizzleAttr::getTrivialSwizzle(MLIRContext *context) { return get(context, 0, 0, 0); }

//===----------------------------------------------------------------------===//
// CoordSwizzleAttr
//===----------------------------------------------------------------------===//

CoordSwizzleAttr CoordSwizzleAttr::get(MLIRContext *ctx, int32_t mask, int32_t baseRow,
                                       ArrayRef<int32_t> modeRow, int32_t baseCol,
                                       ArrayRef<int32_t> modeCol) {
  auto validModes = [](ArrayRef<int32_t> modes) {
    return modes.size() <= core::kMaxBasisModes &&
           llvm::all_of(modes, [](int32_t mode) { return mode >= 0 && mode <= UINT8_MAX; });
  };
  if (!validModes(modeRow) || !validModes(modeCol) || mask < 0 || baseRow < 0 || baseCol < 0)
    return {};
  auto rows = SmallVector<uint8_t>(modeRow.begin(), modeRow.end());
  auto cols = SmallVector<uint8_t>(modeCol.begin(), modeCol.end());
  return get(ctx, core::CoordSwizzle<core::Trait>(mask, baseRow, rows, baseCol, cols));
}

//===----------------------------------------------------------------------===//
// IntTupleAttr
//===----------------------------------------------------------------------===//

/// Mode `path` of a tuple; a leaf is its own (single) mode 0.
static core::IntTupleRef<core::Trait> modeAt(core::IntTupleRef<core::Trait> tuple,
                                             ArrayRef<int32_t> path) {
  for (int32_t idx : path) {
    if (tuple.isLeaf()) {
      assert(idx == 0 && "a leaf has a single mode");
      continue;
    }
    tuple = tuple.at(idx);
  }
  return tuple;
}

IntTupleAttr IntTupleAttr::get(MLIRContext *ctx, Attribute value) {
  if (auto tuple = dyn_cast<IntTupleAttr>(value))
    return tuple;
  if (auto leaf = dyn_cast<LeafAttr>(value))
    return get(ctx, core::IntTuple<core::Trait>::fromLeaf(leaf.getRef()));
  auto array = dyn_cast<ArrayAttr>(value);
  if (!array)
    return {};
  SmallVector<core::IntTuple<core::Trait>> children;
  for (auto child : array) {
    auto tuple = get(ctx, child);
    if (!tuple)
      return {};
    children.push_back(tuple.getRef());
  }
  return get(ctx, core::make_tuple(children));
}

IntTupleAttr IntTupleAttr::getLeafNone(MLIRContext *ctx) {
  return get(ctx, core::IntTuple<core::Trait>::fromLeaf(core::Leaf<core::Trait>::getNone()));
}
IntTupleAttr IntTupleAttr::getLeafStatic(MLIRContext *ctx, int64_t value) {
  return get(ctx, core::IntTuple<core::Trait>::getSInt(value));
}
IntTupleAttr IntTupleAttr::getLeafDynamic(MLIRContext *ctx, int32_t width, int32_t divisibility) {
  return get(LeafAttr::getDynamic(ctx, width, divisibility));
}
bool IntTupleAttr::isLeafInt() const { return isLeaf() && !getRef().getLeaf().isBasis(); }
bool IntTupleAttr::isLeafBasis() const { return isLeaf() && getRef().getLeaf().isBasis(); }

bool IntTupleAttr::isLeafStaticValue(int64_t value) const {
  return isLeaf() && !getRef().getLeaf().isBasis() && getRef().getLeaf().isSInt(value);
}

LeafAttr IntTupleAttr::getLeaf() const {
  assert(isLeaf() && "expected a leaf tuple");
  return LeafAttr::get(getContext(), getRef().getLeaf());
}

bool IntTupleAttr::isLeaf() const { return getRef().isLeaf(); }

bool IntTupleAttr::isStatic() const { return getRef().isStatic(); }
int32_t IntTupleAttr::rank() const { return getRef().rank(); }
int32_t IntTupleAttr::rank(int32_t idx) const { return modeAt(getRef().asRef(), {idx}).rank(); }
int32_t IntTupleAttr::rank(ArrayRef<int32_t> idxs) const {
  return modeAt(getRef().asRef(), idxs).rank();
}

int32_t IntTupleAttr::depth() const { return getRef().depth(); }
int32_t IntTupleAttr::depth(int32_t idx) const { return modeAt(getRef().asRef(), {idx}).depth(); }
int32_t IntTupleAttr::depth(ArrayRef<int32_t> idxs) const {
  return modeAt(getRef().asRef(), idxs).depth();
}

IntTupleAttr IntTupleAttr::at(int32_t idx) const { return get(getContext(), getRef().at(idx)); }
IntTupleAttr IntTupleAttr::at(ArrayRef<int32_t> idxs) const {
  return get(getContext(), modeAt(getRef().asRef(), idxs));
}

//===----------------------------------------------------------------------===//
// LayoutAttr
//===----------------------------------------------------------------------===//

IntTupleAttr LayoutAttr::getShape() const {
  return IntTupleAttr::get(getContext(), getRef().shape());
}
IntTupleAttr LayoutAttr::getStride() const {
  return IntTupleAttr::get(getContext(), getRef().stride());
}
static FailureOr<core::IntTuple<core::Trait>> parseIntTuple(AsmParser &parser);

static FailureOr<core::Layout<core::Trait>> makeParsedLayout(AsmParser &parser,
                                                             core::IntTuple<core::Trait> shape,
                                                             core::IntTuple<core::Trait> stride) {
  auto layout = core::Layout<core::Trait>(std::move(shape), std::move(stride));
  if (layout.isError()) {
    parser.emitError(parser.getCurrentLocation(), "incongruent layout shape and stride");
    return failure();
  }
  return layout;
}

/// `shape:stride`, the shape already parsed.
static FailureOr<core::Layout<core::Trait>> parseLayoutStride(AsmParser &parser,
                                                              core::IntTuple<core::Trait> shape) {
  if (parser.parseColon())
    return failure();
  auto stride = parseIntTuple(parser);
  if (failed(stride))
    return failure();
  return makeParsedLayout(parser, std::move(shape), std::move(*stride));
}

static FailureOr<core::Layout<core::Trait>> parseLayout(AsmParser &parser) {
  auto shape = parseIntTuple(parser);
  if (failed(shape))
    return failure();
  return parseLayoutStride(parser, std::move(*shape));
}

Attribute LayoutAttr::parse(AsmParser &parser, Type) {
  auto layout = parseLayout(parser);
  if (failed(layout))
    return {};
  return LayoutAttr::get(parser.getContext(), std::move(*layout));
}
static void printIntTuple(AsmPrinter &printer, core::IntTupleRef<core::Trait> tuple);

static void printLayout(AsmPrinter &printer, const core::Layout<core::Trait> &layout);

void LayoutAttr::print(AsmPrinter &printer) const { printLayout(printer, getRef()); }

bool LayoutAttr::isStatic() const { return getRef().isStatic(); }

bool LayoutAttr::isStaticShape() const { return getRef().shape().isStatic(); }

bool LayoutAttr::isStaticStride() const { return getRef().stride().isStatic(); }

bool LayoutAttr::isLeaf() const { return getRef().shape().isLeaf(); }

int32_t LayoutAttr::rank() const { return getRef().shape().rank(); }
int32_t LayoutAttr::rank(int32_t idx) const { return modeAt(getRef().shape(), {idx}).rank(); }
int32_t LayoutAttr::rank(ArrayRef<int32_t> idxs) const {
  return modeAt(getRef().shape(), idxs).rank();
}

int32_t LayoutAttr::depth() const { return getRef().shape().depth(); }
int32_t LayoutAttr::depth(int32_t idx) const { return modeAt(getRef().shape(), {idx}).depth(); }
int32_t LayoutAttr::depth(ArrayRef<int32_t> idxs) const {
  return modeAt(getRef().shape(), idxs).depth();
}

LayoutAttr LayoutAttr::at(int32_t idx) const { return at(ArrayRef<int32_t>{idx}); }
LayoutAttr LayoutAttr::at(ArrayRef<int32_t> idxs) const {
  auto &layout = getRef();
  return get(getContext(), core::Layout<core::Trait>(modeAt(layout.shape(), idxs),
                                                     modeAt(layout.stride(), idxs)));
}

//===----------------------------------------------------------------------===//
// ComposedLayoutAttr
//===----------------------------------------------------------------------===//

namespace {
Attribute exportLayoutAttr(MLIRContext *ctx, const core::Layout<core::Trait> &value) {
  return LayoutAttr::get(ctx, value);
}
Attribute exportLayoutAttr(MLIRContext *ctx, const core::ComposedLayout<core::Trait> &value) {
  return ComposedLayoutAttr::get(ctx, value);
}
Attribute exportLayoutAttr(MLIRContext *ctx, core::Swizzle<core::Trait> value) {
  return SwizzleAttr::get(ctx, value);
}
Attribute exportLayoutAttr(MLIRContext *ctx, core::CoordSwizzle<core::Trait> value) {
  return CoordSwizzleAttr::get(ctx, value);
}
core::ComposedOuter<core::Trait> importComposedOuter(Attribute attr) {
  if (auto layout = dyn_cast<LayoutAttr>(attr))
    return layout.getRef();
  return cast<ComposedLayoutAttr>(attr).getRef();
}
core::ComposedInner<core::Trait> importComposedInner(Attribute attr) {
  if (auto composed = dyn_cast<ComposedLayoutAttr>(attr))
    return composed.getRef();
  if (auto layout = dyn_cast<LayoutAttr>(attr))
    return layout.getRef();
  if (auto swizzle = dyn_cast<SwizzleAttr>(attr))
    return swizzle.getRef();
  return cast<CoordSwizzleAttr>(attr).getRef();
}
} // namespace

ComposedLayoutAttr ComposedLayoutAttr::get(MLIRContext *ctx, Attribute inner, IntTupleAttr offset,
                                           Attribute outer) {
  auto outerValue = importComposedOuter(outer);
  return get(ctx, core::ComposedLayout<core::Trait>(importComposedInner(inner), offset.getRef(),
                                                    outerValue));
}
Attribute ComposedLayoutAttr::getInner() const {
  return std::visit([&](const auto &value) { return exportLayoutAttr(getContext(), value); },
                    getRef().inner());
}
IntTupleAttr ComposedLayoutAttr::getOffset() const {
  return IntTupleAttr::get(getContext(), getRef().offset());
}
Attribute ComposedLayoutAttr::getOuter() const {
  return std::visit([&](const auto &value) { return exportLayoutAttr(getContext(), value); },
                    getRef().outer());
}

bool ComposedLayoutAttr::isStatic() const { return getRef().isStatic(); }
bool ComposedLayoutAttr::isStaticOuter() const {
  return std::visit([](const auto &outer) { return outer.isStatic(); }, getRef().outer());
}
bool ComposedLayoutAttr::isStaticOffset() const { return getRef().offset().isStatic(); }
bool ComposedLayoutAttr::isStaticInner() const {
  return std::visit(
      [](const auto &inner) {
        using Inner = std::decay_t<decltype(inner)>;
        if constexpr (std::is_same_v<Inner, core::Swizzle<core::Trait>> ||
                      std::is_same_v<Inner, core::CoordSwizzle<core::Trait>>)
          return true;
        else
          return inner.isStatic();
      },
      getRef().inner());
}

bool ComposedLayoutAttr::isLeaf() const { return getRef().shape().isLeaf(); }

int32_t ComposedLayoutAttr::rank() const { return getRef().shape().rank(); }
int32_t ComposedLayoutAttr::rank(int32_t idx) const {
  return modeAt(getRef().shape(), {idx}).rank();
}
int32_t ComposedLayoutAttr::rank(ArrayRef<int32_t> idxs) const {
  return modeAt(getRef().shape(), idxs).rank();
}
int32_t ComposedLayoutAttr::depth() const { return getRef().shape().depth(); }
int32_t ComposedLayoutAttr::depth(int32_t idx) const {
  return modeAt(getRef().shape(), {idx}).depth();
}
int32_t ComposedLayoutAttr::depth(ArrayRef<int32_t> idxs) const {
  return modeAt(getRef().shape(), idxs).depth();
}

ComposedLayoutAttr ComposedLayoutAttr::at(int32_t idx) const { return at(ArrayRef<int32_t>{idx}); }
ComposedLayoutAttr ComposedLayoutAttr::at(ArrayRef<int32_t> idxs) const {
  auto composed = getRef();
  for (int32_t idx : idxs)
    composed = core::get(composed, idx);
  return get(getContext(), std::move(composed));
}

//===----------------------------------------------------------------------===//
// TileAttr
//===----------------------------------------------------------------------===//

TileAttr TileAttr::get(MLIRContext *ctx, Attribute value) {
  using Tile = core::Tile<core::Trait>;
  if (auto tile = dyn_cast<TileAttr>(value))
    return tile;
  if (auto layout = dyn_cast<LayoutAttr>(value))
    return get(ctx, Tile(layout.getRef()));
  if (auto leaf = dyn_cast<LeafAttr>(value)) {
    if (leaf.isBasis() || leaf.isRatio())
      return {};
    return get(ctx, leaf.isNone() ? Tile::getNone() : Tile(leaf.getRef()));
  }
  auto array = dyn_cast<ArrayAttr>(value);
  if (!array)
    return {};
  SmallVector<Tile> children;
  for (auto child : array) {
    auto tile = get(ctx, child);
    if (!tile)
      return {};
    children.push_back(tile.getRef());
  }
  return get(ctx, Tile::getTuple(children));
}

//===----------------------------------------------------------------------===//
// Parser and Printer
//===----------------------------------------------------------------------===//

static void printScalarLeaf(AsmPrinter &printer, const core::Leaf<core::Trait> &leaf) {
  if (leaf.isNone()) {
    printer << "*";
  } else if (leaf.isRatio()) {
    printer << leaf.num() << "/" << leaf.den();
  } else if (leaf.isStatic()) {
    printer << leaf.staticValue();
  } else {
    printer << "?";
    if (leaf.bitWidth() != 32 || leaf.divisibility() != 1 || leaf.hasFastDivMod()) {
      printer << "{";
      bool delimiter = false;
      if (leaf.bitWidth() != 32) {
        printer << "i" << leaf.bitWidth();
        delimiter = true;
      }
      if (leaf.divisibility() != 1) {
        if (delimiter) {
          printer << " ";
        }
        printer << "div=" << leaf.divisibility();
        delimiter = true;
      }
      if (leaf.hasFastDivMod()) {
        if (delimiter)
          printer << " ";
        printer << "fdm";
      }
      printer << "}";
    }
  }
}

static LeafAttr parseStaticLeaf(AsmParser &parser) {
  int64_t value;
  if (parser.parseDecimalInteger(value))
    return {};
  if (succeeded(parser.parseOptionalSlash())) {
    int32_t denominator;
    if (parser.parseDecimalInteger(denominator))
      return {};
    if (value < INT32_MIN || value > INT32_MAX) {
      parser.emitError(parser.getCurrentLocation(),
                       "ratio numerator exceeds the representable range");
      return {};
    }
    auto leaf = core::Leaf<core::Trait>::getRatio(value, denominator);
    if (leaf.isError()) {
      parser.emitError(parser.getCurrentLocation(), "invalid or unrepresentable ratio");
      return {};
    }
    return LeafAttr::get(parser.getContext(), leaf);
  }
  return LeafAttr::getStatic(parser.getContext(), value);
}

static LeafAttr parseLeafBody(AsmParser &odsParser) {
  auto *ctx = odsParser.getBuilder().getContext();

  LeafAttr valueAttr;
  if (odsParser.parseOptionalStar().succeeded()) {
    valueAttr = LeafAttr::getNone(ctx);
  } else if (odsParser.parseOptionalQuestion().succeeded()) {
    int32_t width = 32;
    int32_t divisibility = 1;
    bool fastDivMod = false;
    if (odsParser.parseOptionalLBrace().succeeded()) {
      if (odsParser.parseOptionalKeyword("i32")) {
        if (odsParser.parseOptionalKeyword("i64").succeeded()) {
          width = 64;
        }
      }
      if (odsParser.parseOptionalKeyword("div").succeeded()) {
        if (odsParser.parseEqual() || odsParser.parseDecimalInteger(divisibility))
          return {};
      }
      fastDivMod = odsParser.parseOptionalKeyword("fdm").succeeded();
      if (odsParser.parseRBrace())
        return {};
    }
    if (divisibility <= 0) {
      odsParser.emitError(odsParser.getCurrentLocation(), "divisibility must be positive");
      return {};
    }
    valueAttr = LeafAttr::getDynamic(ctx, width, divisibility);
    if (fastDivMod)
      valueAttr = LeafAttr::get(ctx, valueAttr.getRef().withFastDivMod());
  } else {
    valueAttr = parseStaticLeaf(odsParser);
    if (!valueAttr)
      return {};
  }

  auto nextLoc = odsParser.getCurrentLocation();
  const char *nextPtr = nextLoc.getPointer();
  if (!nextPtr || *nextPtr != 'E' || !std::isdigit(static_cast<unsigned char>(*(nextPtr + 1))))
    return valueAttr;

  StringRef strRefModes;
  if (failed(odsParser.parseOptionalKeyword(&strRefModes)))
    return valueAttr;

  SmallVector<int32_t> modes;
  SmallVector<StringRef, 8> strRefModeList;

  strRefModes.split(strRefModeList, "E");
  for (StringRef strRefMode : strRefModeList) {
    if (strRefMode.empty())
      continue;
    int32_t mode;
    if (strRefMode.getAsInteger(10, mode))
      return {};
    if (mode < 0 || mode > UINT8_MAX || modes.size() == core::kMaxBasisModes) {
      odsParser.emitError(odsParser.getCurrentLocation(),
                          "basis mode exceeds the representable range");
      return {};
    }
    modes.push_back(mode);
  }
  auto result = LeafAttr::get(ctx, valueAttr, modes);
  if (!result)
    odsParser.emitError(odsParser.getCurrentLocation(), "invalid basis coefficient or modes");
  return result;
}

Attribute LeafAttr::parse(AsmParser &parser, Type) {
  if (parser.parseLess())
    return {};
  auto leaf = parseLeafBody(parser);
  if (!leaf || parser.parseGreater())
    return {};
  return leaf;
}

static void printLeafBody(AsmPrinter &printer, const core::Leaf<core::Trait> &leaf) {
  printScalarLeaf(printer, leaf.scale());
  if (leaf.isBasis())
    for (uint8_t mode : leaf.modes())
      printer << "E" << unsigned(mode);
}

void LeafAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printLeafBody(printer, getRef());
  printer << ">";
}

static void printIntTuple(AsmPrinter &printer, core::IntTupleRef<core::Trait> tuple) {
  if (tuple.isLeaf()) {
    printLeafBody(printer, tuple.getLeaf());
    return;
  }
  printer << "(";
  llvm::interleave(
      tuple, [&](core::IntTupleRef<core::Trait> child) { printIntTuple(printer, child); },
      [&] { printer << ","; });
  printer << ")";
}

/// A tuple: a parenthesized list of tuples, or a single leaf.
static FailureOr<core::IntTuple<core::Trait>> parseIntTuple(AsmParser &parser) {
  if (parser.parseOptionalLParen().succeeded()) {
    SmallVector<core::IntTuple<core::Trait>> children;
    if (succeeded(parser.parseOptionalRParen()))
      return core::IntTuple<core::Trait>::getEmpty();
    do {
      auto child = parseIntTuple(parser);
      if (failed(child))
        return failure();
      children.push_back(std::move(*child));
    } while (parser.parseOptionalComma().succeeded());
    if (parser.parseRParen())
      return failure();
    auto result = core::make_tuple(children);
    if (result.isError()) {
      parser.emitError(parser.getCurrentLocation(), "tuple exceeds the tree capacity");
      return failure();
    }
    return result;
  }
  auto leaf = parseLeafBody(parser);
  if (!leaf)
    return failure();
  return core::IntTuple<core::Trait>::fromLeaf(leaf.getRef());
}

::mlir::Attribute IntTupleAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type) {
  auto tuple = parseIntTuple(odsParser);
  if (failed(tuple))
    return {};
  return IntTupleAttr::get(odsParser.getContext(), std::move(*tuple));
}

void IntTupleAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  printIntTuple(odsPrinter, getRef().asRef());
}

static ParseResult parseInt32Array(::mlir::AsmParser &odsParser, SmallVectorImpl<int32_t> &values) {
  if (odsParser.parseLSquare())
    return failure();
  if (odsParser.parseOptionalRSquare().succeeded())
    return success();
  do {
    int32_t value;
    if (odsParser.parseInteger(value))
      return failure();
    values.push_back(value);
  } while (odsParser.parseOptionalComma().succeeded());
  return odsParser.parseRSquare();
}

static void printModes(AsmPrinter &printer, ArrayRef<uint8_t> modes) {
  printer << "[";
  llvm::interleaveComma(modes, printer, [&](uint8_t mode) { printer << unsigned(mode); });
  printer << "]";
}

static ::mlir::Attribute parseCoordSwizzleBody(::mlir::AsmParser &odsParser, MLIRContext *ctx) {
  int32_t mask, baseRow, baseCol;
  SmallVector<int32_t> modeRow;
  SmallVector<int32_t> modeCol;
  if (odsParser.parseLess() || odsParser.parseInteger(mask) || odsParser.parseComma() ||
      odsParser.parseInteger(baseRow) || odsParser.parseComma() ||
      parseInt32Array(odsParser, modeRow) || odsParser.parseComma() ||
      odsParser.parseInteger(baseCol) || odsParser.parseComma() ||
      parseInt32Array(odsParser, modeCol) || odsParser.parseGreater()) {
    return {};
  }
  return CoordSwizzleAttr::get(ctx, mask, baseRow, modeRow, baseCol, modeCol);
}

::mlir::Attribute CoordSwizzleAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  auto *ctx = odsParser.getBuilder().getContext();
  if (odsParser.parseKeyword("CS"))
    return {};
  return parseCoordSwizzleBody(odsParser, ctx);
}

static void printCoordSwizzle(AsmPrinter &printer, const core::CoordSwizzle<core::Trait> &value) {
  printer << "CS<" << value.mask() << "," << value.rowBase() << ",";
  printModes(printer, value.rowModes());
  printer << "," << value.colBase() << ",";
  printModes(printer, value.colModes());
  printer << ">";
}

void CoordSwizzleAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  printCoordSwizzle(odsPrinter, getRef());
}

/// A tile mode: `[mode|...]`, `shape:stride`, or a leaf that is neither a basis
/// nor a ratio.
static FailureOr<core::Tile<core::Trait>> parseTileMode(AsmParser &parser) {
  using Tile = core::Tile<core::Trait>;
  if (parser.parseOptionalLSquare().succeeded()) {
    SmallVector<Tile> children;
    do {
      auto child = parseTileMode(parser);
      if (failed(child))
        return failure();
      children.push_back(std::move(*child));
    } while (parser.parseOptionalVerticalBar().succeeded());
    if (parser.parseRSquare())
      return failure();
    return Tile::getTuple(children);
  }
  auto shape = parseIntTuple(parser);
  if (failed(shape))
    return failure();
  if (succeeded(parser.parseOptionalColon())) {
    auto stride = parseIntTuple(parser);
    if (failed(stride))
      return failure();
    auto layout = makeParsedLayout(parser, std::move(*shape), std::move(*stride));
    if (failed(layout))
      return failure();
    return Tile(*layout);
  }
  if (!shape->isLeaf() || shape->getLeaf().isBasis() || shape->getLeaf().isRatio())
    return failure();
  auto leaf = shape->getLeaf();
  return leaf.isNone() ? Tile::getNone() : Tile(leaf);
}

::mlir::Attribute TileAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type) {
  auto tile = parseTileMode(odsParser);
  if (failed(tile))
    return {};
  auto result = TileAttr::get(odsParser.getContext(), std::move(*tile));
  if (!result)
    odsParser.emitError(odsParser.getCurrentLocation(), "invalid tile or tree capacity exceeded");
  return result;
}

static void printLayout(AsmPrinter &printer, const core::Layout<core::Trait> &layout) {
  printIntTuple(printer, layout.shape());
  printer << ":";
  printIntTuple(printer, layout.stride());
}

static void printTile(AsmPrinter &printer, const core::Tile<core::Trait> &tile) {
  if (tile.isLayout()) {
    printLayout(printer, tile.getLayout());
  } else if (tile.isNone()) {
    printer << "*";
  } else if (tile.isScalar()) {
    printLeafBody(printer, tile.getScalar());
  } else {
    printer << "[";
    llvm::interleave(
        tile, [&](const core::Tile<core::Trait> &child) { printTile(printer, child); },
        [&] { printer << "|"; });
    printer << "]";
  }
}

void TileAttr::print(::mlir::AsmPrinter &odsPrinter) const { printTile(odsPrinter, getRef()); }

static void printComposedFlat(AsmPrinter &printer,
                              const core::ComposedLayout<core::Trait> &composed) {
  std::visit(
      [&](const auto &inner) {
        using Inner = std::decay_t<decltype(inner)>;
        if constexpr (std::is_same_v<Inner, core::ComposedLayout<core::Trait>>)
          printComposedFlat(printer, inner);
        else if constexpr (std::is_same_v<Inner, core::Swizzle<core::Trait>>)
          printer << "S<" << inner.mask() << "," << inner.base() << "," << inner.shift() << ">";
        else if constexpr (std::is_same_v<Inner, core::CoordSwizzle<core::Trait>>)
          printCoordSwizzle(printer, inner);
        else
          printLayout(printer, inner);
      },
      composed.inner());
  printer << " o ";
  printIntTuple(printer, composed.offset());
  printer << " o ";
  std::visit(
      [&](const auto &outer) {
        if constexpr (std::is_same_v<std::decay_t<decltype(outer)>,
                                     core::ComposedLayout<core::Trait>>) {
          printer << "[";
          printComposedFlat(printer, outer);
          printer << "]";
        } else {
          printLayout(printer, outer);
        }
      },
      composed.outer());
}

::mlir::Attribute ComposedLayoutAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  auto *ctx = odsParser.getBuilder().getContext();
  Attribute inner;
  if (odsParser.parseOptionalKeyword("S").succeeded()) {
    int32_t mask, base, shift;
    if (odsParser.parseLess() || odsParser.parseInteger(mask) || odsParser.parseComma() ||
        odsParser.parseInteger(base) || odsParser.parseComma() || odsParser.parseInteger(shift) ||
        odsParser.parseGreater())
      return {};
    if (!core::Swizzle<core::Trait>::isValid(mask, base, shift)) {
      odsParser.emitError(odsParser.getCurrentLocation(), "invalid swizzle parameters");
      return {};
    }
    inner = SwizzleAttr::get(ctx, mask, base, shift);
  } else if (odsParser.parseOptionalKeyword("CS").succeeded()) {
    inner = parseCoordSwizzleBody(odsParser, ctx);
    if (!inner)
      return {};
  } else {
    auto layout = parseLayout(odsParser);
    if (failed(layout))
      return {};
    inner = LayoutAttr::get(ctx, std::move(*layout));
  }

  while (odsParser.parseOptionalKeyword("o").succeeded()) {
    auto offset = parseIntTuple(odsParser);
    if (failed(offset) || odsParser.parseKeyword("o"))
      return {};

    Attribute outer;
    if (odsParser.parseOptionalLSquare().succeeded()) {
      outer = ComposedLayoutAttr::parse(odsParser, odsType);
      if (!outer || !isa<LayoutAttr, ComposedLayoutAttr>(outer) || odsParser.parseRSquare())
        return {};
    } else {
      auto layout = parseLayout(odsParser);
      if (failed(layout))
        return {};
      outer = LayoutAttr::get(ctx, std::move(*layout));
    }
    inner = ComposedLayoutAttr::get(ctx, inner, IntTupleAttr::get(ctx, std::move(*offset)), outer);
  }
  return inner;
}

void ComposedLayoutAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  printComposedFlat(odsPrinter, getRef());
}

} // namespace mlir::fly
