
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/IntUtils.h"

namespace mlir::fly {

//===----------------------------------------------------------------------===//
// Class Definitions
//===----------------------------------------------------------------------===//

IntTupleAttr IntTupleAttr::getLeafNone(MLIRContext *ctx) { return get(ctx, IntAttr::getNone(ctx)); }
IntTupleAttr IntTupleAttr::getLeafStatic(MLIRContext *ctx, int32_t value) {
  return get(ctx, IntAttr::getStatic(ctx, value));
}
IntTupleAttr IntTupleAttr::getLeafDynamic(MLIRContext *ctx, int32_t width, int32_t divisibility) {
  return get(ctx, IntAttr::getDynamic(ctx, width, divisibility));
}
bool IntTupleAttr::isLeafNone() const {
  if (this->isLeaf()) {
    if (auto intAttr = dyn_cast<IntAttr>(this->getValue())) {
      return intAttr.isNone();
    }
  }
  return false;
}
bool IntTupleAttr::isLeafStaticValue(int32_t value) const {
  if (this->isLeaf()) {
    if (auto intAttr = dyn_cast<IntAttr>(this->getValue())) {
      return intAttr.isStaticValue(value);
    }
  }
  return false;
}

IntAttr IntTupleAttr::getLeafAsInt() const {
  assert(this->isLeaf() && "Non-leaf attribute cannot be converted to IntAttr");
  return cast<IntAttr>(this->getValue());
}
BasisAttr IntTupleAttr::getLeafAsBasis() const {
  assert(this->isLeaf() && "Non-leaf attribute cannot be converted to BasisAttr");
  return cast<BasisAttr>(this->getValue());
}

int32_t IntTupleAttr::dyncLeafCount() const {
  if (this->isLeaf()) {
    return this->isStatic() ? 0 : 1;
  }
  int32_t count = 0;
  for (int32_t i = 0; i < this->rank(); ++i) {
    count += this->at(i).dyncLeafCount();
  }
  return count;
}

//===----------------------------------------------------------------------===//
// Interface methods
//===----------------------------------------------------------------------===//

bool IntAttr::isStatic() const { return getValue() != std::numeric_limits<int32_t>::min(); }

bool BasisAttr::isStatic() const { return cast<IntAttr>(getValue()).isStatic(); }

int32_t BasisAttr::depth() { return static_cast<int32_t>(getModes().size()); }

bool IntTupleAttr::isLeaf() const { return !isa<ArrayAttr>(getValue()); }

bool IntTupleAttr::isStatic() const {
  if (auto tupleAttr = dyn_cast<ArrayAttr>(this->getValue())) {
    for (int i = 0; i < rank(); ++i) {
      if (!at(i).isStatic()) {
        return false;
      }
    }
    return true;
  } else if (auto basisAttr = dyn_cast<BasisAttr>(getValue())) {
    return basisAttr.isStatic();
  } else if (auto intAttr = dyn_cast<IntAttr>(getValue())) {
    return intAttr.isStatic();
  }
  return true;
}

int32_t IntTupleAttr::rank() const {
  if (auto tupleAttr = dyn_cast<ArrayAttr>(this->getValue())) {
    return tupleAttr.size();
  }
  return 1;
}
int32_t IntTupleAttr::rank(int32_t idx) const {
  if (auto tupleAttr = dyn_cast<ArrayAttr>(this->getValue())) {
    return cast<IntTupleAttr>(tupleAttr[idx]).rank();
  }
  assert(idx == 0);
  return 1;
}
int32_t IntTupleAttr::rank(ArrayRef<int32_t> idxs) const {
  IntTupleAttr result = *this;
  for (int32_t idx : idxs) {
    result = result.at(idx);
  }
  return result.rank();
}

int32_t IntTupleAttr::depth() const {
  if (auto tupleAttr = dyn_cast<ArrayAttr>(this->getValue())) {
    int maxLeafDepth = at(0).depth();
    for (int i = 1; i < rank(); ++i) {
      maxLeafDepth = std::max(maxLeafDepth, at(i).depth());
    }
    return 1 + maxLeafDepth;
  }
  return 0;
}
int32_t IntTupleAttr::depth(int32_t idx) const {
  if (auto tupleAttr = dyn_cast<ArrayAttr>(this->getValue())) {
    return cast<IntTupleAttr>(tupleAttr[idx]).depth();
  }
  assert(idx == 0);
  return 0;
}
int32_t IntTupleAttr::depth(ArrayRef<int32_t> idxs) const {
  IntTupleAttr result = *this;
  for (int32_t idx : idxs) {
    result = result.at(idx);
  }
  return result.depth();
}

IntTupleAttr IntTupleAttr::at(int32_t idx) const {
  if (auto tupleAttr = dyn_cast<ArrayAttr>(this->getValue())) {
    return cast<IntTupleAttr>(tupleAttr[idx]);
  }
  assert(idx == 0 && "Index out of bounds for non-array pattern");
  return *this;
}
IntTupleAttr IntTupleAttr::at(ArrayRef<int32_t> idxs) const {
  IntTupleAttr result = *this;
  for (int32_t idx : idxs) {
    result = result.at(idx);
  }
  return result;
}

bool LayoutAttr::isStatic() const { return getShape().isStatic() && getStride().isStatic(); }

bool LayoutAttr::isStaticShape() const { return getShape().isStatic(); }

bool LayoutAttr::isStaticStride() const { return getStride().isStatic(); }

bool LayoutAttr::isLeaf() const { return getShape().isLeaf(); }

int32_t LayoutAttr::rank() const { return getShape().rank(); }
int32_t LayoutAttr::rank(int32_t idx) const { return getShape().rank(idx); }
int32_t LayoutAttr::rank(ArrayRef<int32_t> idxs) const { return getShape().rank(idxs); }

int32_t LayoutAttr::depth() const { return getShape().depth(); }
int32_t LayoutAttr::depth(int32_t idx) const { return getShape().depth(idx); }
int32_t LayoutAttr::depth(ArrayRef<int32_t> idxs) const { return getShape().depth(idxs); }

LayoutAttr LayoutAttr::at(int32_t idx) const {
  return LayoutAttr::get(getContext(), getShape().at(idx), getStride().at(idx));
}
LayoutAttr LayoutAttr::at(ArrayRef<int32_t> idxs) const {
  return LayoutAttr::get(getContext(), getShape().at(idxs), getStride().at(idxs));
}

bool ComposedLayoutAttr::isStatic() const {
  return isStaticOuter() && isStaticOffset() && isStaticInner();
}
bool ComposedLayoutAttr::isStaticOuter() const { return getOuter().isStatic(); }
bool ComposedLayoutAttr::isStaticOffset() const { return getOffset().isStatic(); }
bool ComposedLayoutAttr::isStaticInner() const {
  if (auto inner = dyn_cast<ComposedLayoutAttr>(getInner())) {
    return inner.isStatic();
  } else if (auto layout = dyn_cast<LayoutAttr>(getInner())) {
    return layout.isStatic();
  } else if (auto basis = dyn_cast<SwizzleAttr>(getInner())) {
    return true;
  } else {
    assert(false && "invalid InnerAttr of ComposedLayoutAttr");
    return false;
  }
}

bool ComposedLayoutAttr::isLeaf() const { return getOuter().isLeaf(); }

int32_t ComposedLayoutAttr::rank() const { return getOuter().rank(); }
int32_t ComposedLayoutAttr::rank(int32_t idx) const { return getOuter().rank(idx); }
int32_t ComposedLayoutAttr::rank(ArrayRef<int32_t> idxs) const { return getOuter().rank(idxs); }
int32_t ComposedLayoutAttr::depth() const { return getOuter().depth(); }
int32_t ComposedLayoutAttr::depth(int32_t idx) const { return getOuter().depth(idx); }
int32_t ComposedLayoutAttr::depth(ArrayRef<int32_t> idxs) const { return getOuter().depth(idxs); }

ComposedLayoutAttr ComposedLayoutAttr::at(int32_t idx) const {
  return ComposedLayoutAttr::get(getContext(), getInner(), getOffset(), getOuter().at(idx));
}
ComposedLayoutAttr ComposedLayoutAttr::at(ArrayRef<int32_t> idxs) const {
  return ComposedLayoutAttr::get(getContext(), getInner(), getOffset(), getOuter().at(idxs));
}

int32_t TileAttr::rank() const {
  if (auto arrayAttr = dyn_cast<ArrayAttr>(this->getValue())) {
    return arrayAttr.size();
  }
  assert(false && "invalid TileAttr");
  return 0;
}

bool TileAttr::isLeaf() const { return !isa<ArrayAttr>(this->getValue()); }
Attribute TileAttr::at(int32_t idx) const { return cast<ArrayAttr>(this->getValue())[idx]; }
bool TileAttr::isNoneMode() const {
  if (!isLeaf())
    return false;
  if (auto intAttr = dyn_cast<IntAttr>(this->getValue()))
    return intAttr.isNone();
  return false;
}
bool TileAttr::isNoneMode(int32_t idx) const {
  if (auto intAttr = dyn_cast<IntAttr>(at(idx)))
    return intAttr.isNone();
  return false;
}

//===----------------------------------------------------------------------===//
// Parser and Printer
//===----------------------------------------------------------------------===//

void prettyPrintIntAttr(::mlir::AsmPrinter &odsPrinter, IntAttr attr) {
  if (attr.isStatic()) {
    odsPrinter << attr.getValue();
  } else {
    odsPrinter << "?";
    if (attr.getWidth() != 32 || attr.getDivisibility() != 1) {
      odsPrinter << "{";
      bool delimiter = false;
      if (attr.getWidth() != 32) {
        odsPrinter << "i" << attr.getWidth();
        delimiter = true;
      }
      if (attr.getDivisibility() != 1) {
        if (delimiter) {
          odsPrinter << " ";
        }
        odsPrinter << "div=" << attr.getDivisibility();
      }
      odsPrinter << "}";
    }
  }
}

::mlir::Attribute IntAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  auto *ctx = odsParser.getBuilder().getContext();

  if (odsParser.parseOptionalQuestion().succeeded()) {
    int32_t width = 32;
    int32_t divisibility = 1;
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
      if (odsParser.parseRBrace())
        return {};
    }
    return IntAttr::getDynamic(ctx, width, divisibility);
  }
  int32_t value;
  if (odsParser.parseDecimalInteger(value))
    return {};
  return IntAttr::getStatic(ctx, value);
}

void IntAttr::print(::mlir::AsmPrinter &odsPrinter) const { prettyPrintIntAttr(odsPrinter, *this); }

::mlir::Attribute parseLeafAttr(::mlir::AsmParser &odsParser) {
  auto *ctx = odsParser.getBuilder().getContext();

  Attribute valueAttr;
  if (odsParser.parseOptionalStar().succeeded()) {
    valueAttr = IntAttr::getNone(ctx);
  } else if (odsParser.parseOptionalQuestion().succeeded()) {
    int32_t width = 32;
    int32_t divisibility = 1;
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
      if (odsParser.parseRBrace())
        return {};
    }
    valueAttr = IntAttr::getDynamic(ctx, width, divisibility);
  } else {
    int32_t value;
    if (odsParser.parseDecimalInteger(value))
      return {};
    valueAttr = IntAttr::getStatic(ctx, value);
  }

  SmallString<16> strModes;
  StringRef strRefModes;
  if (odsParser.parseOptionalKeyword(&strRefModes))
    return valueAttr;

  SmallVector<int32_t> modes;
  SmallVector<StringRef, 8> strRefModeList;

  strRefModes.consume_front("E");
  strRefModes.split(strRefModeList, "E");
  for (StringRef strRefMode : strRefModeList) {
    int32_t mode;
    if (strRefMode.getAsInteger(10, mode))
      return {};
    modes.push_back(mode);
  }
  return BasisAttr::get(ctx, valueAttr, modes);
}

::mlir::Attribute BasisAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  auto valueAttr = parseLeafAttr(odsParser);
  if (!isa<BasisAttr>(valueAttr))
    return {};
  return valueAttr;
}

void BasisAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  if (auto intAttr = dyn_cast<IntAttr>(this->getValue())) {
    prettyPrintIntAttr(odsPrinter, intAttr);
  } else {
    llvm_unreachable("invalid BasisAttr value");
  }
  for (int32_t mode : getModes())
    odsPrinter << "E" << mode;
}

::mlir::Attribute IntTupleAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  auto *ctx = odsParser.getBuilder().getContext();
  if (odsParser.parseOptionalLParen().succeeded()) {
    SmallVector<Attribute> elements;
    do {
      elements.push_back(IntTupleAttr::parse(odsParser, odsType));
    } while (odsParser.parseOptionalComma().succeeded());
    if (odsParser.parseRParen())
      return {};
    return IntTupleAttr::get(ArrayAttr::get(ctx, elements));
  } else {
    return IntTupleAttr::get(parseLeafAttr(odsParser));
  }
}

void IntTupleAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  if (auto tupleAttr = dyn_cast<ArrayAttr>(this->getValue())) {
    odsPrinter << "(";
    at(0).print(odsPrinter);
    for (int i = 1; i < rank(); ++i) {
      odsPrinter << ",";
      at(i).print(odsPrinter);
    }
    odsPrinter << ")";
  } else {
    ::llvm::TypeSwitch<Attribute>(this->getValue())
        .Case<IntAttr>([&](IntAttr attr) {
          if (attr.isNone()) {
            odsPrinter << "*";
          } else {
            prettyPrintIntAttr(odsPrinter, attr);
          }
        })
        .Case<BasisAttr>([&](BasisAttr attr) { attr.print(odsPrinter); })
        .DefaultUnreachable("invalid LeafAttr");
  }
}

::mlir::Attribute TileAttr::parse(::mlir::AsmParser &odsParser, ::mlir::Type odsType) {
  auto *ctx = odsParser.getBuilder().getContext();
  auto parseElement = [&]() -> Attribute {
    auto shapeAttr = IntTupleAttr::parse(odsParser, odsType);
    if (!shapeAttr)
      return {};
    auto shape = cast<IntTupleAttr>(shapeAttr);
    if (odsParser.parseOptionalColon().succeeded()) {
      auto strideAttr = IntTupleAttr::parse(odsParser, odsType);
      if (!strideAttr)
        return {};
      auto stride = cast<IntTupleAttr>(strideAttr);
      return LayoutAttr::get(ctx, shape, stride);
    }
    if (!shape.isLeaf())
      return {};
    Attribute leaf = shape.getValue();
    if (isa<IntAttr>(leaf))
      return leaf;
    return {};
  };

  if (odsParser.parseOptionalLSquare().succeeded()) {
    SmallVector<Attribute> elements;
    do {
      Attribute elem = parseElement();
      if (!elem)
        return {};
      elements.push_back(elem);
    } while (odsParser.parseOptionalVerticalBar().succeeded());
    if (odsParser.parseRSquare())
      return {};
    return TileAttr::get(ArrayAttr::get(ctx, elements));
  } else {
    Attribute elem = parseElement();
    if (!elem)
      return {};
    return TileAttr::get(elem);
  }
}

void TileAttr::print(::mlir::AsmPrinter &odsPrinter) const {
  auto elemPrint = [&](Attribute attr) {
    ::llvm::TypeSwitch<Attribute>(attr)
        .Case<IntAttr>([&](IntAttr attr) {
          if (attr.isNone()) {
            odsPrinter << "*";
          } else {
            prettyPrintIntAttr(odsPrinter, attr);
          }
        })
        .Case<LayoutAttr>([&](LayoutAttr attr) { attr.print(odsPrinter); })
        .DefaultUnreachable("invalid LayoutAttr");
  };
  if (isLeaf()) {
    elemPrint(this->getValue());
    return;
  }
  odsPrinter << "[";
  elemPrint(this->at(0));
  for (int i = 1; i < this->rank(); ++i) {
    odsPrinter << "|";
    elemPrint(this->at(i));
  }
  odsPrinter << "]";
}

} // namespace mlir::fly
