//===- FlirTypes.cpp - Flir Type Implementation -------------------------===//

#include "flir/FlirDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>

using namespace mlir;
using namespace mlir::flir;

namespace {
static int computeRankFromPattern(Attribute pat) {
  if (!pat)
    return -1;
  int rank = 0;
  std::function<void(Attribute)> rec = [&](Attribute a) {
    if (auto arr = dyn_cast<ArrayAttr>(a)) {
      for (auto e : arr.getValue())
        rec(e);
      return;
    }
    ++rank;
  };
  rec(pat);
  return rank;
}

static void printPattern(AsmPrinter &printer, Attribute pat) {
  std::function<void(Attribute)> rec = [&](Attribute a) {
    if (!a) {
      printer << "?";
      return;
    }
    if (auto arr = dyn_cast<ArrayAttr>(a)) {
      printer << "(";
      bool first = true;
      for (auto e : arr.getValue()) {
        if (!first)
          printer << ",";
        rec(e);
        first = false;
      }
      printer << ")";
      return;
    }
    if (isa<UnderscoreAttr>(a)) {
      printer << "*";
      return;
    }
    if (auto d64 = dyn_cast<DyncI64Attr>(a)) {
      if (d64.getDivisibility() == 1)
        printer << "?";
      else
        printer << "?{div=" << d64.getDivisibility() << "}";
      return;
    }
    if (auto d32 = dyn_cast<DyncI32Attr>(a)) {
      if (d32.getDivisibility() == 1)
        printer << "?";
      else
        printer << "?{div=" << d32.getDivisibility() << "}";
      return;
    }
    if (auto i = dyn_cast<IntegerAttr>(a)) {
      printer << i.getInt();
      return;
    }
    // Fallback: keep assembly parseable.
    printer << "?";
  };
  rec(pat);
}

static ParseResult parsePatternElem(AsmParser &parser, Attribute &out, int32_t &dyncIdx);

static ParseResult parsePatternTupleAfterLParen(AsmParser &parser, Attribute &out,
                                                int32_t &dyncIdx) {
  MLIRContext *ctx = parser.getContext();
  SmallVector<Attribute, 8> elems;

  if (succeeded(parser.parseOptionalRParen())) {
    out = ArrayAttr::get(ctx, elems);
    return success();
  }

  while (true) {
    Attribute e;
    if (failed(parsePatternElem(parser, e, dyncIdx)))
      return failure();
    elems.push_back(e);
    if (succeeded(parser.parseOptionalComma()))
      continue;
    break;
  }

  if (failed(parser.parseRParen()))
    return failure();

  out = ArrayAttr::get(ctx, elems);
  return success();
}

static ParseResult parsePatternElem(AsmParser &parser, Attribute &out, int32_t &dyncIdx) {
  MLIRContext *ctx = parser.getContext();
  Builder b(ctx);

  if (succeeded(parser.parseOptionalLParen()))
    return parsePatternTupleAfterLParen(parser, out, dyncIdx);

  if (succeeded(parser.parseOptionalStar())) {
    out = UnderscoreAttr::get(ctx);
    return success();
  }

  if (succeeded(parser.parseOptionalQuestion())) {
    int32_t divisibility = 1;
    if (succeeded(parser.parseOptionalLBrace())) {
      if (failed(parser.parseKeyword("div")) || failed(parser.parseEqual()) ||
          failed(parser.parseInteger(divisibility)) || failed(parser.parseRBrace()))
        return failure();
    }
    out = DyncI64Attr::get(ctx, dyncIdx++, divisibility);
    return success();
  }

  int64_t v = 0;
  if (failed(parser.parseInteger(v)))
    return failure();
  out = IntegerAttr::get(b.getIntegerType(64), APInt(64, static_cast<uint64_t>(v), true));
  return success();
}
} // namespace

//===----------------------------------------------------------------------===//
// Rank helpers
//===----------------------------------------------------------------------===//

int ShapeType::getRank() const { return computeRankFromPattern(getPattern()); }
int StrideType::getRank() const { return computeRankFromPattern(getPattern()); }
int CoordType::getRank() const { return computeRankFromPattern(getPattern()); }
int LayoutType::getRank() const { return computeRankFromPattern(getShapePattern()); }

LayoutType LayoutType::get(MLIRContext *ctx, int rank) {
  if (rank < 0)
    return LayoutType::get(ctx, Attribute(), Attribute());
  Attribute pat;
  if (rank == 0) {
    pat = Attribute();
  } else {
    SmallVector<Attribute, 8> elems;
    elems.reserve(rank);
    for (int32_t i = 0; i < rank; ++i)
      elems.push_back(DyncI64Attr::get(ctx, /*dyncElemIdx=*/i, /*divisibility=*/1));
    pat = (rank == 1) ? elems.front() : ArrayAttr::get(ctx, elems);
  }
  return LayoutType::get(ctx, pat, pat);
}

LayoutType LayoutType::get(MLIRContext *ctx, ShapeType shape, StrideType stride) {
  return LayoutType::get(ctx, shape.getPattern(), stride.getPattern());
}

//===----------------------------------------------------------------------===//
// Custom type assembly formats (flyx-aligned patterns)
//===----------------------------------------------------------------------===//

Type ShapeType::parse(AsmParser &parser) {
  Attribute pat;
  int32_t dyncIdx = 0;
  if (failed(parser.parseLess()) || failed(parsePatternElem(parser, pat, dyncIdx)) ||
      failed(parser.parseGreater()))
    return Type();
  return ShapeType::get(parser.getContext(), pat);
}

void ShapeType::print(AsmPrinter &printer) const {
  printer << "<";
  printPattern(printer, getPattern());
  printer << ">";
}

Type StrideType::parse(AsmParser &parser) {
  Attribute pat;
  int32_t dyncIdx = 0;
  if (failed(parser.parseLess()) || failed(parsePatternElem(parser, pat, dyncIdx)) ||
      failed(parser.parseGreater()))
    return Type();
  return StrideType::get(parser.getContext(), pat);
}

void StrideType::print(AsmPrinter &printer) const {
  printer << "<";
  printPattern(printer, getPattern());
  printer << ">";
}

Type CoordType::parse(AsmParser &parser) {
  Attribute pat;
  int32_t dyncIdx = 0;
  if (failed(parser.parseLess()) || failed(parsePatternElem(parser, pat, dyncIdx)) ||
      failed(parser.parseGreater()))
    return Type();
  return CoordType::get(parser.getContext(), pat);
}

void CoordType::print(AsmPrinter &printer) const {
  printer << "<";
  printPattern(printer, getPattern());
  printer << ">";
}

Type LayoutType::parse(AsmParser &parser) {
  Attribute shapePat, stridePat;
  int32_t dyncIdx = 0;
  if (failed(parser.parseLess()) || failed(parsePatternElem(parser, shapePat, dyncIdx)) ||
      failed(parser.parseColon()))
    return Type();
  dyncIdx = 0;
  if (failed(parsePatternElem(parser, stridePat, dyncIdx)) || failed(parser.parseGreater()))
    return Type();
  return LayoutType::get(parser.getContext(), shapePat, stridePat);
}

void LayoutType::print(AsmPrinter &printer) const {
  printer << "<";
  printPattern(printer, getShapePattern());
  printer << ":";
  printPattern(printer, getStridePattern());
  printer << ">";
}

