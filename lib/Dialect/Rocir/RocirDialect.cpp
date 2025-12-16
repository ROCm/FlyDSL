//===- RocirDialect.cpp - Rocir Dialect Implementation --------------------===//

#include "rocir/RocirDialect.h"
#include "rocir/RocirOps.h" // Required for generated RocirOps.cpp.inc op class references.
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include <functional>
#include <string>

using namespace mlir;
using namespace mlir::rocir;

namespace {
/// Strip all ASCII whitespace to canonicalize specs.
static std::string canonicalizeSpec(StringRef spec) {
  std::string out;
  out.reserve(spec.size());
  for (char c : spec) {
    if (!llvm::isSpace(static_cast<unsigned char>(c)))
      out.push_back(c);
  }
  return out;
}

struct ParsedTupleSpec {
  llvm::SmallVector<int32_t, 16> structure;
  llvm::SmallVector<int64_t, 16> dims;
};

/// Parse a tuple spec like "(9,(4,8))" or "(?,(?,?))".
/// Produces:
/// - structure: preorder encoding (tuple -> N, leaf -> -1)
/// - dims: flattened leaf dims (int64), with -1 for '?'
static FailureOr<ParsedTupleSpec> parseTupleSpec(StringRef spec) {
  ParsedTupleSpec parsed;
  std::string canon = canonicalizeSpec(spec);
  StringRef s(canon);
  size_t i = 0;

  auto fail = [&]() -> FailureOr<ParsedTupleSpec> { return failure(); };
  auto peek = [&]() -> char { return i < s.size() ? s[i] : '\0'; };
  auto consume = [&](char c) -> bool {
    if (peek() != c)
      return false;
    ++i;
    return true;
  };

  std::function<LogicalResult()> parseElem;
  std::function<LogicalResult()> parseTuple;

  parseElem = [&]() -> LogicalResult {
    if (peek() == '(')
      return parseTuple();

    if (consume('?')) {
      parsed.structure.push_back(-1);
      parsed.dims.push_back(-1);
      return success();
    }

    bool neg = false;
    if (consume('-'))
      neg = true;
    if (!llvm::isDigit(static_cast<unsigned char>(peek())))
      return failure();
    int64_t value = 0;
    while (llvm::isDigit(static_cast<unsigned char>(peek()))) {
      value = value * 10 + (peek() - '0');
      ++i;
    }
    if (neg)
      value = -value;
    parsed.structure.push_back(-1);
    parsed.dims.push_back(value);
    return success();
  };

  parseTuple = [&]() -> LogicalResult {
    if (!consume('('))
      return failure();

    if (consume(')')) {
      parsed.structure.push_back(0);
      return success();
    }

    int32_t arity = 0;
    size_t headerIdx = parsed.structure.size();
    parsed.structure.push_back(0);

    while (true) {
      if (failed(parseElem()))
        return failure();
      ++arity;
      if (consume(','))
        continue;
      break;
    }

    if (!consume(')'))
      return failure();

    parsed.structure[headerIdx] = arity;
    return success();
  };

  if (failed(parseTuple()))
    return fail();
  if (i != s.size())
    return fail();

  return parsed;
}
} // namespace

//===----------------------------------------------------------------------===//
// IntType
//===----------------------------------------------------------------------===//

IntType IntType::get(MLIRContext *ctx) {
  return Base::get(ctx);
}

//===----------------------------------------------------------------------===//
// ShapeType
//===----------------------------------------------------------------------===//

ShapeType ShapeType::get(MLIRContext *ctx, int rank) {
  return Base::get(ctx, detail::StructuredTypeStorage::KeyTy{rank, /*structure=*/{}, /*dims=*/{}});
}

ShapeType ShapeType::get(MLIRContext *ctx, StringRef spec) {
  auto parsed = parseTupleSpec(spec);
  if (failed(parsed))
    return get(ctx, -1);
  return get(ctx, parsed->structure, parsed->dims);
}

ShapeType ShapeType::get(MLIRContext *ctx,
                         ArrayRef<int32_t> structure,
                         ArrayRef<int64_t> dims) {
  return Base::get(ctx, detail::StructuredTypeStorage::KeyTy{static_cast<int>(dims.size()), structure, dims});
}

int ShapeType::getRank() const {
  return getImpl()->rank;
}

ArrayRef<int32_t> ShapeType::getStructure() const {
  return getImpl()->structure;
}

StringRef ShapeType::getSpec() const {
  return getImpl()->spec;
}

ArrayRef<int64_t> ShapeType::getDims() const {
  return getImpl()->dims;
}

//===----------------------------------------------------------------------===//
// StrideType
//===----------------------------------------------------------------------===//

StrideType StrideType::get(MLIRContext *ctx, int rank) {
  return Base::get(ctx, detail::StructuredTypeStorage::KeyTy{rank, /*structure=*/{}, /*dims=*/{}});
}

int StrideType::getRank() const {
  return getImpl()->rank;
}

StrideType StrideType::get(MLIRContext *ctx, StringRef spec) {
  auto parsed = parseTupleSpec(spec);
  if (failed(parsed))
    return get(ctx, -1);
  return get(ctx, parsed->structure, parsed->dims);
}

StrideType StrideType::get(MLIRContext *ctx,
                           ArrayRef<int32_t> structure,
                           ArrayRef<int64_t> dims) {
  return Base::get(ctx, detail::StructuredTypeStorage::KeyTy{static_cast<int>(dims.size()), structure, dims});
}

ArrayRef<int32_t> StrideType::getStructure() const {
  return getImpl()->structure;
}

StringRef StrideType::getSpec() const {
  return getImpl()->spec;
}

ArrayRef<int64_t> StrideType::getDims() const {
  return getImpl()->dims;
}

//===----------------------------------------------------------------------===//
// LayoutType
//===----------------------------------------------------------------------===//

LayoutType LayoutType::get(MLIRContext *ctx, int rank) {
  return Base::get(ctx, detail::LayoutTypeStorage::KeyTy{/*rank=*/rank,
                                                         /*shapeStructure=*/{},
                                                         /*shapeDims=*/{},
                                                         /*strideStructure=*/{},
                                                         /*strideDims=*/{}});
}

LayoutType LayoutType::get(MLIRContext *ctx, ShapeType shape, StrideType stride) {
  // If shape/stride do not carry structure (rank-only), structure/dims will be empty.
  // Keep rank from the flattened rank.
  int rank = shape.getRank();
  if (rank < 0)
    rank = stride.getRank();
  return Base::get(ctx, detail::LayoutTypeStorage::KeyTy{/*rank=*/rank,
                                                        /*shapeStructure=*/shape.getStructure(),
                                                        /*shapeDims=*/shape.getDims(),
                                                        /*strideStructure=*/stride.getStructure(),
                                                        /*strideDims=*/stride.getDims()});
}

LayoutType LayoutType::get(MLIRContext *ctx,
                           ArrayRef<int32_t> shapeStructure,
                           ArrayRef<int64_t> shapeDims,
                           ArrayRef<int32_t> strideStructure,
                           ArrayRef<int64_t> strideDims) {
  int rank = static_cast<int>(shapeDims.size());
  if (rank == 0)
    rank = static_cast<int>(strideDims.size());
  return Base::get(ctx, detail::LayoutTypeStorage::KeyTy{/*rank=*/rank,
                                                        /*shapeStructure=*/shapeStructure,
                                                        /*shapeDims=*/shapeDims,
                                                        /*strideStructure=*/strideStructure,
                                                        /*strideDims=*/strideDims});
}

int LayoutType::getRank() const { return getImpl()->rank; }

StringRef LayoutType::getShapeSpec() const { return getImpl()->shapeSpec; }
ArrayRef<int32_t> LayoutType::getShapeStructure() const { return getImpl()->shapeStructure; }
ArrayRef<int64_t> LayoutType::getShapeDims() const { return getImpl()->shapeDims; }
StringRef LayoutType::getStrideSpec() const { return getImpl()->strideSpec; }
ArrayRef<int32_t> LayoutType::getStrideStructure() const { return getImpl()->strideStructure; }
ArrayRef<int64_t> LayoutType::getStrideDims() const { return getImpl()->strideDims; }

ShapeType LayoutType::getShapeType() const {
  auto *ctx = getContext();
  if (getShapeStructure().empty())
    return ShapeType::get(ctx, getRank());
  return ShapeType::get(ctx, getShapeStructure(), getShapeDims());
}

StrideType LayoutType::getStrideType() const {
  auto *ctx = getContext();
  if (getStrideStructure().empty())
    return StrideType::get(ctx, getRank());
  return StrideType::get(ctx, getStrideStructure(), getStrideDims());
}

//===----------------------------------------------------------------------===//
// CoordType
//===----------------------------------------------------------------------===//

CoordType CoordType::get(MLIRContext *ctx, int rank) {
  return Base::get(ctx, rank);
}

int CoordType::getRank() const {
  return getImpl()->rank;
}

//===----------------------------------------------------------------------===//
// Dialect initialization
//===----------------------------------------------------------------------===//

#include "rocir/RocirDialect.cpp.inc"

void RocirDialect::initialize() {
  addTypes<IntType, ShapeType, StrideType, LayoutType, CoordType>();
  
  addOperations<
#define GET_OP_LIST
#include "rocir/RocirOps.cpp.inc"
  >();
}

Attribute RocirDialect::parseAttribute(DialectAsmParser &parser, Type type) const {
  return Attribute();
}

void RocirDialect::printAttribute(Attribute attr, DialectAsmPrinter &os) const {
}

Type RocirDialect::parseType(DialectAsmParser &parser) const {
  StringRef mnemonic;
  if (parser.parseKeyword(&mnemonic))
    return Type();

  MLIRContext *ctx = getContext();
  
  if (mnemonic == "int")
    return IntType::get(ctx);

  if (mnemonic == "shape") {
    // Optional: shape<...>
    if (succeeded(parser.parseOptionalLess())) {
      // Supported:
      // - shape<"(...)">   (legacy quoted tuple spec)
      // - shape<(...)>     (tuple spec, unquoted)
      // - shape<rank>
      std::string spec;
      int64_t rank = -1;
      if (succeeded(parser.parseOptionalString(&spec))) {
        if (parser.parseGreater())
          return Type();
        return ShapeType::get(ctx, StringRef(spec));
      }

      // Try tuple spec: <( ... )>
      if (succeeded(parser.parseOptionalLParen())) {
        llvm::SmallVector<int32_t, 16> structure;
        llvm::SmallVector<int64_t, 16> dims;

        std::function<ParseResult()> parseElem;
        std::function<ParseResult()> parseTuple;

        parseElem = [&]() -> ParseResult {
          if (succeeded(parser.parseOptionalLParen()))
            return parseTuple();
          if (succeeded(parser.parseOptionalQuestion())) {
            structure.push_back(-1);
            dims.push_back(-1);
            return success();
          }
          int64_t v = 0;
          if (parser.parseInteger(v))
            return failure();
          structure.push_back(-1);
          dims.push_back(v);
          return success();
        };

        parseTuple = [&]() -> ParseResult {
          // We have already consumed '(' for this tuple.
          // Empty tuple: "()"
          if (succeeded(parser.parseOptionalRParen())) {
            structure.push_back(0);
            return success();
          }

          int32_t arity = 0;
          size_t headerIdx = structure.size();
          structure.push_back(0); // placeholder

          while (true) {
            if (failed(parseElem()))
              return failure();
            ++arity;
            if (succeeded(parser.parseOptionalComma()))
              continue;
            break;
          }

          if (parser.parseRParen())
            return failure();
          structure[headerIdx] = arity;
          return success();
        };

        if (failed(parseTuple()))
          return Type();
        if (parser.parseGreater())
          return Type();
        return ShapeType::get(ctx, structure, dims);
      }

      // Rank form
      if (parser.parseInteger(rank) || parser.parseGreater())
        return Type();
      return ShapeType::get(ctx, static_cast<int>(rank));
    }
    return ShapeType::get(ctx, -1);
  }
  
  if (mnemonic == "stride") {
    if (succeeded(parser.parseOptionalLess())) {
      // Supported:
      // - stride<"(...)">   (legacy quoted tuple spec)
      // - stride<(...)>     (tuple spec, unquoted)
      // - stride<rank>
      std::string spec;
      int64_t rank = -1;
      if (succeeded(parser.parseOptionalString(&spec))) {
        if (parser.parseGreater())
          return Type();
        return StrideType::get(ctx, StringRef(spec));
      }

      if (succeeded(parser.parseOptionalLParen())) {
        llvm::SmallVector<int32_t, 16> structure;
        llvm::SmallVector<int64_t, 16> dims;

        std::function<ParseResult()> parseElem;
        std::function<ParseResult()> parseTuple;

        parseElem = [&]() -> ParseResult {
          if (succeeded(parser.parseOptionalLParen()))
            return parseTuple();
          if (succeeded(parser.parseOptionalQuestion())) {
            structure.push_back(-1);
            dims.push_back(-1);
            return success();
          }
          int64_t v = 0;
          if (parser.parseInteger(v))
            return failure();
          structure.push_back(-1);
          dims.push_back(v);
          return success();
        };

        parseTuple = [&]() -> ParseResult {
          if (succeeded(parser.parseOptionalRParen())) {
            structure.push_back(0);
            return success();
          }

          int32_t arity = 0;
          size_t headerIdx = structure.size();
          structure.push_back(0);

          while (true) {
            if (failed(parseElem()))
              return failure();
            ++arity;
            if (succeeded(parser.parseOptionalComma()))
              continue;
            break;
          }

          if (parser.parseRParen())
            return failure();
          structure[headerIdx] = arity;
          return success();
        };

        if (failed(parseTuple()))
          return Type();
        if (parser.parseGreater())
          return Type();
        return StrideType::get(ctx, structure, dims);
      }

      if (parser.parseInteger(rank) || parser.parseGreater())
        return Type();
      return StrideType::get(ctx, static_cast<int>(rank));
    }
    return StrideType::get(ctx, -1);
  }
  
  if (mnemonic == "layout") {
    if (succeeded(parser.parseOptionalLess())) {
      // Supported:
      // - layout<shapeSpec:strideSpec> (tuple specs, unquoted)
      // - layout<(...)>   (tuple rank spec, backward compatible)
      // - layout<rank>
      int64_t rank = -1;
      if (succeeded(parser.parseOptionalLParen())) {
        // Parse the first tuple spec (we already consumed '(').
        llvm::SmallVector<int32_t, 16> shapeStructure;
        llvm::SmallVector<int64_t, 16> shapeDims;

        std::function<ParseResult()> parseElem;
        std::function<ParseResult()> parseTuple;

        parseElem = [&]() -> ParseResult {
          if (succeeded(parser.parseOptionalLParen()))
            return parseTuple();
          if (succeeded(parser.parseOptionalQuestion())) {
            shapeStructure.push_back(-1);
            shapeDims.push_back(-1);
            return success();
          }
          int64_t v = 0;
          if (parser.parseInteger(v))
            return failure();
          shapeStructure.push_back(-1);
          shapeDims.push_back(v);
          return success();
        };

        parseTuple = [&]() -> ParseResult {
          if (succeeded(parser.parseOptionalRParen())) {
            shapeStructure.push_back(0);
            return success();
          }
          int32_t arity = 0;
          size_t headerIdx = shapeStructure.size();
          shapeStructure.push_back(0);
          while (true) {
            if (failed(parseElem()))
              return failure();
            ++arity;
            if (succeeded(parser.parseOptionalComma()))
              continue;
            break;
          }
          if (parser.parseRParen())
            return failure();
          shapeStructure[headerIdx] = arity;
          return success();
        };

        if (failed(parseTuple()))
          return Type();

        // New layout spec form requires ':' followed by another tuple for stride spec.
        if (succeeded(parser.parseOptionalColon())) {
          if (failed(parser.parseLParen()))
            return Type();

          llvm::SmallVector<int32_t, 16> strideStructure;
          llvm::SmallVector<int64_t, 16> strideDims;

          std::function<ParseResult()> parseElem2;
          std::function<ParseResult()> parseTuple2;

          parseElem2 = [&]() -> ParseResult {
            if (succeeded(parser.parseOptionalLParen()))
              return parseTuple2();
            if (succeeded(parser.parseOptionalQuestion())) {
              strideStructure.push_back(-1);
              strideDims.push_back(-1);
              return success();
            }
            int64_t v = 0;
            if (parser.parseInteger(v))
              return failure();
            strideStructure.push_back(-1);
            strideDims.push_back(v);
            return success();
          };

          parseTuple2 = [&]() -> ParseResult {
            if (succeeded(parser.parseOptionalRParen())) {
              strideStructure.push_back(0);
              return success();
            }
            int32_t arity = 0;
            size_t headerIdx = strideStructure.size();
            strideStructure.push_back(0);
            while (true) {
              if (failed(parseElem2()))
                return failure();
              ++arity;
              if (succeeded(parser.parseOptionalComma()))
                continue;
              break;
            }
            if (parser.parseRParen())
              return failure();
            strideStructure[headerIdx] = arity;
            return success();
          };

          if (failed(parseTuple2()) || parser.parseGreater())
            return Type();
          return LayoutType::get(ctx, shapeStructure, shapeDims, strideStructure, strideDims);
        }

        // Backward compatible: layout<(...)> only encodes rank.
        if (parser.parseGreater())
          return Type();
        return LayoutType::get(ctx, static_cast<int>(shapeDims.size()));
      }

      if (parser.parseInteger(rank) || parser.parseGreater())
        return Type();
      return LayoutType::get(ctx, static_cast<int>(rank));
    }
    return LayoutType::get(ctx, -1);
  }
  
  if (mnemonic == "coord") {
    if (succeeded(parser.parseOptionalLess())) {
      // Supported:
      // - coord<(...)>   (tuple rank spec)
      // - coord<rank>
      int64_t rank = -1;
      if (succeeded(parser.parseOptionalLParen())) {
        int leafCount = 0;
        std::function<ParseResult()> parseElem;
        std::function<ParseResult()> parseTuple;

        parseElem = [&]() -> ParseResult {
          if (succeeded(parser.parseOptionalLParen()))
            return parseTuple();
          if (succeeded(parser.parseOptionalQuestion())) {
            ++leafCount;
            return success();
          }
          int64_t v = 0;
          if (parser.parseInteger(v))
            return failure();
          ++leafCount;
          return success();
        };

        parseTuple = [&]() -> ParseResult {
          if (succeeded(parser.parseOptionalRParen()))
            return success();
          while (true) {
            if (failed(parseElem()))
              return failure();
            if (succeeded(parser.parseOptionalComma()))
              continue;
            break;
          }
          if (parser.parseRParen())
            return failure();
          return success();
        };

        if (failed(parseTuple()) || parser.parseGreater())
          return Type();
        return CoordType::get(ctx, leafCount);
      }

      if (parser.parseInteger(rank) || parser.parseGreater())
        return Type();
      return CoordType::get(ctx, static_cast<int>(rank));
    }
    return CoordType::get(ctx, -1);
  }
  
  parser.emitError(parser.getNameLoc(), "unknown rocir type: ") << mnemonic;
  return Type();
}

void RocirDialect::printType(Type type, DialectAsmPrinter &os) const {
  if (auto intType = llvm::dyn_cast<IntType>(type)) {
    os << "int";
  } else if (auto shapeType = llvm::dyn_cast<ShapeType>(type)) {
    if (!shapeType.getSpec().empty()) {
      // Tuple spec: no quotes.
      os << "shape<" << shapeType.getSpec() << ">";
    } else {
      int r = shapeType.getRank();
      if (r >= 0) {
        os << "shape<(";
        for (int i = 0; i < r; ++i) {
          if (i) os << ",";
          os << "?";
        }
        os << ")>";
      } else {
        os << "shape<" << r << ">";
      }
    }
  } else if (auto strideType = llvm::dyn_cast<StrideType>(type)) {
    if (!strideType.getSpec().empty()) {
      // Tuple spec: no quotes.
      os << "stride<" << strideType.getSpec() << ">";
    } else {
      int r = strideType.getRank();
      if (r >= 0) {
        os << "stride<(";
        for (int i = 0; i < r; ++i) {
          if (i) os << ",";
          os << "?";
        }
        os << ")>";
      } else {
        os << "stride<" << r << ">";
      }
    }
  } else if (auto layoutType = llvm::dyn_cast<LayoutType>(type)) {
    // Prefer printing full shape/stride specs when available (Flyx-like).
    if (!layoutType.getShapeSpec().empty() && !layoutType.getStrideSpec().empty()) {
      os << "layout<" << layoutType.getShapeSpec() << ":" << layoutType.getStrideSpec() << ">";
      return;
    }
    int r = layoutType.getRank();
    if (r >= 0) {
      os << "layout<(";
      for (int i = 0; i < r; ++i) {
        if (i) os << ",";
        os << "?";
      }
      os << ")>";
    } else {
      os << "layout<" << r << ">";
    }
  } else if (auto coordType = llvm::dyn_cast<CoordType>(type)) {
    int r = coordType.getRank();
    if (r >= 0) {
      os << "coord<(";
      for (int i = 0; i < r; ++i) {
        if (i) os << ",";
        os << "?";
      }
      os << ")>";
    } else {
      os << "coord<" << r << ">";
    }
  }
}
