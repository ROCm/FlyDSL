//===- RocirDialect.cpp - Rocir Dialect Implementation --------------------===//

#include "rocir/RocirDialect.h"
#include "rocir/RocirOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
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
  return Base::get(ctx, rank);
}

int LayoutType::getRank() const {
  return getImpl()->rank;
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
      // - shape<(...)>     (cute-like unquoted tuple spec)
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
      // - stride<(...)>     (cute-like unquoted tuple spec)
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
      int64_t rank = -1;
      if (parser.parseInteger(rank) || parser.parseGreater())
        return Type();
      return LayoutType::get(ctx, static_cast<int>(rank));
    }
    return LayoutType::get(ctx, -1);
  }
  
  if (mnemonic == "coord") {
    if (succeeded(parser.parseOptionalLess())) {
      int64_t rank = -1;
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
      // Cute-like: no quotes.
      os << "shape<" << shapeType.getSpec() << ">";
    } else {
      os << "shape<" << shapeType.getRank() << ">";
    }
  } else if (auto strideType = llvm::dyn_cast<StrideType>(type)) {
    if (!strideType.getSpec().empty()) {
      // Cute-like: no quotes.
      os << "stride<" << strideType.getSpec() << ">";
    } else {
      os << "stride<" << strideType.getRank() << ">";
    }
  } else if (auto layoutType = llvm::dyn_cast<LayoutType>(type)) {
    os << "layout<" << layoutType.getRank() << ">";
  } else if (auto coordType = llvm::dyn_cast<CoordType>(type)) {
    os << "coord<" << coordType.getRank() << ">";
  }
}
