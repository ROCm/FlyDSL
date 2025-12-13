//===- RocirDialect.cpp - Rocir Dialect Implementation --------------------===//

#include "rocir/RocirDialect.h"
#include "rocir/RocirOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::rocir;

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
  return Base::get(ctx, detail::StructuredTypeStorage::KeyTy{rank, StringRef("")});
}

ShapeType ShapeType::get(MLIRContext *ctx, StringRef spec) {
  // rank is derived by storage from spec; pass a dummy (-1).
  return Base::get(ctx, detail::StructuredTypeStorage::KeyTy{-1, spec});
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
  return Base::get(ctx, detail::StructuredTypeStorage::KeyTy{rank, StringRef("")});
}

int StrideType::getRank() const {
  return getImpl()->rank;
}

StrideType StrideType::get(MLIRContext *ctx, StringRef spec) {
  return Base::get(ctx, detail::StructuredTypeStorage::KeyTy{-1, spec});
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
      // shape<"(...)"> or shape<rank>
      std::string spec;
      int64_t rank = -1;
      if (succeeded(parser.parseOptionalString(&spec))) {
        if (parser.parseGreater())
          return Type();
        return ShapeType::get(ctx, StringRef(spec));
      }
      if (parser.parseInteger(rank) || parser.parseGreater())
        return Type();
      return ShapeType::get(ctx, static_cast<int>(rank));
    }
    return ShapeType::get(ctx, -1);
  }
  
  if (mnemonic == "stride") {
    if (succeeded(parser.parseOptionalLess())) {
      std::string spec;
      int64_t rank = -1;
      if (succeeded(parser.parseOptionalString(&spec))) {
        if (parser.parseGreater())
          return Type();
        return StrideType::get(ctx, StringRef(spec));
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
      os << "shape<\"" << shapeType.getSpec() << "\">";
    } else {
      os << "shape<" << shapeType.getRank() << ">";
    }
  } else if (auto strideType = llvm::dyn_cast<StrideType>(type)) {
    if (!strideType.getSpec().empty()) {
      os << "stride<\"" << strideType.getSpec() << "\">";
    } else {
      os << "stride<" << strideType.getRank() << ">";
    }
  } else if (auto layoutType = llvm::dyn_cast<LayoutType>(type)) {
    os << "layout<" << layoutType.getRank() << ">";
  } else if (auto coordType = llvm::dyn_cast<CoordType>(type)) {
    os << "coord<" << coordType.getRank() << ">";
  }
}
