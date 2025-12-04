//===- RocirDialect.cpp - Rocir Dialect Implementation --------------------===//

#include "rocir/RocirDialect.h"
#include "rocir/RocirOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

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
  return Base::get(ctx, rank);
}

int ShapeType::getRank() const {
  return getImpl()->rank;
}

//===----------------------------------------------------------------------===//
// StrideType
//===----------------------------------------------------------------------===//

StrideType StrideType::get(MLIRContext *ctx, int rank) {
  return Base::get(ctx, rank);
}

int StrideType::getRank() const {
  return getImpl()->rank;
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
    
  if (mnemonic.starts_with("shape")) {
    int rank = -1;
    if (mnemonic.size() > 5 && mnemonic[5] == '<') {
      StringRef rankStr = mnemonic.substr(6, mnemonic.size() - 7);
      if (rankStr.getAsInteger(10, rank))
        return Type();
    }
    return ShapeType::get(ctx, rank);
  }
  
  if (mnemonic.starts_with("stride")) {
    int rank = -1;
    if (mnemonic.size() > 6 && mnemonic[6] == '<') {
      StringRef rankStr = mnemonic.substr(7, mnemonic.size() - 8);
      if (rankStr.getAsInteger(10, rank))
        return Type();
    }
    return StrideType::get(ctx, rank);
  }
  
  if (mnemonic.starts_with("layout")) {
    int rank = -1;
    if (mnemonic.size() > 6 && mnemonic[6] == '<') {
      StringRef rankStr = mnemonic.substr(7, mnemonic.size() - 8);
      if (rankStr.getAsInteger(10, rank))
        return Type();
    }
    return LayoutType::get(ctx, rank);
  }
  
  if (mnemonic.starts_with("coord")) {
    int rank = -1;
    if (mnemonic.size() > 5 && mnemonic[5] == '<') {
      StringRef rankStr = mnemonic.substr(6, mnemonic.size() - 7);
      if (rankStr.getAsInteger(10, rank))
        return Type();
    }
    return CoordType::get(ctx, rank);
  }
  
  parser.emitError(parser.getNameLoc(), "unknown rocir type: ") << mnemonic;
  return Type();
}

void RocirDialect::printType(Type type, DialectAsmPrinter &os) const {
  if (auto intType = llvm::dyn_cast<IntType>(type)) {
    os << "int";
  } else if (auto shapeType = llvm::dyn_cast<ShapeType>(type)) {
    os << "shape<" << shapeType.getRank() << ">";
  } else if (auto strideType = llvm::dyn_cast<StrideType>(type)) {
    os << "stride<" << strideType.getRank() << ">";
  } else if (auto layoutType = llvm::dyn_cast<LayoutType>(type)) {
    os << "layout<" << layoutType.getRank() << ">";
  } else if (auto coordType = llvm::dyn_cast<CoordType>(type)) {
    os << "coord<" << coordType.getRank() << ">";
  }
}
