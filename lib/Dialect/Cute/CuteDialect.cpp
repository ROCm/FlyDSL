#include "cute/CuteDialect.h"
#include "cute/CuteTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::cute;

namespace mlir::cute::detail {

struct IntTypeStorage : public TypeStorage {
  using KeyTy = unsigned; 
  IntTypeStorage() = default;
  bool operator==(const KeyTy &key) const { return true; }
  static IntTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &) {
    return new (allocator.allocate<IntTypeStorage>()) IntTypeStorage();
  }
};

struct RankedTypeStorage : public TypeStorage {
  using KeyTy = int;
  RankedTypeStorage(int rank) : rank(rank) {}
  bool operator==(const KeyTy &key) const { return rank == key; }
  static RankedTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<RankedTypeStorage>()) RankedTypeStorage(key);
  }
  int rank;
};

} // namespace mlir::cute::detail

IntType IntType::get(MLIRContext *ctx) {
  return Base::get(ctx, 0);
}

ShapeType ShapeType::get(MLIRContext *ctx, int rank) {
  return Base::get(ctx, rank);
}

int ShapeType::getRank() const {
  return getImpl()->rank;
}

StrideType StrideType::get(MLIRContext *ctx, int rank) {
  return Base::get(ctx, rank);
}

int StrideType::getRank() const {
  return getImpl()->rank;
}

LayoutType LayoutType::get(MLIRContext *ctx, int rank) {
  return Base::get(ctx, rank);
}

int LayoutType::getRank() const {
  return getImpl()->rank;
}

CoordType CoordType::get(MLIRContext *ctx, int rank) {
  return Base::get(ctx, rank);
}

int CoordType::getRank() const {
  return getImpl()->rank;
}

#include "cute/CuteDialect.cpp.inc"

void CuteDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "cute/CuteOps.cpp.inc"
  >();
  addTypes<IntType, ShapeType, StrideType, LayoutType, CoordType>();
}

Attribute CuteDialect::parseAttribute(DialectAsmParser &parser, Type type) const {
  return Attribute();
}

void CuteDialect::printAttribute(Attribute attr, DialectAsmPrinter &os) const {
}

#define GET_OP_CLASSES
#include "cute/CuteOps.cpp.inc"

Type CuteDialect::parseType(DialectAsmParser &parser) const {
  StringRef mnemonic;
  if (parser.parseKeyword(&mnemonic))
    return Type();

  MLIRContext *ctx = getContext();
  
  if (mnemonic == "int")
    return IntType::get(ctx);
    
  if (mnemonic == "shape" || mnemonic == "stride" || 
      mnemonic == "layout" || mnemonic == "coord") {
    int rank;
    if (parser.parseLess() || parser.parseInteger(rank) || parser.parseGreater())
      return Type();
      
    if (mnemonic == "shape") return ShapeType::get(ctx, rank);
    if (mnemonic == "stride") return StrideType::get(ctx, rank);
    if (mnemonic == "layout") return LayoutType::get(ctx, rank);
    if (mnemonic == "coord") return CoordType::get(ctx, rank);
  }
  
  parser.emitError(parser.getNameLoc(), "unknown type: ") << mnemonic;
  return Type();
}

void CuteDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (llvm::isa<IntType>(type)) {
    printer << "int";
  } else if (auto t = llvm::dyn_cast<ShapeType>(type)) {
    printer << "shape<" << t.getRank() << ">";
  } else if (auto t = llvm::dyn_cast<StrideType>(type)) {
    printer << "stride<" << t.getRank() << ">";
  } else if (auto t = llvm::dyn_cast<LayoutType>(type)) {
    printer << "layout<" << t.getRank() << ">";
  } else if (auto t = llvm::dyn_cast<CoordType>(type)) {
    printer << "coord<" << t.getRank() << ">";
  }
}
