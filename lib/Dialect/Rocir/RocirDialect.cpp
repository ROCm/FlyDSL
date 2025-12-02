#include "rocir/RocirDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::rocir;

namespace mlir::rocir::detail {

struct IntTypeStorage : public TypeStorage {
  using KeyTy = unsigned; 
  IntTypeStorage() = default;
  bool operator==(const KeyTy &key) const { return true; }
  static IntTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &) {
    return new (allocator.allocate<IntTypeStorage>()) IntTypeStorage();
  }
};

struct StructureTypeStorage : public TypeStorage {
  using KeyTy = ArrayRef<int32_t>;
  StructureTypeStorage(KeyTy key) : structure(key) {}
  bool operator==(const KeyTy &key) const { return key == structure; }
  static StructureTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<StructureTypeStorage>()) StructureTypeStorage(allocator.copyInto(key));
  }
  ArrayRef<int32_t> structure;
};

struct RankedTypeStorage : public TypeStorage {
  using KeyTy = int;
  RankedTypeStorage(KeyTy key) : rank(key) {}
  bool operator==(const KeyTy &key) const { return key == rank; }
  static RankedTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<RankedTypeStorage>()) RankedTypeStorage(key);
  }
  int rank;
};

} // namespace mlir::rocir::detail

IntType IntType::get(MLIRContext *ctx) {
  return Base::get(ctx, 0);
}

ShapeType ShapeType::get(MLIRContext *ctx, int rank) {
  // Legacy support: create flat tuple structure
  SmallVector<int32_t> structure;
  if (rank == 0) {
    structure.push_back(0); // Empty tuple
  } else {
    structure.push_back(rank);
    for (int i = 0; i < rank; ++i) structure.push_back(-1); // -1 for Leaf
  }
  return Base::get(ctx, structure);
}

ShapeType ShapeType::get(MLIRContext *ctx, ArrayRef<int32_t> structure) {
  return Base::get(ctx, structure);
}

int ShapeType::getRank() const {
  // Count leaves (-1)
  int leaves = 0;
  for (auto s : getImpl()->structure) {
    if (s == -1) leaves++;
  }
  return leaves;
}

ArrayRef<int32_t> ShapeType::getStructure() const {
  return getImpl()->structure;
}

StrideType StrideType::get(MLIRContext *ctx, int rank) {
  SmallVector<int32_t> structure;
  if (rank == 0) {
    structure.push_back(0);
  } else {
    structure.push_back(rank);
    for (int i = 0; i < rank; ++i) structure.push_back(-1);
  }
  return Base::get(ctx, structure);
}

StrideType StrideType::get(MLIRContext *ctx, ArrayRef<int32_t> structure) {
  return Base::get(ctx, structure);
}

int StrideType::getRank() const {
  int leaves = 0;
  for (auto s : getImpl()->structure) {
    if (s == -1) leaves++;
  }
  return leaves;
}

ArrayRef<int32_t> StrideType::getStructure() const {
  return getImpl()->structure;
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

#include "rocir/RocirDialect.cpp.inc"

void RocirDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "rocir/RocirOps.cpp.inc"
  >();
  addTypes<IntType, ShapeType, StrideType, LayoutType, CoordType>();
}

Attribute RocirDialect::parseAttribute(DialectAsmParser &parser, Type type) const {
  return Attribute();
}

void RocirDialect::printAttribute(Attribute attr, DialectAsmPrinter &os) const {
}

#define GET_OP_CLASSES
#include "rocir/RocirOps.cpp.inc"

Type RocirDialect::parseType(DialectAsmParser &parser) const {
  StringRef mnemonic;
  if (parser.parseKeyword(&mnemonic))
    return Type();

  MLIRContext *ctx = getContext();
  
  if (mnemonic == "int")
    return IntType::get(ctx);
    
  if (mnemonic == "shape" || mnemonic == "stride") {
    SmallVector<int32_t> structure;
    if (parser.parseLess()) return Type();
    
    // Parse comma-separated integers
    do {
      int32_t val;
      if (parser.parseInteger(val)) return Type();
      structure.push_back(val);
    } while (succeeded(parser.parseOptionalComma()));
    
    if (parser.parseGreater()) return Type();
    
    // Backward compatibility: if single integer N >= 0, treat as rank (flat tuple)
    // But if N=-1, it is a leaf structure (invalid as top level usually, but maybe).
    if (structure.size() == 1 && structure[0] >= 0) {
       if (mnemonic == "shape") return ShapeType::get(ctx, structure[0]);
       if (mnemonic == "stride") return StrideType::get(ctx, structure[0]);
    }
    
    if (mnemonic == "shape") return ShapeType::get(ctx, structure);
    if (mnemonic == "stride") return StrideType::get(ctx, structure);
  }

  if (mnemonic == "layout" || mnemonic == "coord") {
    int rank;
    if (parser.parseLess() || parser.parseInteger(rank) || parser.parseGreater())
      return Type();
    if (mnemonic == "layout") return LayoutType::get(ctx, rank);
    if (mnemonic == "coord") return CoordType::get(ctx, rank);
  }
  
  parser.emitError(parser.getNameLoc(), "unknown type: ") << mnemonic;
  return Type();
}

void RocirDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (llvm::isa<IntType>(type)) {
    printer << "int";
  } else if (auto t = llvm::dyn_cast<ShapeType>(type)) {
    printer << "shape<";
    auto s = t.getStructure();
    for (size_t i = 0; i < s.size(); ++i) {
      if (i > 0) printer << ", ";
      printer << s[i];
    }
    printer << ">";
  } else if (auto t = llvm::dyn_cast<StrideType>(type)) {
    printer << "stride<";
    auto s = t.getStructure();
    for (size_t i = 0; i < s.size(); ++i) {
      if (i > 0) printer << ", ";
      printer << s[i];
    }
    printer << ">";
  } else if (auto t = llvm::dyn_cast<LayoutType>(type)) {
    printer << "layout<" << t.getRank() << ">";
  } else if (auto t = llvm::dyn_cast<CoordType>(type)) {
    printer << "coord<" << t.getRank() << ">";
  }
}
