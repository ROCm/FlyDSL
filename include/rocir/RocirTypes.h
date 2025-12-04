#ifndef ROCIR_TYPES_H
#define ROCIR_TYPES_H

#include "mlir/IR/Types.h"
#include "mlir/IR/TypeSupport.h"

namespace mlir::rocir {

namespace detail {

// Storage for IntType - a simple type without parameters
struct IntTypeStorage : public TypeStorage {
  using KeyTy = int; // Dummy key
  
  IntTypeStorage() = default;
  
  bool operator==(const KeyTy &) const { return true; }
  
  static IntTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &) {
    return new (allocator.allocate<IntTypeStorage>()) IntTypeStorage();
  }
};

// Storage for ranked types (Shape, Stride, Layout, Coord)
struct RankedTypeStorage : public TypeStorage {
  using KeyTy = int; // rank
  
  RankedTypeStorage(int rank) : rank(rank) {}
  
  bool operator==(const KeyTy &key) const { return rank == key; }
  
  static RankedTypeStorage *construct(TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<RankedTypeStorage>()) RankedTypeStorage(key);
  }
  
  int rank;
};

// Alias for compatibility
using StructureTypeStorage = RankedTypeStorage;

} // namespace detail

} // namespace mlir::rocir

#endif // ROCIR_TYPES_H

