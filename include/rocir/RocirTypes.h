#ifndef ROCIR_TYPES_H
#define ROCIR_TYPES_H

#include "mlir/IR/Types.h"
#include "mlir/IR/TypeSupport.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <algorithm>
#include <memory>
#include <string>
#include <utility>

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

// Storage for structured types (Shape, Stride).
// - rank: number of leaf dimensions (i.e. flattened rank)
// - spec: optional canonical textual spec, e.g. "(9,(4,8))" or "(?,(?,?))"
// - structure: preorder encoding of tuple tree: tuple node -> N children, leaf -> -1
// - dims: optional flattened leaf dims; -1 means dynamic/unknown
struct StructuredTypeStorage : public TypeStorage {
  struct KeyTy {
    int rank;
    llvm::ArrayRef<int32_t> structure;
    llvm::ArrayRef<int64_t> dims;
  };

  StructuredTypeStorage(int rank,
                        llvm::StringRef spec,
                        llvm::ArrayRef<int32_t> structure,
                        llvm::ArrayRef<int64_t> dims)
      : rank(rank), spec(spec), structure(structure), dims(dims) {}

  bool operator==(const KeyTy &key) const {
    return rank == key.rank && structure == key.structure && dims == key.dims;
  }

  static StructuredTypeStorage *construct(TypeStorageAllocator &allocator,
                                          const KeyTy &key) {
    int rank = key.rank;
    llvm::ArrayRef<int32_t> structure = allocator.copyInto(key.structure);
    llvm::ArrayRef<int64_t> dims = allocator.copyInto(key.dims);

    // Derive canonical spec from (structure,dims) for printing. If structure is
    // empty, we treat it as rank-only and leave spec empty.
    llvm::StringRef spec = "";
    if (!structure.empty()) {
      size_t si = 0;
      size_t di = 0;
      std::function<std::string()> emit = [&]() -> std::string {
        if (si >= structure.size())
          return "?";
        int32_t code = structure[si++];
        if (code == -1) {
          int64_t v = (di < dims.size()) ? dims[di++] : -1;
          if (v < 0)
            return "?";
          return std::to_string(v);
        }
        // Tuple node.
        std::string out = "(";
        for (int32_t i = 0; i < code; ++i) {
          if (i) out += ",";
          out += emit();
        }
        out += ")";
        return out;
      };
      std::string s = emit();
      spec = allocator.copyInto(llvm::StringRef(s));
      // If rank was not provided consistently, recompute from dims size.
      rank = static_cast<int>(dims.size());
    }

    return new (allocator.allocate<StructuredTypeStorage>())
        StructuredTypeStorage(rank, spec, structure, dims);
  }

  int rank;
  llvm::StringRef spec;
  llvm::ArrayRef<int32_t> structure;
  llvm::ArrayRef<int64_t> dims;
};

} // namespace detail

} // namespace mlir::rocir

namespace llvm {
template <>
struct DenseMapInfo<mlir::rocir::detail::StructuredTypeStorage::KeyTy> {
  using KeyTy = mlir::rocir::detail::StructuredTypeStorage::KeyTy;
  static inline KeyTy getEmptyKey() {
    return KeyTy{/*rank=*/-2, /*structure=*/{}, /*dims=*/{}};
  }
  static inline KeyTy getTombstoneKey() {
    return KeyTy{/*rank=*/-3, /*structure=*/{}, /*dims=*/{}};
  }
  static unsigned getHashValue(const KeyTy &k) {
    // Note: rank participates even for structured keys (so shape<2> != shape<3> even if empty arrays).
    auto h = llvm::hash_combine(k.rank,
                                llvm::hash_combine_range(k.structure.begin(), k.structure.end()),
                                llvm::hash_combine_range(k.dims.begin(), k.dims.end()));
    return static_cast<unsigned>(h);
  }
  static bool isEqual(const KeyTy &a, const KeyTy &b) {
    return a.rank == b.rank && a.structure == b.structure && a.dims == b.dims;
  }
};
} // namespace llvm

#endif // ROCIR_TYPES_H

