#ifndef ROCIR_TYPES_H
#define ROCIR_TYPES_H

#include "mlir/IR/Types.h"
#include "mlir/IR/TypeSupport.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringExtras.h"
#include <algorithm>
#include <memory>
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
  /// Key is (rank, spec). If spec is non-empty, rank is derived from parsing
  /// the spec and the provided rank is ignored.
  using KeyTy = std::pair<int, llvm::StringRef>;

  StructuredTypeStorage(int rank,
                        llvm::StringRef spec,
                        llvm::ArrayRef<int32_t> structure,
                        llvm::ArrayRef<int64_t> dims)
      : rank(rank), spec(spec), structure(structure), dims(dims) {}

  bool operator==(const KeyTy &key) const {
    // When spec is present, it uniquely identifies structure+dims.
    if (!spec.empty())
      return spec == key.second;
    return rank == key.first && key.second.empty();
  }

  static StructuredTypeStorage *construct(TypeStorageAllocator &allocator,
                                          const KeyTy &key) {
    int rank = key.first;
    llvm::StringRef spec = key.second;
    llvm::SmallVector<int32_t, 16> structureTmp;
    llvm::SmallVector<int64_t, 16> dimsTmp;

    auto canonicalize = [](llvm::StringRef in) -> std::string {
      std::string out;
      out.reserve(in.size());
      for (char c : in) {
        if (!llvm::isSpace(static_cast<unsigned char>(c)))
          out.push_back(c);
      }
      return out;
    };

    auto parseTupleSpec = [&](llvm::StringRef in) -> bool {
      // Parse a tuple spec like "(9,(4,8))" or "(?,(?,?))".
      // Produces preorder structure encoding and flattened dims (-1 for '?').
      std::string canon = canonicalize(in);
      llvm::StringRef s(canon);
      size_t i = 0;
      auto peek = [&]() -> char { return i < s.size() ? s[i] : '\0'; };
      auto consume = [&](char c) -> bool {
        if (peek() != c) return false;
        ++i;
        return true;
      };

      std::function<bool()> parseElem;
      std::function<bool()> parseTuple;

      parseElem = [&]() -> bool {
        if (peek() == '(') return parseTuple();
        if (consume('?')) {
          structureTmp.push_back(-1);
          dimsTmp.push_back(-1);
          return true;
        }
        bool neg = false;
        if (consume('-')) neg = true;
        if (!llvm::isDigit(static_cast<unsigned char>(peek()))) return false;
        int64_t value = 0;
        while (llvm::isDigit(static_cast<unsigned char>(peek()))) {
          value = value * 10 + (peek() - '0');
          ++i;
        }
        if (neg) value = -value;
        structureTmp.push_back(-1);
        dimsTmp.push_back(value);
        return true;
      };

      parseTuple = [&]() -> bool {
        if (!consume('(')) return false;
        // Empty tuple "()"
        if (consume(')')) {
          structureTmp.push_back(0);
          return true;
        }
        size_t headerIdx = structureTmp.size();
        structureTmp.push_back(0); // placeholder
        int32_t arity = 0;
        while (true) {
          if (!parseElem()) return false;
          ++arity;
          if (consume(',')) continue;
          break;
        }
        if (!consume(')')) return false;
        structureTmp[headerIdx] = arity;
        return true;
      };

      if (!parseTuple()) return false;
      return i == s.size();
    };

    llvm::ArrayRef<int32_t> structure;
    llvm::ArrayRef<int64_t> dims;

    if (!spec.empty()) {
      if (!parseTupleSpec(spec)) {
        // On parse failure, treat as unranked opaque.
        rank = -1;
        structure = {};
        dims = {};
        spec = "";
      } else {
        rank = static_cast<int>(dimsTmp.size());
        std::string canon = canonicalize(spec);
        spec = allocator.copyInto(llvm::StringRef(canon));
        structure = allocator.copyInto(llvm::ArrayRef<int32_t>(structureTmp));
        dims = allocator.copyInto(llvm::ArrayRef<int64_t>(dimsTmp));
      }
    } else {
      // Rank-only type.
      structure = {};
      dims = {};
      spec = "";
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

#endif // ROCIR_TYPES_H

