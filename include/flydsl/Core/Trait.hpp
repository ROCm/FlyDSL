// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_CORE_TRAIT_HPP
#define FLYDSL_CORE_TRAIT_HPP

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"

#include "flydsl/Core/Algebra.hpp"
#include "flydsl/Core/Visitor.hpp"

#include <cstdint>
#include <optional>

namespace mlir::fly::core {

/// Static type-inference category.
struct Trait {};

//===----------------------------------------------------------------------===//
// Leaf<Trait>
//===----------------------------------------------------------------------===//

template <> struct Leaf<Trait> : public LeafProperty {
public:
  using Cat = Trait;

  ///
  /// Constructors
  ///
  Leaf() : LeafProperty{} {}
  Leaf(const LeafProperty &property) : LeafProperty(property) {}

  static Leaf getSInt(int64_t value);
  static Leaf getDInt(LogWidth logWidth = LogWidth::I32, int32_t div = 1);
  static Leaf getRatio(int32_t num, int32_t den);
  static Leaf getNone();
  static Leaf getError(ErrorInfo info);
  static Leaf getBasis(Leaf scale, llvm::ArrayRef<int32_t> modes);

  static std::optional<Leaf> fromString(llvm::StringRef text);

  static Leaf getOne() { return getSInt(1); }
  static Leaf getZero() { return getSInt(0); }

  /// Basis accessors returning the concrete category.
  Leaf scale() const;

  ///
  /// Modifiers
  ///
  Leaf withModes(Leaf proto) const;

  Leaf withFastDivMod() const;
  /// Returns this leaf unchanged when range inference is disabled.
  Leaf withRange(int32_t lo, int32_t hi) const;
};

//===----------------------------------------------------------------------===//
// Pointer<Trait>
//===----------------------------------------------------------------------===//

template <> struct Pointer<Trait> : public PointerProperty {
public:
  using PointerProperty::PointerProperty;
  explicit Pointer(PointerProperty property) : PointerProperty(std::move(property)) {}

  Swizzle<Trait> swizzle() const { return PointerProperty::swizzle<Trait>(); }
};

#define FLYDSL_CORE_CATEGORY Trait
#include "flydsl/Core/Category.def"

llvm::hash_code hash_value(Leaf<Trait> leaf);

/// Forget evaluation payloads, retaining structure, complete LeafProperty and
/// opaque metadata. Owning results never borrow storage from the source Cat.
template <class T> auto to_trait(const T &value) {
  return transform_value<Trait>(
      value, [](const auto &leaf) { return Leaf<Trait>(leaf); },
      [](const auto &pointer) { return Pointer<Trait>(pointer); });
}

} // namespace mlir::fly::core

#endif // FLYDSL_CORE_TRAIT_HPP
