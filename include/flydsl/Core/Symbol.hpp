// SPDX-FileCopyrightText: Copyright (c) 2026 FlyDSL Project Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLYDSL_CORE_SYMBOL_HPP
#define FLYDSL_CORE_SYMBOL_HPP

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include "flydsl/Core/Algebra.hpp"
#include "flydsl/Core/Trait.hpp"

#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

namespace mlir::fly::core::symbol {

struct SymbolNode;
struct SymbolAccess;

using SymbolExprRef = llvm::IntrusiveRefCntPtr<const SymbolNode>;

} // namespace mlir::fly::core::symbol

namespace llvm {

template <> struct IntrusiveRefCntPtrInfo<const mlir::fly::core::symbol::SymbolNode> {
  static void retain(const mlir::fly::core::symbol::SymbolNode *node);
  static void release(const mlir::fly::core::symbol::SymbolNode *node);
};

} // namespace llvm

namespace mlir::fly::core {

//===----------------------------------------------------------------------===//
// Leaf<Symbol>
//===----------------------------------------------------------------------===//

/// Dynamic integers identified by a symbolic expression.
struct Symbol {};

/// Dual-rail leaf: Trait-stage inference plus a symbolic expression.
///
/// The Trait-stage abstract half is stored in the inherited `LeafProperty`.
/// Static leaves (`isStatic()`) carry no expression: their property is already
/// complete. A dynamic result holds an expression whose operators match
/// `Leaf<Trait>` arithmetic and whose operands are the expressions of the
/// leaves it was computed from; static operands appear in it as constants.
/// Leaves compare by the structure of their expressions.
template <> struct Leaf<Symbol> : public LeafProperty {
public:
  using Cat = Symbol;

  Leaf() : LeafProperty{} {}

  static Leaf getNone() { return Leaf(Leaf<Trait>::getNone(), nullptr); }
  static Leaf getError(ErrorInfo info) { return Leaf(Leaf<Trait>::getError(info), nullptr); }

  static Leaf getSInt(int64_t v) { return Leaf(Leaf<Trait>::getSInt(v), nullptr); }
  static Leaf getOne() { return getSInt(1); }
  static Leaf getZero() { return getSInt(0); }

  /// A fresh symbol, distinct from every other symbol.
  static Leaf getDInt(LogWidth logWidth = LogWidth::I32, int32_t div = 1);

  /// The symbol `name`; symbols with the same name are equal.
  static Leaf getSym(llvm::StringRef name, LogWidth logWidth = LogWidth::I32, int32_t div = 1);

  static Leaf getRatio(int32_t num, int32_t den) {
    return Leaf(Leaf<Trait>::getRatio(num, den), nullptr);
  }

  static Leaf getDyn(llvm::StringRef name, LogWidth logWidth = LogWidth::I32, int32_t div = 1,
                     std::optional<std::pair<int32_t, int32_t>> range = std::nullopt,
                     bool fastDivMod = false);

  static Leaf getBasis(Leaf scale, llvm::ArrayRef<int32_t> modes) {
    return Leaf(Leaf<Trait>::getBasis(Leaf<Trait>(scale), modes), scale.expr_);
  }

  static std::optional<Leaf> fromString(llvm::StringRef text);

  Leaf scale() const { return Leaf(Leaf<Trait>(*this).scale(), expr_); }
  Leaf withModes(Leaf proto) const {
    return Leaf(Leaf<Trait>(*this).withModes(Leaf<Trait>(proto)), expr_);
  }
  Leaf withFastDivMod() const;
  Leaf withRange(int32_t lo, int32_t hi) const;

  /// Static leaves must not carry an expr. Dynamic integer leaves must.
  bool isSound() const;

  friend bool operator==(const Leaf &a, const Leaf &b);
  friend bool operator!=(const Leaf &a, const Leaf &b) { return !(a == b); }

private:
  friend struct symbol::SymbolAccess;

  Leaf(LeafProperty property, symbol::SymbolExprRef expr)
      : LeafProperty(property), expr_(property.isStatic() ? nullptr : std::move(expr)) {}

  symbol::SymbolExprRef expr_;
};

//===----------------------------------------------------------------------===//
// Pointer<Symbol>
//===----------------------------------------------------------------------===//

/// Symbol-stage pointer whose address is a symbolic expression measured in
/// elements. Pointer arithmetic therefore remains symbolic without changing
/// the category-independent tensor algorithms.
template <> struct Pointer<Symbol> : public PointerProperty {
public:
  /// A pointer whose address is the symbol `base`.
  Pointer(OpaqueValue elemType, int32_t elemBitWidth, int32_t byteAlignment, llvm::StringRef base,
          Swizzle<Symbol> swizzle = {}, int64_t bitAlignment = 0,
          OpaqueValue addressSpace = OpaqueValue::getType<void>(),
          int32_t storageElementBitWidth = 0);

  /// The address as a symbolic i64: `base` plus every offset added since.
  Leaf<Symbol> value() const;
  Swizzle<Symbol> swizzle() const { return PointerProperty::swizzle<Symbol>(); }

private:
  friend struct symbol::SymbolAccess;

  Pointer(PointerProperty property, symbol::SymbolExprRef address)
      : PointerProperty(std::move(property)), address_(std::move(address)) {}

  symbol::SymbolExprRef address_;
};

#define FLYDSL_CORE_CATEGORY Symbol
#include "flydsl/Core/Category.def"

} // namespace mlir::fly::core

#endif // FLYDSL_CORE_SYMBOL_HPP
