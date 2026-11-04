// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_CORE_RUNTIME_HPP
#define FLYDSL_CORE_RUNTIME_HPP

#include "flydsl/Core/Algebra.hpp"
#include "flydsl/Core/Trait.hpp"

#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>

namespace mlir::fly::core {

/// Inference paired with a concrete integer evaluation.
struct Runtime {};

/// A leaf carrying both the static inference and a concrete int64.
template <> struct Leaf<Runtime> : public LeafProperty {
public:
  using Cat = Runtime;

  int64_t value;

  Leaf() : LeafProperty{}, value(0) {}
  Leaf(LeafProperty property, int64_t value) : LeafProperty(property), value(value) {}

  static Leaf getNone() { return Leaf(Leaf<Trait>::getNone(), 0); }
  static Leaf getError(ErrorInfo info) { return Leaf(Leaf<Trait>::getError(info), 0); }

  /// A compile-time-known integer; prints as `_v`.
  static Leaf getSInt(int64_t v) { return Leaf(Leaf<Trait>::getSInt(v), v); }
  static Leaf getOne() { return getSInt(1); }
  static Leaf getZero() { return getSInt(0); }

  static Leaf getDInt(LogWidth logWidth = LogWidth::I32, int32_t div = 1) {
    return Leaf(Leaf<Trait>::getDInt(logWidth, div), 0);
  }

  static Leaf getRatio(int32_t num, int32_t den) {
    return Leaf(Leaf<Trait>::getRatio(num, den), 0);
  }

  static Leaf getDyn(int64_t v, LogWidth logWidth = LogWidth::I32, int32_t div = 1,
                     std::optional<std::pair<int32_t, int32_t>> range = std::nullopt,
                     bool fastDivMod = false);

  static Leaf getBasis(Leaf scale, llvm::ArrayRef<int32_t> modes) {
    return Leaf(Leaf<Trait>::getBasis(Leaf<Trait>(scale), modes), scale.value);
  }

  static std::optional<Leaf> fromString(llvm::StringRef text);

  int64_t staticValue() const { return value; }

  Leaf scale() const { return Leaf(Leaf<Trait>(*this).scale(), value); }
  Leaf withModes(Leaf proto) const {
    return Leaf(Leaf<Trait>(*this).withModes(Leaf<Trait>(proto)), value);
  }
  Leaf withFastDivMod() const { return Leaf(Leaf<Trait>(*this).withFastDivMod(), value); }
  Leaf withRange(int32_t lo, int32_t hi) const;

  /// Whether `value` really satisfies what the inherited property claims. This
  /// is the invariant a differential test asserts after every operation: the
  /// abstract stage is allowed to be less precise than the concrete one, never
  /// wrong.
  bool isSound() const;

  friend bool operator==(Leaf a, Leaf b) {
    return static_cast<const LeafProperty &>(a) == static_cast<const LeafProperty &>(b) &&
           a.value == b.value;
  }
  friend bool operator!=(Leaf a, Leaf b) { return !(a == b); }
};

//===----------------------------------------------------------------------===//
// Pointer<Runtime>
//===----------------------------------------------------------------------===//

/// Runtime pointer backed by a native C++ address. The pointee type remains
/// type-erased because its identity and bit width live in the common metadata.
template <> struct Pointer<Runtime> : public PointerProperty {
public:
  Pointer(OpaqueValue elemType, int32_t elemBitWidth, int32_t byteAlignment, void *value,
          Swizzle<Runtime> swizzle = {}, int64_t bitAlignment = 0,
          OpaqueValue addressSpace = OpaqueValue::getType<void>(),
          int32_t storageElementBitWidth = 0);

  void *value() const { return value_; }
  Swizzle<Runtime> swizzle() const { return PointerProperty::swizzle<Runtime>(); }

private:
  void *value_;
};

#define FLYDSL_CORE_CATEGORY Runtime
#include "flydsl/Core/Category.def"

//===----------------------------------------------------------------------===//
// Literals
//===----------------------------------------------------------------------===//

namespace runtime {

template <class... Ts>
constexpr uint64_t parse_int_digits(uint64_t result, int digit, Ts... digits) {
  if constexpr (sizeof...(Ts) == 0)
    return 10 * result + static_cast<uint64_t>(digit);
  else
    return parse_int_digits(10 * result + static_cast<uint64_t>(digit), digits...);
}

template <char... digits> constexpr int64_t parse_literal() {
  static_assert((('0' <= digits && digits <= '9') && ...),
                "Expected 0 <= digit <= 9 for each digit of the integer.");
  return static_cast<int64_t>(parse_int_digits(0, (digits - '0')...));
}

} // namespace runtime

/// `8_c` is the static leaf 8 and `8_d` a dynamic i32 leaf holding 8; `8_ct` and
/// `8_dt` are the same leaves as tuples.
template <char... digits> auto operator""_c() {
  constexpr auto value = runtime::parse_literal<digits...>();
  return Leaf<Runtime>::getSInt(value);
}

template <char... digits> auto operator""_d() {
  constexpr auto value = runtime::parse_literal<digits...>();
  return Leaf<Runtime>::getDyn(value);
}

template <char... digits> auto operator""_ct() {
  constexpr auto value = runtime::parse_literal<digits...>();
  return IntTuple<Runtime>::getSInt(value);
}

template <char... digits> auto operator""_dt() {
  constexpr auto value = runtime::parse_literal<digits...>();
  return IntTuple<Runtime>::fromLeaf(Leaf<Runtime>::getDyn(value));
}

} // namespace mlir::fly::core

#endif // FLYDSL_CORE_RUNTIME_HPP
