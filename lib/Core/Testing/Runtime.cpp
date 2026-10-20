// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#include "flydsl/Core/Runtime.hpp"

#include <cstddef>
#include <string>

namespace mlir::fly::core {

//===----------------------------------------------------------------------===//
// Leaf<Runtime> and Pointer<Runtime>
//===----------------------------------------------------------------------===//

Leaf<Runtime> Leaf<Runtime>::getDyn(int64_t v, LogWidth logWidth, int32_t div,
                                    std::optional<std::pair<int32_t, int32_t>> range,
                                    bool fastDivMod) {
  auto type = Leaf<Trait>::getDInt(logWidth, div);
  if (range)
    type = type.withRange(range->first, range->second);
  if (fastDivMod && !type.isError())
    type = type.withFastDivMod();
  return Leaf(type, type.isError() ? 0 : v);
}

Leaf<Runtime> Leaf<Runtime>::withRange(int32_t lo, int32_t hi) const {
  if constexpr (!FLYDSL_CORE_ENABLE_RANGE_INFERENCE)
    return *this;
  return Leaf(Leaf<Trait>(*this).withRange(lo, hi), value);
}

bool Leaf<Runtime>::isSound() const {
  auto type = Leaf<Trait>(*this).scale();
  if (type.isSInt())
    return value == type.staticValue();
  if (type.isDInt()) {
    auto bits = type.bitWidth();
    if (bits == 1 ? (value < 0 || value > 1)
                  : (bits < 64 && (value < -(int64_t{1} << (bits - 1)) ||
                                   value > (int64_t{1} << (bits - 1)) - 1)))
      return false;
    auto div = type.divisibility();
    if (div != 0 && value % div != 0)
      return false;
    if (type.hasRange() && (value < type.rangeMin() || value > type.rangeMax()))
      return false;
    return true;
  }
  return true; // None / Error / Ratio carry no value
}

std::optional<Leaf<Runtime>> Leaf<Runtime>::fromString(llvm::StringRef text) {
  // A dynamic leaf prints its value right after the `?` of its Trait type.
  auto value = int64_t{0};
  auto typeText = text.str();
  if (text.consume_front("?")) {
    if (text.consumeInteger(10, value))
      return std::nullopt;
    typeText = "?" + text.str();
  }
  auto type = Leaf<Trait>::fromString(typeText);
  if (!type)
    return std::nullopt;
  auto scale = type->scale();
  return Leaf(*type, scale.isSInt() ? scale.staticValue() : value);
}

Pointer<Runtime>::Pointer(OpaqueValue elemType, int32_t elemBitWidth, int32_t byteAlignment,
                          void *value, Swizzle<Runtime> swizzle, int64_t bitAlignment,
                          OpaqueValue addressSpace, int32_t storageElementBitWidth)
    : PointerProperty(std::move(elemType), elemBitWidth, byteAlignment,
                      bitAlignment == 0 ? int64_t{byteAlignment} * 8 : bitAlignment, swizzle,
                      std::move(addressSpace), storageElementBitWidth),
      value_(value) {}

//===----------------------------------------------------------------------===//
// Leaf operations
//===----------------------------------------------------------------------===//

namespace runtime {
namespace {

using Rule = Leaf<Trait> (*)(Leaf<Trait>, Leaf<Trait>);
using UnaryRule = Leaf<Trait> (*)(Leaf<Trait>);

/// The concrete half of a Runtime operation runs the very same Leaf<Trait> rule
/// over static leaves, so the two halves share one implementation and cannot
/// drift apart.  A result that does not fold to an integer carries no value.
int64_t concrete(Leaf<Trait> folded) { return folded.isSInt() ? folded.staticValue() : 0; }

/// Over plain static operands `isSound()` gives the property
/// `getSInt(a.value)`,
/// so the two halves are literally the same call and the abstract result is
/// already the concrete one.  The branch is on the operands rather than on the
/// folded result: a result-dependent branch would sit between the two
/// evaluations and stop them from overlapping, which costs the mixed case more
/// than the static case gains.
Leaf<Runtime> binary(AlgebraOp op, Rule rule, Leaf<Runtime> a, Leaf<Runtime> b) {
  if ((a.kind_ | b.kind_) == 0u) {
    auto folded = rule(Leaf<Trait>(a), Leaf<Trait>(b));
    return Leaf<Runtime>(folded, concrete(folded));
  }
  auto property = rule(Leaf<Trait>(a), Leaf<Trait>(b));
  if (property.isError())
    return Leaf<Runtime>(property, 0);
  auto concreteOperand = [](Leaf<Runtime> leaf) {
    auto scale = leaf.scale();
    return scale.isRatio() ? Leaf<Trait>(scale) : Leaf<Trait>::getSInt(leaf.value);
  };
  auto lhs = concreteOperand(a), rhs = concreteOperand(b);
  // The only supported ratio/dynamic operation is an exactly aligned product.
  // Match EmitOp's divide-before-multiply order, including at i64 limits.
  if (property.scale().isDInt() && a.scale().isRatio() && b.scale().isDInt()) {
    lhs = Leaf<Trait>::getSInt(a.scale().num());
    rhs = Leaf<Trait>::getSInt(b.value / a.scale().den());
  } else if (property.scale().isDInt() && b.scale().isRatio() && a.scale().isDInt()) {
    lhs = Leaf<Trait>::getSInt(a.value / b.scale().den());
    rhs = Leaf<Trait>::getSInt(b.scale().num());
  }
  // safe_div checks exactness only between static operands; with a dynamic
  // operand its value is the plain truncating quotient.
  auto folded = op == AlgebraOp::Safe ? lhs / rhs : rule(lhs, rhs);
  if (folded.isError())
    return Leaf<Runtime>(folded, 0);
  auto value = concrete(folded);
  auto type = property.scale();
  if (type.isDInt()) {
    auto bits = type.bitWidth();
    if (bits == 1 ? (value < 0 || value > 1)
                  : (bits < 64 && (value < -(int64_t{1} << (bits - 1)) ||
                                   value > (int64_t{1} << (bits - 1)) - 1)))
      return Leaf<Runtime>::getError(ErrorCode::IntegerOverflow);
  }
  return Leaf<Runtime>(property, value);
}

Leaf<Runtime> unary(UnaryRule rule, Leaf<Runtime> a) {
  auto property = rule(Leaf<Trait>(a));
  if (property.isError())
    return Leaf<Runtime>(property, 0);
  // A static result is fully described by its property. Recomputing it from
  // `a.value` would be wrong for a static ratio, which carries no value.
  if (property.isStatic())
    return Leaf<Runtime>(property, concrete(property.scale()));
  auto folded = rule(Leaf<Trait>::getSInt(a.value));
  if (folded.isError())
    return Leaf<Runtime>(folded, 0);
  auto result = Leaf<Runtime>(property, concrete(folded));
  // Unary abs/negation may overflow a narrow dynamic integer.
  auto type = property.scale();
  if (type.isDInt() &&
      !Leaf<Runtime>(Leaf<Trait>::getDInt(type.logWidth()), result.value).isSound())
    return Leaf<Runtime>::getError(ErrorCode::IntegerOverflow);
  return result;
}

} // namespace
} // namespace runtime

#define FLYDSL_CORE_LEAF_BINARY(NAME, OP)                                                          \
  Leaf<Runtime> NAME(Leaf<Runtime> a, Leaf<Runtime> b) {                                           \
    return runtime::binary(                                                                        \
        AlgebraOp::OP, [](Leaf<Trait> x, Leaf<Trait> y) { return NAME(x, y); }, a, b);             \
  }
#define FLYDSL_CORE_LEAF_UNARY(NAME, OP)                                                           \
  Leaf<Runtime> NAME(Leaf<Runtime> a) {                                                            \
    return runtime::unary([](Leaf<Trait> x) { return NAME(x); }, a);                               \
  }
#include "flydsl/Core/Category.def"

void print(Leaf<Runtime> l, llvm::raw_ostream &os) {
  auto type = Leaf<Trait>(l).scale();
  if (!type.isDInt()) {
    print(Leaf<Trait>(l), os);
    return;
  }
  os << '?' << l.value;
  detail::printDynIntSuffix(type, os);
  detail::printModes(l, os);
}

//===----------------------------------------------------------------------===//
// Pointer operations
//===----------------------------------------------------------------------===//

bool operator==(const Pointer<Runtime> &lhs, const Pointer<Runtime> &rhs) {
  return static_cast<const PointerProperty &>(lhs) == static_cast<const PointerProperty &>(rhs) &&
         lhs.value() == rhs.value();
}

Pointer<Runtime> add_offset(const Pointer<Runtime> &pointer, IntTupleRef<Runtime> offset) {
  FLYDSL_CORE_ASSERT(offset.isLeaf() && offset.isInt());
  auto bitOffset = pointer.storageElementBitWidth() * offset.getLeaf().value;
  FLYDSL_CORE_ASSERT(bitOffset % 8 == 0);
  auto *address = static_cast<std::byte *>(pointer.value());
  if (bitOffset != 0) {
    FLYDSL_CORE_ASSERT(address);
    address += bitOffset / 8;
  }
  auto bitAlignment = detail::offsetBitAlignment(pointer, offset);
  return Pointer<Runtime>(pointer.elementType(), pointer.elementBitWidth(),
                          static_cast<int32_t>((bitAlignment + 7) / 8), address, pointer.swizzle(),
                          bitAlignment, pointer.addressSpace(), pointer.storageElementBitWidth());
}

} // namespace mlir::fly::core
