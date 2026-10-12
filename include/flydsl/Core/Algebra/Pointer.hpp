// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_CORE_ALGEBRA_POINTER_HPP
#define FLYDSL_CORE_ALGEBRA_POINTER_HPP

#include "flydsl/Core/Algebra/Config.hpp"
#include "flydsl/Core/Algebra/OpaqueValue.hpp"
#include "flydsl/Core/Algebra/Swizzle.hpp"

#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <type_traits>
#include <utility>

namespace mlir::fly::core {

/// Category-independent pointer metadata. Concrete `Pointer<Cat>` injection
/// points inherit this storage while choosing their own payload.
struct PointerProperty {
public:
  PointerProperty(OpaqueValue elemType, int32_t elemBitWidth, int32_t byteAlignment,
                  OpaqueValue addressSpace = OpaqueValue::getType<void>(),
                  int32_t storageElementBitWidth = 0);

  PointerProperty(OpaqueValue elemType, int32_t elemBitWidth, int32_t byteAlignment,
                  int64_t bitAlignment, OpaqueValue addressSpace = OpaqueValue::getType<void>(),
                  int32_t storageElementBitWidth = 0);

  template <class Cat>
  PointerProperty(OpaqueValue elemType, int32_t elemBitWidth, int32_t byteAlignment,
                  Swizzle<Cat> swizzle, OpaqueValue addressSpace = OpaqueValue::getType<void>(),
                  int32_t storageElementBitWidth = 0);

  template <class Cat>
  PointerProperty(OpaqueValue elemType, int32_t elemBitWidth, int32_t byteAlignment,
                  int64_t bitAlignment, Swizzle<Cat> swizzle,
                  OpaqueValue addressSpace = OpaqueValue::getType<void>(),
                  int32_t storageElementBitWidth = 0);

private:
  void validate() const;

public:
  const OpaqueValue &elementType() const { return elemType_; }
  /// The void type tag denotes an unspecified address space.
  const OpaqueValue &addressSpace() const { return addressSpace_; }
  int32_t elementBitWidth() const { return elemBitWidth_; }
  /// Physical bit stride of adjacent stored elements; defaults to logical width.
  int32_t storageElementBitWidth() const { return storageElemBitWidth_; }
  int32_t byteAlignment() const { return byteAlignment_; }
  int64_t storageBitAlignment() const { return bitAlignment_; }
  int32_t swizzleMask() const { return swizzleMask_; }
  int32_t swizzleBase() const { return swizzleBase_; }
  int32_t swizzleShift() const { return swizzleShift_; }
  template <class Cat> Swizzle<Cat> swizzle() const {
    return Swizzle<Cat>(swizzleMask_, swizzleBase_, swizzleShift_);
  }

  int64_t bitAlignment() const;

  friend bool operator==(const PointerProperty &lhs, const PointerProperty &rhs) {
    return lhs.elemType_ == rhs.elemType_ && lhs.addressSpace_ == rhs.addressSpace_ &&
           lhs.elemBitWidth_ == rhs.elemBitWidth_ &&
           lhs.storageElemBitWidth_ == rhs.storageElemBitWidth_ &&
           lhs.byteAlignment_ == rhs.byteAlignment_ && lhs.bitAlignment_ == rhs.bitAlignment_ &&
           lhs.swizzleMask_ == rhs.swizzleMask_ && lhs.swizzleBase_ == rhs.swizzleBase_ &&
           lhs.swizzleShift_ == rhs.swizzleShift_;
  }
  friend bool operator!=(const PointerProperty &lhs, const PointerProperty &rhs) {
    return !(lhs == rhs);
  }

private:
  OpaqueValue elemType_;
  OpaqueValue addressSpace_;
  int32_t elemBitWidth_;
  int32_t storageElemBitWidth_;
  int32_t byteAlignment_;
  int64_t bitAlignment_;
  int32_t swizzleMask_ = 0;
  int32_t swizzleBase_ = 0;
  int32_t swizzleShift_ = 0;
};

llvm::hash_code hash_value(const PointerProperty &pointer);
/// Debug text only: opaque element type and address space are not printed, so
/// pointers have no fromString.
void print(const PointerProperty &pointer, llvm::raw_ostream &os);

namespace detail {

template <class Offset, class Cat = category_of<Offset>,
          std::enable_if_t<is_int_tuple_v<Offset>, int> = 0>
int64_t offsetBitAlignment(const Pointer<Cat> &pointer, const Offset &offset);

} // namespace detail

inline PointerProperty::PointerProperty(OpaqueValue elemType, int32_t elemBitWidth,
                                        int32_t byteAlignment, OpaqueValue addressSpace,
                                        int32_t storageElementBitWidth)
    : elemType_(std::move(elemType)), addressSpace_(std::move(addressSpace)),
      elemBitWidth_(elemBitWidth),
      storageElemBitWidth_(storageElementBitWidth ? storageElementBitWidth : elemBitWidth),
      byteAlignment_(byteAlignment), bitAlignment_(int64_t{byteAlignment} * 8) {
  validate();
}

inline PointerProperty::PointerProperty(OpaqueValue elemType, int32_t elemBitWidth,
                                        int32_t byteAlignment, int64_t bitAlignment,
                                        OpaqueValue addressSpace, int32_t storageElementBitWidth)
    : elemType_(std::move(elemType)), addressSpace_(std::move(addressSpace)),
      elemBitWidth_(elemBitWidth),
      storageElemBitWidth_(storageElementBitWidth ? storageElementBitWidth : elemBitWidth),
      byteAlignment_(byteAlignment), bitAlignment_(bitAlignment) {
  validate();
  FLYDSL_CORE_ASSERT(bitAlignment > 0 && bitAlignment <= int64_t{byteAlignment} * 8);
}

template <class Cat>
PointerProperty::PointerProperty(OpaqueValue elemType, int32_t elemBitWidth, int32_t byteAlignment,
                                 Swizzle<Cat> swizzle, OpaqueValue addressSpace,
                                 int32_t storageElementBitWidth)
    : PointerProperty(std::move(elemType), elemBitWidth, byteAlignment, int64_t{byteAlignment} * 8,
                      swizzle, std::move(addressSpace), storageElementBitWidth) {}

template <class Cat>
PointerProperty::PointerProperty(OpaqueValue elemType, int32_t elemBitWidth, int32_t byteAlignment,
                                 int64_t bitAlignment, Swizzle<Cat> swizzle,
                                 OpaqueValue addressSpace, int32_t storageElementBitWidth)
    : elemType_(std::move(elemType)), addressSpace_(std::move(addressSpace)),
      elemBitWidth_(elemBitWidth),
      storageElemBitWidth_(storageElementBitWidth ? storageElementBitWidth : elemBitWidth),
      byteAlignment_(byteAlignment), bitAlignment_(bitAlignment), swizzleMask_(swizzle.mask()),
      swizzleBase_(swizzle.base()), swizzleShift_(swizzle.shift()) {
  validate();
  FLYDSL_CORE_ASSERT(bitAlignment > 0 && bitAlignment <= int64_t{byteAlignment} * 8);
}

inline void PointerProperty::validate() const {
  FLYDSL_CORE_ASSERT(elemBitWidth_ > 0 && storageElemBitWidth_ > 0);
  FLYDSL_CORE_ASSERT(byteAlignment_ > 0);
}

inline int64_t PointerProperty::bitAlignment() const {
  if (swizzleMask_ == 0)
    return bitAlignment_;
  // swizzle_ptr operates on byte addresses, independent of element type.
  if (swizzleBase_ >= 60)
    return int64_t{1} << llvm::countr_zero(static_cast<uint64_t>(bitAlignment_));
  auto swizzleBits = int64_t{8} << swizzleBase_;
  return std::gcd(bitAlignment_, swizzleBits);
}

inline llvm::hash_code hash_value(const PointerProperty &pointer) {
  return llvm::hash_combine(
      pointer.elementType(), pointer.addressSpace(), pointer.elementBitWidth(),
      pointer.storageElementBitWidth(), pointer.byteAlignment(), pointer.storageBitAlignment(),
      llvm::hash_combine(pointer.swizzleMask(), pointer.swizzleBase(), pointer.swizzleShift()));
}

inline void print(const PointerProperty &pointer, llvm::raw_ostream &os) {
  os << "ptr<opaque:" << pointer.elementBitWidth() << ", alignment=" << pointer.byteAlignment();
  if (pointer.storageElementBitWidth() != pointer.elementBitWidth())
    os << ", storage_bits=" << pointer.storageElementBitWidth();
  if (pointer.storageBitAlignment() != int64_t{pointer.byteAlignment()} * 8)
    os << ", bit_alignment=" << pointer.storageBitAlignment();
  if (pointer.swizzleMask() != 0)
    os << ", swizzle=S<" << pointer.swizzleMask() << ',' << pointer.swizzleBase() << ','
       << pointer.swizzleShift() << '>';
  os << '>';
}

namespace detail {

template <class Offset, class Cat, std::enable_if_t<is_int_tuple_v<Offset>, int>>
int64_t offsetBitAlignment(const Pointer<Cat> &pointer, const Offset &offset) {
  auto value = offset.asRef();
  FLYDSL_CORE_ASSERT(value.isLeaf() && value.isInt());
  if (value.isSInt(0))
    return pointer.storageBitAlignment();

  auto elementOffsetDivisibility = [&] {
    if (value.isSInt()) {
      auto staticOffset = value.staticValue();
      auto magnitude = static_cast<uint64_t>(staticOffset);
      return staticOffset < 0 ? uint64_t{0} - magnitude : magnitude;
    }
    return static_cast<uint64_t>(value.divisibility());
  }();
  auto pointerBitAlignment = static_cast<uint64_t>(pointer.storageBitAlignment());
  auto elementBitWidth = static_cast<uint64_t>(pointer.storageElementBitWidth());
  // gcd(A, E * D) = gcd(A, E) * gcd(A / gcd(A, E), D).
  // Each product is bounded by A, even for INT64_MIN or large E * D.
  auto elementAlignment = std::gcd(pointerBitAlignment, elementBitWidth);
  return static_cast<int64_t>(elementAlignment * std::gcd(pointerBitAlignment / elementAlignment,
                                                          elementOffsetDivisibility));
}

} // namespace detail

} // namespace mlir::fly::core

#endif // FLYDSL_CORE_ALGEBRA_POINTER_HPP
