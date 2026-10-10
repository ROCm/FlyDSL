// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_CORE_ALGEBRA_SWIZZLE_HPP
#define FLYDSL_CORE_ALGEBRA_SWIZZLE_HPP

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "flydsl/Core/Algebra/Config.hpp"
#include "flydsl/Core/Algebra/IntTuple.hpp"
#include "flydsl/Core/Algebra/Parse.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <optional>
#include <type_traits>

namespace mlir::fly::core {

template <class Cat> struct Swizzle {
  Swizzle() = default;
  Swizzle(int32_t mask, int32_t base, int32_t shift);

  static std::optional<Swizzle> fromString(llvm::StringRef text);

  static bool isValid(int32_t mask, int32_t base, int32_t shift);

  int32_t mask() const { return mask_; }
  int32_t base() const { return base_; }
  int32_t shift() const { return shift_; }

  int32_t num_bits() const { return mask(); }
  int32_t num_base() const { return base(); }
  int32_t num_shift() const { return shift(); }

  uint32_t y_mask() const;
  uint32_t z_mask() const;

  bool isIdentity() const { return mask_ == 0; }

  template <class T> IntTuple<Cat> operator()(const T &offset) const;

  friend bool operator==(Swizzle lhs, Swizzle rhs) {
    return lhs.base_ == rhs.base_ && lhs.mask_ == rhs.mask_ && lhs.shift_ == rhs.shift_;
  }
  friend bool operator!=(Swizzle lhs, Swizzle rhs) { return !(lhs == rhs); }

  int32_t mask_ = 0;
  int32_t base_ = 0;
  int32_t shift_ = 0;
};

template <class Cat> struct CoordSwizzle {
  CoordSwizzle() = default;

  CoordSwizzle(int32_t mask, int32_t rowBase, llvm::ArrayRef<uint8_t> rowModes, int32_t colBase,
               llvm::ArrayRef<uint8_t> colModes);

  static std::optional<CoordSwizzle> fromString(llvm::StringRef text);

  static bool isValid(int32_t mask, int32_t rowBase, llvm::ArrayRef<uint8_t> rowModes,
                      int32_t colBase, llvm::ArrayRef<uint8_t> colModes);

  bool isIdentity() const { return mask_ == 0; }

  int32_t mask() const { return mask_; }
  int32_t rowBase() const { return row_base_; }
  int32_t colBase() const { return col_base_; }

  llvm::ArrayRef<uint8_t> rowModes() const {
    return llvm::ArrayRef<uint8_t>(row_modes_, row_nmodes_);
  }
  llvm::ArrayRef<uint8_t> colModes() const {
    return llvm::ArrayRef<uint8_t>(col_modes_, col_nmodes_);
  }

  friend bool operator==(const CoordSwizzle &lhs, const CoordSwizzle &rhs) {
    return lhs.mask_ == rhs.mask_ && lhs.row_base_ == rhs.row_base_ &&
           lhs.col_base_ == rhs.col_base_ && lhs.rowModes() == rhs.rowModes() &&
           lhs.colModes() == rhs.colModes();
  }
  friend bool operator!=(const CoordSwizzle &lhs, const CoordSwizzle &rhs) { return !(lhs == rhs); }

private:
  int32_t mask_ = 0;
  int32_t row_base_ = 0;
  int32_t col_base_ = 0;
  uint8_t row_modes_[kMaxBasisModes] = {};
  uint8_t row_nmodes_ = 0;
  uint8_t col_modes_[kMaxBasisModes] = {};
  uint8_t col_nmodes_ = 0;
};

template <class Cat> IntTuple<Cat> max_alignment(Swizzle<Cat> value);
/// The swizzle with these y and z masks, if one exists.
template <class Cat> std::optional<Swizzle<Cat>> make_swizzle(uint32_t y_mask, uint32_t z_mask);

/// The single swizzle equal to applying `rhs` and then `lhs`.  Both must have
/// the same shift and the result must be one swizzle; otherwise none.
template <class Cat> std::optional<Swizzle<Cat>> composition(Swizzle<Cat> lhs, Swizzle<Cat> rhs);

template <class Cat> Swizzle<Cat> right_inverse(Swizzle<Cat> swizzle);
template <class Cat> Swizzle<Cat> left_inverse(Swizzle<Cat> swizzle);

template <class Cat> Swizzle<Cat> upcast(int32_t n, Swizzle<Cat> swizzle);
template <class Cat> Swizzle<Cat> downcast(int32_t n, Swizzle<Cat> swizzle);

template <class Cat> CoordSwizzle<Cat> upcast(int32_t n, const CoordSwizzle<Cat> &swizzle);
template <class Cat> CoordSwizzle<Cat> downcast(int32_t n, const CoordSwizzle<Cat> &swizzle);

template <class Cat>
Swizzle<Cat> recast_layout(int32_t newTypeBits, int32_t oldTypeBits, Swizzle<Cat> swizzle);

template <class T, class Cat = category_of<T>,
          std::enable_if_t<is_int_tuple_v<T> && std::is_same_v<Cat, category_of<T>>, int> = 0>
IntTuple<Cat> swizzle_apply(Swizzle<Cat> swizzle, const T &value);

template <class T, class Cat = category_of<T>,
          std::enable_if_t<is_int_tuple_v<T> && std::is_same_v<Cat, category_of<T>>, int> = 0>
IntTuple<Cat> swizzle_apply(CoordSwizzle<Cat> swizzle, const T &value);

template <class Cat> llvm::hash_code hash_value(Swizzle<Cat> value);
template <class Cat> llvm::hash_code hash_value(const CoordSwizzle<Cat> &value);

template <class Cat> void print(Swizzle<Cat> swizzle, llvm::raw_ostream &os);
template <class Cat> void print(const CoordSwizzle<Cat> &swizzle, llvm::raw_ostream &os);

//===----------------------------------------------------------------------===//
// Definitions
//===----------------------------------------------------------------------===//

template <class Cat>
Swizzle<Cat>::Swizzle(int32_t mask, int32_t base, int32_t shift)
    : mask_(mask), base_(base), shift_(shift) {
  FLYDSL_CORE_ASSERT(isValid(mask, base, shift));
}

template <class Cat> bool Swizzle<Cat>::isValid(int32_t mask, int32_t base, int32_t shift) {
  if (mask < 0 || base < 0)
    return false;
  auto absShift = shift < 0 ? -int64_t{shift} : int64_t{shift};
  return absShift >= mask;
}

template <class Cat> uint32_t Swizzle<Cat>::y_mask() const {
  if (mask_ == 0)
    return 0;
  auto bits = (uint32_t{1} << mask_) - 1;
  auto offset = base_ + (shift_ > 0 ? shift_ : 0);
  return static_cast<uint32_t>(bits << offset);
}

template <class Cat> uint32_t Swizzle<Cat>::z_mask() const {
  if (mask_ == 0)
    return 0;
  auto bits = (uint32_t{1} << mask_) - 1;
  auto offset = base_ - (shift_ < 0 ? shift_ : 0);
  return static_cast<uint32_t>(bits << offset);
}

template <class Cat>
CoordSwizzle<Cat>::CoordSwizzle(int32_t mask, int32_t rowBase, llvm::ArrayRef<uint8_t> rowModes,
                                int32_t colBase, llvm::ArrayRef<uint8_t> colModes)
    : mask_(mask), row_base_(rowBase), col_base_(colBase),
      row_nmodes_(static_cast<uint8_t>(std::min(rowModes.size(), size_t{kMaxBasisModes}))),
      col_nmodes_(static_cast<uint8_t>(std::min(colModes.size(), size_t{kMaxBasisModes}))) {
  FLYDSL_CORE_ASSERT(isValid(mask, rowBase, rowModes, colBase, colModes));
  // Copy no more modes than the storage holds even when assertions are off.
  std::copy_n(rowModes.begin(), row_nmodes_, row_modes_);
  std::copy_n(colModes.begin(), col_nmodes_, col_modes_);
}

template <class Cat>
bool CoordSwizzle<Cat>::isValid(int32_t mask, int32_t rowBase, llvm::ArrayRef<uint8_t> rowModes,
                                int32_t colBase, llvm::ArrayRef<uint8_t> colModes) {
  return mask >= 0 && rowBase >= 0 && colBase >= 0 && rowModes.size() <= kMaxBasisModes &&
         colModes.size() <= kMaxBasisModes;
}

template <class Cat>
template <class T>
IntTuple<Cat> Swizzle<Cat>::operator()(const T &offset) const {
  return swizzle_apply(*this, offset);
}

template <class Cat> IntTuple<Cat> max_alignment(Swizzle<Cat> value) {
  if (value.isIdentity())
    return IntTuple<Cat>::getSInt(1024); // regard it as the maximum alignment
  else
    return IntTuple<Cat>::getSInt(int64_t{1} << value.base());
}

template <class Cat> std::optional<Swizzle<Cat>> make_swizzle(uint32_t y_mask, uint32_t z_mask) {
  if (llvm::popcount(y_mask) != llvm::popcount(z_mask))
    return std::nullopt;
  auto bits = static_cast<int32_t>(llvm::popcount(z_mask));
  auto trailingY = y_mask == 0 ? 0 : static_cast<int32_t>(llvm::countr_zero(y_mask));
  auto trailingZ = z_mask == 0 ? 0 : static_cast<int32_t>(llvm::countr_zero(z_mask));
  auto base = std::min(trailingY, trailingZ) % 32;
  auto shift = trailingY - trailingZ;

  if (!Swizzle<Cat>::isValid(bits, base, shift))
    return std::nullopt;
  auto result = Swizzle<Cat>(bits, base, shift);
  if (result.y_mask() != y_mask || result.z_mask() != z_mask)
    return std::nullopt;
  return result;
}

template <class Cat> std::optional<Swizzle<Cat>> composition(Swizzle<Cat> lhs, Swizzle<Cat> rhs) {
  if (lhs.shift() != rhs.shift())
    return std::nullopt;
  return make_swizzle<Cat>(lhs.y_mask() ^ rhs.y_mask(), lhs.z_mask() ^ rhs.z_mask());
}

template <class Cat> Swizzle<Cat> right_inverse(Swizzle<Cat> swizzle) { return swizzle; }

template <class Cat> Swizzle<Cat> left_inverse(Swizzle<Cat> swizzle) { return swizzle; }

template <class Cat> Swizzle<Cat> upcast(int32_t n, Swizzle<Cat> swizzle) {
  FLYDSL_CORE_ASSERT(llvm::has_single_bit(static_cast<uint32_t>(n)));
  auto shift = static_cast<int32_t>(llvm::Log2_32(n));
  auto newBase = swizzle.base() - shift;
  if (newBase >= 0)
    return Swizzle<Cat>(swizzle.mask(), newBase, swizzle.shift());
  else
    return Swizzle<Cat>(std::max(swizzle.mask() + newBase, 0), 0, swizzle.shift());
}

template <class Cat> CoordSwizzle<Cat> upcast(int32_t n, const CoordSwizzle<Cat> &swizzle) {
  FLYDSL_CORE_ASSERT(llvm::has_single_bit(static_cast<uint32_t>(n)));
  auto shift = static_cast<int32_t>(llvm::Log2_32(n));
  auto dropped = std::max(shift - swizzle.colBase(), 0);
  return CoordSwizzle<Cat>(std::max(swizzle.mask() - dropped, 0), swizzle.rowBase() + dropped,
                           swizzle.rowModes(), swizzle.colBase() + dropped - shift,
                           swizzle.colModes());
}

template <class Cat> Swizzle<Cat> downcast(int32_t n, Swizzle<Cat> swizzle) {
  FLYDSL_CORE_ASSERT(llvm::has_single_bit(static_cast<uint32_t>(n)));
  return Swizzle<Cat>(swizzle.mask(), swizzle.base() + llvm::Log2_32(n), swizzle.shift());
}

template <class Cat> CoordSwizzle<Cat> downcast(int32_t n, const CoordSwizzle<Cat> &swizzle) {
  FLYDSL_CORE_ASSERT(llvm::has_single_bit(static_cast<uint32_t>(n)));
  // Downcasting multiplies only the column coordinate by n.
  return CoordSwizzle<Cat>(swizzle.mask(), swizzle.rowBase(), swizzle.rowModes(),
                           swizzle.colBase() + llvm::Log2_32(n), swizzle.colModes());
}

template <class Cat>
Swizzle<Cat> recast_layout(int32_t newTypeBits, int32_t oldTypeBits, Swizzle<Cat> swizzle) {
  FLYDSL_CORE_ASSERT(oldTypeBits > 0 && newTypeBits > 0);
  auto divisor = std::gcd(oldTypeBits, newTypeBits);
  auto numerator = newTypeBits / divisor;
  auto denominator = oldTypeBits / divisor;
  if (numerator == 1 && denominator == 1)
    return swizzle;
  if (numerator == 1)
    return downcast(denominator, swizzle);
  if (denominator == 1)
    return upcast(numerator, swizzle);
  return downcast(denominator, upcast(numerator, swizzle));
}

template <class Cat>
CoordSwizzle<Cat> recast_layout(int32_t newTypeBits, int32_t oldTypeBits,
                                const CoordSwizzle<Cat> &swizzle) {
  FLYDSL_CORE_ASSERT(oldTypeBits > 0 && newTypeBits > 0);
  auto divisor = std::gcd(oldTypeBits, newTypeBits);
  auto numerator = newTypeBits / divisor;
  auto denominator = oldTypeBits / divisor;
  if (numerator == 1 && denominator == 1)
    return swizzle;
  if (numerator == 1)
    return downcast(denominator, swizzle);
  if (denominator == 1)
    return upcast(numerator, swizzle);
  return downcast(denominator, upcast(numerator, swizzle));
}

template <class Cat> inline llvm::hash_code hash_value(Swizzle<Cat> value) {
  return llvm::hash_combine(value.mask(), value.base(), value.shift());
}

template <class Cat> inline llvm::hash_code hash_value(const CoordSwizzle<Cat> &value) {
  return llvm::hash_combine(
      value.mask(), value.rowBase(),
      llvm::hash_combine_range(value.rowModes().begin(), value.rowModes().end()), value.colBase(),
      llvm::hash_combine_range(value.colModes().begin(), value.colModes().end()));
}

template <class Cat> void print(Swizzle<Cat> swizzle, llvm::raw_ostream &os) {
  os << "S<" << swizzle.mask() << ',' << swizzle.base() << ',' << swizzle.shift() << '>';
}

template <class Cat> std::optional<Swizzle<Cat>> Swizzle<Cat>::fromString(llvm::StringRef text) {
  if (!text.consume_front("S<") || !text.consume_back(">"))
    return std::nullopt;
  llvm::SmallVector<llvm::StringRef, 3> fields;
  text.split(fields, ',');
  if (fields.size() != 3)
    return std::nullopt;
  auto mask = int32_t{0};
  auto base = int32_t{0};
  auto shift = int32_t{0};
  if (fields[0].getAsInteger(10, mask) || fields[1].getAsInteger(10, base) ||
      fields[2].getAsInteger(10, shift) || !isValid(mask, base, shift))
    return std::nullopt;
  return Swizzle(mask, base, shift);
}

template <class Cat> void print(const CoordSwizzle<Cat> &swizzle, llvm::raw_ostream &os) {
  os << "CS<" << swizzle.mask() << ',' << swizzle.rowBase() << ",[";
  llvm::interleaveComma(swizzle.rowModes(), os, [&](uint8_t mode) { os << int32_t{mode}; });
  os << "]," << swizzle.colBase() << ",[";
  llvm::interleaveComma(swizzle.colModes(), os, [&](uint8_t mode) { os << int32_t{mode}; });
  os << "]>";
}

template <class Cat>
std::optional<CoordSwizzle<Cat>> CoordSwizzle<Cat>::fromString(llvm::StringRef text) {
  if (!text.consume_front("CS<") || !text.consume_back(">"))
    return std::nullopt;
  auto first = text.find(",[");
  if (first == llvm::StringRef::npos)
    return std::nullopt;
  auto head = text.take_front(first).split(',');
  auto mask = int32_t{0};
  auto rowBase = int32_t{0};
  if (head.second.empty() || head.first.getAsInteger(10, mask) ||
      head.second.getAsInteger(10, rowBase))
    return std::nullopt;
  auto rowEnd = text.find("],", first + 2);
  if (rowEnd == llvm::StringRef::npos)
    return std::nullopt;
  auto colModesStart = text.find(",[", rowEnd + 2);
  if (colModesStart == llvm::StringRef::npos || !text.ends_with(']'))
    return std::nullopt;
  auto colBaseText = text.slice(rowEnd + 2, colModesStart);
  auto colBase = int32_t{0};
  if (colBaseText.getAsInteger(10, colBase))
    return std::nullopt;

  auto parseModes = [](llvm::StringRef body) -> std::optional<llvm::SmallVector<uint8_t, 4>> {
    llvm::SmallVector<uint8_t, 4> modes;
    if (body.empty())
      return modes;
    llvm::SmallVector<llvm::StringRef, 4> fields;
    body.split(fields, ',');
    for (auto field : fields) {
      auto mode = uint8_t{0};
      if (field.trim().getAsInteger(10, mode) || modes.size() == kMaxBasisModes)
        return std::nullopt;
      modes.push_back(mode);
    }
    return modes;
  };
  auto rowModes = parseModes(text.slice(first + 2, rowEnd));
  auto colModes = parseModes(text.slice(colModesStart + 2, text.size() - 1));
  if (!rowModes || !colModes || mask < 0 || rowBase < 0 || colBase < 0)
    return std::nullopt;
  return CoordSwizzle(mask, rowBase, *rowModes, colBase, *colModes);
}

template <class T, class Cat,
          std::enable_if_t<is_int_tuple_v<T> && std::is_same_v<Cat, category_of<T>>, int>>
IntTuple<Cat> swizzle_apply(Swizzle<Cat> swizzle, const T &value) {
  auto tuple = value.asRef();
  if (!tuple.isLeaf())
    return detail::operandError<Cat>({tuple}, ErrorCode::ExpectedLeafOperand,
                                     AlgebraOp::SwizzleApply);
  if (swizzle.isIdentity())
    return tuple;
  auto amount = IntTuple<Cat>::getSInt(swizzle.shift() < 0 ? -swizzle.shift() : swizzle.shift());
  auto masked = leaf_bit_and(tuple, IntTuple<Cat>::getSInt(swizzle.y_mask()));
  auto shifted = swizzle.shift() < 0 ? leaf_shl(masked, amount) : leaf_shr(masked, amount);
  return leaf_bit_xor(tuple, shifted);
}

namespace detail {

template <class Cat>
IntTuple<Cat> replace_path(IntTupleRef<Cat> tuple, llvm::ArrayRef<uint8_t> path,
                           IntTupleRef<Cat> replacement) {
  if (path.empty())
    return replacement;

  FLYDSL_CORE_ASSERT(!tuple.isLeaf());
  auto index = path.front();
  auto rank = tuple.rank();
  FLYDSL_CORE_ASSERT(index < rank);
  llvm::SmallVector<IntTuple<Cat>, 8> children;
  children.reserve(static_cast<size_t>(rank));
  auto i = int32_t{0};
  for (auto child : tuple) {
    if (i++ == index)
      children.push_back(replace_path(child, path.drop_front(), replacement));
    else
      children.emplace_back(child);
  }
  return make_tuple(children);
}

} // namespace detail

template <class T, class Cat,
          std::enable_if_t<is_int_tuple_v<T> && std::is_same_v<Cat, category_of<T>>, int>>
IntTuple<Cat> swizzle_apply(CoordSwizzle<Cat> swizzle, const T &value) {
  auto input = value.asRef();
  if (swizzle.isIdentity())
    return input;

  // A coordinate whose other modes are zero collapses into a basis leaf, or
  // into 0; read it as the arithmetic tuple it denotes, with the zero modes up
  // to the row and column paths filled back in.
  auto zerosAlong = [](llvm::ArrayRef<uint8_t> path) {
    auto tuple = IntTuple<Cat>::getZero();
    for (auto i = path.size(); i-- > 0;) {
      llvm::SmallVector<IntTuple<Cat>, 8> modes(path[i] + size_t{1}, IntTuple<Cat>::getZero());
      modes.back() = std::move(tuple);
      tuple = make_tuple(modes);
    }
    return tuple;
  };
  auto expanded = IntTuple<Cat>(input);
  if (input.isBasis() || (input.isLeaf() && input.isZero()))
    expanded = as_arithmetic_tuple(input) + zerosAlong(swizzle.rowModes()) +
               zerosAlong(swizzle.colModes());
  auto coord = expanded.asRef();

  auto hasPath = [&](llvm::ArrayRef<uint8_t> path) {
    auto cur = coord;
    for (auto mode : path) {
      if (cur.isLeaf() || mode >= cur.rank())
        return false;
      cur = cur.at(mode);
    }
    return true;
  };
  if (!hasPath(swizzle.rowModes()) || !hasPath(swizzle.colModes()))
    return detail::operandError<Cat>({coord}, ErrorCode::BasisModeOutOfRange,
                                     AlgebraOp::SwizzleApply);

  auto row = coord.at(swizzle.rowModes());
  auto col = coord.at(swizzle.colModes());
  if (!row.isInt() || !col.isInt()) {
    auto reason = row.isLeaf() && col.isLeaf() ? ErrorCode::UnsupportedBitwiseOperand
                                               : ErrorCode::ExpectedLeafOperand;
    return detail::operandError<Cat>({row, col}, reason, AlgebraOp::SwizzleApply);
  }

  auto mask = IntTuple<Cat>::getSInt((int64_t{1} << swizzle.mask()) - 1);
  auto rowBase = IntTuple<Cat>::getSInt(swizzle.rowBase());
  auto colBase = IntTuple<Cat>::getSInt(swizzle.colBase());
  auto rowBits = leaf_bit_and(leaf_shr(row, rowBase), mask);
  auto replacement = leaf_bit_xor(col, leaf_shl(rowBits, colBase));
  return detail::replace_path(coord, swizzle.colModes(), replacement.asRef());
}

} // namespace mlir::fly::core

#endif // FLYDSL_CORE_ALGEBRA_SWIZZLE_HPP
