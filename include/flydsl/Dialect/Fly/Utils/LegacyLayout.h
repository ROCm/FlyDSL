// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLY_UTILS_LEGACYLAYOUT_H
#define FLYDSL_DIALECT_FLY_UTILS_LEGACYLAYOUT_H

#include "flydsl/Core/Algebra.hpp"

namespace mlir::fly::core {
namespace detail {

/// Unwrap one outer singleton mode, using Core accessors for tuples,
/// layouts and tensors. get preserves a tensor's engine and composed offsets.
template <class T> T legacy_unwrap(T value) {
  // Own the child before the parent is destroyed: get(IntTuple) returns a view.
  if (!core::shape(value).isLeaf() && value.rank() == 1)
    return T(core::get(value, 0));
  return value;
}

} // namespace detail

/// A single selected mode retains its own tuple profile.
template <class T> auto legacy_take(const T &value, int32_t begin, int32_t end) {
  return detail::legacy_unwrap(core::take(value, begin, end));
}
template <class T> auto legacy_select(const T &value, llvm::ArrayRef<int32_t> indices) {
  return detail::legacy_unwrap(core::select(value, indices));
}

/// Tuple coordinates remove one result wrapper, preserving the selected child.
/// Scalar coordinates preserve Core's result.
template <class Coord, class T> auto legacy_slice(const Coord &coord, const T &value) {
  auto result = core::slice(coord, value);
  return coord.isLeaf() ? result : detail::legacy_unwrap(std::move(result));
}
template <class Coord, class T> auto legacy_dice(const Coord &coord, const T &value) {
  auto result = core::dice(coord, value);
  return coord.isLeaf() ? result : detail::legacy_unwrap(std::move(result));
}

} // namespace mlir::fly::core

#endif // FLYDSL_DIALECT_FLY_UTILS_LEGACYLAYOUT_H
