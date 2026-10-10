// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_CORE_ALGEBRA_COMPOSEDLAYOUT_HPP
#define FLYDSL_CORE_ALGEBRA_COMPOSEDLAYOUT_HPP

#include "flydsl/Core/Algebra/Config.hpp"
#include "flydsl/Core/Algebra/Layout.hpp"
#include "flydsl/Core/Algebra/Swizzle.hpp"

#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

namespace mlir::fly::core {

template <class Cat> struct ComposedLayout;

template <class Cat> using AnyLayout = std::variant<Layout<Cat>, ComposedLayout<Cat>>;

template <class Cat> using ComposedOuter = AnyLayout<Cat>;

template <class Cat>
using ComposedInner =
    std::variant<Layout<Cat>, Swizzle<Cat>, CoordSwizzle<Cat>, ComposedLayout<Cat>>;

template <class Cat> struct ComposedLayout {
public:
  using IntTuple = core::IntTuple<Cat>;
  using IntTupleRef = core::IntTupleRef<Cat>;

  struct Storage;

  ComposedLayout();
  static ComposedLayout getError(ErrorInfo info);
  bool isError() const;
  /// The first error, in offset, outer, then inner order.
  ErrorInfo errorInfo() const;
  ErrorCode errorCode() const { return errorInfo().reason(); }
  ComposedLayout(ComposedInner<Cat> inner, IntTuple offset, ComposedOuter<Cat> outer);

  static std::optional<ComposedLayout> fromString(llvm::StringRef text);

  const ComposedInner<Cat> &inner() const;
  IntTupleRef offset() const FLYDSL_CORE_LIFETIMEBOUND;
  const ComposedOuter<Cat> &outer() const;

  const ComposedInner<Cat> &layout_a() const { return inner(); }
  const ComposedOuter<Cat> &layout_b() const { return outer(); }
  const ComposedLayout &layout() const { return *this; }

  IntTupleRef shape() const FLYDSL_CORE_LIFETIMEBOUND;

  int32_t rank() const;
  int32_t depth() const;
  auto size() const;

  bool isStatic() const;

  template <class Coord, std::enable_if_t<is_int_tuple_v<Coord>, int> = 0>
  auto operator()(const Coord &coord) const;
  template <class Coord0, class Coord1, class... Coords,
            std::enable_if_t<is_make_tuple_arg_v<Coord0, Coord1, Coords...>, int> = 0>
  auto operator()(const Coord0 &c0, const Coord1 &c1, const Coords &...coords) const;

  template <class Other> auto compose(const Other &other) const;
  template <class Other0, class Other1, class... Others>
  auto compose(const Other0 &other0, const Other1 &other1, const Others &...others) const;

  template <class Other> auto tile(const Other &other) const;
  template <class Other0, class Other1, class... Others>
  auto tile(const Other0 &other0, const Other1 &other1, const Others &...others) const;

  template <class Shape, std::enable_if_t<is_int_tuple_v<Shape>, int> = 0>
  auto with_shape(const Shape &shape) const;
  template <class Shape0, class Shape1, class... Shapes>
  auto with_shape(const Shape0 &shape0, const Shape1 &shape1, const Shapes &...shapes) const;

  friend bool operator==(const ComposedLayout &lhs, const ComposedLayout &rhs) {
    return lhs.inner() == rhs.inner() && lhs.offset() == rhs.offset() && lhs.outer() == rhs.outer();
  }
  friend bool operator!=(const ComposedLayout &lhs, const ComposedLayout &rhs) {
    return !(lhs == rhs);
  }

private:
  std::shared_ptr<const Storage> storage_;
};

template <class Cat> struct ComposedLayout<Cat>::Storage {
  Storage(ComposedInner<Cat> inner, IntTuple offset, ComposedOuter<Cat> outer);

  ComposedInner<Cat> inner;
  IntTuple offset;
  ComposedOuter<Cat> outer;
};

template <class Cat> llvm::hash_code hash_value(const ComposedLayout<Cat> &layout);

template <class Cat>
ComposedLayout<Cat> layout(const ComposedLayout<Cat> &value, llvm::ArrayRef<int32_t> path);

template <int... Is, class Cat> ComposedLayout<Cat> layout(const ComposedLayout<Cat> &value);

template <int... Is, class Cat>
IntTupleRef<Cat> shape(const ComposedLayout<Cat> &value FLYDSL_CORE_LIFETIMEBOUND);

template <int... Is, class Cat> IntTuple<Cat> size(const ComposedLayout<Cat> &value);

template <int... Is, class Cat> int32_t rank(const ComposedLayout<Cat> &value);

template <int... Is, class Cat> int32_t depth(const ComposedLayout<Cat> &value);

template <class Cat> ComposedLayout<Cat> get(const ComposedLayout<Cat> &value, int32_t i);

template <int... Is, class Cat> ComposedLayout<Cat> get(const ComposedLayout<Cat> &value);

template <class Cat>
ComposedLayout<Cat> take(const ComposedLayout<Cat> &value, int32_t begin, int32_t end);

template <class Cat>
ComposedLayout<Cat> select(const ComposedLayout<Cat> &value, llvm::ArrayRef<int32_t> indices);

template <class Cat> ComposedLayout<Cat> flatten(const ComposedLayout<Cat> &value);

template <class Cat, class Profile>
ComposedLayout<Cat> unflatten(const ComposedLayout<Cat> &value, const Profile &targetProfile);

template <class Cat>
ComposedLayout<Cat> group(const ComposedLayout<Cat> &value, int32_t begin, int32_t end);

template <class Cat>
ComposedLayout<Cat> append(const ComposedLayout<Cat> &value, const Layout<Cat> &sub);

template <class Cat>
ComposedLayout<Cat> append(const ComposedLayout<Cat> &value, const Layout<Cat> &sub, int32_t n);

template <class Cat>
ComposedLayout<Cat> prepend(const ComposedLayout<Cat> &value, const Layout<Cat> &sub);

template <class Cat>
ComposedLayout<Cat> prepend(const ComposedLayout<Cat> &value, const Layout<Cat> &sub, int32_t n);

template <class Cat>
ComposedLayout<Cat> replace(const ComposedLayout<Cat> &value, int32_t i, const Layout<Cat> &sub);

template <class Coord, class Cat>
std::pair<ComposedLayout<Cat>, IntTuple<Cat>> slice_and_offset(const Coord &coord,
                                                               const ComposedLayout<Cat> &value);

template <class Coord, class Cat>
ComposedLayout<Cat> dice(const Coord &coord, const ComposedLayout<Cat> &value);

template <class Coord, class Cat>
std::pair<ComposedLayout<Cat>, IntTuple<Cat>> domain_offset(const Coord &coord,
                                                            const ComposedLayout<Cat> &value);

template <class Cat, class Tiler>
ComposedLayout<Cat> composition(const ComposedLayout<Cat> &lhs, const Tiler &rhs);

template <class Cat>
ComposedLayout<Cat> composition(const Layout<Cat> &lhs, const ComposedLayout<Cat> &rhs);

template <class Cotarget, class Cat, std::enable_if_t<is_int_tuple_v<Cotarget>, int>>
Layout<Cat> complement(const ComposedLayout<Cat> &value, const Cotarget &cotarget);

template <class Cat> ComposedLayout<Cat> zip(const ComposedLayout<Cat> &value);

template <class Cat, class Shape>
ComposedLayout<Cat> tile_to_shape(const ComposedLayout<Cat> &value, const Shape &targetShape,
                                  LayoutLeft order = {});

template <class Cat, class Shape, class Order, std::enable_if_t<is_int_tuple_v<Order>, int>>
ComposedLayout<Cat> tile_to_shape(const ComposedLayout<Cat> &value, const Shape &targetShape,
                                  const Order &order);

template <class Cat> ComposedLayout<Cat> filter_zeros(const ComposedLayout<Cat> &value);

template <class Cat, class Profile>
ComposedLayout<Cat> filter_zeros(const ComposedLayout<Cat> &value, const Profile &targetProfile);

template <class Cat> ComposedLayout<Cat> filter(const ComposedLayout<Cat> &value);

template <class Cat, class Profile>
ComposedLayout<Cat> filter(const ComposedLayout<Cat> &value, const Profile &targetProfile);

template <class Cat> ComposedLayout<Cat> coalesce(const ComposedLayout<Cat> &value);

template <class Cat, class Profile>
ComposedLayout<Cat> coalesce(const ComposedLayout<Cat> &value, const Profile &targetProfile);

template <class Cat> ComposedLayout<Cat> coalesce_x(const ComposedLayout<Cat> &value);

template <class Cat, class Profile>
ComposedLayout<Cat> coalesce_x(const ComposedLayout<Cat> &value, const Profile &targetProfile);

template <class Cat> ComposedLayout<Cat> upcast(int32_t n, const ComposedLayout<Cat> &value);

template <class Cat> ComposedLayout<Cat> downcast(int32_t n, const ComposedLayout<Cat> &value);

template <class Cat>
ComposedLayout<Cat> recast_layout(int32_t newTypeBits, int32_t oldTypeBits,
                                  const ComposedLayout<Cat> &value);

template <class Cat> auto max_alignment(const ComposedLayout<Cat> &value);

template <class Cat> void print(const ComposedLayout<Cat> &value, llvm::raw_ostream &os);

template <class Cat> void print(const AnyLayout<Cat> &value, llvm::raw_ostream &os);

template <class Cat> void print(const ComposedInner<Cat> &value, llvm::raw_ostream &os);

template <class Cat> llvm::hash_code hash_value(const AnyLayout<Cat> &value);

template <class Cat> llvm::hash_code hash_value(const ComposedInner<Cat> &value);

//===----------------------------------------------------------------------===//
// Definitions
//===----------------------------------------------------------------------===//

// An error value still retains a traversable layout tree.
template <class Cat>
ComposedLayout<Cat>::ComposedLayout()
    : ComposedLayout(Swizzle<Cat>{}, IntTuple::getZero(),
                     Layout<Cat>(IntTuple::getError(ErrorCode::InvalidLayout),
                                 IntTuple::getError(ErrorCode::InvalidLayout))) {}

template <class Cat> ComposedLayout<Cat> ComposedLayout<Cat>::getError(ErrorInfo info) {
  return ComposedLayout(Swizzle<Cat>{}, IntTuple::getZero(), Layout<Cat>::getError(info));
}

/// The constructor collapses an invalid part into an invalid outer layout.
template <class Cat> bool ComposedLayout<Cat>::isError() const {
  auto *layout = std::get_if<Layout<Cat>>(&outer());
  return layout && layout->isError();
}

template <class Cat> ErrorInfo ComposedLayout<Cat>::errorInfo() const {
  if (isError())
    return std::get<Layout<Cat>>(outer()).errorInfo();
  FLYDSL_CORE_ASSERT(false && "composed layout has no error");
  return ErrorCode::InvalidLayout;
}

namespace detail {

/// Records `op` in the error of an invalid `value`; a valid value is returned
/// unchanged.
template <class Cat>
ComposedLayout<Cat> tagError(ComposedLayout<Cat> value, AlgebraOp op, int64_t mode = -1) {
  if (!value.isError())
    return value;
  return ComposedLayout<Cat>::getError(value.errorInfo().withFrame(op, mode));
}

} // namespace detail

template <class Cat> IntTupleRef<Cat> ComposedLayout<Cat>::shape() const {
  return std::visit([](const auto &value) { return value.shape(); }, outer());
}

template <class Cat> int32_t ComposedLayout<Cat>::rank() const { return shape().rank(); }

template <class Cat> int32_t ComposedLayout<Cat>::depth() const { return shape().depth(); }

template <class Cat> auto ComposedLayout<Cat>::size() const { return product(shape()); }

template <class Cat> bool ComposedLayout<Cat>::isStatic() const {
  auto innerStatic = std::visit(
      [](const auto &value) {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, Layout<Cat>> || std::is_same_v<T, ComposedLayout<Cat>>)
          return value.isStatic();
        else
          return true;
      },
      inner());
  auto outerStatic = std::visit([](const auto &value) { return value.isStatic(); }, outer());
  return innerStatic && storage_->offset.isStatic() && outerStatic;
}

template <class Cat>
template <class Coord, std::enable_if_t<is_int_tuple_v<Coord>, int>>
auto ComposedLayout<Cat>::operator()(const Coord &coord) const {
  static_assert(std::is_same_v<Cat, category_of<Coord>>);
  return slice(coord, *this);
}

template <class Cat>
template <class Coord0, class Coord1, class... Coords,
          std::enable_if_t<is_make_tuple_arg_v<Coord0, Coord1, Coords...>, int>>
auto ComposedLayout<Cat>::operator()(const Coord0 &c0, const Coord1 &c1,
                                     const Coords &...coords) const {
  auto coord = make_tuple(c0, c1, coords...);
  return operator()(coord.asRef());
}

template <class Cat>
template <class Other>
auto ComposedLayout<Cat>::compose(const Other &other) const {
  return composition(*this, other);
}

template <class Cat>
template <class Other0, class Other1, class... Others>
auto ComposedLayout<Cat>::compose(const Other0 &other0, const Other1 &other1,
                                  const Others &...others) const {
  return composition(*this, make_tile(other0, other1, others...));
}

template <class Cat>
template <class Other>
auto ComposedLayout<Cat>::tile(const Other &other) const {
  return tiled_divide(*this, other);
}

template <class Cat>
template <class Other0, class Other1, class... Others>
auto ComposedLayout<Cat>::tile(const Other0 &other0, const Other1 &other1,
                               const Others &...others) const {
  return tiled_divide(*this, make_tile(other0, other1, others...));
}

template <class Cat>
template <class Shape, std::enable_if_t<is_int_tuple_v<Shape>, int>>
auto ComposedLayout<Cat>::with_shape(const Shape &newShape) const {
  return composition(*this, make_layout(newShape));
}

template <class Cat>
template <class Shape0, class Shape1, class... Shapes>
auto ComposedLayout<Cat>::with_shape(const Shape0 &shape0, const Shape1 &shape1,
                                     const Shapes &...shapes) const {
  return composition(*this, make_layout(make_tuple(shape0, shape1, shapes...)));
}

template <class Cat>
ComposedLayout<Cat>::Storage::Storage(ComposedInner<Cat> inner, IntTuple offset,
                                      ComposedOuter<Cat> outer)
    : inner(std::move(inner)), offset(std::move(offset)), outer(std::move(outer)) {}

template <class Cat>
ComposedLayout<Cat>::ComposedLayout(ComposedInner<Cat> inner, IntTuple offset,
                                    ComposedOuter<Cat> outer) {
  // An invalid part keeps only its error, the first in offset, outer, then
  // inner order, as the invalid outer layout of an identity composition.
  auto layoutError = [](const auto &value) -> std::optional<ErrorInfo> {
    using T = std::decay_t<decltype(value)>;
    if constexpr (std::is_same_v<T, Layout<Cat>> || std::is_same_v<T, ComposedLayout<Cat>>) {
      if (value.isError())
        return value.errorInfo();
    }
    return std::nullopt;
  };
  auto info = offset.isError() ? std::optional(offset.errorInfo()) : std::visit(layoutError, outer);
  if (!info)
    info = std::visit(layoutError, inner);
  if (info) {
    inner = Swizzle<Cat>{};
    offset = IntTuple::getZero();
    outer = Layout<Cat>::getError(*info);
  }
  storage_ = std::make_shared<Storage>(std::move(inner), std::move(offset), std::move(outer));
}

template <class Cat> const ComposedInner<Cat> &ComposedLayout<Cat>::inner() const {
  return storage_->inner;
}

template <class Cat> IntTupleRef<Cat> ComposedLayout<Cat>::offset() const {
  return storage_->offset.asRef();
}

template <class Cat> const ComposedOuter<Cat> &ComposedLayout<Cat>::outer() const {
  return storage_->outer;
}

template <class Cat> llvm::hash_code hash_value(const ComposedLayout<Cat> &layout) {
  return llvm::hash_combine(hash_value(layout.inner()), hash_value(layout.offset()),
                            hash_value(layout.outer()));
}

template <class Cat>
ComposedLayout<Cat> make_composed_layout(ComposedInner<Cat> inner, IntTuple<Cat> offset,
                                         ComposedOuter<Cat> outer) {
  return ComposedLayout<Cat>(std::move(inner), std::move(offset), std::move(outer));
}

template <class Inner, class Offset, class Outer, class Cat = category_of<Offset>,
          std::enable_if_t<is_int_tuple_v<Offset>, int> = 0>
ComposedLayout<Cat> make_composed_layout(const Inner &inner, const Offset &offset,
                                         const Outer &outer) {
  return ComposedLayout<Cat>(inner, offset.asRef(), ComposedOuter<Cat>(outer));
}

template <class Inner, class Offset, class Outer, class Cat = category_of<Offset>,
          std::enable_if_t<is_int_tuple_v<Offset>, int> = 0>
ComposedLayout<Cat> composition(const Inner &inner, const Offset &offset, const Outer &outer) {
  return make_composed_layout(inner, offset, outer);
}

template <class Cat> ComposedLayout<Cat> composition(Swizzle<Cat> inner, const Layout<Cat> &outer) {
  return composition(inner, IntTuple<Cat>::getZero(), outer);
}

template <class Cat>
ComposedLayout<Cat> composition(const CoordSwizzle<Cat> &inner, const Layout<Cat> &outer) {
  return composition(inner, IntTuple<Cat>::getZero(), outer);
}

template <class Coord, class Cat = category_of<Coord>,
          std::enable_if_t<is_int_tuple_v<Coord>, int> = 0>
IntTuple<Cat> crd2idx(const Coord &coord, const ComposedLayout<Cat> &layout) {
  auto outerResult =
      std::visit([&](const auto &value) { return crd2idx(coord, value); }, layout.outer());
  auto intermediate = layout.offset() + outerResult;
  return std::visit(
      [&](const auto &fn) -> IntTuple<Cat> {
        using T = std::decay_t<decltype(fn)>;
        if constexpr (std::is_same_v<T, Layout<Cat>> || std::is_same_v<T, ComposedLayout<Cat>>) {
          return crd2idx(intermediate.asRef(), fn);
        } else if constexpr (std::is_same_v<T, Swizzle<Cat>>) {
          return swizzle_apply(fn, intermediate.asRef());
        } else {
          static_assert(std::is_same_v<T, CoordSwizzle<Cat>>);
          return swizzle_apply(fn, intermediate.asRef());
        }
      },
      layout.inner());
}

template <class Cat>
ComposedLayout<Cat> layout(const ComposedLayout<Cat> &value, llvm::ArrayRef<int32_t> path) {
  if (path.empty())
    return value;
  auto newOuter = std::visit(
      [&](const auto &outer) -> AnyLayout<Cat> { return layout(outer, path); }, value.outer());
  return make_composed_layout(value.inner(), value.offset(), std::move(newOuter));
}
template <int... Is, class Cat> ComposedLayout<Cat> layout(const ComposedLayout<Cat> &value) {
  if constexpr (sizeof...(Is) == 0)
    return value;
  else
    return layout(value, {Is...});
}

template <class Cat>
IntTupleRef<Cat> shape(const ComposedLayout<Cat> &value FLYDSL_CORE_LIFETIMEBOUND,
                       llvm::ArrayRef<int32_t> path) {
  return value.shape().at(path);
}
template <int... Is, class Cat>
IntTupleRef<Cat> shape(const ComposedLayout<Cat> &value FLYDSL_CORE_LIFETIMEBOUND) {
  if constexpr (sizeof...(Is) == 0)
    return value.shape();
  else
    return shape(value, {Is...});
}

template <class Cat> auto size(const ComposedLayout<Cat> &value, llvm::ArrayRef<int32_t> path) {
  return product(shape(value, path));
}
template <int... Is, class Cat> IntTuple<Cat> size(const ComposedLayout<Cat> &value) {
  if constexpr (sizeof...(Is) == 0)
    return value.size();
  else
    return size(value, {Is...});
}

template <class Cat> int32_t rank(const ComposedLayout<Cat> &value, llvm::ArrayRef<int32_t> path) {
  return shape(value, path).rank();
}
template <int... Is, class Cat> int32_t rank(const ComposedLayout<Cat> &value) {
  if constexpr (sizeof...(Is) == 0)
    return value.rank();
  else
    return rank(value, {Is...});
}

template <class Cat> int32_t depth(const ComposedLayout<Cat> &value, llvm::ArrayRef<int32_t> path) {
  return shape(value, path).depth();
}
template <int... Is, class Cat> int32_t depth(const ComposedLayout<Cat> &value) {
  if constexpr (sizeof...(Is) == 0)
    return value.depth();
  else
    return depth(value, {Is...});
}

template <class Cat> IntTuple<Cat> coprofile(const ComposedLayout<Cat> &value) {
  return std::visit([](const auto &outer) { return coprofile(outer); }, value.outer());
}

template <class Cat> IntTuple<Cat> coshape(const ComposedLayout<Cat> &value) {
  return std::visit([](const auto &outer) { return coshape(outer); }, value.outer());
}

template <class Cat> IntTuple<Cat> cosize(const ComposedLayout<Cat> &value) {
  return std::visit([](const auto &outer) { return cosize(outer); }, value.outer());
}
template <class Cat>
IntTuple<Cat> cosize(const ComposedLayout<Cat> &value, llvm::ArrayRef<int32_t> path) {
  return std::visit([&](const auto &outer) { return cosize(outer, path); }, value.outer());
}
template <int I, int... Is, class Cat> auto cosize(const ComposedLayout<Cat> &value) {
  return cosize(value, {I, Is...});
}

template <class Cat> ComposedLayout<Cat> get(const ComposedLayout<Cat> &value, int32_t i) {
  auto newOuter =
      std::visit([&](const auto &outer) -> AnyLayout<Cat> { return get(outer, i); }, value.outer());
  return make_composed_layout(value.inner(), value.offset(), std::move(newOuter));
}
template <int... Is, class Cat> ComposedLayout<Cat> get(const ComposedLayout<Cat> &value) {
  if constexpr (sizeof...(Is) == 0)
    return value;
  else
    return layout<Is...>(value);
}

template <class Cat>
ComposedLayout<Cat> take(const ComposedLayout<Cat> &value, int32_t begin, int32_t end) {
  auto newOuter = std::visit(
      [&](const auto &outer) -> AnyLayout<Cat> { return take(outer, begin, end); }, value.outer());
  return make_composed_layout(value.inner(), value.offset(), std::move(newOuter));
}
template <int Begin, int End, class Cat>
ComposedLayout<Cat> take(const ComposedLayout<Cat> &value) {
  return take(value, Begin, End);
}

template <class Cat>
ComposedLayout<Cat> select(const ComposedLayout<Cat> &value, llvm::ArrayRef<int32_t> indices) {
  auto newOuter = std::visit(
      [&](const auto &outer) -> AnyLayout<Cat> { return select(outer, indices); }, value.outer());
  return make_composed_layout(value.inner(), value.offset(), std::move(newOuter));
}

template <class Cat> ComposedLayout<Cat> flatten(const ComposedLayout<Cat> &value) {
  auto newOuter =
      std::visit([](const auto &outer) -> AnyLayout<Cat> { return flatten(outer); }, value.outer());
  return make_composed_layout(value.inner(), value.offset(), std::move(newOuter));
}

template <class Cat, class Profile>
ComposedLayout<Cat> unflatten(const ComposedLayout<Cat> &value, const Profile &targetProfile) {
  auto newOuter = std::visit(
      [&](const auto &outer) -> AnyLayout<Cat> { return unflatten(outer, targetProfile); },
      value.outer());
  return make_composed_layout(value.inner(), value.offset(), std::move(newOuter));
}

template <class Cat>
ComposedLayout<Cat> group(const ComposedLayout<Cat> &value, int32_t begin, int32_t end) {
  auto newOuter = std::visit(
      [&](const auto &outer) -> AnyLayout<Cat> { return group(outer, begin, end); }, value.outer());
  return make_composed_layout(value.inner(), value.offset(), std::move(newOuter));
}
template <int Begin, int End, class Cat>
ComposedLayout<Cat> group(const ComposedLayout<Cat> &value) {
  return group(value, Begin, End);
}

template <class Cat>
ComposedLayout<Cat> append(const ComposedLayout<Cat> &value, const Layout<Cat> &sub) {
  auto newOuter = std::visit(
      [&](const auto &outer) -> AnyLayout<Cat> { return append(outer, sub); }, value.outer());
  return make_composed_layout(value.inner(), value.offset(), std::move(newOuter));
}

template <class Cat>
ComposedLayout<Cat> append(const ComposedLayout<Cat> &value, const Layout<Cat> &sub, int32_t n) {
  auto newOuter = std::visit(
      [&](const auto &outer) -> AnyLayout<Cat> { return append(outer, sub, n); }, value.outer());
  return make_composed_layout(value.inner(), value.offset(), std::move(newOuter));
}
template <int N, class Cat>
ComposedLayout<Cat> append(const ComposedLayout<Cat> &value, const Layout<Cat> &sub) {
  return append(value, sub, N);
}

template <class Cat> ComposedLayout<Cat> append(const ComposedLayout<Cat> &value, int32_t n) {
  auto unit = make_layout(IntTuple<Cat>::getOne(), IntTuple<Cat>::getZero());
  return append(value, unit, n);
}

template <class Cat>
ComposedLayout<Cat> prepend(const ComposedLayout<Cat> &value, const Layout<Cat> &sub) {
  auto newOuter = std::visit(
      [&](const auto &outer) -> AnyLayout<Cat> { return prepend(outer, sub); }, value.outer());
  return make_composed_layout(value.inner(), value.offset(), std::move(newOuter));
}

template <class Cat>
ComposedLayout<Cat> prepend(const ComposedLayout<Cat> &value, const Layout<Cat> &sub, int32_t n) {
  auto newOuter = std::visit(
      [&](const auto &outer) -> AnyLayout<Cat> { return prepend(outer, sub, n); }, value.outer());
  return make_composed_layout(value.inner(), value.offset(), std::move(newOuter));
}
template <int N, class Cat>
ComposedLayout<Cat> prepend(const ComposedLayout<Cat> &value, const Layout<Cat> &sub) {
  return prepend(value, sub, N);
}

template <class Cat> ComposedLayout<Cat> prepend(const ComposedLayout<Cat> &value, int32_t n) {
  auto unit = make_layout(IntTuple<Cat>::getOne(), IntTuple<Cat>::getZero());
  return prepend(value, unit, n);
}

template <class Cat>
ComposedLayout<Cat> replace(const ComposedLayout<Cat> &value, int32_t i, const Layout<Cat> &sub) {
  auto newOuter = std::visit(
      [&](const auto &outer) -> AnyLayout<Cat> { return replace(outer, i, sub); }, value.outer());
  return make_composed_layout(value.inner(), value.offset(), std::move(newOuter));
}

template <class Coord, class Cat = category_of<Coord>>
std::pair<ComposedLayout<Cat>, IntTuple<Cat>> slice_and_offset(const Coord &coord,
                                                               const ComposedLayout<Cat> &value) {
  return std::visit(
      [&](const auto &outer) {
        auto [slicedOuter, offset] = slice_and_offset(coord, outer);
        IntTuple<Cat> baseOffset = value.offset();
        auto newOffset = offset.isZero()       ? std::move(baseOffset)
                         : baseOffset.isZero() ? std::move(offset)
                                               : baseOffset + offset;
        auto sliced = make_composed_layout(value.inner(), std::move(newOffset),
                                           ComposedOuter<Cat>(std::move(slicedOuter)));
        return std::make_pair(std::move(sliced), IntTuple<Cat>::getZero());
      },
      value.outer());
}

template <class Coord, class Cat = category_of<Coord>>
ComposedLayout<Cat> slice(const Coord &coord, const ComposedLayout<Cat> &value) {
  return slice_and_offset(coord, value).first;
}

template <class Coord, class Cat = category_of<Coord>>
ComposedLayout<Cat> dice(const Coord &coord, const ComposedLayout<Cat> &value) {
  auto newOuter = std::visit(
      [&](const auto &outer) -> AnyLayout<Cat> { return dice(coord, outer); }, value.outer());
  return make_composed_layout(value.inner(), value.offset(), std::move(newOuter));
}

template <class Coord, class Cat = category_of<Coord>>
std::pair<ComposedLayout<Cat>, IntTuple<Cat>> domain_offset(const Coord &coord,
                                                            const ComposedLayout<Cat> &value) {
  auto outerOffset =
      std::visit([&](const auto &outer) { return crd2idx(coord, outer); }, value.outer());
  auto result = make_composed_layout(value.inner(), value.offset() + outerOffset, value.outer());
  return std::make_pair(std::move(result), IntTuple<Cat>::getZero());
}

template <class Cat, class Tiler>
ComposedLayout<Cat> composition(const ComposedLayout<Cat> &lhs, const Tiler &rhs) {
  auto newOuter = std::visit(
      [&](const auto &outer) -> AnyLayout<Cat> { return composition(outer, rhs); }, lhs.outer());
  return make_composed_layout(lhs.inner(), lhs.offset(), std::move(newOuter));
}

template <class Cat>
ComposedLayout<Cat> composition(const Layout<Cat> &lhs, const ComposedLayout<Cat> &rhs) {
  if (auto error = detail::firstError(lhs, rhs))
    return ComposedLayout<Cat>::getError(error->withFrame(AlgebraOp::Composition));
  if (!rhs.offset().isZero())
    return ComposedLayout<Cat>::getError(
        ErrorInfo(ErrorCode::InvalidLayout).withFrame(AlgebraOp::Composition));
  return std::visit(
      [&](const auto &inner) -> ComposedLayout<Cat> {
        using T = std::decay_t<decltype(inner)>;
        if constexpr (std::is_same_v<T, Layout<Cat>> || std::is_same_v<T, ComposedLayout<Cat>>) {
          auto composed = composition(lhs, inner);
          if (composed.isError())
            return ComposedLayout<Cat>::getError(
                composed.errorInfo().withFrame(AlgebraOp::Composition));
          return make_composed_layout(composed, rhs.offset(), rhs.outer());
        } else {
          return ComposedLayout<Cat>::getError(ErrorInfo(ErrorCode::UnsupportedSwizzleComposition)
                                                   .withFrame(AlgebraOp::Composition));
        }
      },
      rhs.inner());
}

template <class Cotarget, class Cat = category_of<Cotarget>,
          std::enable_if_t<is_int_tuple_v<Cotarget>, int> = 0>
Layout<Cat> complement(const ComposedLayout<Cat> &value, const Cotarget &cotarget) {
  return std::visit([&](const auto &outer) { return complement(outer, cotarget); }, value.outer());
}

template <class Cat> Layout<Cat> complement(const ComposedLayout<Cat> &value) {
  return std::visit([](const auto &outer) { return complement(outer); }, value.outer());
}

template <class Cat> ComposedLayout<Cat> zip(const ComposedLayout<Cat> &value) {
  auto newOuter =
      std::visit([](const auto &outer) -> AnyLayout<Cat> { return zip(outer); }, value.outer());
  return make_composed_layout(value.inner(), value.offset(), std::move(newOuter));
}

#define FLYDSL_CORE_COMPOSED_OUTER_ALGORITHM(name)                                                 \
  template <class Cat, class Tiler>                                                                \
  ComposedLayout<Cat> name(const ComposedLayout<Cat> &value, const Tiler &tiler) {                 \
    auto newOuter = std::visit(                                                                    \
        [&](const auto &outer) -> AnyLayout<Cat> { return name(outer, tiler); }, value.outer());   \
    return make_composed_layout(value.inner(), value.offset(), std::move(newOuter));               \
  }

FLYDSL_CORE_COMPOSED_OUTER_ALGORITHM(logical_divide)
FLYDSL_CORE_COMPOSED_OUTER_ALGORITHM(tile_unzip)
FLYDSL_CORE_COMPOSED_OUTER_ALGORITHM(tiled_divide)
FLYDSL_CORE_COMPOSED_OUTER_ALGORITHM(zipped_divide)
FLYDSL_CORE_COMPOSED_OUTER_ALGORITHM(flat_divide)
FLYDSL_CORE_COMPOSED_OUTER_ALGORITHM(logical_product)
FLYDSL_CORE_COMPOSED_OUTER_ALGORITHM(zipped_product)
FLYDSL_CORE_COMPOSED_OUTER_ALGORITHM(tiled_product)
FLYDSL_CORE_COMPOSED_OUTER_ALGORITHM(flat_product)
FLYDSL_CORE_COMPOSED_OUTER_ALGORITHM(blocked_product)
FLYDSL_CORE_COMPOSED_OUTER_ALGORITHM(raked_product)

#undef FLYDSL_CORE_COMPOSED_OUTER_ALGORITHM

template <class Cat, class Shape>
ComposedLayout<Cat> tile_to_shape(const ComposedLayout<Cat> &value, const Shape &targetShape,
                                  LayoutLeft order) {
  auto newOuter = std::visit(
      [&](const auto &outer) -> AnyLayout<Cat> { return tile_to_shape(outer, targetShape, order); },
      value.outer());
  return make_composed_layout(value.inner(), value.offset(), std::move(newOuter));
}

template <class Cat, class Shape, class Order, std::enable_if_t<is_int_tuple_v<Order>, int> = 0>
ComposedLayout<Cat> tile_to_shape(const ComposedLayout<Cat> &value, const Shape &targetShape,
                                  const Order &order) {
  auto newOuter = std::visit(
      [&](const auto &outer) -> AnyLayout<Cat> { return tile_to_shape(outer, targetShape, order); },
      value.outer());
  return make_composed_layout(value.inner(), value.offset(), std::move(newOuter));
}

template <class Cat> ComposedLayout<Cat> filter_zeros(const ComposedLayout<Cat> &value) {
  auto newOuter = std::visit(
      [](const auto &outer) -> AnyLayout<Cat> { return filter_zeros(outer); }, value.outer());
  return make_composed_layout(value.inner(), value.offset(), std::move(newOuter));
}

template <class Cat, class Profile>
ComposedLayout<Cat> filter_zeros(const ComposedLayout<Cat> &value, const Profile &targetProfile) {
  auto newOuter = std::visit(
      [&](const auto &outer) -> AnyLayout<Cat> { return filter_zeros(outer, targetProfile); },
      value.outer());
  return make_composed_layout(value.inner(), value.offset(), std::move(newOuter));
}

template <class Cat> ComposedLayout<Cat> filter(const ComposedLayout<Cat> &value) {
  auto newOuter =
      std::visit([](const auto &outer) -> AnyLayout<Cat> { return filter(outer); }, value.outer());
  return make_composed_layout(value.inner(), value.offset(), std::move(newOuter));
}

template <class Cat, class Profile>
ComposedLayout<Cat> filter(const ComposedLayout<Cat> &value, const Profile &targetProfile) {
  auto newOuter =
      std::visit([&](const auto &outer) -> AnyLayout<Cat> { return filter(outer, targetProfile); },
                 value.outer());
  return make_composed_layout(value.inner(), value.offset(), std::move(newOuter));
}

template <class Cat> ComposedLayout<Cat> coalesce(const ComposedLayout<Cat> &value) {
  auto newOuter = std::visit([](const auto &outer) -> AnyLayout<Cat> { return coalesce(outer); },
                             value.outer());
  return make_composed_layout(value.inner(), value.offset(), std::move(newOuter));
}

template <class Cat, class Profile>
ComposedLayout<Cat> coalesce(const ComposedLayout<Cat> &value, const Profile &targetProfile) {
  auto newOuter = std::visit(
      [&](const auto &outer) -> AnyLayout<Cat> { return coalesce(outer, targetProfile); },
      value.outer());
  return make_composed_layout(value.inner(), value.offset(), std::move(newOuter));
}

template <class Cat> ComposedLayout<Cat> coalesce_x(const ComposedLayout<Cat> &value) {
  auto newOuter = std::visit([](const auto &outer) -> AnyLayout<Cat> { return coalesce_x(outer); },
                             value.outer());
  return make_composed_layout(value.inner(), value.offset(), std::move(newOuter));
}

template <class Cat, class Profile>
ComposedLayout<Cat> coalesce_x(const ComposedLayout<Cat> &value, const Profile &targetProfile) {
  auto newOuter = std::visit(
      [&](const auto &outer) -> AnyLayout<Cat> { return coalesce_x(outer, targetProfile); },
      value.outer());
  return make_composed_layout(value.inner(), value.offset(), std::move(newOuter));
}

template <class Cat> ComposedLayout<Cat> upcast(int32_t n, const ComposedLayout<Cat> &value) {
  if (value.isError())
    return detail::tagError(value, AlgebraOp::Upcast);
  auto hasSwizzle = std::holds_alternative<Swizzle<Cat>>(value.inner()) ||
                    std::holds_alternative<CoordSwizzle<Cat>>(value.inner());
  if (n <= 0 || (hasSwizzle && (n & (n - 1)) != 0))
    return ComposedLayout<Cat>::getError(
        ErrorInfo(ErrorCode::InvalidRecastFactor).withFrame(AlgebraOp::Upcast));
  auto inner = std::visit([&](const auto &item) -> ComposedInner<Cat> { return upcast(n, item); },
                          value.inner());
  auto factor = IntTuple<Cat>::getSInt(n);
  auto offset = transform_leaf(value.offset(),
                               [&](IntTupleRef<Cat> leaf) { return leaf_safe_div(leaf, factor); });
  if (offset.isError())
    return ComposedLayout<Cat>::getError(offset.errorInfo().withFrame(AlgebraOp::Upcast));
  auto outer = std::visit([&](const auto &item) -> AnyLayout<Cat> { return upcast(n, item); },
                          value.outer());
  auto result = make_composed_layout(std::move(inner), std::move(offset), std::move(outer));
  return detail::tagError(std::move(result), AlgebraOp::Upcast);
}

template <class Cat> ComposedLayout<Cat> downcast(int32_t n, const ComposedLayout<Cat> &value) {
  if (value.isError())
    return detail::tagError(value, AlgebraOp::Downcast);
  auto hasSwizzle = std::holds_alternative<Swizzle<Cat>>(value.inner()) ||
                    std::holds_alternative<CoordSwizzle<Cat>>(value.inner());
  if (n <= 0 || (hasSwizzle && (n & (n - 1)) != 0))
    return ComposedLayout<Cat>::getError(
        ErrorInfo(ErrorCode::InvalidRecastFactor).withFrame(AlgebraOp::Downcast));
  auto inner = std::visit([&](const auto &item) -> ComposedInner<Cat> { return downcast(n, item); },
                          value.inner());
  auto factor = IntTuple<Cat>::getSInt(n);
  auto offset =
      transform_leaf(value.offset(), [&](IntTupleRef<Cat> leaf) { return leaf * factor; });
  auto outer = std::visit([&](const auto &item) -> AnyLayout<Cat> { return downcast(n, item); },
                          value.outer());
  auto result = make_composed_layout(std::move(inner), std::move(offset), std::move(outer));
  return detail::tagError(std::move(result), AlgebraOp::Downcast);
}

template <class Cat>
ComposedLayout<Cat> recast_layout(int32_t newTypeBits, int32_t oldTypeBits,
                                  const ComposedLayout<Cat> &value) {
  if (oldTypeBits <= 0 || newTypeBits <= 0)
    return ComposedLayout<Cat>::getError(
        ErrorInfo(ErrorCode::InvalidRecastFactor).withFrame(AlgebraOp::RecastLayout));
  auto divisor = std::gcd(oldTypeBits, newTypeBits);
  auto numerator = newTypeBits / divisor;
  auto denominator = oldTypeBits / divisor;
  if (numerator == 1 && denominator == 1)
    return value;
  if (numerator == 1)
    return detail::tagError(downcast(denominator, value), AlgebraOp::RecastLayout);
  if (denominator == 1)
    return detail::tagError(upcast(numerator, value), AlgebraOp::RecastLayout);
  return detail::tagError(downcast(denominator, upcast(numerator, value)), AlgebraOp::RecastLayout);
}

template <class Cat> auto max_alignment(const ComposedLayout<Cat> &value) {
  const auto *swizzle = std::get_if<Swizzle<Cat>>(&value.inner());
  if (!swizzle || !value.offset().isSInt())
    return IntTuple<Cat>::getOne();
  auto outerAlignment =
      std::visit([](const auto &outer) { return max_alignment(outer); }, value.outer());
  if (!outerAlignment.isSInt())
    return IntTuple<Cat>::getOne();
  auto swizzleAlignment = max_alignment(*swizzle).staticValue();
  auto offset = value.offset().staticValue();
  auto offsetMagnitude = offset < 0 ? -offset : offset;
  auto alignment = std::gcd(swizzleAlignment, outerAlignment.staticValue());
  alignment = std::gcd(alignment, offsetMagnitude);
  return IntTuple<Cat>::getSInt(alignment);
}

template <class Cat> Layout<Cat> nullspace(const ComposedLayout<Cat> &) {
  return make_layout(IntTuple<Cat>::getOne(), IntTuple<Cat>::getZero());
}

template <class Cat> void print(const ComposedLayout<Cat> &value, llvm::raw_ostream &os) {
  bool nestedInner = std::holds_alternative<ComposedLayout<Cat>>(value.inner());
  if (nestedInner)
    os << '[';
  print(value.inner(), os);
  if (nestedInner)
    os << ']';
  os << " o ";
  print(value.offset(), os);
  os << " o ";
  print(value.outer(), os);
}

namespace detail {

template <class Cat> std::optional<ComposedInner<Cat>> parseComposedInner(llvm::StringRef text) {
  if (text.starts_with('[') && detail::findMatching(text, 0, '[', ']') == text.size() - 1) {
    auto value = ComposedLayout<Cat>::fromString(text.drop_front().drop_back());
    return value ? std::optional<ComposedInner<Cat>>(*value) : std::nullopt;
  }
  if (text.starts_with("CS<")) {
    auto value = CoordSwizzle<Cat>::fromString(text);
    return value ? std::optional<ComposedInner<Cat>>(*value) : std::nullopt;
  }
  if (text.starts_with("S<")) {
    auto value = Swizzle<Cat>::fromString(text);
    return value ? std::optional<ComposedInner<Cat>>(*value) : std::nullopt;
  }
  auto value = Layout<Cat>::fromString(text);
  return value ? std::optional<ComposedInner<Cat>>(*value) : std::nullopt;
}

template <class Cat> std::optional<AnyLayout<Cat>> parseAnyLayout(llvm::StringRef text);

} // namespace detail

template <class Cat>
std::optional<ComposedLayout<Cat>> ComposedLayout<Cat>::fromString(llvm::StringRef text) {
  auto first = detail::findTopLevel(text, " o ");
  if (first == llvm::StringRef::npos)
    return std::nullopt;
  auto remainder = text.drop_front(first + 3);
  auto second = detail::findTopLevel(remainder, " o ");
  if (second == llvm::StringRef::npos)
    return std::nullopt;
  auto inner = detail::parseComposedInner<Cat>(text.take_front(first));
  auto offset = IntTuple::fromString(remainder.take_front(second));
  auto outer = detail::parseAnyLayout<Cat>(remainder.drop_front(second + 3));
  if (!inner || !offset || !outer)
    return std::nullopt;
  return ComposedLayout(std::move(*inner), std::move(*offset), std::move(*outer));
}

namespace detail {

template <class Cat> std::optional<AnyLayout<Cat>> parseAnyLayout(llvm::StringRef text) {
  if (findTopLevel(text, " o ") != llvm::StringRef::npos) {
    auto value = ComposedLayout<Cat>::fromString(text);
    return value ? std::optional<AnyLayout<Cat>>(*value) : std::nullopt;
  }
  auto value = Layout<Cat>::fromString(text);
  return value ? std::optional<AnyLayout<Cat>>(*value) : std::nullopt;
}

} // namespace detail

template <class Cat> void print(const AnyLayout<Cat> &value, llvm::raw_ostream &os) {
  std::visit([&](const auto &item) { print(item, os); }, value);
}

template <class Cat> void print(const ComposedInner<Cat> &value, llvm::raw_ostream &os) {
  std::visit([&](const auto &item) { print(item, os); }, value);
}

template <class Cat> llvm::hash_code hash_value(const AnyLayout<Cat> &value) {
  return std::visit(
      [&](const auto &item) { return llvm::hash_combine(value.index(), hash_value(item)); }, value);
}

template <class Cat> llvm::hash_code hash_value(const ComposedInner<Cat> &value) {
  return std::visit(
      [&](const auto &item) { return llvm::hash_combine(value.index(), hash_value(item)); }, value);
}

namespace detail {

template <class Cat> struct FromString<AnyLayout<Cat>> {
  static std::optional<AnyLayout<Cat>> parse(llvm::StringRef text) {
    return parseAnyLayout<Cat>(text);
  }
};

template <class Cat> struct FromString<ComposedInner<Cat>> {
  static std::optional<ComposedInner<Cat>> parse(llvm::StringRef text) {
    return parseComposedInner<Cat>(text);
  }
};

} // namespace detail

template <class Cat> Layout<Cat> make_fragment_like(const ComposedLayout<Cat> &layout) {
  return std::visit([](const auto &outer) -> Layout<Cat> { return make_fragment_like(outer); },
                    layout.outer());
}

} // namespace mlir::fly::core

#endif // FLYDSL_CORE_ALGEBRA_COMPOSEDLAYOUT_HPP
