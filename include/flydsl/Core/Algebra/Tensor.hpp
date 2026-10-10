// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_CORE_ALGEBRA_TENSOR_HPP
#define FLYDSL_CORE_ALGEBRA_TENSOR_HPP

#include "flydsl/Core/Algebra/ComposedLayout.hpp"
#include "flydsl/Core/Algebra/Config.hpp"
#include "flydsl/Core/Algebra/Pointer.hpp"

#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

namespace mlir::fly::core {

template <class Cat, class Engine> struct Tensor {
public:
  using engine_type = Engine;
  using layout_type = AnyLayout<Cat>;

  Tensor(Engine engine, AnyLayout<Cat> layout)
      : engine_(std::move(engine)), layout_(std::move(layout)) {}

  template <class LayoutT>
  Tensor(Engine engine, LayoutT layout)
      : Tensor(std::move(engine), AnyLayout<Cat>(std::move(layout))) {}

  /// Only coordinate tensors parse: a pointer engine's text omits its opaque
  /// metadata and address payload.
  static std::optional<Tensor> fromString(llvm::StringRef text);

  const Engine &data() const { return engine_; }
  Engine &data() { return engine_; }
  const AnyLayout<Cat> &layout() const { return layout_; }
  const Tensor &tensor() const { return *this; }

  /// A tensor is invalid when its layout is; the engine carries no error.
  bool isError() const {
    return std::visit([](const auto &value) { return value.isError(); }, layout_);
  }
  ErrorInfo errorInfo() const {
    return std::visit([](const auto &value) { return value.errorInfo(); }, layout_);
  }
  ErrorCode errorCode() const { return errorInfo().reason(); }

  auto shape() const FLYDSL_CORE_LIFETIMEBOUND {
    return std::visit([](const auto &value) { return value.shape(); }, layout_);
  }
  IntTuple<Cat> stride() const;

  auto size() const {
    return std::visit([](const auto &value) { return value.size(); }, layout_);
  }
  auto rank() const {
    return std::visit([](const auto &value) { return value.rank(); }, layout_);
  }
  auto depth() const {
    return std::visit([](const auto &value) { return value.depth(); }, layout_);
  }

  template <class Index> IntTuple<Cat> get_1d_coord(const Index &index) const;

  template <class Index> IntTuple<Cat> get_hier_coord(const Index &index) const;

  template <class Index> IntTuple<Cat> get_flat_coord(const Index &index) const;

  template <class Tiler> auto compose(const Tiler &tiler) const;

  template <class Tiler0, class Tiler1, class... Tilers>
  auto compose(const Tiler0 &tiler0, const Tiler1 &tiler1, const Tilers &...tilers) const {
    return compose(make_tile(tiler0, tiler1, tilers...));
  }

  template <class Tiler> auto tile(const Tiler &tiler) const;

  template <class Tiler0, class Tiler1, class... Tilers>
  auto tile(const Tiler0 &tiler0, const Tiler1 &tiler1, const Tilers &...tilers) const {
    return tile(make_tile(tiler0, tiler1, tilers...));
  }

  template <class Coord> auto operator[](const Coord &coord) const;

  template <class Coord> auto operator()(const Coord &coord) const;

  template <class Coord0, class Coord1, class... Coords,
            std::enable_if_t<is_make_tuple_arg_v<Coord0, Coord1, Coords...>, int> = 0>
  auto operator()(const Coord0 &c0, const Coord1 &c1, const Coords &...coords) const;

private:
  template <class NewLayout> auto with_layout(NewLayout layout) const {
    return Tensor<Cat, Engine>(engine_, std::move(layout));
  }

  Engine engine_;
  AnyLayout<Cat> layout_;
};

template <class Cat> using MemRef = Tensor<Cat, Pointer<Cat>>;
template <class Cat> using CoordTensor = Tensor<Cat, IntTuple<Cat>>;

template <class Cat, class Shape, class Stride,
          std::enable_if_t<is_int_tuple_v<Shape> && is_int_tuple_v<Stride>, int>>
auto make_tensor(Pointer<Cat> pointer, const Shape &shape, const Stride &stride);

template <class LayoutT, class Cat> auto make_coord_tensor(LayoutT layout);

template <class Cat, class Engine>
auto layout(const Tensor<Cat, Engine> &tensor, llvm::ArrayRef<int32_t> path = {});

template <class Cat, class Engine>
IntTuple<Cat> stride(const Tensor<Cat, Engine> &tensor, llvm::ArrayRef<int32_t> path);

template <class Cat, class Engine>
auto size(const Tensor<Cat, Engine> &tensor, llvm::ArrayRef<int32_t> path);

template <class Cat, class Engine>
auto rank(const Tensor<Cat, Engine> &tensor, llvm::ArrayRef<int32_t> path);

template <class Cat, class Engine>
auto depth(const Tensor<Cat, Engine> &tensor, llvm::ArrayRef<int32_t> path);

template <class Cat, class Engine, class Profile>
auto filter_zeros(const Tensor<Cat, Engine> &tensor, const Profile &profile);

template <class Coord, class Cat, class Engine>
auto slice(const Coord &coord, const Tensor<Cat, Engine> &tensor);

template <class Coord, class Cat, class Engine>
auto domain_offset(const Coord &coord, const Tensor<Cat, Engine> &tensor);

/// Rebase a composed-layout tensor at its origin and keep its linear layout.
/// For layout L, returns (data + L(0), B), where B is the plain Layout reached
/// by following outer(). The result keeps B's shape and strides; discarded
/// mappings no longer affect coordinates away from the origin.
/// A tensor with a plain Layout is returned unchanged.
template <class Cat, class Engine> Tensor<Cat, Engine> decompose(const Tensor<Cat, Engine> &tensor);

template <class Cat, class Engine, class Tiler, class Coord>
auto inner_partition(const Tensor<Cat, Engine> &tensor, const Tiler &tiler, const Coord &coord);

template <class Cat, class Engine, class Tiler, class Coord>
auto outer_partition(const Tensor<Cat, Engine> &tensor, const Tiler &tiler, const Coord &coord);

template <class Cat, class Engine, class TileLayout, class Index,
          std::enable_if_t<is_int_tuple_v<Index>, int>>
auto local_partition(const Tensor<Cat, Engine> &tensor, const TileLayout &tile, const Index &index);

template <class Cat> auto max_common_layout(const MemRef<Cat> &src, const MemRef<Cat> &dst);

template <class Cat> auto max_common_vector(const MemRef<Cat> &src, const MemRef<Cat> &dst);

template <class Cat, class Engine> auto max_alignment(const Tensor<Cat, Engine> &tensor);

template <class Cat, class Engine>
void print(const Tensor<Cat, Engine> &tensor, llvm::raw_ostream &os);

namespace detail {
template <class Cat, class Engine, class F>
auto transform_tensor(const Tensor<Cat, Engine> &tensor, F &&f);

} // namespace detail

//===----------------------------------------------------------------------===//
// Definitions
//===----------------------------------------------------------------------===//

template <class Cat, class Engine> IntTuple<Cat> Tensor<Cat, Engine>::stride() const {
  return std::visit(
      [](const auto &value) -> IntTuple<Cat> {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, Layout<Cat>>)
          return value.stride();
        else
          return IntTuple<Cat>::getError(ErrorCode::InvalidLayout);
      },
      layout_);
}

template <class Cat, class Engine>
template <class Index>
IntTuple<Cat> Tensor<Cat, Engine>::get_1d_coord(const Index &index) const {
  return std::visit(
      [&](const auto &value) -> IntTuple<Cat> {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, Layout<Cat>>)
          return value.get_1d_coord(index);
        else
          return IntTuple<Cat>::getError(ErrorCode::InvalidLayout);
      },
      layout_);
}

template <class Cat, class Engine>
template <class Index>
IntTuple<Cat> Tensor<Cat, Engine>::get_hier_coord(const Index &index) const {
  return std::visit(
      [&](const auto &value) -> IntTuple<Cat> {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, Layout<Cat>>)
          return value.get_hier_coord(index);
        else
          return IntTuple<Cat>::getError(ErrorCode::InvalidLayout);
      },
      layout_);
}

template <class Cat, class Engine>
template <class Index>
IntTuple<Cat> Tensor<Cat, Engine>::get_flat_coord(const Index &index) const {
  return std::visit(
      [&](const auto &value) -> IntTuple<Cat> {
        using T = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<T, Layout<Cat>>)
          return value.get_flat_coord(index);
        else
          return IntTuple<Cat>::getError(ErrorCode::InvalidLayout);
      },
      layout_);
}

template <class Cat, class Engine>
template <class Tiler>
auto Tensor<Cat, Engine>::compose(const Tiler &tiler) const {
  return with_layout(std::visit(
      [&](const auto &value) { return AnyLayout<Cat>(composition(value, tiler)); }, layout_));
}

template <class Cat, class Engine>
template <class Tiler>
auto Tensor<Cat, Engine>::tile(const Tiler &tiler) const {
  return with_layout(
      std::visit([&](const auto &value) { return AnyLayout<Cat>(value.tile(tiler)); }, layout_));
}

template <class Cat, class Engine>
template <class Coord>
auto Tensor<Cat, Engine>::operator[](const Coord &coord) const {
  auto offset = std::visit([&](const auto &value) { return crd2idx(coord, value); }, layout_);
  return add_offset(engine_, offset);
}

template <class Coord, class Offset, class Cat = category_of<Coord>,
          std::enable_if_t<is_int_tuple_v<Coord> && is_int_tuple_v<Offset>, int> = 0>
IntTuple<Cat> add_offset(const Coord &coord, const Offset &offset) {
  static_assert(std::is_same_v<Cat, category_of<Offset>>);
  return coord + offset;
}

template <class Cat, class LayoutT, std::enable_if_t<is_layout_v<LayoutT>, int> = 0>
auto make_tensor(Pointer<Cat> pointer, LayoutT layout) {
  return MemRef<Cat>(std::move(pointer), std::move(layout));
}

template <class Cat, class Shape, std::enable_if_t<is_int_tuple_v<Shape>, int> = 0>
auto make_tensor(Pointer<Cat> pointer, const Shape &shape) {
  static_assert(std::is_same_v<Cat, category_of<Shape>>);
  return make_tensor(std::move(pointer), make_layout(shape));
}

template <class Cat, class Shape, class Stride,
          std::enable_if_t<is_int_tuple_v<Shape> && is_int_tuple_v<Stride>, int> = 0>
auto make_tensor(Pointer<Cat> pointer, const Shape &shape, const Stride &stride) {
  static_assert(std::is_same_v<Cat, category_of<Shape>>);
  static_assert(std::is_same_v<Cat, category_of<Stride>>);
  return make_tensor(std::move(pointer), make_layout(shape, stride));
}

template <class LayoutT, class Cat = category_of<LayoutT>> auto make_coord_tensor(LayoutT layout) {
  auto coord = coprofile(layout);
  return CoordTensor<Cat>(std::move(coord), std::move(layout));
}

template <class Shape, class Cat = category_of<Shape>>
auto make_identity_tensor(const Shape &shape) {
  return make_coord_tensor(make_identity_layout(shape));
}

template <class Cat, class Engine>
auto shape(const Tensor<Cat, Engine> &tensor FLYDSL_CORE_LIFETIMEBOUND) {
  return tensor.shape();
}

template <class Cat, class Engine> auto stride(const Tensor<Cat, Engine> &tensor) {
  return tensor.stride();
}

template <class Cat, class Engine> auto size(const Tensor<Cat, Engine> &tensor) {
  return tensor.size();
}

template <class Cat, class Engine> auto rank(const Tensor<Cat, Engine> &tensor) {
  return tensor.rank();
}

template <class Cat, class Engine> auto depth(const Tensor<Cat, Engine> &tensor) {
  return tensor.depth();
}

template <class Cat, class Engine>
auto layout(const Tensor<Cat, Engine> &tensor, llvm::ArrayRef<int32_t> path) {
  return std::visit([&](const auto &value) { return AnyLayout<Cat>(layout(value, path)); },
                    tensor.layout());
}
template <int I, int... Is, class Cat, class Engine>
auto layout(const Tensor<Cat, Engine> &tensor) {
  return layout(tensor, {I, Is...});
}

template <class Cat, class Engine>
auto shape(const Tensor<Cat, Engine> &tensor FLYDSL_CORE_LIFETIMEBOUND,
           llvm::ArrayRef<int32_t> path) {
  return std::visit([&](const auto &value) { return shape(value, path); }, tensor.layout());
}
template <int I, int... Is, class Cat, class Engine>
auto shape(const Tensor<Cat, Engine> &tensor FLYDSL_CORE_LIFETIMEBOUND) {
  return shape(tensor, {I, Is...});
}

template <class Cat, class Engine>
IntTuple<Cat> stride(const Tensor<Cat, Engine> &tensor, llvm::ArrayRef<int32_t> path) {
  auto all = tensor.stride();
  if (all.isError())
    return all;
  return all.asRef().at(path);
}
template <int I, int... Is, class Cat, class Engine>
IntTuple<Cat> stride(const Tensor<Cat, Engine> &tensor) {
  return stride(tensor, {I, Is...});
}

template <class Cat, class Engine>
auto size(const Tensor<Cat, Engine> &tensor, llvm::ArrayRef<int32_t> path) {
  return std::visit([&](const auto &value) { return size(value, path); }, tensor.layout());
}
template <int I, int... Is, class Cat, class Engine> auto size(const Tensor<Cat, Engine> &tensor) {
  return size(tensor, {I, Is...});
}

template <class Cat, class Engine>
auto rank(const Tensor<Cat, Engine> &tensor, llvm::ArrayRef<int32_t> path) {
  return std::visit([&](const auto &value) { return shape(value, path).rank(); }, tensor.layout());
}
template <int I, int... Is, class Cat, class Engine> auto rank(const Tensor<Cat, Engine> &tensor) {
  return rank(tensor, {I, Is...});
}

template <class Cat, class Engine>
auto depth(const Tensor<Cat, Engine> &tensor, llvm::ArrayRef<int32_t> path) {
  return std::visit([&](const auto &value) { return shape(value, path).depth(); }, tensor.layout());
}
template <int I, int... Is, class Cat, class Engine> auto depth(const Tensor<Cat, Engine> &tensor) {
  return depth(tensor, {I, Is...});
}

namespace detail {

template <class Cat, class Engine, class F>
auto transform_tensor(const Tensor<Cat, Engine> &tensor, F &&f) {
  auto newLayout =
      std::visit([&](const auto &layout) -> AnyLayout<Cat> { return f(layout); }, tensor.layout());
  return Tensor<Cat, Engine>(tensor.data(), std::move(newLayout));
}

/// A value of the kind of `value` that reports `info`; a tensor keeps its engine.
template <class Cat> Layout<Cat> errorLike(const Layout<Cat> &, ErrorInfo info) {
  return Layout<Cat>::getError(info);
}
template <class Cat> ComposedLayout<Cat> errorLike(const ComposedLayout<Cat> &, ErrorInfo info) {
  return ComposedLayout<Cat>::getError(info);
}
template <class Cat, class Engine>
Tensor<Cat, Engine> errorLike(const Tensor<Cat, Engine> &tensor, ErrorInfo info) {
  return transform_tensor(tensor, [&](const auto &) { return Layout<Cat>::getError(info); });
}

/// `tensor` with `op` recorded in the error of its invalid layout; a valid
/// tensor is returned unchanged.
template <class Cat, class Engine>
Tensor<Cat, Engine> tagError(Tensor<Cat, Engine> tensor, AlgebraOp op) {
  if (!tensor.isError())
    return tensor;
  return transform_tensor(tensor, [&](const auto &layout) { return tagError(layout, op); });
}

} // namespace detail

template <class Cat, class Engine> auto get(const Tensor<Cat, Engine> &tensor, int32_t i) {
  return detail::transform_tensor(tensor, [&](const auto &layout) { return get(layout, i); });
}

template <class Cat, class Engine> auto flatten(const Tensor<Cat, Engine> &tensor) {
  return detail::transform_tensor(tensor, [](const auto &layout) { return flatten(layout); });
}

template <class Cat, class Engine> auto coalesce(const Tensor<Cat, Engine> &tensor) {
  return detail::transform_tensor(tensor, [](const auto &layout) { return coalesce(layout); });
}

template <class Cat, class Engine, class Profile>
auto coalesce(const Tensor<Cat, Engine> &tensor, const Profile &profile) {
  return detail::transform_tensor(tensor,
                                  [&](const auto &layout) { return coalesce(layout, profile); });
}

template <class Cat, class Engine> auto filter_zeros(const Tensor<Cat, Engine> &tensor) {
  return detail::transform_tensor(tensor, [](const auto &layout) { return filter_zeros(layout); });
}

template <class Cat, class Engine, class Profile>
auto filter_zeros(const Tensor<Cat, Engine> &tensor, const Profile &profile) {
  return detail::transform_tensor(
      tensor, [&](const auto &layout) { return filter_zeros(layout, profile); });
}

template <class Cat, class Engine> auto filter(const Tensor<Cat, Engine> &tensor) {
  return detail::transform_tensor(tensor, [](const auto &layout) { return filter(layout); });
}

template <class Cat, class Engine>
auto group(const Tensor<Cat, Engine> &tensor, int32_t begin, int32_t end) {
  return detail::transform_tensor(tensor,
                                  [&](const auto &layout) { return group(layout, begin, end); });
}

template <class Cat, class Engine>
auto take(const Tensor<Cat, Engine> &tensor, int32_t begin, int32_t end) {
  return detail::transform_tensor(tensor,
                                  [&](const auto &layout) { return take(layout, begin, end); });
}

template <class Cat, class Engine>
auto select(const Tensor<Cat, Engine> &tensor, llvm::ArrayRef<int32_t> indices) {
  return detail::transform_tensor(tensor,
                                  [&](const auto &layout) { return select(layout, indices); });
}

template <class Cat, class Engine, class Profile>
auto unflatten(const Tensor<Cat, Engine> &tensor, const Profile &profile) {
  return detail::transform_tensor(tensor,
                                  [&](const auto &layout) { return unflatten(layout, profile); });
}

template <class Coord, class Cat, class Engine>
auto dice(const Coord &coord, const Tensor<Cat, Engine> &tensor) {
  return detail::transform_tensor(tensor, [&](const auto &layout) { return dice(coord, layout); });
}

template <class Coord, class Cat, class Engine>
auto slice(const Coord &coord, const Tensor<Cat, Engine> &tensor) {
  auto [newLayout, offset] = std::visit(
      [&](const auto &layout) {
        auto [result, offset] = slice_and_offset(coord, layout);
        return std::pair(AnyLayout<Cat>(std::move(result)), std::move(offset));
      },
      tensor.layout());
  auto newEngine = add_offset(tensor.data(), offset);
  return Tensor<Cat, Engine>(std::move(newEngine), std::move(newLayout));
}

template <class Cat, class Engine>
template <class Coord>
auto Tensor<Cat, Engine>::operator()(const Coord &coord) const {
  static_assert(is_int_tuple_v<Coord>);
  return slice(coord, *this);
}

template <class Cat, class Engine>
template <class Coord0, class Coord1, class... Coords,
          std::enable_if_t<is_make_tuple_arg_v<Coord0, Coord1, Coords...>, int>>
auto Tensor<Cat, Engine>::operator()(const Coord0 &c0, const Coord1 &c1,
                                     const Coords &...coords) const {
  return operator()(make_tuple(c0, c1, coords...));
}

template <class Coord, class Cat, class Engine>
auto domain_offset(const Coord &coord, const Tensor<Cat, Engine> &tensor) {
  auto [newLayout, offset] = std::visit(
      [&](const auto &layout) {
        auto [result, offset] = domain_offset(coord, layout);
        return std::pair(AnyLayout<Cat>(std::move(result)), std::move(offset));
      },
      tensor.layout());
  auto newEngine = add_offset(tensor.data(), offset);
  return Tensor<Cat, Engine>(std::move(newEngine), std::move(newLayout));
}

template <class Cat, class Engine>
Tensor<Cat, Engine> decompose(const Tensor<Cat, Engine> &tensor) {
  if (std::holds_alternative<Layout<Cat>>(tensor.layout()))
    return tensor;
  const auto *linear = &tensor.layout();
  while (const auto *composed = std::get_if<ComposedLayout<Cat>>(linear))
    linear = &composed->outer();
  return Tensor<Cat, Engine>(tensor[IntTuple<Cat>::getZero()], *linear);
}

#define FLYDSL_CORE_TENSOR_LAYOUT_ALGORITHM(name)                                                  \
  template <class Cat, class Engine, class Tiler>                                                  \
  auto name(const Tensor<Cat, Engine> &tensor, const Tiler &tiler) {                               \
    return detail::transform_tensor(tensor,                                                        \
                                    [&](const auto &layout) { return name(layout, tiler); });      \
  }

FLYDSL_CORE_TENSOR_LAYOUT_ALGORITHM(composition)
FLYDSL_CORE_TENSOR_LAYOUT_ALGORITHM(logical_divide)
FLYDSL_CORE_TENSOR_LAYOUT_ALGORITHM(zipped_divide)
FLYDSL_CORE_TENSOR_LAYOUT_ALGORITHM(tiled_divide)
FLYDSL_CORE_TENSOR_LAYOUT_ALGORITHM(flat_divide)

#undef FLYDSL_CORE_TENSOR_LAYOUT_ALGORITHM

template <class Cat, class Engine, class Tiler, class Coord>
auto inner_partition(const Tensor<Cat, Engine> &tensor, const Tiler &tiler, const Coord &coord) {
  auto tiled = zipped_divide(tensor, tiler);
  if (tiled.isError())
    return detail::tagError(tiled, AlgebraOp::InnerPartition);
  FLYDSL_CORE_ASSERT(tiled.shape().rank() == 2);
  auto tileCoord = repeat(IntTuple<Cat>::getNone(), tiled.shape().at(0).rank());
  auto restCoord = [&]() -> IntTuple<Cat> {
    auto value = coord.asRef();
    if (value.isLeaf())
      return value;
    return append(value, IntTuple<Cat>::getNone(), tiled.shape().at(1).rank());
  }();
  return slice(make_tuple(tileCoord, restCoord), tiled);
}

template <class Cat, class Engine, class Tiler, class Coord>
auto outer_partition(const Tensor<Cat, Engine> &tensor, const Tiler &tiler, const Coord &coord) {
  auto tiled = zipped_divide(tensor, tiler);
  if (tiled.isError())
    return detail::tagError(tiled, AlgebraOp::OuterPartition);
  FLYDSL_CORE_ASSERT(tiled.shape().rank() == 2);
  auto tileCoord = [&]() -> IntTuple<Cat> {
    auto value = coord.asRef();
    if (value.isLeaf())
      return value;
    return append(value, IntTuple<Cat>::getNone(), tiled.shape().at(0).rank());
  }();
  auto restCoord = repeat(IntTuple<Cat>::getNone(), tiled.shape().at(1).rank());
  return slice(make_tuple(tileCoord, restCoord), tiled);
}

template <class Cat, class Engine, class Tiler, class Coord>
auto local_tile(const Tensor<Cat, Engine> &tensor, const Tiler &tiler, const Coord &coord) {
  return inner_partition(tensor, tiler, coord);
}

template <class Cat, class Engine, class Tiler, class Coord, class Projection>
auto local_tile(const Tensor<Cat, Engine> &tensor, const Tiler &tiler, const Coord &coord,
                const Projection &projection) {
  return local_tile(tensor, dice(projection, tiler), dice(projection, coord));
}

template <class Cat, class Engine, class TileLayout, class Index,
          std::enable_if_t<is_int_tuple_v<Index>, int> = 0>
auto local_partition(const Tensor<Cat, Engine> &tensor, const TileLayout &tile,
                     const Index &index) {
  return outer_partition(tensor, product_each(tile.shape()), tile.get_flat_coord(index));
}

template <class Cat, class Engine, class TileLayout, class Index, class Projection,
          std::enable_if_t<is_int_tuple_v<Index>, int> = 0>
auto local_partition(const Tensor<Cat, Engine> &tensor, const TileLayout &tile, const Index &index,
                     const Projection &projection) {
  return local_partition(tensor, dice(projection, tile), index);
}

template <class Cat> auto max_common_layout(const MemRef<Cat> &src, const MemRef<Cat> &dst) {
  if (src.data().elementType() != dst.data().elementType() ||
      src.data().elementBitWidth() != dst.data().elementBitWidth())
    return Layout<Cat>::getError(
        ErrorInfo(ErrorCode::IncompatibleElementType).withFrame(AlgebraOp::MaxCommonLayout));
  const auto *srcLayout = std::get_if<Layout<Cat>>(&src.layout());
  const auto *dstLayout = std::get_if<Layout<Cat>>(&dst.layout());
  if (!srcLayout || !dstLayout)
    return Layout<Cat>::getError(
        ErrorInfo(ErrorCode::UnsupportedSwizzleComposition).withFrame(AlgebraOp::MaxCommonLayout));
  return max_common_layout(*srcLayout, *dstLayout);
}

template <class Cat> auto max_common_vector(const MemRef<Cat> &src, const MemRef<Cat> &dst) {
  if (src.data().elementType() != dst.data().elementType() ||
      src.data().elementBitWidth() != dst.data().elementBitWidth())
    return IntTuple<Cat>::getError(
        ErrorInfo(ErrorCode::IncompatibleElementType).withFrame(AlgebraOp::MaxCommonVector));
  const auto *srcLayout = std::get_if<Layout<Cat>>(&src.layout());
  const auto *dstLayout = std::get_if<Layout<Cat>>(&dst.layout());
  if (!srcLayout || !dstLayout)
    return IntTuple<Cat>::getError(
        ErrorInfo(ErrorCode::UnsupportedSwizzleComposition).withFrame(AlgebraOp::MaxCommonVector));
  return max_common_vector(*srcLayout, *dstLayout);
}

template <class Cat, class Engine> auto max_alignment(const Tensor<Cat, Engine> &tensor) {
  if constexpr (std::is_same_v<Engine, Pointer<Cat>>) {
    auto pointerBits = IntTuple<Cat>::getSInt(tensor.data().bitAlignment());
    auto layoutBits =
        std::visit([](const auto &layout) { return max_alignment(layout); }, tensor.layout()) *
        IntTuple<Cat>::getSInt(tensor.data().storageElementBitWidth());
    return leaf_gcd(pointerBits, layoutBits);
  } else {
    return IntTuple<Cat>::getZero();
  }
}

template <class Cat, class Engine>
void print(const Tensor<Cat, Engine> &tensor, llvm::raw_ostream &os) {
  print(tensor.data(), os);
  os << " o ";
  print(tensor.layout(), os);
}

template <class Cat, class Engine>
std::optional<Tensor<Cat, Engine>> Tensor<Cat, Engine>::fromString(llvm::StringRef text) {
  static_assert(std::is_same_v<Engine, IntTuple<Cat>>, "only coordinate tensors parse");
  auto separator = detail::findTopLevel(text, " o ");
  if (separator == llvm::StringRef::npos)
    return std::nullopt;
  auto engine = IntTuple<Cat>::fromString(text.take_front(separator));
  auto layout = detail::parseAnyLayout<Cat>(text.drop_front(separator + 3));
  if (!engine || !layout)
    return std::nullopt;
  return Tensor(std::move(*engine), std::move(*layout));
}

} // namespace mlir::fly::core

#endif // FLYDSL_CORE_ALGEBRA_TENSOR_HPP
