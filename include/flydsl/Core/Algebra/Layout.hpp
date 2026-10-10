// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_CORE_ALGEBRA_LAYOUT_HPP
#define FLYDSL_CORE_ALGEBRA_LAYOUT_HPP

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include "flydsl/Core/Algebra/Config.hpp"
#include "flydsl/Core/Algebra/IntTupleAlgorithms.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

namespace mlir::fly::core {

template <class Cat> struct Layout {
public:
  using Leaf = core::Leaf<Cat>;
  using IntTuple = core::IntTuple<Cat>;
  using IntTupleRef = core::IntTupleRef<Cat>;

  ///
  /// Constructors
  ///

  Layout(IntTuple shape, IntTuple stride);
  /// An invalid layout whose shape and stride are both the error `info`.
  static Layout getError(ErrorInfo info);

  static std::optional<Layout> fromString(llvm::StringRef text);

  ///
  /// Accessors
  ///

  const Layout &layout() const { return *this; }

  IntTupleRef shape() const FLYDSL_CORE_LIFETIMEBOUND;
  IntTupleRef stride() const FLYDSL_CORE_LIFETIMEBOUND;

  int32_t rank() const;
  int32_t depth() const;
  auto size() const;

  bool isLeaf() const;
  bool isStatic() const;
  /// The constructor collapses an invalid shape or stride into both.
  bool isError() const { return shape().isError(); }
  /// The first error, in shape then stride order.
  ErrorInfo errorInfo() const { return shape().errorInfo(); }
  ErrorCode errorCode() const { return errorInfo().reason(); }

  template <class Coord> auto operator()(const Coord &coord) const;

  /// Constrained so that is_invocable-style probes (e.g. raw_ostream's
  /// function_ref overload of `<<`) fail softly instead of instantiating.
  template <class Coord0, class Coord1, class... Coords,
            std::enable_if_t<is_make_tuple_arg_v<Coord0, Coord1, Coords...>, int> = 0>
  auto operator()(const Coord0 &c0, const Coord1 &c1, const Coords &...coords) const;

  template <class OtherLayout> auto compose(const OtherLayout &other) const;
  template <class Shape> auto with_shape(const Shape &shape) const;
  template <class OtherLayout> auto tile(const OtherLayout &other) const;

  template <class OtherLayout0, class OtherLayout1, class... Layouts>
  auto compose(const OtherLayout0 &layout0, const OtherLayout1 &layout1,
               const Layouts &...layouts) const;
  template <class Shape0, class Shape1, class... Shapes>
  auto with_shape(const Shape0 &shape0, const Shape1 &shape1, const Shapes &...shapes) const;
  template <class OtherLayout0, class OtherLayout1, class... Layouts>
  auto tile(const OtherLayout0 &layout0, const OtherLayout1 &layout1,
            const Layouts &...layouts) const;

  template <class Index> auto get_hier_coord(const Index &index) const;
  template <class Index> auto get_flat_coord(const Index &index) const;
  template <class Index> auto get_1d_coord(const Index &index) const;

  friend bool operator==(const Layout &a, const Layout &b) {
    return a.shape() == b.shape() && a.stride() == b.stride();
  }
  friend bool operator!=(const Layout &a, const Layout &b) { return !(a == b); }

private:
  // Shape and stride are congruent, so their views share this tree topology.
  llvm::SmallVector<Node, 10> nodes_;
  llvm::SmallVector<Leaf, 6> shapeLeaves_;
  llvm::SmallVector<Leaf, 6> strideLeaves_;
};

template <class Cat> struct Tile {
public:
  struct Storage;

  struct Iterator {
  public:
    Iterator() = default;

    Tile operator*() const;
    Iterator &operator++();
    friend bool operator==(const Iterator &a, const Iterator &b) {
      return a.storage_ == b.storage_ && a.nodeOffset_ == b.nodeOffset_ &&
             a.endOffset_ == b.endOffset_;
    }
    friend bool operator!=(const Iterator &a, const Iterator &b) { return !(a == b); }

  private:
    friend struct Tile;
    Iterator(std::shared_ptr<const Storage> storage, int32_t nodeOffset, int32_t endOffset);

    std::shared_ptr<const Storage> storage_;
    int32_t nodeOffset_ = 0;
    int32_t endOffset_ = 0;
  };

  Tile();
  Tile(const Layout<Cat> &layout);
  explicit Tile(Leaf<Cat> scalar);

  static Tile getNone();
  static Tile getError(ErrorInfo info);
  static Tile getTuple(llvm::ArrayRef<Tile> children);
  static Tile getTuple(std::initializer_list<Tile> children);

  static std::optional<Tile> fromString(llvm::StringRef text);

  Tile asRef() const { return *this; }
  IntTuple<Cat> shape() const;

  bool isLeaf() const;
  bool isLayout() const;
  bool isScalar() const;
  bool isNone() const;
  bool empty() const;
  int32_t rank() const;
  int32_t leafCount() const;
  int32_t depth() const;
  bool isStatic() const;
  bool isError() const;
  /// The first error in leaf order.
  ErrorInfo errorInfo() const;
  ErrorCode errorCode() const { return errorInfo().reason(); }

  const Layout<Cat> &getLayout() const;
  Leaf<Cat> getScalar() const;

  Tile at(int32_t i) const;
  Iterator begin() const;
  Iterator end() const;

  llvm::SmallVector<Tile, 8> children() const;

  friend bool operator==(const Tile &lhs, const Tile &rhs) { return equals(lhs, rhs); }
  friend bool operator!=(const Tile &lhs, const Tile &rhs) { return !(lhs == rhs); }

private:
  static bool equals(const Tile &lhs, const Tile &rhs);

  Tile(std::shared_ptr<const Storage> storage, int32_t nodeOffset, int32_t nodeCount,
       int32_t leafOffset, int32_t leafCount);

  std::shared_ptr<const Storage> storage_;
  int32_t nodeOffset_ = 0;
  int32_t nodeCount_ = 0;
  int32_t leafOffset_ = 0;
  int32_t leafCount_ = 0;
};

template <class Cat> struct Tile<Cat>::Storage {
  using LeafValue = std::variant<std::monostate, Layout<Cat>, Leaf<Cat>>;

  llvm::SmallVector<Node, 16> nodes;
  llvm::SmallVector<LeafValue, 8> leaves;
};

//===----------------------------------------------------------------------===//
// Declarations
//===----------------------------------------------------------------------===//

namespace detail {

template <class Cat> struct BasisComplementMode {
  IntTuple<Cat> shape = IntTuple<Cat>::getEmpty();
  IntTuple<Cat> stride = IntTuple<Cat>::getEmpty();
  bool used = false;
};
template <class Cat> struct ComplementPrefix {
  IntTuple<Cat> result_shape = IntTuple<Cat>::getEmpty();
  IntTuple<Cat> result_stride = IntTuple<Cat>::getEmpty();
  IntTuple<Cat> new_stride = IntTuple<Cat>::getEmpty();
  bool ok = true;
  ErrorCode code = ErrorCode::InvalidLayout;
};
template <class Cat> struct BasisInverseBucket {
  llvm::SmallVector<int32_t, kMaxBasisModes> path;
  llvm::SmallVector<IntTuple<Cat>, 4> shapes;
  llvm::SmallVector<IntTuple<Cat>, 4> strides;
  IntTuple<Cat> current = IntTuple<Cat>::getOne();
};

} // namespace detail

template <class Cat> llvm::hash_code hash_value(const Tile<Cat> &tile);
template <class Cat> void print(const Tile<Cat> &tile, llvm::raw_ostream &os);

template <class Cat, class... Ts>
Tile<Cat> make_tile(const Layout<Cat> &value, const Ts &...values);

template <class Cat, class... Ts> Tile<Cat> make_tile(const Tile<Cat> &value, const Ts &...values);

template <class T, class... Ts, class Cat, std::enable_if_t<is_int_tuple_v<T>, int>>
Tile<Cat> make_tile(const T &value, const Ts &...values);

template <class Cat, class... Ts> Tile<Cat> make_tile(Leaf<Cat> value, const Ts &...values);

template <class Cat> Tile<Cat> append(const Tile<Cat> &tile, const Tile<Cat> &sub);

template <class Cat> Tile<Cat> append(const Tile<Cat> &tile, const Tile<Cat> &sub, int32_t n);

template <int N, class Cat> Tile<Cat> append(const Tile<Cat> &tile, const Tile<Cat> &sub);

template <class T, class Cat> Tile<Cat> shape_to_tile(const T &shape);

template <class Cat> auto size(const Tile<Cat> &tile);

template <class Shape, class Stride, class Cat,
          std::enable_if_t<is_int_tuple_v<Shape> && is_int_tuple_v<Stride>, int>>
Layout<Cat> make_layout(Shape &&shape, Stride &&stride);

template <class Shape, class Cat = category_of<Shape>,
          std::enable_if_t<is_int_tuple_v<Shape>, int> = 0>
Layout<Cat> make_layout(Shape &&shape, LayoutLeft = {});

template <class Cat, class... Layouts>
Layout<Cat> make_layout(const Layout<Cat> &layout0, const Layout<Cat> &layout1,
                        const Layouts &...layouts);

template <class Cat> auto make_layout_like(const Layout<Cat> &layout);

template <class Cat> auto make_fragment_like(const Layout<Cat> &layout);

template <int... Is, class Cat> int32_t rank(const Layout<Cat> &layout);

template <int... Is, class Cat> int32_t depth(const Layout<Cat> &layout);

template <int... Is, class Cat>
IntTupleRef<Cat> shape(const Layout<Cat> &layout FLYDSL_CORE_LIFETIMEBOUND);

template <int... Is, class Cat>
IntTupleRef<Cat> stride(const Layout<Cat> &layout FLYDSL_CORE_LIFETIMEBOUND);

template <int... Is, class Cat> Layout<Cat> layout(const Layout<Cat> &value);

template <int... Is, class Cat> IntTuple<Cat> size(const Layout<Cat> &layout);

template <int... Is, class Cat> Layout<Cat> get(const Layout<Cat> &layout);

template <int I, int... Is, class Cat> auto get(const Tile<Cat> &tile);

template <class Cat, class F> Layout<Cat> transform_layout(const Layout<Cat> &layout, F &&f);

template <class T1, class F, class Cat>
Layout<Cat> transform_layout(const Layout<Cat> &t0, const T1 &t1, F &&f);

template <class Cat> IntTuple<Cat> coshape(const Layout<Cat> &value, llvm::ArrayRef<int32_t> path);

template <class Cat> Layout<Cat> coalesce_x(const Layout<Cat> &layout);

template <class Profile, class Cat>
Layout<Cat> coalesce_x(const Layout<Cat> &layout, const Profile &targetProfile);

template <class Profile, class Cat>
Layout<Cat> coalesce(const Layout<Cat> &layout, const Profile &targetProfile);

template <class T, class Cat> IntTuple<Cat> coalesce(const T &shape);

template <class Profile, class Cat>
Layout<Cat> filter(const Layout<Cat> &layout, const Profile &targetProfile);

template <class Cat> Layout<Cat> composition(const Layout<Cat> &lhs, const Layout<Cat> &rhs);

template <class Cat> Layout<Cat> composition(const Layout<Cat> &lhs, const Tile<Cat> &rhs);

template <class Cat> Layout<Cat> complement(const Layout<Cat> &layout);

template <class Cotarget, class Cat, std::enable_if_t<is_int_tuple_v<Cotarget>, int>>
Layout<Cat> complement(const Layout<Cat> &layout, const Cotarget &cotarget);

template <class Cat> Layout<Cat> right_inverse(const Layout<Cat> &layout);

template <class Cat> Layout<Cat> left_inverse(const Layout<Cat> &layout);

template <class Cat> Layout<Cat> max_common_layout(const Layout<Cat> &a, const Layout<Cat> &b);

template <class Cat> IntTuple<Cat> max_common_vector(const Layout<Cat> &a, const Layout<Cat> &b);

template <class A, class B, class Cat> Layout<Cat> domain_distribute(const A &a, const B &b);

template <class Cat> Layout<Cat> nullspace(const Layout<Cat> &layout);

template <class Cat> Layout<Cat> zip(const Layout<Cat> &layoutA, const Layout<Cat> &layoutB);

template <class Cat> Layout<Cat> tile_unzip(const Layout<Cat> &layout, const Tile<Cat> &tiler);

template <class Cat> Layout<Cat> logical_divide(const Layout<Cat> &layout, const Tile<Cat> &tiler);

template <class Tiler, class Cat, std::enable_if_t<!is_int_tuple_v<Tiler>, int> = 0>
Layout<Cat> tiled_divide(const Layout<Cat> &layout, const Tiler &tiler);

template <class Tiler, class Cat, std::enable_if_t<!is_int_tuple_v<Tiler>, int>>
Layout<Cat> flat_divide(const Layout<Cat> &layout, const Tiler &tiler);

template <class Cat> Layout<Cat> logical_product(const Layout<Cat> &block, const Tile<Cat> &tiler);

template <class Tiler, class Cat, std::enable_if_t<!is_int_tuple_v<Tiler>, int>>
Layout<Cat> tiled_product(const Layout<Cat> &block, const Tiler &tiler);

template <class Tiler, class Cat, std::enable_if_t<!is_int_tuple_v<Tiler>, int>>
Layout<Cat> flat_product(const Layout<Cat> &block, const Tiler &tiler);

template <class Cat>
Layout<Cat> blocked_product(const Layout<Cat> &block, const Layout<Cat> &tiler);

template <class Cat> Layout<Cat> raked_product(const Layout<Cat> &block, const Layout<Cat> &tiler);

template <class TargetShape, class Order, class Cat>
Layout<Cat> tile_to_shape(const Layout<Cat> &block, const TargetShape &targetShape,
                          const Order &order = {});

template <class Cat>
Layout<Cat> recast_layout(int32_t newTypeBits, int32_t oldTypeBits, const Layout<Cat> &layout);

template <class Cat> auto max_alignment(const Layout<Cat> &layout);

template <class Cat> void print(const Layout<Cat> &layout, llvm::raw_ostream &os);

//===----------------------------------------------------------------------===//
// Definitions
//===----------------------------------------------------------------------===//

///
/// Layout members
///

template <class Cat> Layout<Cat>::Layout(IntTuple shape, IntTuple stride) {
  // An invalid layout is one error leaf shared by its shape and stride. An
  // operand that already failed explains a mismatch better.
  if (shape.isError())
    stride = shape;
  else if (stride.isError())
    shape = stride;
  else if (!Node::congruent(shape.nodes(), stride.nodes()))
    shape = stride = IntTuple::getError(ErrorCode::InvalidLayout);
  nodes_ = std::move(shape.nodes_);
  shapeLeaves_ = std::move(shape.leaves_);
  strideLeaves_ = std::move(stride.leaves_);
}

template <class Cat> Layout<Cat> Layout<Cat>::getError(ErrorInfo info) {
  auto error = IntTuple::getError(info);
  return Layout(error, error);
}

template <class Cat>
typename Layout<Cat>::IntTupleRef Layout<Cat>::shape() const FLYDSL_CORE_LIFETIMEBOUND {
  return IntTupleRef(nodes_, shapeLeaves_);
}

template <class Cat>
typename Layout<Cat>::IntTupleRef Layout<Cat>::stride() const FLYDSL_CORE_LIFETIMEBOUND {
  return IntTupleRef(nodes_, strideLeaves_);
}

template <class Cat> int32_t Layout<Cat>::rank() const { return Node::rank(nodes_); }

template <class Cat> int32_t Layout<Cat>::depth() const { return Node::depth(nodes_); }

template <class Cat> auto Layout<Cat>::size() const {
  return product(IntTupleRef(nodes_, shapeLeaves_));
}

template <class Cat> bool Layout<Cat>::isLeaf() const { return Node::is_leaf(nodes_); }
template <class Cat> bool Layout<Cat>::isStatic() const {
  if (isError())
    return false;
  return llvm::all_of(shapeLeaves_, [](const auto &leaf) { return leaf.isStatic(); }) &&
         llvm::all_of(strideLeaves_, [](const auto &leaf) { return leaf.isStatic(); });
}

template <class Cat> template <class Coord> auto Layout<Cat>::operator()(const Coord &coord) const {
  return slice(coord, *this);
}

template <class Cat>
template <class Coord0, class Coord1, class... Coords,
          std::enable_if_t<is_make_tuple_arg_v<Coord0, Coord1, Coords...>, int>>
auto Layout<Cat>::operator()(const Coord0 &c0, const Coord1 &c1, const Coords &...coords) const {
  return operator()(make_tuple(c0, c1, coords...));
}

template <class Cat>
template <class OtherLayout>
auto Layout<Cat>::compose(const OtherLayout &other) const {
  return composition(*this, other);
}

template <class Cat>
template <class OtherLayout0, class OtherLayout1, class... Layouts>
auto Layout<Cat>::compose(const OtherLayout0 &layout0, const OtherLayout1 &layout1,
                          const Layouts &...layouts) const {
  return composition(*this, make_tile(layout0, layout1, layouts...));
}

template <class Cat> template <class Shape> auto Layout<Cat>::with_shape(const Shape &shape) const {
  return composition(*this, make_layout(shape));
}

template <class Cat>
template <class Shape0, class Shape1, class... Shapes>
auto Layout<Cat>::with_shape(const Shape0 &shape0, const Shape1 &shape1,
                             const Shapes &...shapes) const {
  return composition(*this, make_layout(make_tuple(shape0, shape1, shapes...)));
}

template <class Cat>
template <class OtherLayout>
auto Layout<Cat>::tile(const OtherLayout &other) const {
  return tiled_divide(*this, other);
}

template <class Cat>
template <class OtherLayout0, class OtherLayout1, class... Layouts>
auto Layout<Cat>::tile(const OtherLayout0 &layout0, const OtherLayout1 &layout1,
                       const Layouts &...layouts) const {
  return tiled_divide(*this, make_tile(layout0, layout1, layouts...));
}

template <class Cat>
template <class Index>
auto Layout<Cat>::get_hier_coord(const Index &index) const {
  return idx2crd(index, shape(), stride());
}

template <class Cat>
template <class Index>
auto Layout<Cat>::get_flat_coord(const Index &index) const {
  return crd2crd(get_hier_coord(index), shape(), repeat(IntTuple::getOne(), rank()));
}

template <class Cat>
template <class Index>
auto Layout<Cat>::get_1d_coord(const Index &index) const {
  return crd2idx(get_hier_coord(index), shape());
}

template <class Cat>
Tile<Cat>::Iterator::Iterator(std::shared_ptr<const Storage> storage, int32_t nodeOffset,
                              int32_t endOffset)
    : storage_(std::move(storage)), nodeOffset_(nodeOffset), endOffset_(endOffset) {}

template <class Cat>
Tile<Cat>::Tile(std::shared_ptr<const Storage> storage, int32_t nodeOffset, int32_t nodeCount,
                int32_t leafOffset, int32_t leafCount)
    : storage_(std::move(storage)), nodeOffset_(nodeOffset), nodeCount_(nodeCount),
      leafOffset_(leafOffset), leafCount_(leafCount) {}

template <class Cat> Tile<Cat>::Tile() {
  auto storage = std::make_shared<Storage>();
  storage->nodes.push_back({1, 0});
  storage->nodes.push_back({0, 0});
  storage_ = std::move(storage);
  nodeCount_ = 2;
}

template <class Cat> Tile<Cat>::Tile(const Layout<Cat> &layout) {
  auto storage = std::make_shared<Storage>();
  storage->nodes.push_back({1, 0});
  storage->nodes.push_back({0, 1});
  storage->leaves.emplace_back(layout);
  storage_ = std::move(storage);
  nodeCount_ = 2;
  leafCount_ = 1;
}

template <class Cat> Tile<Cat>::Tile(Leaf<Cat> scalar) {
  FLYDSL_CORE_ASSERT(!scalar.isBasis() && !scalar.isNone());
  auto storage = std::make_shared<Storage>();
  storage->nodes.push_back({1, 0});
  storage->nodes.push_back({0, 1});
  storage->leaves.emplace_back(scalar);
  storage_ = std::move(storage);
  nodeCount_ = 2;
  leafCount_ = 1;
}

template <class Cat> Tile<Cat> Tile<Cat>::getNone() {
  auto storage = std::make_shared<Storage>();
  storage->nodes.push_back({1, 0});
  storage->nodes.push_back({0, 1});
  storage->leaves.emplace_back(std::monostate{});
  return Tile(std::move(storage), 0, 2, 0, 1);
}

template <class Cat> Tile<Cat> Tile<Cat>::getError(ErrorInfo info) {
  return Tile(Leaf<Cat>::getError(info));
}

template <class Cat> Tile<Cat> Tile<Cat>::getTuple(llvm::ArrayRef<Tile> children) {
  auto storage = std::make_shared<Storage>();
  storage->nodes.push_back({0, 0});
  for (const auto &child : children) {
    // A tile with an invalid child is that child's error.
    if (child.isError())
      return child;
    auto childNodes =
        llvm::ArrayRef<Node>(child.storage_->nodes).slice(child.nodeOffset_, child.nodeCount_);
    if (storage->nodes.size() + childNodes.size() - 1 > Node::maxSpan ||
        storage->leaves.size() + child.leafCount_ > UINT16_MAX)
      return getError(ErrorCode::TupleCapacityExceeded);
    Node::appendSubtree(storage->nodes, childNodes, storage->leaves.size());
    auto childLeaves = llvm::ArrayRef<typename Storage::LeafValue>(child.storage_->leaves)
                           .slice(child.leafOffset_, child.leafCount_);
    storage->leaves.append(childLeaves.begin(), childLeaves.end());
  }
  Node::finish(storage->nodes, storage->leaves.size());
  auto nodeCount = static_cast<int32_t>(storage->nodes.size());
  auto leafCount = static_cast<int32_t>(storage->leaves.size());
  return Tile(std::move(storage), 0, nodeCount, 0, leafCount);
}

template <class Cat> Tile<Cat> Tile<Cat>::getTuple(std::initializer_list<Tile> children) {
  return getTuple(llvm::ArrayRef<Tile>(children.begin(), children.size()));
}

template <class Cat> bool Tile<Cat>::isLeaf() const {
  auto nodes = llvm::ArrayRef<Node>(storage_->nodes).slice(nodeOffset_, nodeCount_);
  return Node::is_leaf(nodes);
}

template <class Cat> bool Tile<Cat>::isLayout() const {
  return isLeaf() && std::holds_alternative<Layout<Cat>>(storage_->leaves[leafOffset_]);
}

template <class Cat> bool Tile<Cat>::isScalar() const {
  return isLeaf() && std::holds_alternative<Leaf<Cat>>(storage_->leaves[leafOffset_]);
}

template <class Cat> Leaf<Cat> Tile<Cat>::getScalar() const {
  FLYDSL_CORE_ASSERT(isScalar());
  return std::get<Leaf<Cat>>(storage_->leaves[leafOffset_]);
}

template <class Cat> bool Tile<Cat>::isNone() const {
  return isLeaf() && std::holds_alternative<std::monostate>(storage_->leaves[leafOffset_]);
}

template <class Cat> bool Tile<Cat>::empty() const { return !isLeaf() && rank() == 0; }

template <class Cat> int32_t Tile<Cat>::rank() const {
  auto nodes = llvm::ArrayRef<Node>(storage_->nodes).slice(nodeOffset_, nodeCount_);
  return Node::rank(nodes);
}

template <class Cat> int32_t Tile<Cat>::leafCount() const { return leafCount_; }

template <class Cat> int32_t Tile<Cat>::depth() const {
  auto nodes = llvm::ArrayRef<Node>(storage_->nodes).slice(nodeOffset_, nodeCount_);
  return Node::depth(nodes);
}

template <class Cat> bool Tile<Cat>::isStatic() const {
  auto leaves =
      llvm::ArrayRef<typename Storage::LeafValue>(storage_->leaves).slice(leafOffset_, leafCount_);
  return llvm::all_of(leaves, [](const auto &leaf) {
    if (auto *layout = std::get_if<Layout<Cat>>(&leaf))
      return layout->isStatic();
    if (auto *scalar = std::get_if<Leaf<Cat>>(&leaf))
      return scalar->isStatic();
    return true;
  });
}

/// `getTuple` collapses an invalid tile into its invalid leaf, so only a leaf
/// tile can be invalid.
template <class Cat> bool Tile<Cat>::isError() const {
  if (!isLeaf())
    return false;
  const auto &leaf = storage_->leaves[leafOffset_];
  if (auto *layout = std::get_if<Layout<Cat>>(&leaf))
    return layout->isError();
  if (auto *scalar = std::get_if<Leaf<Cat>>(&leaf))
    return scalar->isError();
  return false;
}

template <class Cat> ErrorInfo Tile<Cat>::errorInfo() const {
  if (isError()) {
    const auto &leaf = storage_->leaves[leafOffset_];
    if (auto *layout = std::get_if<Layout<Cat>>(&leaf))
      return layout->errorInfo();
    return std::get<Leaf<Cat>>(leaf).errorInfo();
  }
  FLYDSL_CORE_ASSERT(false && "tile has no error");
  return ErrorCode::InvalidLayout;
}

namespace detail {

/// Records `op` in the error of an invalid `value`, at `mode` for a mode-wise
/// step; a valid value is returned unchanged. Taken by value so that a valid
/// temporary or moved result passes through without a copy.
template <class Cat> Layout<Cat> tagError(Layout<Cat> value, AlgebraOp op, int64_t mode = -1) {
  if (!value.isError())
    return value;
  return Layout<Cat>::getError(value.errorInfo().withFrame(op, mode));
}

template <class Cat> Tile<Cat> tagError(Tile<Cat> value, AlgebraOp op, int64_t mode = -1) {
  if (!value.isError())
    return value;
  return Tile<Cat>::getError(value.errorInfo().withFrame(op, mode));
}

/// The first error carried by `operands`, in argument order.
template <class T, class... Ts>
std::optional<ErrorInfo> firstError(const T &operand, const Ts &...operands) {
  if (operand.isError())
    return operand.errorInfo();
  if constexpr (sizeof...(Ts) == 0)
    return std::nullopt;
  else
    return firstError(operands...);
}

} // namespace detail

template <class Cat> const Layout<Cat> &Tile<Cat>::getLayout() const {
  FLYDSL_CORE_ASSERT(isLayout());
  return std::get<Layout<Cat>>(storage_->leaves[leafOffset_]);
}

template <class Cat> Tile<Cat> Tile<Cat>::at(int32_t i) const {
  FLYDSL_CORE_ASSERT(!isLeaf() && i >= 0 && i < rank());
  auto nodes = llvm::ArrayRef<Node>(storage_->nodes).slice(nodeOffset_, nodeCount_);
  auto childOffset = Node::childOffset(nodes, i);
  auto childNode = nodeOffset_ + childOffset;
  auto childNodeCount = storage_->nodes[childNode].span + 1;
  auto firstLeaf = storage_->nodes[childNode].firstLeaf;
  auto nextLeaf = storage_->nodes[childNode + childNodeCount - 1].firstLeaf;
  return Tile(storage_, childNode, childNodeCount, firstLeaf, nextLeaf - firstLeaf);
}

template <class Cat> Tile<Cat> Tile<Cat>::Iterator::operator*() const {
  auto nodeCount = storage_->nodes[nodeOffset_].span + 1;
  auto firstLeaf = storage_->nodes[nodeOffset_].firstLeaf;
  auto nextLeaf = storage_->nodes[nodeOffset_ + nodeCount - 1].firstLeaf;
  return Tile(storage_, nodeOffset_, nodeCount, firstLeaf, nextLeaf - firstLeaf);
}

template <class Cat> typename Tile<Cat>::Iterator &Tile<Cat>::Iterator::operator++() {
  nodeOffset_ += storage_->nodes[nodeOffset_].span;
  return *this;
}

template <class Cat> typename Tile<Cat>::Iterator Tile<Cat>::begin() const {
  return Iterator(storage_, nodeOffset_ + 1, nodeOffset_ + nodeCount_ - 1);
}

template <class Cat> typename Tile<Cat>::Iterator Tile<Cat>::end() const {
  auto sentinel = nodeOffset_ + nodeCount_ - 1;
  return Iterator(storage_, sentinel, sentinel);
}

template <class Cat> llvm::SmallVector<Tile<Cat>, 8> Tile<Cat>::children() const {
  FLYDSL_CORE_ASSERT(!isLeaf());
  llvm::SmallVector<Tile, 8> result;
  result.reserve(rank());
  for (auto child : *this)
    result.push_back(child);
  return result;
}

template <class Cat> IntTuple<Cat> Tile<Cat>::shape() const {
  if (isNone())
    return IntTuple<Cat>::getNone();
  if (isLayout())
    return getLayout().shape();
  if (isScalar())
    return IntTuple<Cat>::fromLeaf(getScalar());
  llvm::SmallVector<IntTuple<Cat>, 8> children;
  children.reserve(static_cast<size_t>(rank()));
  for (auto child : *this)
    children.push_back(child.shape());
  return make_tuple(children);
}

template <class Cat> bool Tile<Cat>::equals(const Tile &lhs, const Tile &rhs) {
  if (lhs.isNone() || rhs.isNone())
    return lhs.isNone() && rhs.isNone();
  if (lhs.isScalar() || rhs.isScalar())
    return lhs.isScalar() && rhs.isScalar() && lhs.getScalar() == rhs.getScalar();
  if (lhs.isLayout() || rhs.isLayout())
    return lhs.isLayout() && rhs.isLayout() && lhs.getLayout() == rhs.getLayout();
  if (lhs.rank() != rhs.rank())
    return false;
  auto lhsIt = lhs.begin();
  auto rhsIt = rhs.begin();
  for (auto end = lhs.end(); lhsIt != end; ++lhsIt, ++rhsIt)
    if (*lhsIt != *rhsIt)
      return false;
  return true;
}

template <class Cat> llvm::hash_code hash_value(const Layout<Cat> &layout) {
  return llvm::hash_combine(hash_value(layout.shape()), hash_value(layout.stride()));
}

template <class Cat> llvm::hash_code hash_value(const Tile<Cat> &tile) {
  if (tile.isNone())
    return llvm::hash_value(uint8_t{0});
  if (tile.isScalar())
    return llvm::hash_combine(uint8_t{3}, hash_value(tile.getScalar()));
  if (tile.isLayout())
    return llvm::hash_combine(uint8_t{1}, hash_value(tile.getLayout()));
  auto children = tile.children();
  return llvm::hash_combine(uint8_t{2}, llvm::hash_combine_range(children.begin(), children.end()));
}

namespace detail {
template <class Cat> Tile<Cat> toTile(const Tile<Cat> &value) { return value; }

template <class Cat> Tile<Cat> toTile(const Layout<Cat> &value) { return Tile<Cat>(value); }

template <class T, class Cat = category_of<T>, std::enable_if_t<is_int_tuple_v<T>, int> = 0>
Tile<Cat> toTile(const T &value) {
  if (value.isNone())
    return Tile<Cat>::getNone();
  return Tile<Cat>(make_layout(value));
}

template <class Cat> Tile<Cat> toTile(Leaf<Cat> value) {
  return toTile(IntTuple<Cat>::fromLeaf(value));
}

} // namespace detail

template <class Cat, class... Ts>
Tile<Cat> make_tile(const Layout<Cat> &value, const Ts &...values) {
  auto children =
      std::array<Tile<Cat>, 1 + sizeof...(Ts)>{Tile<Cat>(value), detail::toTile(values)...};
  return Tile<Cat>::getTuple(children);
}

template <class Cat, class... Ts> Tile<Cat> make_tile(const Tile<Cat> &value, const Ts &...values) {
  auto children = std::array<Tile<Cat>, 1 + sizeof...(Ts)>{value, detail::toTile(values)...};
  return Tile<Cat>::getTuple(children);
}

template <class T, class... Ts, class Cat = category_of<T>,
          std::enable_if_t<is_int_tuple_v<T>, int> = 0>
Tile<Cat> make_tile(const T &value, const Ts &...values) {
  auto children =
      std::array<Tile<Cat>, 1 + sizeof...(Ts)>{detail::toTile(value), detail::toTile(values)...};
  return Tile<Cat>::getTuple(children);
}

template <class Cat, class... Ts> Tile<Cat> make_tile(Leaf<Cat> value, const Ts &...values) {
  auto children =
      std::array<Tile<Cat>, 1 + sizeof...(Ts)>{detail::toTile(value), detail::toTile(values)...};
  return Tile<Cat>::getTuple(children);
}

template <class Cat> Tile<Cat> make_tile() { return Tile<Cat>(); }

template <class Cat> Tile<Cat> append(const Tile<Cat> &tile, const Tile<Cat> &sub) {
  auto children = tile.isLeaf() ? llvm::SmallVector<Tile<Cat>, 8>{tile} : tile.children();
  children.push_back(sub);
  return Tile<Cat>::getTuple(children);
}

template <class Cat> Tile<Cat> append(const Tile<Cat> &tile, const Tile<Cat> &sub, int32_t n) {
  auto tileRank = tile.rank();
  if (n < tileRank)
    return Tile<Cat>::getError(ErrorInfo(ErrorCode::IndexOutOfRange).withFrame(AlgebraOp::Append));
  if (n == tileRank)
    return tile;
  auto children = tile.isLeaf() ? llvm::SmallVector<Tile<Cat>, 8>{tile} : tile.children();
  children.reserve(static_cast<size_t>(n));
  for (auto i = tileRank; i < n; ++i)
    children.push_back(sub);
  return Tile<Cat>::getTuple(children);
}

template <int N, class Cat> Tile<Cat> append(const Tile<Cat> &tile, const Tile<Cat> &sub) {
  return append(tile, sub, N);
}

template <class T, class Cat = category_of<T>> Tile<Cat> shape_to_tile(const T &shape) {
  auto value = shape.asRef();
  if (value.isLeaf())
    return detail::toTile(value);

  llvm::SmallVector<Tile<Cat>, 8> children;
  children.reserve(static_cast<size_t>(value.rank()));
  for (auto child : value)
    children.push_back(shape_to_tile(child));
  return Tile<Cat>::getTuple(children);
}

template <class Cat> auto size(const Tile<Cat> &tile) {
  if (tile.isNone())
    return IntTuple<Cat>::getZero();
  if (tile.isLayout())
    return size(tile.getLayout());
  if (tile.isScalar())
    return IntTuple<Cat>::fromLeaf(tile.getScalar());
  auto result = IntTuple<Cat>::getOne();
  for (auto child : tile)
    result = result * size(child);
  return result;
}

template <class Tiler, class Cat, std::enable_if_t<!is_int_tuple_v<Tiler>, int> = 0>
Layout<Cat> tiled_product(const Layout<Cat> &block, const Tiler &tiler);

//===----------------------------------------------------------------------===//
// Layout Constructors
//===----------------------------------------------------------------------===//

template <class Shape, class Stride, class Cat = category_of<Shape>,
          std::enable_if_t<is_int_tuple_v<Shape> && is_int_tuple_v<Stride>, int> = 0>
Layout<Cat> make_layout(Shape &&shape, Stride &&stride) {
  static_assert(std::is_same_v<Cat, category_of<Stride>>);
  return Layout<Cat>(std::forward<Shape>(shape), std::forward<Stride>(stride));
}

template <class Shape, class Cat, std::enable_if_t<is_int_tuple_v<Shape>, int>>
Layout<Cat> make_layout(Shape &&shape, LayoutLeft) {
  IntTuple<Cat> ownedShape = std::forward<Shape>(shape);
  auto stride = compact_col_major(ownedShape);
  return make_layout(std::move(ownedShape), std::move(stride));
}

template <class Shape, class Cat = category_of<Shape>,
          std::enable_if_t<is_int_tuple_v<Shape>, int> = 0>
Layout<Cat> make_layout(Shape &&shape, LayoutRight) {
  auto stride = compact_row_major(shape);
  return make_layout(std::forward<Shape>(shape), std::move(stride));
}

template <class Cat> Layout<Cat> make_layout(const Layout<Cat> &layout) {
  return make_layout(make_tuple(layout.shape()), make_tuple(layout.stride()));
}

template <class Cat, class... Layouts>
Layout<Cat> make_layout(const Layout<Cat> &layout0, const Layout<Cat> &layout1,
                        const Layouts &...layouts) {
  auto shapes = make_tuple(layout0.shape(), layout1.shape(), layouts.shape()...);
  auto strides = make_tuple(layout0.stride(), layout1.stride(), layouts.stride()...);
  return make_layout(std::move(shapes), std::move(strides));
}

template <class Shape, class Order, class Cat = category_of<Shape>>
auto make_ordered_layout(const Shape &shape, const Order &order) {
  return make_layout(shape, compact_order(shape, order));
}

template <class Cat> auto make_layout_like(const Layout<Cat> &layout) {
  return make_layout(layout.shape(),
                     compact_order(filter_zeros(layout.stride(), layout.shape()), layout.stride()));
}

template <class Cat> auto make_fragment_like(const Layout<Cat> &layout) {
  if (layout.rank() > 1 && layout.shape().isStatic()) {
    auto first = get(layout, 0);
    auto rest = take(layout, 1, layout.rank());
    return tiled_product(
        make_layout(first.shape(), compact_col_major(filter_zeros(first.stride(), first.shape()))),
        make_ordered_layout(rest.shape(), rest.stride()));
  }
  return make_layout(layout.shape());
}

template <class T, class Cat = category_of<T>> auto make_fragment_like(const T &shape) {
  return make_layout(shape);
}

template <class T, class Cat = category_of<T>> auto make_identity_layout(const T &shape) {
  return make_layout(shape, make_basis_like(shape));
}

///
/// Accessors
///

template <int... Is, class Cat> int32_t rank(const Layout<Cat> &layout) {
  if constexpr (sizeof...(Is) == 0)
    return layout.rank();
  else
    return shape(layout, {Is...}).rank();
}
template <int... Is, class Cat> int32_t depth(const Layout<Cat> &layout) {
  if constexpr (sizeof...(Is) == 0)
    return layout.depth();
  else
    return shape(layout, {Is...}).depth();
}

template <class Cat>
IntTupleRef<Cat> shape(const Layout<Cat> &layout FLYDSL_CORE_LIFETIMEBOUND,
                       llvm::ArrayRef<int32_t> path) {
  return layout.shape().at(path);
}
template <int... Is, class Cat>
IntTupleRef<Cat> shape(const Layout<Cat> &layout FLYDSL_CORE_LIFETIMEBOUND) {
  if constexpr (sizeof...(Is) == 0)
    return layout.shape();
  else
    return shape(layout, {Is...});
}

template <class Cat>
IntTupleRef<Cat> stride(const Layout<Cat> &layout FLYDSL_CORE_LIFETIMEBOUND,
                        llvm::ArrayRef<int32_t> path) {
  return layout.stride().at(path);
}
template <int... Is, class Cat>
IntTupleRef<Cat> stride(const Layout<Cat> &layout FLYDSL_CORE_LIFETIMEBOUND) {
  if constexpr (sizeof...(Is) == 0)
    return layout.stride();
  else
    return stride(layout, {Is...});
}

template <class Cat> Layout<Cat> layout(const Layout<Cat> &value, llvm::ArrayRef<int32_t> path) {
  return make_layout(shape(value, path), stride(value, path));
}
template <int... Is, class Cat> Layout<Cat> layout(const Layout<Cat> &value) {
  if constexpr (sizeof...(Is) == 0)
    return value;
  else
    return layout(value, {Is...});
}

template <class Cat> IntTuple<Cat> size(const Layout<Cat> &layout, llvm::ArrayRef<int32_t> path) {
  return product(shape(layout, path));
}
template <int... Is, class Cat> IntTuple<Cat> size(const Layout<Cat> &layout) {
  if constexpr (sizeof...(Is) == 0)
    return layout.size();
  else
    return size(layout, {Is...});
}

template <class Cat> Layout<Cat> get(const Layout<Cat> &layout, int32_t i) {
  return make_layout(layout.shape().at(i), layout.stride().at(i));
}
template <int... Is, class Cat> Layout<Cat> get(const Layout<Cat> &layout) {
  if constexpr (sizeof...(Is) == 0)
    return layout;
  else
    return make_layout(shape<Is...>(layout), stride<Is...>(layout));
}

template <class Cat> Tile<Cat> get(const Tile<Cat> &tile, int32_t i) { return tile.at(i); }
template <int I, int... Is, class Cat> auto get(const Tile<Cat> &tile) {
  static_assert(I >= 0);
  auto child = tile.at(I);
  if constexpr (sizeof...(Is) == 0)
    return child;
  else
    return get<Is...>(child);
}

template <class Cat> Layout<Cat> take(const Layout<Cat> &layout, int32_t begin, int32_t end) {
  return make_layout(take(layout.shape(), begin, end), take(layout.stride(), begin, end));
}
template <int B, int E, class Cat> Layout<Cat> take(const Layout<Cat> &layout) {
  return take(layout, B, E);
}

template <class Cat>
Layout<Cat> select(const Layout<Cat> &layout, llvm::ArrayRef<int32_t> indices) {
  return make_layout(select(layout.shape(), indices), select(layout.stride(), indices));
}
template <int... Is, class Cat> Layout<Cat> select(const Layout<Cat> &layout) {
  return select(layout, {Is...});
}

template <class Cat> Layout<Cat> group(const Layout<Cat> &layout, int32_t begin, int32_t end) {
  return make_layout(group(layout.shape(), begin, end), group(layout.stride(), begin, end));
}
template <int B, int E, class Cat> Layout<Cat> group(const Layout<Cat> &layout) {
  return group(layout, B, E);
}

template <class Cat> Layout<Cat> flatten(const Layout<Cat> &layout) {
  return make_layout(flatten(layout.shape()), flatten(layout.stride()));
}
template <class Cat, class Profile>
Layout<Cat> unflatten(const Layout<Cat> &layout, const Profile &targetProfile) {
  return make_layout(unflatten(layout.shape(), targetProfile),
                     unflatten(layout.stride(), targetProfile));
}

template <class Cat> Layout<Cat> append(const Layout<Cat> &layout, const Layout<Cat> &sub) {
  return make_layout(append(layout.shape(), sub.shape()), append(layout.stride(), sub.stride()));
}

template <class Cat>
Layout<Cat> append(const Layout<Cat> &layout, const Layout<Cat> &sub, int32_t n) {
  return make_layout(append(layout.shape(), sub.shape(), n),
                     append(layout.stride(), sub.stride(), n));
}
template <class Cat> Layout<Cat> append(const Layout<Cat> &layout, int32_t n) {
  auto unit = make_layout(IntTuple<Cat>::getOne(), IntTuple<Cat>::getZero());
  return append(layout, unit, n);
}

template <int N, class Cat> Layout<Cat> append(const Layout<Cat> &layout) {
  return append(layout, N);
}
template <int N, class Cat> Layout<Cat> append(const Layout<Cat> &layout, const Layout<Cat> &sub) {
  return append(layout, sub, N);
}

template <class Cat> Layout<Cat> prepend(const Layout<Cat> &layout, const Layout<Cat> &sub) {
  return make_layout(prepend(layout.shape(), sub.shape()), prepend(layout.stride(), sub.stride()));
}
template <class Cat>
Layout<Cat> prepend(const Layout<Cat> &layout, const Layout<Cat> &sub, int32_t n) {
  return make_layout(prepend(layout.shape(), sub.shape(), n),
                     prepend(layout.stride(), sub.stride(), n));
}
template <class Cat> Layout<Cat> prepend(const Layout<Cat> &layout, int32_t n) {
  auto unit = make_layout(IntTuple<Cat>::getOne(), IntTuple<Cat>::getZero());
  return prepend(layout, unit, n);
}

template <int N, class Cat> Layout<Cat> prepend(const Layout<Cat> &layout) {
  return prepend(layout, N);
}
template <int N, class Cat> Layout<Cat> prepend(const Layout<Cat> &layout, const Layout<Cat> &sub) {
  return prepend(layout, sub, N);
}

template <class Cat>
Layout<Cat> replace(const Layout<Cat> &layout, int32_t i, const Layout<Cat> &sub) {
  return make_layout(replace(layout.shape(), i, sub.shape()),
                     replace(layout.stride(), i, sub.stride()));
}
template <int I, class Cat> Layout<Cat> replace(const Layout<Cat> &layout, const Layout<Cat> &sub) {
  return replace(layout, I, sub);
}

template <class Cat, class F> Layout<Cat> transform_layout(const Layout<Cat> &layout, F &&f) {
  llvm::SmallVector<IntTuple<Cat>, 8> shapes;
  llvm::SmallVector<IntTuple<Cat>, 8> strides;
  auto shapeIt = layout.shape().begin();
  auto strideIt = layout.stride().begin();
  for (auto end = layout.shape().end(); shapeIt != end; ++shapeIt, ++strideIt) {
    auto transformed = f(make_layout(*shapeIt, *strideIt));
    shapes.emplace_back(transformed.shape());
    strides.emplace_back(transformed.stride());
  }
  return make_layout(make_tuple(shapes), make_tuple(strides));
}

template <class T1, class F, class Cat>
Layout<Cat> transform_layout(const Layout<Cat> &t0, const T1 &t1, F &&f) {
  auto rank0 = t0.rank();
  auto rank1 = t1.rank();
  auto commonRank = std::min(rank0, rank1);
  llvm::SmallVector<IntTuple<Cat>, 8> shapes;
  llvm::SmallVector<IntTuple<Cat>, 8> strides;
  auto shapeIt = t0.shape().begin();
  auto strideIt = t0.stride().begin();
  for (auto i = int32_t{0}; i < commonRank; ++i, ++shapeIt, ++strideIt) {
    auto transformed = f(make_layout(*shapeIt, *strideIt), get(t1, i));
    // Stop at the first invalid mode; the calling operation names itself when
    // it reports the error and takes mode `i` for its frame.
    if (transformed.isError())
      return Layout<Cat>::getError(transformed.errorInfo().withMode(i));
    shapes.emplace_back(transformed.shape());
    strides.emplace_back(transformed.stride());
  }
  for (auto i = commonRank; i < rank0; ++i, ++shapeIt, ++strideIt) {
    shapes.emplace_back(*shapeIt);
    strides.emplace_back(*strideIt);
  }
  for (auto i = commonRank; i < rank1; ++i) {
    if constexpr (is_layout_v<T1>) {
      auto child = get(t1, i);
      shapes.emplace_back(child.shape());
      strides.emplace_back(child.stride());
    } else if constexpr (is_tile_v<std::decay_t<T1>>) {
      auto child = get(t1, i);
      FLYDSL_CORE_ASSERT(child.isLayout() || child.isScalar());
      auto layout = child.isScalar() ? make_layout(IntTuple<Cat>::fromLeaf(child.getScalar()))
                                     : child.getLayout();
      shapes.emplace_back(layout.shape());
      strides.emplace_back(layout.stride());
    } else {
      static_assert(is_int_tuple_v<T1>);
    }
  }
  return make_layout(make_tuple(shapes), make_tuple(strides));
}

template <class Coord, class Cat = category_of<Coord>,
          std::enable_if_t<is_int_tuple_v<Coord>, int> = 0>
auto crd2idx(const Coord &coord, const Layout<Cat> &layout) {
  return crd2idx(coord, layout.shape(), layout.stride());
}

template <class Coord, class Cat = category_of<Coord>>
auto slice(const Coord &coord, const Layout<Cat> &layout) {
  return make_layout(slice(coord, layout.shape()), slice(coord, layout.stride()));
}

template <class Coord, class Cat = category_of<Coord>>
auto dice(const Coord &coord, const Layout<Cat> &layout) {
  return make_layout(dice(coord, layout.shape()), dice(coord, layout.stride()));
}

template <class Coord, class Cat = category_of<Coord>>
auto slice_and_offset(const Coord &coord, const Layout<Cat> &layout) {
  return std::make_pair(slice(coord, layout), crd2idx(coord, layout));
}

template <class Coord, class Cat = category_of<Coord>>
auto domain_offset(const Coord &coord, const Layout<Cat> &layout) {
  return std::make_pair(layout, crd2idx(coord, layout));
}

template <class Cat>
IntTuple<Cat> coprofile(const Layout<Cat> &value, llvm::ArrayRef<int32_t> path) {
  return repeat_like(as_arithmetic_tuple(sum(stride(value, path))), IntTuple<Cat>::getZero());
}
template <int... Is, class Cat> IntTuple<Cat> coprofile(const Layout<Cat> &value) {
  return coprofile(value, {Is...});
}

template <class Cat> IntTuple<Cat> coshape(const Layout<Cat> &value, llvm::ArrayRef<int32_t> path) {
  auto m1_shapes = transform_leaf(shape(value, path),
                                  [](IntTupleRef<Cat> s) { return s - IntTuple<Cat>::getOne(); });
  auto abs_strides =
      transform_leaf(stride(value, path), [](IntTupleRef<Cat> d) { return leaf_abs(d); });
  auto co_coord = as_arithmetic_tuple(inner_product(m1_shapes, abs_strides));
  return transform_leaf(co_coord, [](IntTupleRef<Cat> c) { return c + IntTuple<Cat>::getOne(); });
}
template <int... Is, class Cat> IntTuple<Cat> coshape(const Layout<Cat> &value) {
  return coshape(value, {Is...});
}

template <class Cat> auto cosize(const Layout<Cat> &value, llvm::ArrayRef<int32_t> path) {
  return size(coshape(value, path));
}
template <int... Is, class Cat> auto cosize(const Layout<Cat> &value) {
  return cosize(value, {Is...});
}

template <class Cat> Layout<Cat> coalesce(const Layout<Cat> &layout);

template <class Cat> Layout<Cat> filter_zeros(const Layout<Cat> &layout) {
  return make_layout(filter_zeros(layout.stride(), layout.shape()), layout.stride());
}

template <class Profile, class Cat = category_of<Profile>>
Layout<Cat> filter_zeros(const Layout<Cat> &layout, const Profile &targetProfile) {
  return make_layout(filter_zeros(targetProfile, layout.shape()), layout.stride());
}

template <class Cat> Layout<Cat> filter(const Layout<Cat> &layout) {
  auto result = coalesce(filter_zeros(layout));
  if (result.shape().isLeaf() && result.shape().isOne())
    return make_layout(IntTuple<Cat>::getOne(), IntTuple<Cat>::getZero());
  return result;
}

//===----------------------------------------------------------------------===//
// Layout Algebra
//===----------------------------------------------------------------------===//

template <class Cat> Layout<Cat> coalesce(const Layout<Cat> &layout) {
  if (layout.shape().isLeaf())
    return layout;

  Layout<Cat> flat = flatten(layout);
  llvm::SmallVector<IntTuple<Cat>, 8> shapes;
  llvm::SmallVector<IntTuple<Cat>, 8> strides;
  for (int32_t i = flat.rank(); i-- > 0;) {
    IntTuple<Cat> s = get(flat.shape(), i);
    IntTuple<Cat> d = get(flat.stride(), i);
    if (s.isOne())
      continue;
    if (!shapes.empty() && shapes.front().isStatic()) {
      auto boundary = leaf_mul(s, d);
      if (boundary.isStatic() && strides.front().isStatic() && boundary == strides.front()) {
        shapes.front() = leaf_mul(s, shapes.front());
        strides.front() = d;
        continue;
      }
    }
    shapes.insert(shapes.begin(), s);
    strides.insert(strides.begin(), d);
  }
  if (shapes.empty()) {
    if (flat.rank() > 0)
      return make_layout(IntTuple<Cat>::getOne(), get(flat.stride(), flat.rank() - 1));
    return make_layout(IntTuple<Cat>::getOne(), IntTuple<Cat>::getZero());
  }
  return make_layout(detail::tupleFromLeaves<Cat>(shapes), detail::tupleFromLeaves<Cat>(strides));
}

template <class Cat> Layout<Cat> coalesce_x(const Layout<Cat> &layout) {
  auto flat = flatten(layout);
  llvm::SmallVector<IntTuple<Cat>, 8> shapes;
  llvm::SmallVector<IntTuple<Cat>, 8> strides;
  for (auto i = int32_t{0}, r = flat.rank(); i < r; ++i) {
    auto s = get(flat.shape(), i);
    auto d = get(flat.stride(), i);
    while (!shapes.empty() && shapes.back().isOne()) {
      shapes.pop_back();
      strides.pop_back();
    }
    if (!shapes.empty()) {
      auto &sA = shapes.back();
      auto &dA = strides.back();
      if (sA.isStatic() && s.isStatic() && dA.isStatic() && d.isStatic() && leaf_mul(sA, dA) == d) {
        sA = leaf_mul(sA, s);
        continue;
      }
    }
    shapes.push_back(s);
    strides.push_back(d);
  }
  if (shapes.empty())
    return make_layout(IntTuple<Cat>::getOne(), IntTuple<Cat>::getZero());
  return make_layout(detail::tupleFromLeaves<Cat>(shapes), detail::tupleFromLeaves<Cat>(strides));
}

template <class Profile, class Cat = category_of<Profile>>
Layout<Cat> coalesce_x(const Layout<Cat> &layout, const Profile &targetProfile) {
  if (targetProfile.isLeaf())
    return coalesce_x(layout);
  if (layout.isError())
    return detail::tagError(layout, AlgebraOp::CoalesceX);
  if (targetProfile.rank() > layout.rank())
    return Layout<Cat>::getError(
        ErrorInfo(ErrorCode::TilerRankExceeded).withFrame(AlgebraOp::CoalesceX));
  return detail::tagError(
      transform_layout(layout, targetProfile,
                       [](auto const &l, auto const &t) { return coalesce_x(l, t); }),
      AlgebraOp::CoalesceX);
}

template <class Profile, class Cat = category_of<Profile>>
Layout<Cat> coalesce(const Layout<Cat> &layout, const Profile &targetProfile) {
  if (targetProfile.isLeaf())
    return coalesce(layout);
  if (layout.isError())
    return detail::tagError(layout, AlgebraOp::Coalesce);
  if (targetProfile.rank() > layout.rank())
    return Layout<Cat>::getError(
        ErrorInfo(ErrorCode::TilerRankExceeded).withFrame(AlgebraOp::Coalesce));

  return detail::tagError(
      transform_layout(layout, targetProfile,
                       [](auto const &l, auto const &t) { return coalesce(l, t); }),
      AlgebraOp::Coalesce);
}

template <class T, class Cat = category_of<T>> IntTuple<Cat> coalesce(const T &shape) {
  auto flat = flatten(shape);

  llvm::SmallVector<IntTuple<Cat>, 8> result;
  auto append = [&](IntTupleRef<Cat> value) {
    IntTuple<Cat> leaf = value;
    if (!result.empty() && result.back().isStatic() == leaf.isStatic())
      result.back() = leaf_mul(result.back(), leaf);
    else
      result.push_back(std::move(leaf));
  };
  for (auto child : flat.asRef())
    append(child);
  return detail::tupleFromLeaves<Cat>(result);
}

template <class Profile, class Cat = category_of<Profile>>
Layout<Cat> filter(const Layout<Cat> &layout, const Profile &targetProfile) {
  if (targetProfile.isLeaf())
    return filter(layout);
  else
    return detail::tagError(
        transform_layout(layout, targetProfile,
                         [](auto const &l, auto const &t) { return filter(l, t); }),
        AlgebraOp::Filter);
}

namespace detail {
/// Whether `IntTupleRef::at(path)` names a mode of `tuple`.
template <class Cat> bool hasModePath(IntTupleRef<Cat> tuple, llvm::ArrayRef<int32_t> path) {
  for (auto i : path) {
    if (tuple.isLeaf() ? i != 0 : i < 0 || i >= tuple.rank())
      return false;
    tuple = tuple.at(i);
  }
  return true;
}

template <class Cat> bool sameBasisPath(IntTupleRef<Cat> basis, llvm::ArrayRef<int32_t> path) {
  if (!basis.isBasis() || basis.modeCount() != path.size())
    return false;
  for (auto i = size_t{0}; i < path.size(); ++i)
    if (basis.mode(i) != path[i])
      return false;
  return true;
}

template <class Cat>
Layout<Cat> makeBasisComplementBucket(llvm::ArrayRef<int32_t> path,
                                      llvm::SmallVectorImpl<BasisComplementMode<Cat>> &modes) {
  llvm::SmallVector<BasisComplementMode<Cat> *, 8> selected;
  for (auto &mode : modes) {
    if (sameBasisPath(mode.stride.asRef(), path)) {
      mode.used = true;
      if (!(mode.shape.isSInt() && mode.shape.isOne()) &&
          !(mode.stride.scale().isSInt() && mode.stride.scale().isZero()))
        selected.push_back(&mode);
    }
  }

  auto allStatic =
      llvm::all_of(selected, [](const auto *mode) { return mode->stride.scale().isSInt(); });
  if (!allStatic && selected.size() > 1)
    return Layout<Cat>::getError(ErrorCode::UnsupportedDynamicComplement);
  if (allStatic)
    std::stable_sort(selected.begin(), selected.end(), [](const auto *a, const auto *b) {
      return a->stride.scale().staticValue() < b->stride.scale().staticValue();
    });

  llvm::SmallVector<IntTuple<Cat>, 8> shapes;
  llvm::SmallVector<IntTuple<Cat>, 8> strides;
  auto current = IntTuple<Cat>::getOne();
  for (const auto *mode : selected) {
    auto scale = mode->stride.scale();
    if (scale.isSInt() && current.isSInt() && scale.staticValue() < current.staticValue())
      return Layout<Cat>::getError(ErrorCode::NonInjectiveLayout);
    auto gap = leaf_div(scale, current);
    if (!gap.isOne()) {
      shapes.push_back(gap);
      strides.push_back(IntTuple<Cat>::getBasis(current, path));
    }
    current = leaf_mul(scale, mode->shape);
  }

  shapes.push_back(IntTuple<Cat>::getOne());
  strides.push_back(IntTuple<Cat>::getBasis(current, path));
  return make_layout(tupleFromLeaves<Cat>(shapes), tupleFromLeaves<Cat>(strides));
}

template <class Cat>
Layout<Cat> makeBasisComplement(IntTupleRef<Cat> coprofile, llvm::SmallVectorImpl<int32_t> &path,
                                llvm::SmallVectorImpl<BasisComplementMode<Cat>> &modes) {
  if (coprofile.isLeaf()) {
    auto scalarPath = llvm::SmallVector<int32_t, kMaxBasisModes>(path.begin(), path.end());
    if (scalarPath.empty())
      scalarPath.push_back(0);
    return makeBasisComplementBucket(scalarPath, modes);
  }

  llvm::SmallVector<IntTuple<Cat>, 8> shapes;
  llvm::SmallVector<IntTuple<Cat>, 8> strides;
  auto i = int32_t{0};
  for (auto coprofileChild : coprofile) {
    path.push_back(i++);
    auto child = makeBasisComplement(coprofileChild, path, modes);
    path.pop_back();
    shapes.emplace_back(child.shape());
    strides.emplace_back(child.stride());
  }
  return make_layout(make_tuple(shapes), make_tuple(strides));
}

template <class Cat>
Layout<Cat> extendBasisComplementBucket(const Layout<Cat> &base, IntTupleRef<Cat> shape) {
  if (base.stride().leafCount() == 0)
    return Layout<Cat>::getError(ErrorCode::InvalidLayout);
  auto lastStride = base.stride().leaves().back();
  auto size = lastStride.isBasis() ? lastStride.scale() : lastStride;
  if (!size.isInt() || (size.isSInt() && size.isZero()))
    return Layout<Cat>::getError(ErrorCode::UnsupportedDynamicComplement);

  llvm::SmallVector<IntTuple<Cat>, 8> shapeR;
  shapeR.reserve(shape.leafCount());
  for (auto extent : shape.leaves()) {
    shapeR.push_back(IntTuple<Cat>::fromLeaf(ceil_div(extent, size)));
    size = ceil_div(size, extent);
  }
  auto extension = make_tuple(shapeR);
  auto extensionStride = compact_col_major(extension, IntTuple<Cat>::fromLeaf(lastStride));
  auto newShape = base.shape().isLeaf() ? std::move(extension)
                                        : replace(base.shape(), base.rank() - 1, extension);
  auto newStride = base.stride().isLeaf()
                       ? std::move(extensionStride)
                       : replace(base.stride(), base.rank() - 1, extensionStride);
  return coalesce(make_layout(std::move(newShape), std::move(newStride)));
}

template <class Cat>
Layout<Cat> complement_basis_layout(IntTupleRef<Cat> coprofile, const Layout<Cat> &base,
                                    IntTupleRef<Cat> shape, IntTupleRef<Cat> basis) {
  if (coprofile.isLeaf())
    return extendBasisComplementBucket(base, shape);

  auto shapeMode = [&](int32_t i) { return shape.isLeaf() ? shape : shape.at(i); };
  auto basisMode = [&](int32_t i) { return basis.isLeaf() ? basis : basis.at(i); };
  auto coprofileRank = coprofile.rank();
  if (!shape.isLeaf() && shape.rank() < coprofileRank)
    return Layout<Cat>::getError(
        ErrorInfo(ErrorCode::IncompatibleProfile).withFrame(AlgebraOp::Complement));
  auto totalRank = std::max(coprofileRank, shape.rank());
  llvm::SmallVector<IntTuple<Cat>, 8> shapes;
  llvm::SmallVector<IntTuple<Cat>, 8> strides;
  shapes.reserve(totalRank);
  strides.reserve(totalRank);
  for (auto i = int32_t{0}; i < totalRank; ++i) {
    auto child = i < coprofileRank ? complement_basis_layout(coprofile.at(i), get(base, i),
                                                             shapeMode(i), basisMode(i))
                                   : make_layout(shapeMode(i), basisMode(i));
    if (child.isError())
      return tagError(child, AlgebraOp::Complement, i);
    shapes.emplace_back(child.shape());
    strides.emplace_back(child.stride());
  }
  return make_layout(make_tuple(shapes), make_tuple(strides));
}

template <class Cat>
Layout<Cat> composition_impl(IntTupleRef<Cat> lhsShape, IntTupleRef<Cat> lhsStride,
                             IntTupleRef<Cat> rhsShape, IntTupleRef<Cat> rhsStride) {
  if (!rhsShape.isLeaf()) {
    FLYDSL_CORE_ASSERT(!rhsStride.isLeaf() && congruent(rhsShape, rhsStride));
    llvm::SmallVector<IntTuple<Cat>, 8> resultShapes;
    llvm::SmallVector<IntTuple<Cat>, 8> resultStrides;
    auto shapeIt = rhsShape.begin();
    auto strideIt = rhsStride.begin();
    for (auto end = rhsShape.end(); shapeIt != end; ++shapeIt, ++strideIt) {
      Layout<Cat> child = composition_impl(lhsShape, lhsStride, *shapeIt, *strideIt);
      if (child.isError())
        return tagError(child, AlgebraOp::Composition, resultShapes.size());
      resultShapes.emplace_back(child.shape());
      resultStrides.emplace_back(child.stride());
    }
    return make_layout(make_tuple(resultShapes), make_tuple(resultStrides));
  }

  FLYDSL_CORE_ASSERT(rhsStride.isLeaf());
  IntTuple<Cat> rshape = rhsShape;
  IntTuple<Cat> rstride = rhsStride;

  if (rstride.isBasis()) {
    llvm::SmallVector<int32_t, kMaxBasisModes> path;
    for (uint8_t mode : rstride.modes())
      path.push_back(mode);
    // A leaf lhs only has mode 0.
    if (lhsShape.isLeaf() ? path.size() != 1 || path.front() != 0 : !hasModePath(lhsShape, path))
      return Layout<Cat>::getError(
          ErrorInfo(ErrorCode::BasisModeOutOfRange).withFrame(AlgebraOp::Composition));
    auto selectedShape = lhsShape;
    auto selectedStride = lhsStride;
    if (!lhsShape.isLeaf()) {
      selectedShape = lhsShape.at(path);
      selectedStride = lhsStride.at(path);
    }
    auto scale = rstride.scale();
    return composition_impl(selectedShape, selectedStride, rhsShape, scale.asRef());
  }
  if (rstride.isZero())
    return make_layout(rhsShape, rhsStride);
  if (lhsShape.isLeaf()) {
    FLYDSL_CORE_ASSERT(lhsStride.isLeaf());
    return make_layout(rhsShape, rstride * lhsStride);
  }

  FLYDSL_CORE_ASSERT(!lhsStride.isLeaf() && congruent(lhsShape, lhsStride));
  llvm::SmallVector<IntTuple<Cat>, 8> resultShape;
  llvm::SmallVector<IntTuple<Cat>, 8> resultStride;
  auto restShape = rshape;
  auto restStride = rstride;
  int32_t lhsRank = lhsShape.rank();
  FLYDSL_CORE_ASSERT(lhsRank > 0);

  for (int32_t i = 0; i + 1 < lhsRank; ++i) {
    IntTupleRef<Cat> shapeMode = lhsShape.at(i);
    IntTupleRef<Cat> strideMode = lhsStride.at(i);
    FLYDSL_CORE_ASSERT(shapeMode.isLeaf() && strideMode.isLeaf());
    auto currShape = shapeMode;
    auto currStride = strideMode;
    if (currShape.isSInt() && currShape.isZero())
      return Layout<Cat>::getError(
          ErrorInfo(ErrorCode::DivisionByZero).withFrame(AlgebraOp::Composition));
    if (currShape.isSInt() && restStride.isSInt() &&
        restStride.staticValue() % currShape.staticValue() != 0 &&
        restStride.staticValue() >= currShape.staticValue())
      return Layout<Cat>::getError(
          ErrorInfo(ErrorCode::IncompatibleShapeDivision).withFrame(AlgebraOp::Composition));
    auto nextShape = leaf_ceil_div(currShape, leaf_abs(restStride));
    auto nextStride = leaf_ceil_div(leaf_abs(restStride), currShape) * leaf_signum(restStride);

    if (nextShape.isOne() || restShape.isOne()) {
      restStride = nextStride;
      continue;
    }

    auto newShape = leaf_min(nextShape, restShape);
    if (newShape.isSInt() && newShape.isZero())
      return Layout<Cat>::getError(
          ErrorInfo(ErrorCode::DivisionByZero).withFrame(AlgebraOp::Composition));
    if (newShape.isSInt() && restShape.isSInt() &&
        restShape.staticValue() % newShape.staticValue() != 0)
      return Layout<Cat>::getError(
          ErrorInfo(ErrorCode::IncompatibleShapeDivision).withFrame(AlgebraOp::Composition));
    resultShape.push_back(newShape);
    resultStride.push_back(restStride * currStride);
    restShape = restShape / newShape;
    restStride = nextStride;
  }

  IntTupleRef<Cat> lastStride = lhsStride.at(lhsRank - 1);
  FLYDSL_CORE_ASSERT(lastStride.isLeaf());
  auto finalStride = restStride * lastStride;
  if (resultShape.empty())
    return make_layout(std::move(restShape), std::move(finalStride));
  if (restShape.isSInt() && restShape.isOne())
    return make_layout(unwrap(tupleFromLeaves<Cat>(resultShape)),
                       unwrap(tupleFromLeaves<Cat>(resultStride)));
  resultShape.push_back(restShape);
  resultStride.push_back(finalStride);
  return make_layout(tupleFromLeaves<Cat>(resultShape), tupleFromLeaves<Cat>(resultStride));
}

} // namespace detail

template <class Cat> Layout<Cat> composition(const Layout<Cat> &lhs, const Layout<Cat> &rhs) {
  if (auto error = detail::firstError(lhs, rhs))
    return Layout<Cat>::getError(error->withFrame(AlgebraOp::Composition));
  auto flatLhs = coalesce_x(lhs, coprofile(rhs));
  if (flatLhs.isError())
    return detail::tagError(flatLhs, AlgebraOp::Composition);
  auto result =
      detail::composition_impl(flatLhs.shape(), flatLhs.stride(), rhs.shape(), rhs.stride());
  return detail::tagError(std::move(result), AlgebraOp::Composition);
}

template <class Cat> Layout<Cat> composition(const Layout<Cat> &lhs, const Tile<Cat> &rhs) {
  if (auto error = detail::firstError(lhs, rhs))
    return Layout<Cat>::getError(error->withFrame(AlgebraOp::Composition));
  if (rhs.isNone())
    return lhs;
  if (rhs.isLayout())
    return composition(lhs, rhs.getLayout());
  if (rhs.isScalar())
    return composition(
        lhs, make_layout(IntTuple<Cat>::fromLeaf(rhs.getScalar()), IntTuple<Cat>::getOne()));
  if (rhs.rank() > lhs.rank())
    return Layout<Cat>::getError(
        ErrorInfo(ErrorCode::TilerRankExceeded).withFrame(AlgebraOp::Composition));
  llvm::SmallVector<IntTuple<Cat>, 8> shapes;
  llvm::SmallVector<IntTuple<Cat>, 8> strides;
  auto shapeIt = lhs.shape().begin();
  auto strideIt = lhs.stride().begin();
  auto rhsIt = rhs.begin();
  for (auto end = rhs.end(); rhsIt != end; ++rhsIt, ++shapeIt, ++strideIt) {
    auto child = composition(make_layout(*shapeIt, *strideIt), *rhsIt);
    if (child.isError())
      return detail::tagError(child, AlgebraOp::Composition, shapes.size());
    shapes.emplace_back(child.shape());
    strides.emplace_back(child.stride());
  }
  return make_layout(make_tuple(shapes), make_tuple(strides));
}

template <class Tiler, class Cat = category_of<Tiler>>
Layout<Cat> composition(const Layout<Cat> &lhs, const Tiler &rhs) {
  return composition(lhs, shape_to_tile(rhs));
}

namespace detail {

template <class Cat>
ComplementPrefix<Cat> complement_prefix(IntTupleRef<Cat> shape, IntTupleRef<Cat> stride) {
  auto flat = flatten(make_layout(shape, stride));
  ComplementPrefix<Cat> out;
  auto rank = flat.rank();
  llvm::SmallVector<int32_t, 8> indices;
  indices.reserve(static_cast<size_t>(rank));
  for (auto i = int32_t{0}; i < rank; ++i) {
    if (rank > 1 && !get(flat.stride(), i).isSInt()) {
      out.ok = false;
      out.code = ErrorCode::UnsupportedDynamicComplement;
      return out;
    }
    indices.push_back(i);
  }
  if (rank > 1)
    std::stable_sort(indices.begin(), indices.end(), [&](int32_t a, int32_t b) {
      return get(flat.stride(), a).staticValue() < get(flat.stride(), b).staticValue();
    });

  llvm::SmallVector<IntTuple<Cat>, 8> resultShape;
  llvm::SmallVector<IntTuple<Cat>, 8> resultStride;
  auto current = IntTuple<Cat>::getOne();
  for (auto index : indices) {
    IntTuple<Cat> modeShape = get(flat.shape(), index);
    IntTuple<Cat> modeStride = get(flat.stride(), index);
    auto gap = modeStride / current;
    if (gap.isSInt() && gap.staticValue() <= 0) {
      out.ok = false;
      out.code = ErrorCode::NonInjectiveLayout;
      return out;
    }
    resultShape.push_back(gap);
    resultStride.push_back(current);
    current = modeStride * modeShape;
  }

  out.result_shape = tupleFromLeaves<Cat>(resultShape);
  out.result_stride = tupleFromLeaves<Cat>(resultStride);
  out.new_stride = std::move(current);
  return out;
}

template <class Cat>
Layout<Cat> complement(IntTupleRef<Cat> shape, IntTupleRef<Cat> stride, IntTupleRef<Cat> cotarget) {
  if (stride.isZero())
    return make_layout(coalesce(cotarget));
  auto prefix = complement_prefix(shape, stride);
  if (!prefix.ok)
    return Layout<Cat>::getError(prefix.code);
  auto restShape = coalesce(ceil_div(cotarget, prefix.new_stride));
  auto restStride = compact_col_major(restShape, prefix.new_stride);
  return coalesce(make_layout(make_tuple(prefix.result_shape, restShape),
                              make_tuple(prefix.result_stride, restStride)));
}

} // namespace detail

template <class Cat> bool has_basis_stride(const Layout<Cat> &layout) {
  return llvm::any_of(layout.stride().leaves(), [](const auto &value) { return value.isBasis(); });
}

template <class Cat> Layout<Cat> complement(const Layout<Cat> &layout) {
  if (layout.isError())
    return detail::tagError(layout, AlgebraOp::Complement);
  if (has_basis_stride(layout)) {
    // TODO(error-model): ratio-valued basis scales, more than one dynamic
    // stride leaf globally, and mixtures of basis strides with nonzero integer
    // strides are invalid.
    llvm::SmallVector<detail::BasisComplementMode<Cat>, 8> modes;
    auto flatShape = flatten(layout.shape());
    auto flatStride = flatten(layout.stride());
    if (flatShape.rank() != flatStride.rank())
      return Layout<Cat>::getError(
          ErrorInfo(ErrorCode::InvalidLayout).withFrame(AlgebraOp::Complement));
    auto shapeIt = flatShape.asRef().begin();
    auto strideIt = flatStride.asRef().begin();
    for (auto end = flatStride.asRef().end(); strideIt != end; ++strideIt, ++shapeIt) {
      IntTuple<Cat> stride = *strideIt;
      if (!stride.isBasis() && !(stride.isSInt() && stride.isZero()))
        return Layout<Cat>::getError(
            ErrorInfo(ErrorCode::InvalidLayout).withFrame(AlgebraOp::Complement));
      if (stride.isBasis())
        modes.push_back({*shapeIt, std::move(stride)});
    }
    auto path = llvm::SmallVector<int32_t, kMaxBasisModes>{};
    auto profile = coprofile(layout);
    auto result = detail::makeBasisComplement(profile.asRef(), path, modes);
    if (llvm::any_of(modes, [](const auto &mode) { return !mode.used; }))
      return Layout<Cat>::getError(
          ErrorInfo(ErrorCode::InvalidLayout).withFrame(AlgebraOp::Complement));
    return detail::tagError(std::move(result), AlgebraOp::Complement);
  } else {
    auto filter_layout = filter(layout);
    if (filter_layout.stride().isZero())
      return make_layout(IntTuple<Cat>::getOne(), IntTuple<Cat>::getOne());
    auto prefix = detail::complement_prefix(filter_layout.shape(), filter_layout.stride());
    if (!prefix.ok)
      return Layout<Cat>::getError(ErrorInfo(prefix.code).withFrame(AlgebraOp::Complement));
    auto result = coalesce(make_layout(prefix.result_shape, prefix.result_stride));
    auto terminal = make_layout(IntTuple<Cat>::getOne(), std::move(prefix.new_stride));
    if (size(result).isOne())
      return terminal;
    return append(result, terminal);
  }
}

template <class Cotarget, class Cat = category_of<Cotarget>,
          std::enable_if_t<is_int_tuple_v<Cotarget>, int> = 0>
Layout<Cat> complement(const Layout<Cat> &layout, const Cotarget &cotarget) {
  if (auto error = detail::firstError(layout, cotarget))
    return Layout<Cat>::getError(error->withFrame(AlgebraOp::Complement));
  if (has_basis_stride(layout)) {
    auto base = complement(layout);
    if (base.shape().isError())
      return base;
    auto profile = coprofile(layout);
    auto basis = make_basis_like(cotarget);
    auto result =
        detail::complement_basis_layout(profile.asRef(), base, cotarget.asRef(), basis.asRef());
    if (result.isError())
      return detail::tagError(result, AlgebraOp::Complement);
    if (profile.isLeaf())
      return make_layout(make_tuple(result.shape()), make_tuple(result.stride()));
    return result;
  } else {
    auto filter_layout = filter(layout);
    auto result =
        detail::complement(filter_layout.shape(), filter_layout.stride(), cotarget.asRef());
    return detail::tagError(std::move(result), AlgebraOp::Complement);
  }
}

namespace detail {

template <class Cat>
BasisInverseBucket<Cat> *findBasisBucket(llvm::SmallVectorImpl<BasisInverseBucket<Cat>> &buckets,
                                         IntTupleRef<Cat> basis) {
  for (auto &bucket : buckets) {
    if (bucket.path.size() != basis.modeCount())
      continue;
    auto matches = true;
    for (auto i = size_t{0}; i < bucket.path.size(); ++i)
      matches &= bucket.path[i] == basis.mode(i);
    if (matches)
      return &bucket;
  }
  BasisInverseBucket<Cat> bucket;
  for (auto mode : basis.modes())
    bucket.path.push_back(mode);
  buckets.push_back(std::move(bucket));
  return &buckets.back();
}

template <class Cat>
IntTuple<Cat> buildBasisInverseTuple(IntTupleRef<Cat> profile, llvm::ArrayRef<int32_t> path,
                                     llvm::ArrayRef<BasisInverseBucket<Cat>> buckets,
                                     bool buildShape) {
  if (profile.isLeaf()) {
    for (const auto &bucket : buckets) {
      if (bucket.path != path)
        continue;
      const auto &values = buildShape ? bucket.shapes : bucket.strides;
      if (!values.empty())
        return tupleFromLeaves<Cat>(values);
      break;
    }
    return IntTuple<Cat>::getSInt(buildShape ? 1 : 0);
  }

  llvm::SmallVector<IntTuple<Cat>, 8> children;
  llvm::SmallVector<int32_t, kMaxBasisModes> childPath(path.begin(), path.end());
  auto i = int32_t{0};
  for (auto profileChild : profile) {
    childPath.push_back(i++);
    children.push_back(buildBasisInverseTuple(profileChild, llvm::ArrayRef<int32_t>(childPath),
                                              buckets, buildShape));
    childPath.pop_back();
  }
  return make_tuple(children);
}

template <class Cat> Layout<Cat> basis_inverse(const Layout<Cat> &layout, bool left) {
  auto flat = flatten(coalesce(layout));
  llvm::SmallVector<IntTuple<Cat>, 8> prefixProducts;
  llvm::SmallVector<int32_t, 8> indices;
  auto prefix = IntTuple<Cat>::getOne();
  auto shapeIt = flat.shape().begin();
  auto strideIt = flat.stride().begin();
  auto i = int32_t{0};
  for (auto end = flat.shape().end(); shapeIt != end; ++shapeIt, ++strideIt, ++i) {
    IntTuple<Cat> modeShape = *shapeIt;
    IntTuple<Cat> modeStride = *strideIt;
    if (!modeShape.isSInt() || (!modeStride.isBasis() && !modeStride.isZero()) ||
        (modeStride.isBasis() && !modeStride.scale().isSInt()))
      return Layout<Cat>::getError(ErrorCode::InvalidLayout);
    prefixProducts.push_back(prefix);
    prefix = prefix * modeShape;
    if (modeStride.isBasis())
      indices.push_back(i);
  }

  std::stable_sort(indices.begin(), indices.end(), [&](int32_t lhs, int32_t rhs) {
    auto a = get(flat.stride(), lhs);
    auto b = get(flat.stride(), rhs);
    auto common = std::min(a.modeCount(), b.modeCount());
    for (auto i = uint8_t{0}; i < common; ++i) {
      if (a.mode(i) != b.mode(i))
        return a.mode(i) < b.mode(i);
    }
    if (a.modeCount() != b.modeCount())
      return a.modeCount() < b.modeCount();
    return a.scale().staticValue() < b.scale().staticValue();
  });

  llvm::SmallVector<BasisInverseBucket<Cat>, 8> buckets;
  for (auto index : indices) {
    IntTuple<Cat> shape = get(flat.shape(), index);
    IntTuple<Cat> basis = get(flat.stride(), index);
    auto scale = basis.scale();
    auto *bucket = findBasisBucket(buckets, basis.asRef());
    if (scale.isZero() || shape.isOne())
      continue;

    if (!left) {
      if (bucket->current != scale)
        continue;
      bucket->shapes.push_back(shape);
      bucket->strides.push_back(prefixProducts[static_cast<size_t>(index)]);
      bucket->current = shape * scale;
      continue;
    }

    auto remainder = scale % bucket->current;
    if (!remainder.isZero())
      return Layout<Cat>::getError(ErrorCode::NonInjectiveLayout);
    auto gap = scale / bucket->current;
    if (bucket->shapes.empty()) {
      bucket->shapes.push_back(gap);
      bucket->strides.push_back(IntTuple<Cat>::getZero());
    } else {
      // A smaller gap than the preceding extent overlaps coordinates already
      // covered by that mode and is invalid.
      if (gap.isSInt() && bucket->shapes.back().isSInt() &&
          gap.staticValue() < bucket->shapes.back().staticValue())
        return Layout<Cat>::getError(ErrorCode::NonInjectiveLayout);
      bucket->shapes.back() = gap;
    }
    bucket->current = bucket->current * gap;
    bucket->shapes.push_back(shape);
    bucket->strides.push_back(prefixProducts[static_cast<size_t>(index)]);
  }

  auto profile = coprofile(layout);
  auto path = llvm::SmallVector<int32_t, kMaxBasisModes>{};
  auto result =
      make_layout(buildBasisInverseTuple(profile.asRef(), llvm::ArrayRef<int32_t>(path),
                                         llvm::ArrayRef<BasisInverseBucket<Cat>>(buckets), true),
                  buildBasisInverseTuple(profile.asRef(), llvm::ArrayRef<int32_t>(path),
                                         llvm::ArrayRef<BasisInverseBucket<Cat>>(buckets), false));
  return coalesce(result, profile);
}

} // namespace detail

template <class Cat> Layout<Cat> right_inverse(const Layout<Cat> &layout) {
  if (layout.isError())
    return detail::tagError(layout, AlgebraOp::RightInverse);
  if (llvm::any_of(layout.stride().leaves(), [](const auto &value) { return value.isBasis(); }))
    return detail::tagError(detail::basis_inverse(layout, false), AlgebraOp::RightInverse);
  if (llvm::any_of(layout.stride().leaves(), [](const auto &value) { return !value.isInt(); }))
    return Layout<Cat>::getError(
        ErrorInfo(ErrorCode::InvalidLayout).withFrame(AlgebraOp::RightInverse));

  auto flat = flatten(coalesce(layout));
  auto rank = flat.rank();
  llvm::SmallVector<IntTuple<Cat>, 8> prefixProducts;
  auto prefix = IntTuple<Cat>::getOne();
  for (auto i = int32_t{0}; i < rank; ++i) {
    prefixProducts.push_back(prefix);
    prefix = prefix * get(flat.shape(), i);
  }

  llvm::SmallVector<int32_t, 8> indices;
  for (auto i = int32_t{0}; i < rank; ++i)
    if (get(flat.stride(), i).isSInt())
      indices.push_back(i);
  std::stable_sort(indices.begin(), indices.end(), [&](int32_t a, int32_t b) {
    return get(flat.stride(), a).staticValue() < get(flat.stride(), b).staticValue();
  });

  llvm::SmallVector<IntTuple<Cat>, 8> resultShape{IntTuple<Cat>::getOne()};
  llvm::SmallVector<IntTuple<Cat>, 8> resultStride{IntTuple<Cat>::getZero()};
  auto current = IntTuple<Cat>::getOne();
  for (auto index : indices) {
    IntTuple<Cat> modeShape = get(flat.shape(), index);
    IntTuple<Cat> modeStride = get(flat.stride(), index);
    if (modeStride == current) {
      resultShape.push_back(modeShape);
      resultStride.push_back(prefixProducts[index]);
      current = modeShape * modeStride;
    }
  }
  auto result = coalesce(make_layout(detail::tupleFromLeaves<Cat>(resultShape),
                                     detail::tupleFromLeaves<Cat>(resultStride)));
  return detail::tagError(std::move(result), AlgebraOp::RightInverse);
}

template <class Cat> Layout<Cat> left_inverse(const Layout<Cat> &layout) {
  if (layout.isError())
    return detail::tagError(layout, AlgebraOp::LeftInverse);
  if (llvm::any_of(layout.stride().leaves(), [](const auto &value) { return value.isBasis(); }))
    return detail::tagError(detail::basis_inverse(layout, true), AlgebraOp::LeftInverse);
  if (llvm::any_of(layout.stride().leaves(), [](const auto &value) { return !value.isInt(); }))
    return Layout<Cat>::getError(
        ErrorInfo(ErrorCode::InvalidLayout).withFrame(AlgebraOp::LeftInverse));
  auto flat = flatten(coalesce(layout));
  auto rank = flat.rank();
  if (rank == 0)
    return make_layout(IntTuple<Cat>::getOne(), IntTuple<Cat>::getZero());
  llvm::SmallVector<IntTuple<Cat>, 8> prefixProducts;
  llvm::SmallVector<int32_t, 8> indices;
  auto prefix = IntTuple<Cat>::getOne();
  auto dynamicCount = int32_t{0};
  for (auto i = int32_t{0}; i < rank; ++i) {
    auto modeStride = get(flat.stride(), i);
    dynamicCount += modeStride.isDInt();
    prefixProducts.push_back(prefix);
    prefix = prefix * get(flat.shape(), i);
    indices.push_back(i);
  }
  if (dynamicCount > 1 || (dynamicCount == 1 && rank != 1))
    return Layout<Cat>::getError(
        ErrorInfo(ErrorCode::InvalidLayout).withFrame(AlgebraOp::LeftInverse));
  std::stable_sort(indices.begin(), indices.end(), [&](int32_t a, int32_t b) {
    if (!get(flat.stride(), a).isSInt())
      return false;
    if (!get(flat.stride(), b).isSInt())
      return true;
    return get(flat.stride(), a).staticValue() < get(flat.stride(), b).staticValue();
  });

  llvm::SmallVector<IntTuple<Cat>, 8> resultShape{IntTuple<Cat>::getOne()};
  llvm::SmallVector<IntTuple<Cat>, 8> resultStride{IntTuple<Cat>::getZero()};
  auto resultSize = IntTuple<Cat>::getOne();
  for (auto index : indices) {
    IntTuple<Cat> modeStride = get(flat.stride(), index);
    IntTuple<Cat> modeShape = get(flat.shape(), index);
    if (modeStride.isZero())
      continue;
    if (modeShape.isOne())
      continue;
    auto remainder = modeStride % resultSize;
    if (!remainder.isZero())
      return Layout<Cat>::getError(
          ErrorInfo(ErrorCode::NonInjectiveLayout).withFrame(AlgebraOp::LeftInverse));
    auto gap = modeStride / resultSize;
    resultShape.back() = gap;
    resultSize = resultSize * gap;
    resultShape.push_back(modeShape);
    resultStride.push_back(prefixProducts[static_cast<size_t>(index)]);
  }
  if (resultShape.size() == 1)
    return make_layout(IntTuple<Cat>::getOne(), IntTuple<Cat>::getZero());
  auto result = coalesce(make_layout(detail::tupleFromLeaves<Cat>(resultShape),
                                     detail::tupleFromLeaves<Cat>(resultStride)));
  return detail::tagError(std::move(result), AlgebraOp::LeftInverse);
}

template <class Cat> Layout<Cat> max_common_layout(const Layout<Cat> &a, const Layout<Cat> &b) {
  auto inverseB = right_inverse(b);
  auto common = coalesce(composition(a, inverseB));

  if (shape<0>(common).isStatic() && stride<0>(common).isOne()) {
    return composition(inverseB, layout<0>(common));
  } else
    return make_layout(IntTuple<Cat>::getOne(), IntTuple<Cat>::getZero());
}

template <class Cat> IntTuple<Cat> max_common_vector(const Layout<Cat> &a, const Layout<Cat> &b) {
  auto common = coalesce(composition(a, right_inverse(b)));

  if (shape<0>(common).isStatic() && stride<0>(common).isOne())
    return shape<0>(common);
  else
    return IntTuple<Cat>::getOne();
}

template <class A, class B, class Cat = category_of<A>>
Layout<Cat> domain_distribute(const A &a, const B &b) {
  auto flatA = flatten(shape(a));
  if (flatA.isError())
    return Layout<Cat>::getError(flatA.errorInfo().withFrame(AlgebraOp::DomainDistribute));
  if (!flatA.isStatic())
    return Layout<Cat>::getError(
        ErrorInfo(ErrorCode::ExpectedStaticOperand).withFrame(AlgebraOp::DomainDistribute));

  IntTuple<Cat> rest = b.asRef();
  llvm::SmallVector<IntTuple<Cat>, 8> resultShape;
  for (auto value : flatA.asRef()) {
    auto factor = leaf_gcd(value, rest);
    resultShape.push_back(factor);
    rest = leaf_div(rest, factor);
  }
  auto result = detail::tupleFromLeaves<Cat>(resultShape);
  return coalesce(make_layout(std::move(result), compact_col_major(flatA)));
}

template <class Cat> Layout<Cat> nullspace(const Layout<Cat> &layout) {
  auto flat = flatten(layout);
  auto compact = compact_col_major(flat.shape());
  llvm::SmallVector<IntTuple<Cat>, 8> shapes;
  llvm::SmallVector<IntTuple<Cat>, 8> strides;
  auto shapeIt = flat.shape().begin();
  auto strideIt = flat.stride().begin();
  auto compactIt = compact.asRef().begin();
  for (auto end = flat.shape().end(); shapeIt != end; ++shapeIt, ++strideIt, ++compactIt) {
    if ((*strideIt).isZero()) {
      shapes.push_back(*shapeIt);
      strides.push_back(*compactIt);
    }
  }
  if (shapes.empty())
    return make_layout(IntTuple<Cat>::getOne(), IntTuple<Cat>::getZero());
  return make_layout(detail::tupleFromLeaves<Cat>(shapes), detail::tupleFromLeaves<Cat>(strides));
}

template <class Cat> Layout<Cat> zip(const Layout<Cat> &layout) {
  return make_layout(zip(layout.shape()), zip(layout.stride()));
}

template <class Cat> Layout<Cat> zip(const Layout<Cat> &layoutA, const Layout<Cat> &layoutB) {
  return make_layout(zip(llvm::ArrayRef<IntTupleRef<Cat>>{layoutA.shape(), layoutB.shape()}),
                     zip(llvm::ArrayRef<IntTupleRef<Cat>>{layoutA.stride(), layoutB.stride()}));
}

template <class Tiler, class Cat = category_of<Tiler>>
Layout<Cat> tile_unzip(const Layout<Cat> &layout, const Tiler &tiler) {
  return make_layout(zip2_by(layout.shape(), tiler), zip2_by(layout.stride(), tiler));
}

template <class Cat> Layout<Cat> tile_unzip(const Layout<Cat> &layout, const Layout<Cat> &) {
  return layout;
}

template <class Cat> Layout<Cat> tile_unzip(const Layout<Cat> &layout, const Tile<Cat> &tiler) {
  if (tiler.isLeaf())
    return layout;
  return make_layout(zip2_by(layout.shape(), tiler), zip2_by(layout.stride(), tiler));
}

template <class Cat>
Layout<Cat> logical_divide(const Layout<Cat> &layout, const Layout<Cat> &tiler) {
  if (auto error = detail::firstError(layout, tiler))
    return Layout<Cat>::getError(error->withFrame(AlgebraOp::LogicalDivide));
  auto result = composition(layout, make_layout(tiler, complement(tiler, shape(coalesce(layout)))));
  return detail::tagError(std::move(result), AlgebraOp::LogicalDivide);
}

template <class Cat> Layout<Cat> logical_divide(const Layout<Cat> &layout, const Tile<Cat> &tiler) {
  if (auto error = detail::firstError(layout, tiler))
    return Layout<Cat>::getError(error->withFrame(AlgebraOp::LogicalDivide));
  if (!tiler.isLeaf()) {
    if (tiler.rank() > layout.rank())
      return Layout<Cat>::getError(
          ErrorInfo(ErrorCode::TilerRankExceeded).withFrame(AlgebraOp::LogicalDivide));
    return detail::tagError(
        transform_layout(layout, tiler,
                         [](const auto &l, const auto &t) { return logical_divide(l, t); }),
        AlgebraOp::LogicalDivide);
  } else if (tiler.isNone())
    return layout;
  else if (tiler.isScalar())
    return logical_divide(layout, make_layout(IntTuple<Cat>::fromLeaf(tiler.getScalar())));
  else
    return logical_divide(layout, tiler.getLayout());
}

template <class Tiler, class Cat = category_of<Tiler>,
          std::enable_if_t<is_int_tuple_v<Tiler>, int> = 0>
Layout<Cat> logical_divide(const Layout<Cat> &layout, const Tiler &tiler) {
  return logical_divide(layout, shape_to_tile(tiler));
}

template <class Tiler, class Cat = category_of<Tiler>,
          std::enable_if_t<!is_int_tuple_v<Tiler>, int> = 0>
Layout<Cat> zipped_divide(const Layout<Cat> &layout, const Tiler &tiler) {
  return detail::tagError(tile_unzip(logical_divide(layout, tiler), tiler),
                          AlgebraOp::ZippedDivide);
}

template <class Tiler, class Cat = category_of<Tiler>,
          std::enable_if_t<is_int_tuple_v<Tiler>, int> = 0>
Layout<Cat> zipped_divide(const Layout<Cat> &layout, const Tiler &tiler) {
  return zipped_divide(layout, shape_to_tile(tiler));
}

template <class Tiler, class Cat, std::enable_if_t<!is_int_tuple_v<Tiler>, int>>
Layout<Cat> tiled_divide(const Layout<Cat> &layout, const Tiler &tiler) {
  auto result = zipped_divide(layout, tiler);
  if (result.isError())
    return detail::tagError(result, AlgebraOp::TiledDivide);
  auto R1 = rank<1>(result);
  return slice(make_tuple(IntTuple<Cat>::getNone(), repeat(IntTuple<Cat>::getNone(), R1)), result);
}

template <class Tiler, class Cat = category_of<Tiler>,
          std::enable_if_t<is_int_tuple_v<Tiler>, int> = 0>
Layout<Cat> tiled_divide(const Layout<Cat> &layout, const Tiler &tiler) {
  return tiled_divide(layout, shape_to_tile(tiler));
}

template <class Tiler, class Cat = category_of<Tiler>,
          std::enable_if_t<!is_int_tuple_v<Tiler>, int> = 0>
Layout<Cat> flat_divide(const Layout<Cat> &layout, const Tiler &tiler) {
  auto result = zipped_divide(layout, tiler);
  if (result.isError())
    return detail::tagError(result, AlgebraOp::FlatDivide);
  auto R0 = rank<0>(result);
  auto R1 = rank<1>(result);
  return slice(
      make_tuple(repeat(IntTuple<Cat>::getNone(), R0), repeat(IntTuple<Cat>::getNone(), R1)),
      result);
}

template <class Tiler, class Cat = category_of<Tiler>,
          std::enable_if_t<is_int_tuple_v<Tiler>, int> = 0>
Layout<Cat> flat_divide(const Layout<Cat> &layout, const Tiler &tiler) {
  return flat_divide(layout, shape_to_tile(tiler));
}

template <class Cat>
Layout<Cat> logical_product(const Layout<Cat> &block, const Layout<Cat> &tiler) {
  if (auto error = detail::firstError(block, tiler))
    return Layout<Cat>::getError(error->withFrame(AlgebraOp::LogicalProduct));
  auto rest = composition(complement(block), tiler);
  if (rest.isError())
    return detail::tagError(rest, AlgebraOp::LogicalProduct);
  return make_layout(block, rest);
}

template <class Cat> Layout<Cat> logical_product(const Layout<Cat> &block, const Tile<Cat> &tiler) {
  if (auto error = detail::firstError(block, tiler))
    return Layout<Cat>::getError(error->withFrame(AlgebraOp::LogicalProduct));
  if (!tiler.isLeaf()) {
    if (tiler.rank() > block.rank())
      return Layout<Cat>::getError(
          ErrorInfo(ErrorCode::TilerRankExceeded).withFrame(AlgebraOp::LogicalProduct));
    return detail::tagError(
        transform_layout(block, tiler,
                         [](const auto &b, const auto &t) { return logical_product(b, t); }),
        AlgebraOp::LogicalProduct);
  } else if (tiler.isNone())
    return block;
  else if (tiler.isScalar())
    return logical_product(block, make_layout(IntTuple<Cat>::fromLeaf(tiler.getScalar())));
  else
    return logical_product(block, tiler.getLayout());
}

template <class Tiler, class Cat = category_of<Tiler>,
          std::enable_if_t<is_int_tuple_v<Tiler>, int> = 0>
Layout<Cat> logical_product(const Layout<Cat> &block, const Tiler &tiler) {
  return logical_product(block, shape_to_tile(tiler));
}

template <class Tiler, class Cat, std::enable_if_t<!is_int_tuple_v<Tiler>, int> = 0>
Layout<Cat> zipped_product(const Layout<Cat> &block, const Tiler &tiler) {
  return detail::tagError(tile_unzip(logical_product(block, tiler), tiler),
                          AlgebraOp::ZippedProduct);
}

template <class Tiler, class Cat = category_of<Tiler>,
          std::enable_if_t<is_int_tuple_v<Tiler>, int> = 0>
Layout<Cat> zipped_product(const Layout<Cat> &block, const Tiler &tiler) {
  return zipped_product(block, shape_to_tile(tiler));
}

template <class Tiler, class Cat, std::enable_if_t<!is_int_tuple_v<Tiler>, int>>
Layout<Cat> tiled_product(const Layout<Cat> &block, const Tiler &tiler) {
  auto result = zipped_product(block, tiler);
  if (result.isError())
    return detail::tagError(result, AlgebraOp::TiledProduct);
  return slice(
      make_tuple(IntTuple<Cat>::getNone(), repeat(IntTuple<Cat>::getNone(), rank<1>(result))),
      result);
}

template <class Tiler, class Cat = category_of<Tiler>,
          std::enable_if_t<is_int_tuple_v<Tiler>, int> = 0>
Layout<Cat> tiled_product(const Layout<Cat> &block, const Tiler &tiler) {
  return tiled_product(block, shape_to_tile(tiler));
}

template <class Tiler, class Cat, std::enable_if_t<!is_int_tuple_v<Tiler>, int> = 0>
Layout<Cat> flat_product(const Layout<Cat> &block, const Tiler &tiler) {
  auto result = zipped_product(block, tiler);
  if (result.isError())
    return detail::tagError(result, AlgebraOp::FlatProduct);
  return slice(make_tuple(repeat(IntTuple<Cat>::getNone(), rank<0>(result)),
                          repeat(IntTuple<Cat>::getNone(), rank<1>(result))),
               result);
}

template <class Tiler, class Cat = category_of<Tiler>,
          std::enable_if_t<is_int_tuple_v<Tiler>, int> = 0>
Layout<Cat> flat_product(const Layout<Cat> &block, const Tiler &tiler) {
  return flat_product(block, shape_to_tile(tiler));
}

template <class Cat>
Layout<Cat> blocked_product(const Layout<Cat> &block, const Layout<Cat> &tiler) {
  if (auto error = detail::firstError(block, tiler))
    return Layout<Cat>::getError(error->withFrame(AlgebraOp::BlockedProduct));
  auto R = std::max(block.rank(), tiler.rank());
  auto result = logical_product(append(block, R), append(tiler, R));
  if (result.isError())
    return detail::tagError(result, AlgebraOp::BlockedProduct);
  return detail::tagError(zip(get(result, 0), get(result, 1)), AlgebraOp::BlockedProduct);
}

template <class Cat> Layout<Cat> raked_product(const Layout<Cat> &block, const Layout<Cat> &tiler) {
  if (auto error = detail::firstError(block, tiler))
    return Layout<Cat>::getError(error->withFrame(AlgebraOp::RakedProduct));
  auto R = std::max(block.rank(), tiler.rank());
  auto result = logical_product(append(block, R), append(tiler, R));
  if (result.isError())
    return detail::tagError(result, AlgebraOp::RakedProduct);
  return detail::tagError(zip(get(result, 1), get(result, 0)), AlgebraOp::RakedProduct);
}

template <class TargetShape, class Order = LayoutLeft, class Cat = category_of<TargetShape>>
Layout<Cat> tile_to_shape(const Layout<Cat> &block, const TargetShape &targetShape,
                          const Order &order) {
  auto padded_block = append(block, targetShape.rank());
  if (padded_block.isError())
    return detail::tagError(padded_block, AlgebraOp::TileToShape);

  auto block_shape = product_each(shape(padded_block));
  auto target_shape = product_each(shape(targetShape));

  auto targetIt = target_shape.asRef().begin();
  for (auto tile : block_shape.asRef()) {
    auto target = *targetIt++;
    if (target.isSInt() && tile.isSInt() && !tile.isZero() &&
        target.staticValue() % tile.staticValue() != 0)
      return Layout<Cat>::getError(
          ErrorInfo(ErrorCode::IncompatibleShapeDivision).withFrame(AlgebraOp::TileToShape));
  }

  auto product_shape = ceil_div(target_shape, block_shape);
  return detail::tagError(blocked_product(padded_block, make_ordered_layout(product_shape, order)),
                          AlgebraOp::TileToShape);
}

template <class Target, class Cat = category_of<Target>>
IntTuple<Cat> ceil_div(const Target &target, const Layout<Cat> &tiler) {
  return shape(complement(tiler, shape(target)));
}

namespace detail {
template <class Cat>
Layout<Cat> upcastImpl(IntTupleRef<Cat> shape, IntTupleRef<Cat> stride, int32_t n) {
  if (!shape.isLeaf()) {
    llvm::SmallVector<IntTuple<Cat>, 8> shapes;
    llvm::SmallVector<IntTuple<Cat>, 8> strides;
    auto is = shape.begin();
    auto id = stride.begin();
    for (auto end = shape.end(); is != end; ++is, ++id) {
      auto child = upcastImpl(*is, *id, n);
      if (child.isError())
        return child;
      shapes.emplace_back(child.shape());
      strides.emplace_back(child.stride());
    }
    return make_layout(make_tuple(shapes), make_tuple(strides));
  } else {
    if (stride.isZero())
      return make_layout(shape, stride);
    auto factor = IntTuple<Cat>::getSInt(n);
    if (stride.isSInt()) {
      if (stride.staticValue() % n != 0 && n % stride.staticValue() != 0)
        return Layout<Cat>::getError(
            ErrorInfo(ErrorCode::InvalidRecastFactor).withFrame(AlgebraOp::Upcast));
      auto absStride = leaf_abs(stride);
      auto sign = leaf_signum(stride);
      return make_layout(leaf_ceil_div(shape, leaf_ceil_div(factor, absStride)),
                         sign * leaf_ceil_div(absStride, factor));
    }
    return make_layout(shape, leaf_safe_div(stride, factor));
  }
}

template <class Cat>
Layout<Cat> downcastImpl(IntTupleRef<Cat> shape, IntTupleRef<Cat> stride, int32_t n) {
  if (!shape.isLeaf()) {
    llvm::SmallVector<IntTuple<Cat>, 8> shapes;
    llvm::SmallVector<IntTuple<Cat>, 8> strides;
    auto is = shape.begin();
    auto id = stride.begin();
    for (auto end = shape.end(); is != end; ++is, ++id) {
      auto child = downcastImpl(*is, *id, n);
      shapes.emplace_back(child.shape());
      strides.emplace_back(child.stride());
    }
    return make_layout(make_tuple(shapes), make_tuple(strides));
  } else {
    auto factor = IntTuple<Cat>::getSInt(n);
    if (stride.isOne() || stride.isSInt(-1))
      return make_layout(leaf_mul(shape, factor), stride);
    return make_layout(shape, leaf_mul(stride, factor));
  }
}

} // namespace detail

template <class Cat> Layout<Cat> upcast(int32_t n, const Layout<Cat> &layout) {
  if (n <= 0)
    return Layout<Cat>::getError(
        ErrorInfo(ErrorCode::InvalidRecastFactor).withFrame(AlgebraOp::Upcast));
  if (has_basis_stride(layout))
    return Layout<Cat>::getError(
        ErrorInfo(ErrorCode::UnsupportedBasisOperation).withFrame(AlgebraOp::Upcast));
  return detail::upcastImpl(layout.shape(), layout.stride(), n);
}

template <class Cat> Layout<Cat> downcast(int32_t n, const Layout<Cat> &layout) {
  if (n <= 0)
    return Layout<Cat>::getError(
        ErrorInfo(ErrorCode::InvalidRecastFactor).withFrame(AlgebraOp::Downcast));
  if (has_basis_stride(layout))
    return Layout<Cat>::getError(
        ErrorInfo(ErrorCode::UnsupportedBasisOperation).withFrame(AlgebraOp::Downcast));
  return detail::downcastImpl(layout.shape(), layout.stride(), n);
}

template <class Cat>
Layout<Cat> recast_layout(int32_t newTypeBits, int32_t oldTypeBits, const Layout<Cat> &layout) {
  if (oldTypeBits <= 0 || newTypeBits <= 0)
    return Layout<Cat>::getError(
        ErrorInfo(ErrorCode::InvalidRecastFactor).withFrame(AlgebraOp::RecastLayout));
  auto divisor = std::gcd(oldTypeBits, newTypeBits);
  auto numerator = newTypeBits / divisor;
  auto denominator = oldTypeBits / divisor;

  if (numerator == 1 && denominator == 1)
    return layout;
  if (numerator == 1)
    return downcast(denominator, layout);
  if (denominator == 1)
    return upcast(numerator, layout);
  return downcast(denominator, upcast(numerator, layout));
}

template <class Cat> auto max_alignment(const Layout<Cat> &layout) {
  if (has_basis_stride(layout))
    return IntTuple<Cat>::getOne();

  auto flat_layout = coalesce(layout);
  auto static_shape = transform(flat_layout.shape(), [](IntTupleRef<Cat> s) {
    return s.isStatic() ? s : IntTuple<Cat>::getOne();
  });
  auto static_stride = transform(flat_layout.stride(), [](IntTupleRef<Cat> d) {
    return d.isStatic() ? d : IntTuple<Cat>::getZero();
  });
  auto filter_layout = make_layout(std::move(static_shape), std::move(static_stride));
  auto permuted = logical_divide(filter_layout, right_inverse(filter_layout));
  return gcd(size<0>(permuted), stride<1>(permuted));
}

///
/// Printer
///

template <class Cat> void print(const Layout<Cat> &layout, llvm::raw_ostream &os) {
  print(layout.shape(), os);
  os << ':';
  print(layout.stride(), os);
}

template <class Cat> std::optional<Layout<Cat>> Layout<Cat>::fromString(llvm::StringRef text) {
  auto colon = detail::findTopLevel(text, ":");
  if (colon == llvm::StringRef::npos)
    return std::nullopt;
  auto shape = IntTuple::fromString(text.take_front(colon));
  auto stride = IntTuple::fromString(text.drop_front(colon + 1));
  if (!shape || !stride || !congruent(*shape, *stride))
    return std::nullopt;
  return Layout(std::move(*shape), std::move(*stride));
}

template <class Cat> void print(const Tile<Cat> &tile, llvm::raw_ostream &os) {
  if (tile.isNone()) {
    os << '_';
    return;
  }
  if (tile.isScalar()) {
    print(tile.getScalar(), os);
    return;
  }
  if (tile.isLayout()) {
    print(tile.getLayout(), os);
    return;
  }
  os << '[';
  llvm::interleave(tile, os, [&](const Tile<Cat> &child) { print(child, os); }, "|");
  os << ']';
}

template <class Cat> std::optional<Tile<Cat>> Tile<Cat>::fromString(llvm::StringRef text) {
  if (text == "_")
    return getNone();
  if (!text.starts_with('[')) {
    auto layout = Layout<Cat>::fromString(text);
    if (layout)
      return Tile(*layout);
    auto scalar = IntTuple<Cat>::fromString(text);
    if (scalar && scalar->isLeaf() && !scalar->isBasis() && !scalar->isNone())
      return Tile(scalar->getLeaf());
    return std::nullopt;
  }
  auto close = detail::findMatching(text, 0, '[', ']');
  if (close != text.size() - 1)
    return std::nullopt;
  auto body = text.slice(1, close);
  llvm::SmallVector<Tile, 8> children;
  if (!body.empty()) {
    while (true) {
      auto separator = detail::findTopLevel(body, "|");
      auto item = separator == llvm::StringRef::npos ? body : body.take_front(separator);
      auto child = Tile::fromString(item);
      if (!child)
        return std::nullopt;
      children.push_back(std::move(*child));
      if (separator == llvm::StringRef::npos)
        break;
      body = body.drop_front(separator + 1);
    }
  }
  return getTuple(children);
}

} // namespace mlir::fly::core

#endif // FLYDSL_CORE_ALGEBRA_LAYOUT_HPP
