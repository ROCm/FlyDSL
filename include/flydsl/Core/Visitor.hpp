// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_CORE_VISITOR_HPP
#define FLYDSL_CORE_VISITOR_HPP

#include "flydsl/Core/Algebra.hpp"

namespace mlir::fly::core {
namespace detail {

template <class T, class U> struct dependent_on {
  using type = T;
};

template <class T> using WalkParts = llvm::SmallVector<T, 8>;

template <class T, class To> struct rebind_cat;

template <template <class> class S, class From, class To> struct rebind_cat<S<From>, To> {
  using type = S<To>;
};

/// The single structural traversal of Core values. Only the leaf and pointer
/// callbacks know about the evaluation category.
/// Traversal order is explicit:
///   tuple: preorder;
///   layout: shape, stride;
///   composed_layout: outer, offset, inner;
///   tensor: engine, layout;
///   tile: preorder.
///   Opaque descriptors are copied.
template <class Derived> struct ValueWalker {
  template <class From> auto operator()(Leaf<From> value) { return self().leaf(value); }

  template <class From> auto operator()(const Pointer<From> &value) {
    return self().pointer(value);
  }

  template <class T, std::enable_if_t<is_int_tuple_v<T>, int> = 0> auto operator()(const T &value) {
    auto ref = value.asRef();
    auto leaves = WalkParts<decltype(self().leaf(std::declval<Leaf<category_of<T>>>()))>{};
    for (const auto &item : ref.leaves())
      leaves.push_back(self().leaf(item));
    return self().build(ref, std::move(leaves));
  }

  template <class From> auto operator()(const Layout<From> &value) {
    auto shape = (*this)(value.shape());
    auto stride = (*this)(value.stride());
    return self().build(value, std::move(shape), std::move(stride));
  }

  template <class From> auto operator()(Swizzle<From> value) { return self().build(value); }

  template <class From> auto operator()(const CoordSwizzle<From> &value) {
    return self().build(value);
  }

  template <class From> auto operator()(const ComposedLayout<From> &value) {
    auto outer = (*this)(value.outer());
    auto offset = (*this)(value.offset());
    auto inner = (*this)(value.inner());
    return self().build(value, std::move(outer), std::move(offset), std::move(inner));
  }

  /// AnyLayout and ComposedInner. Their first alternative is a plain layout,
  /// so it names the result type without deducing through composed layouts,
  /// which contain these variants again.
  template <class T, class... Ts>
  auto operator()(const std::variant<T, Ts...> &value)
      -> decltype(std::declval<typename dependent_on<Derived, T>::type &>().build(
          value, std::declval<typename dependent_on<ValueWalker, T>::type &>()(
                     std::declval<const T &>()))) {
    using Result = decltype(self().build(value, (*this)(std::declval<const T &>())));
    return std::visit([&](const auto &v) -> Result { return self().build(value, (*this)(v)); },
                      value);
  }

  template <class From> auto operator()(const Tile<From> &value) {
    if (value.isNone())
      return self().build(value);
    if (value.isLayout())
      return self().build(value, (*this)(value.getLayout()));
    if (value.isScalar())
      return self().build(value, (*this)(value.getScalar()));
    auto children = WalkParts<decltype(self().build(value))>{};
    for (auto child : value)
      children.push_back((*this)(child));
    return self().build(value, std::move(children));
  }

  template <class From, class Engine> auto operator()(const Tensor<From, Engine> &value) {
    auto engine = (*this)(value.data());
    auto layout = (*this)(value.layout());
    return self().build(value, std::move(engine), std::move(layout));
  }

  template <class From> auto operator()(const CopyAtom<From> &value) {
    auto thr = (*this)(value.thrLayout());
    auto src = (*this)(value.thrBitLayoutSrc());
    auto dst = (*this)(value.thrBitLayoutDst());
    auto ref = (*this)(value.thrBitLayoutRef());
    return self().build(value, std::move(thr), std::move(src), std::move(dst), std::move(ref));
  }

  template <class From> auto operator()(const TiledCopy<From> &value) {
    auto atom = (*this)(value.copyAtom());
    auto layout = (*this)(value.layoutCopyTV());
    auto tiler = (*this)(value.tilerMN());
    return self().build(value, std::move(atom), std::move(layout), std::move(tiler));
  }

  template <class From> auto operator()(const MMAAtom<From> &value) {
    auto thr = (*this)(value.thrLayout());
    auto shape = (*this)(value.shapeMNK());
    auto a = (*this)(value.thrValLayoutA());
    auto b = (*this)(value.thrValLayoutB());
    auto c = (*this)(value.thrValLayoutC());
    return self().build(value, std::move(thr), std::move(shape), std::move(a), std::move(b),
                        std::move(c));
  }

  template <class From> auto operator()(const TiledMMA<From> &value) {
    auto atom = (*this)(value.mmaAtom());
    auto layout = (*this)(value.atomLayoutMNK());
    auto permutation = (*this)(value.permutationMNK());
    return self().build(value, std::move(atom), std::move(layout), std::move(permutation));
  }

  template <class From> auto operator()(const ThrCopy<From> &value) {
    auto tiled = (*this)(value.tiledCopy());
    auto index = (*this)(value.thr_idx());
    return self().build(value, std::move(tiled), std::move(index));
  }

  template <class From> auto operator()(const ThrMMA<From> &value) {
    auto tiled = (*this)(value.tiledMMA());
    auto index = (*this)(value.thr_vmnk());
    return self().build(value, std::move(tiled), std::move(index));
  }

private:
  Derived &self() { return static_cast<Derived &>(*this); }
};

} // namespace detail

template <class LeafFn, class PointerFn>
struct ValueVisitor : detail::ValueWalker<ValueVisitor<LeafFn, PointerFn>> {
  struct Visited {};

  ValueVisitor(LeafFn leafFn, PointerFn pointerFn)
      : leafFn(std::move(leafFn)), pointerFn(std::move(pointerFn)) {}

  template <class From> Visited leaf(Leaf<From> value) {
    leafFn(value);
    return {};
  }
  template <class From> Visited pointer(const Pointer<From> &value) {
    pointerFn(value);
    return {};
  }

  template <class... Parts> Visited build(const Parts &...) { return {}; }

private:
  LeafFn leafFn;
  PointerFn pointerFn;
};

/// Rebuilds the walked structure in category `To`: `leafFn` and `pointerFn` map the leaves and
/// pointers, everything else keeps its shape and opaque metadata.
template <class To, class LeafFn, class PointerFn>
struct ValueTransformer : detail::ValueWalker<ValueTransformer<To, LeafFn, PointerFn>> {
  template <class T> using Parts = detail::WalkParts<T>;

  ValueTransformer(LeafFn leafFn, PointerFn pointerFn)
      : leafFn(std::move(leafFn)), pointerFn(std::move(pointerFn)) {}

  template <class From> Leaf<To> leaf(Leaf<From> value) { return leafFn(value); }
  template <class From> Pointer<To> pointer(const Pointer<From> &value) { return pointerFn(value); }

  template <class From> IntTuple<To> build(IntTupleRef<From> value, Parts<Leaf<To>> leaves) {
    return IntTuple<To>::fromParts(value.nodes(), leaves);
  }

  template <class From>
  Layout<To> build(const Layout<From> &, IntTuple<To> shape, IntTuple<To> stride) {
    return Layout<To>(std::move(shape), std::move(stride));
  }

  template <class From> Swizzle<To> build(Swizzle<From> v) {
    return Swizzle<To>(v.mask(), v.base(), v.shift());
  }

  template <class From> CoordSwizzle<To> build(const CoordSwizzle<From> &v) {
    return CoordSwizzle<To>(v.mask(), v.rowBase(), v.rowModes(), v.colBase(), v.colModes());
  }

  template <class From>
  ComposedLayout<To> build(const ComposedLayout<From> &, ComposedOuter<To> outer,
                           IntTuple<To> offset, ComposedInner<To> inner) {
    return ComposedLayout<To>(std::move(inner), std::move(offset), std::move(outer));
  }

  template <class... Ts, class T> auto build(const std::variant<Ts...> &, T value) {
    return std::variant<typename detail::rebind_cat<Ts, To>::type...>(std::move(value));
  }

  template <class From> Tile<To> build(const Tile<From> &) { return Tile<To>::getNone(); }
  template <class From> Tile<To> build(const Tile<From> &, Layout<To> layout) {
    return Tile<To>(std::move(layout));
  }
  template <class From> Tile<To> build(const Tile<From> &, Leaf<To> scalar) {
    return Tile<To>(scalar);
  }
  template <class From> Tile<To> build(const Tile<From> &, Parts<Tile<To>> children) {
    return Tile<To>::getTuple(children);
  }

  template <class From, class Engine, class E>
  Tensor<To, E> build(const Tensor<From, Engine> &, E engine, AnyLayout<To> layout) {
    return Tensor<To, E>(std::move(engine), std::move(layout));
  }

  template <class From, class Thr, class Src, class Dst, class Ref>
  CopyAtom<To> build(const CopyAtom<From> &v, Thr thr, Src src, Dst dst, Ref ref) {
    return CopyAtom<To>(v.copyOperation(), v.copyInternalType(), v.valueBitWidth(), std::move(thr),
                        std::move(src), std::move(dst), std::move(ref));
  }

  template <class From, class Atom, class LayoutTV, class Tiler>
  TiledCopy<To> build(const TiledCopy<From> &, Atom atom, LayoutTV layout, Tiler tiler) {
    return TiledCopy<To>(std::move(atom), std::move(layout), std::move(tiler));
  }

  template <class From, class Thr, class Shape, class A, class B, class C>
  MMAAtom<To> build(const MMAAtom<From> &v, Thr thr, Shape shape, A a, B b, C c) {
    return MMAAtom<To>(v.mmaOperation(), std::move(thr), std::move(shape), v.valueTypes(),
                       std::move(a), std::move(b), std::move(c));
  }

  template <class From, class Atom, class AtomLayout, class Permutation>
  TiledMMA<To> build(const TiledMMA<From> &, Atom atom, AtomLayout layout,
                     Permutation permutation) {
    return TiledMMA<To>(std::move(atom), std::move(layout), std::move(permutation));
  }

  template <class From, class Tiled, class Index>
  ThrCopy<To> build(const ThrCopy<From> &, Tiled tiled, Index index) {
    return ThrCopy<To>(std::move(tiled), std::move(index));
  }

  template <class From, class Tiled, class Index>
  ThrMMA<To> build(const ThrMMA<From> &, Tiled tiled, Index index) {
    return ThrMMA<To>(std::move(tiled), std::move(index));
  }

private:
  LeafFn leafFn;
  PointerFn pointerFn;
};

/// Reduces the walked structure bottom-up to an `R`: `leafFn` maps each leaf
/// and `combineFn` folds a node's component results, in traversal order, into
/// the node's result. Tuple nesting is transparent: an int tuple combines its
/// leaves, a tile its layouts and scalars in preorder.
/// Pointers have no generic result.
template <class R, class LeafFn, class CombineFn>
struct ValueReducer : detail::ValueWalker<ValueReducer<R, LeafFn, CombineFn>> {
  /// The results of a tile's elements, combined by the node holding the tile.
  using Fields = detail::WalkParts<R>;

  ValueReducer(LeafFn leafFn, CombineFn combineFn)
      : leafFn(std::move(leafFn)), combineFn(std::move(combineFn)) {}

  template <class From> R leaf(Leaf<From> value) { return leafFn(value); }

  template <class From> R build(IntTupleRef<From>, Fields leaves) { return collapse(leaves); }

  template <class... Ts> R build(const std::variant<Ts...> &, R value) { return value; }

  template <class From> Fields build(const Tile<From> &) { return {}; }
  template <class From> Fields build(const Tile<From> &, R element) { return {element}; }
  template <class From> Fields build(const Tile<From> &, detail::WalkParts<Fields> children) {
    auto fields = Fields{};
    for (const auto &child : children)
      fields.append(child.begin(), child.end());
    return fields;
  }

  template <class Node, class... Parts> R build(const Node &, Parts... parts) {
    auto items = Fields{collapse(std::move(parts))...};
    return collapse(items);
  }

  R collapse(R value) { return value; }
  R collapse(const Fields &fields) { return combineFn(llvm::ArrayRef<R>(fields)); }

private:
  LeafFn leafFn;
  CombineFn combineFn;
};

template <class T, class LeafFn, class PointerFn>
void visit_value(const T &value, LeafFn leafFn, PointerFn pointerFn) {
  ValueVisitor<LeafFn, PointerFn>(std::move(leafFn), std::move(pointerFn))(value);
}

template <class To, class T, class LeafFn, class PointerFn>
auto transform_value(const T &value, LeafFn leafFn, PointerFn pointerFn) {
  return ValueTransformer<To, LeafFn, PointerFn>(std::move(leafFn), std::move(pointerFn))(value);
}

template <class R, class T, class LeafFn, class CombineFn>
R reduce_value(const T &value, LeafFn leafFn, CombineFn combineFn) {
  auto reducer = ValueReducer<R, LeafFn, CombineFn>(std::move(leafFn), std::move(combineFn));
  return reducer.collapse(reducer(value));
}

} // namespace mlir::fly::core

#endif // FLYDSL_CORE_VISITOR_HPP
