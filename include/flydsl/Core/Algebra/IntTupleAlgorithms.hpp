// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_CORE_ALGEBRA_INTTUPLEALGORITHMS_HPP
#define FLYDSL_CORE_ALGEBRA_INTTUPLEALGORITHMS_HPP

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "flydsl/Core/Algebra/Config.hpp"
#include "flydsl/Core/Algebra/IntTuple.hpp"

namespace mlir::fly::core {

struct LayoutLeft {};
struct LayoutRight {};

//===----------------------------------------------------------------------===//
// Function Declarations
//===----------------------------------------------------------------------===//

template <class T, class F, class Cat = category_of<T>> IntTuple<Cat> transform(const T &t, F &&f);

template <class T0, class T1, class F, class Cat = category_of<T0>>
IntTuple<Cat> transform(const T0 &t0, const T1 &t1, F &&f);

template <class T0, class T1, class T2, class F, class Cat = category_of<T0>>
IntTuple<Cat> transform(const T0 &t0, const T1 &t1, const T2 &t2, F &&f);

template <class T, class F, class Cat = category_of<T>>
IntTuple<Cat> transform_leaf(const T &t, F &&f);

template <class T0, class T1, class F, class Cat = category_of<T0>>
IntTuple<Cat> transform_leaf(const T0 &t0, const T1 &t1, F &&f);

template <class T0, class T1, class T2, class F, class Cat = category_of<T0>>
IntTuple<Cat> transform_leaf(const T0 &t0, const T1 &t1, const T2 &t2, F &&f);

template <class T0, class T1, class F> void for_each(const T0 &t0, const T1 &t1, F &&f);

template <class T, class F> void for_each_leaf(const T &t, F &&f);

template <class T, class F> int32_t find_if(const T &t, F &&f);

template <class T, class F> bool any_of(const T &t, F &&f);

template <class T, class F> bool all_of(const T &t, F &&f);

template <class T, class F, class Cat = category_of<T>>
IntTuple<Cat> filter_tuple(const T &t, F &&f);

template <class T0, class T1, class F, class Cat = category_of<T0>>
IntTuple<Cat> filter_tuple(const T0 &t0, const T1 &t1, F &&f);

template <class T0, class T1, class T2, class F, class Cat = category_of<T0>>
IntTuple<Cat> filter_tuple(const T0 &t0, const T1 &t1, const T2 &t2, F &&f);

template <class T, class Cat = category_of<T>> IntTuple<Cat> front(const T &tuple);

template <class T, class Cat = category_of<T>> IntTuple<Cat> back(const T &tuple);

template <class T, class Cat = category_of<T>> IntTuple<Cat> flatten(const T &t);

template <class Flat, class Profile, class Cat = category_of<Flat>>
IntTuple<Cat> unflatten(const Flat &flatTuple, const Profile &targetProfile);

template <class T, class Cat = category_of<T>> IntTuple<Cat> wrap(const T &tuple);

template <class T, class Cat = category_of<T>> IntTuple<Cat> unwrap(const T &tuple);

template <class T, class Cat = category_of<T>>
IntTuple<Cat> take(const T &tuple, int32_t begin, int32_t end);

template <class T, class Cat = category_of<T>>
IntTuple<Cat> select(const T &tuple, llvm::ArrayRef<int32_t> indices);

template <class T, class Cat = category_of<T>>
IntTuple<Cat> group(const T &tuple, int32_t begin, int32_t end);

template <class A, class B, class Cat = category_of<A>>
IntTuple<Cat> append(const A &tuple, const B &sub);

template <class A, class B, class Cat = category_of<A>>
IntTuple<Cat> append(const A &tuple, const B &sub, int32_t n);

template <class A, class B, class Cat = category_of<A>>
IntTuple<Cat> prepend(const A &tuple, const B &sub);

template <class A, class B, class Cat = category_of<A>>
IntTuple<Cat> prepend(const A &tuple, const B &sub, int32_t n);

template <class A, class B, class Cat = category_of<A>>
IntTuple<Cat> replace(const A &tuple, int32_t i, const B &sub);

template <int N, class T, class X, class Cat = category_of<T>>
IntTuple<Cat> insert(const T &tuple, const X &sub);

template <int N, class T, class Cat = category_of<T>> IntTuple<Cat> remove(const T &tuple);

template <int N, class X, class Cat = category_of<X>> IntTuple<Cat> repeat(const X &value);

template <class X, class Cat = category_of<X>> IntTuple<Cat> repeat(const X &value, int N);

template <class T, class Cat = category_of<T>> IntTuple<Cat> sum(const T &t);

template <class T, class Cat = category_of<T>> IntTuple<Cat> gcd(const T &t);

template <class T0, class T1, class... Ts, class Cat = category_of<T0>>
IntTuple<Cat> gcd(const T0 &t0, const T1 &t1, const Ts &...ts);

template <class T, class Cat = category_of<T>> IntTuple<Cat> zip(const T &tuple);

template <class T, class TG, class Cat = category_of<T>>
IntTuple<Cat> zip2_by(const T &t, const TG &guide);

template <class T, class Cat = category_of<T>> IntTuple<Cat> reverse(const T &tuple);

template <class A, class B, class Cat = category_of<A>>
IntTuple<Cat> compatible(const A &a, const B &b);

template <int... Is, class T, class Cat = category_of<T>> int32_t rank(const T &tuple);

template <int... Is, class T, class Cat = category_of<T>> int32_t depth(const T &tuple);

template <int... Is, class T, class Cat = category_of<T>> IntTuple<Cat> shape(const T &tuple);

template <class T, class Cat = category_of<T>> IntTuple<Cat> product(const T &t);

template <int... Is, class T, class Cat = category_of<T>> IntTuple<Cat> size(const T &t);

template <class Tuple, class Guide, class Cat = category_of<Tuple>>
IntTuple<Cat> product_like(const Tuple &tuple, const Guide &guide);

template <class A, class B, class Cat = category_of<A>>
IntTuple<Cat> inner_product(const A &a, const B &b);

template <class A, class B, class Cat = category_of<A>> auto ceil_div(const A &a, const B &b);

template <class A, class B, class Cat = category_of<A>> auto shape_div(const A &a, const B &b);

template <class A, class B, class Cat = category_of<A>>
IntTuple<Cat> elem_scale(const A &a, const B &b);

template <class T, class V, class Cat = category_of<T>>
IntTuple<Cat> repeat_like(const T &guide, const V &value);

template <class A, class B, class Cat = category_of<A>>
IntTuple<Cat> filter_zeros(const A &mask, const B &value);

template <class Coord, class T, class Cat = category_of<T>>
IntTuple<Cat> slice(const Coord &coord, const T &value);

template <class Coord, class T, class Cat = category_of<T>>
IntTuple<Cat> dice(const Coord &coord, const T &value);

template <class T, class Current = IntTuple<category_of<T>>, class Cat = category_of<T>>
auto compact_col_major(const T &shape, const Current &current = IntTuple<Cat>::getOne());

template <class T, class Current = IntTuple<category_of<T>>, class Cat = category_of<T>>
auto compact_row_major(const T &shape, const Current &current = IntTuple<Cat>::getOne());

template <class Index, class Shape, class Cat = category_of<Shape>>
IntTuple<Cat> idx2crd(const Index &index, const Shape &shape);

template <class Coord, class Shape, class Cat = category_of<Shape>>
IntTuple<Cat> crd2idx(const Coord &coord, const Shape &shape);

template <class Coord, class SrcShape, class DstShape, class Cat = category_of<Coord>>
IntTuple<Cat> crd2crd(const Coord &coord, const SrcShape &srcShape, const DstShape &dstShape);

template <class Shape, class Order, class Cat = category_of<Shape>>
IntTuple<Cat> compact_order(const Shape &shape, const Order &order);

template <class T, class Cat = category_of<T>, std::enable_if_t<is_int_tuple_v<T>, int> = 0>
void print(const T &value, llvm::raw_ostream &os);

//===----------------------------------------------------------------------===//
// Functional
//===----------------------------------------------------------------------===//

// F :: IntTuple -> IntTuple
template <class T, class F, class Cat> IntTuple<Cat> transform(const T &t, F &&f) {
  auto tuple = t.asRef();
  if (tuple.isLeaf())
    return f(tuple);
  else
    return make_tuple(llvm::map_to_vector<8>(tuple, [&](auto c) { return f(c); }));
}

// F :: (IntTuple, IntTuple) -> IntTuple
template <class T0, class T1, class F, class Cat>
IntTuple<Cat> transform(const T0 &t0, const T1 &t1, F &&f) {
  auto tuple0 = t0.asRef();
  auto tuple1 = t1.asRef();

  if (tuple0.isLeaf())
    return f(tuple0, tuple1);
  else {
    if (tuple1.isLeaf() || tuple0.rank() != tuple1.rank())
      return detail::operandError<Cat>({tuple0, tuple1}, ErrorCode::IncompatibleProfile,
                                       AlgebraOp::Transform);
    auto children = llvm::SmallVector<IntTuple<Cat>, 8>{};
    auto child1 = tuple1.begin();
    for (auto child0 : tuple0) {
      children.push_back(f(child0, *child1));
      ++child1;
    }
    return make_tuple(children);
  }
}

// F :: (IntTuple, IntTuple, IntTuple) -> IntTuple
template <class T0, class T1, class T2, class F, class Cat>
IntTuple<Cat> transform(const T0 &t0, const T1 &t1, const T2 &t2, F &&f) {
  auto tuple0 = t0.asRef();
  auto tuple1 = t1.asRef();
  auto tuple2 = t2.asRef();

  if (tuple0.isLeaf())
    return f(tuple0, tuple1, tuple2);
  else {
    if (tuple1.isLeaf() || tuple2.isLeaf() || tuple0.rank() != tuple1.rank() ||
        tuple0.rank() != tuple2.rank())
      return detail::operandError<Cat>({tuple0, tuple1, tuple2}, ErrorCode::IncompatibleProfile,
                                       AlgebraOp::Transform);

    auto children = llvm::SmallVector<IntTuple<Cat>, 8>{};
    auto child1 = tuple1.begin();
    auto child2 = tuple2.begin();
    for (auto child0 : tuple0) {
      children.push_back(f(child0, *child1, *child2));
      ++child1;
      ++child2;
    }
    return make_tuple(children);
  }
}

// F :: IntTuple -> IntTuple
template <class T, class F, class Cat> IntTuple<Cat> transform_leaf(const T &t, F &&f) {
  auto tuple = t.asRef();

  if (tuple.isLeaf())
    return f(tuple);
  else
    return transform(tuple, [&](IntTupleRef<Cat> a) { return transform_leaf(a, f); });
}

// F :: (IntTuple, IntTuple) -> IntTuple
template <class T0, class T1, class F, class Cat>
IntTuple<Cat> transform_leaf(const T0 &t0, const T1 &t1, F &&f) {
  auto tuple0 = t0.asRef();
  auto tuple1 = t1.asRef();

  if (tuple0.isLeaf())
    return f(tuple0, tuple1);
  else
    return transform(tuple0, tuple1, [&](IntTupleRef<Cat> a, IntTupleRef<Cat> b) {
      return transform_leaf(a, b, f);
    });
}

// F :: (IntTuple, IntTuple, IntTuple) -> IntTuple
template <class T0, class T1, class T2, class F, class Cat>
IntTuple<Cat> transform_leaf(const T0 &t0, const T1 &t1, const T2 &t2, F &&f) {
  auto tuple0 = t0.asRef();
  auto tuple1 = t1.asRef();
  auto tuple2 = t2.asRef();

  if (tuple0.isLeaf())
    return f(tuple0, tuple1, tuple2);
  else
    return transform(tuple0, tuple1, tuple2,
                     [&](IntTupleRef<Cat> a, IntTupleRef<Cat> b, IntTupleRef<Cat> c) {
                       return transform_leaf(a, b, c, f);
                     });
}

// F :: IntTuple -> ()
template <class T, class F> void for_each(const T &t, F &&f) {
  auto tuple = t.asRef();
  for (auto child : tuple)
    f(child);
}

// F :: (IntTuple, IntTuple) -> ()
template <class T0, class T1, class F> void for_each(const T0 &t0, const T1 &t1, F &&f) {
  auto tuple0 = t0.asRef();
  auto tuple1 = t1.asRef();

  if (tuple0.isLeaf()) {
    f(tuple0, tuple1);
    return;
  }
  FLYDSL_CORE_ASSERT(!tuple1.isLeaf() && tuple0.rank() == tuple1.rank());
  for (auto [child0, child1] : llvm::zip(tuple0, tuple1))
    f(child0, child1);
}

// F :: IntTuple -> ()
template <class T, class F> void for_each_leaf(const T &t, F &&f) {
  auto tuple = t.asRef();
  if (tuple.isLeaf()) {
    f(tuple);
    return;
  }
  for (auto child : tuple)
    for_each_leaf(child, f);
}

// F :: IntTuple -> bool
template <class T, class F> int32_t find_if(const T &t, F &&f) {
  auto tuple = t.asRef();

  if (tuple.isLeaf())
    return f(tuple) ? 0 : 1;

  auto i = int32_t{0};
  for (auto child : tuple) {
    if (f(child))
      return i;
    ++i;
  }
  return tuple.rank();
}

template <class T, class X, class Cat = category_of<T>> int32_t find(const T &t, const X &target) {
  return find_if(t, [&](auto value) { return eq(value, target).isOne(); });
}

// F :: IntTuple -> bool
template <class T, class F> bool any_of(const T &t, F &&f) {
  auto tuple = t.asRef();
  if (tuple.isLeaf())
    return f(tuple);
  else {
    for (auto child : tuple)
      if (f(child))
        return true;
    return false;
  }
}

// F :: IntTuple -> bool
template <class T, class F> bool all_of(const T &t, F &&f) {
  auto tuple = t.asRef();
  if (tuple.isLeaf())
    return f(tuple);
  else {
    for (auto child : tuple)
      if (!f(child))
        return false;
    return true;
  }
}

// F :: IntTuple -> bool
template <class T, class F> bool none_of(const T &t, F &&f) { return !any_of(t, f); }

// F :: IntTuple -> IntTuple
template <class T, class F, class Cat> IntTuple<Cat> filter_tuple(const T &t, F &&f) {
  auto tuple = t.asRef();

  if (tuple.isLeaf())
    return f(tuple);
  else {
    auto results = llvm::SmallVector<IntTuple<Cat>, 8>{};
    for (auto child : tuple)
      results.emplace_back(f(child));
    return concat(results);
  }
}

// F :: (IntTuple, IntTuple) -> IntTuple
template <class T0, class T1, class F, class Cat>
IntTuple<Cat> filter_tuple(const T0 &t0, const T1 &t1, F &&f) {
  auto tuple0 = t0.asRef();
  auto tuple1 = t1.asRef();

  if (tuple0.isLeaf())
    return f(tuple0, tuple1);
  else {
    if (tuple1.isLeaf() || tuple0.rank() != tuple1.rank())
      return detail::operandError<Cat>({tuple0, tuple1}, ErrorCode::IncompatibleProfile,
                                       AlgebraOp::FilterTuple);
    auto results = llvm::SmallVector<IntTuple<Cat>, 8>{};
    auto child1 = tuple1.begin();
    for (auto child0 : tuple0) {
      results.emplace_back(f(child0, *child1));
      ++child1;
    }
    return concat(results);
  }
}

// F :: (IntTuple, IntTuple, IntTuple) -> IntTuple
template <class T0, class T1, class T2, class F, class Cat>
IntTuple<Cat> filter_tuple(const T0 &t0, const T1 &t1, const T2 &t2, F &&f) {
  auto tuple0 = t0.asRef();
  auto tuple1 = t1.asRef();
  auto tuple2 = t2.asRef();

  if (tuple0.isLeaf())
    return f(tuple0, tuple1, tuple2);
  else {
    if (tuple1.isLeaf() || tuple2.isLeaf() || tuple0.rank() != tuple1.rank() ||
        tuple0.rank() != tuple2.rank())
      return detail::operandError<Cat>({tuple0, tuple1, tuple2}, ErrorCode::IncompatibleProfile,
                                       AlgebraOp::FilterTuple);
    auto results = llvm::SmallVector<IntTuple<Cat>, 8>{};
    auto child1 = tuple1.begin();
    auto child2 = tuple2.begin();
    for (auto child0 : tuple0) {
      results.emplace_back(f(child0, *child1, *child2));
      ++child1;
      ++child2;
    }
    return concat(results);
  }
}

//===----------------------------------------------------------------------===//
// Structural operations
//===----------------------------------------------------------------------===//

template <class T, class Cat> IntTuple<Cat> front(const T &tuple) {
  auto result = tuple.asRef();
  while (!result.isLeaf()) {
    if (result.rank() == 0)
      return IntTuple<Cat>::getError(
          ErrorInfo(ErrorCode::IndexOutOfRange).withFrame(AlgebraOp::Front));
    result = result.at(0);
  }
  return result;
}

template <class T, class Cat> IntTuple<Cat> back(const T &tuple) {
  auto result = tuple.asRef();
  while (!result.isLeaf()) {
    // at(rank() - 1) would walk the span chain twice; one forward pass keeping
    // the last child does the same in O(rank).
    auto it = result.begin();
    auto end = result.end();
    if (it == end)
      return IntTuple<Cat>::getError(
          ErrorInfo(ErrorCode::IndexOutOfRange).withFrame(AlgebraOp::Back));
    auto last = *it;
    for (++it; it != end; ++it)
      last = *it;
    result = last;
  }
  return result;
}

template <class T, class Cat> IntTuple<Cat> flatten(const T &t) {
  auto tuple = t.asRef();
  if (tuple.isLeaf())
    return t;
  else if (tuple.isFlat()) {
    return t;
  } else {
    llvm::SmallVector<Node, 16> nodes;
    Node::make_flat(nodes, tuple.leafCount());
    return IntTuple<Cat>::fromParts(nodes, tuple.leaves());
  }
}

template <class Flat, class Profile, class Cat>
IntTuple<Cat> unflatten(const Flat &flatTuple, const Profile &targetProfile) {
  auto flat = flatTuple.asRef();
  auto profile = targetProfile.asRef();
  if (flat.isError() || profile.isError() || flat.leafCount() != profile.leafCount())
    return detail::operandError<Cat>({flat, profile}, ErrorCode::IncompatibleProfile,
                                     AlgebraOp::Unflatten);
  return IntTuple<Cat>::fromParts(profile.nodes(), flat.leaves());
}

template <class T, class Cat> IntTuple<Cat> wrap(const T &tuple) {
  auto t = tuple.asRef();
  if (!t.isLeaf())
    return t;
  else
    return make_tuple(t);
}

template <class T, class Cat> IntTuple<Cat> unwrap(const T &tuple) {
  auto t = tuple.asRef();
  // rank() would walk the whole span chain only to compare it against 1.
  // "Exactly one child" is the O(1) test that the first child is also the last.
  while (!t.isLeaf()) {
    auto first = t.begin();
    auto second = first;
    if (first == t.end() || ++second != t.end())
      break;
    t = *first;
  }
  return t;
}

template <class T, class Cat> IntTuple<Cat> take(const T &tuple, int32_t begin, int32_t end) {
  auto t = tuple.asRef();
  if (begin < 0 || (end != -1 && begin > end) || begin > t.rank() || (end != -1 && end > t.rank()))
    return detail::operandError<Cat>({t}, ErrorCode::IndexOutOfRange, AlgebraOp::Take);
  llvm::SmallVector<IntTupleRef<Cat>, 8> parts;
  auto i = int32_t{0};
  for (auto c : t) {
    if (i >= begin && (end == -1 || i < end))
      parts.push_back(c);
    ++i;
  }
  return make_tuple(parts);
}

template <int B, int E, class T, class Cat = category_of<T>> IntTuple<Cat> take(const T &tuple) {
  return take(tuple, B, E);
}

template <class T, class Cat>
IntTuple<Cat> select(const T &tuple, llvm::ArrayRef<int32_t> indices) {
  auto t = tuple.asRef();
  if (llvm::any_of(indices, [&](int32_t i) { return i < 0 || i >= t.rank(); }))
    return detail::operandError<Cat>({t}, ErrorCode::IndexOutOfRange, AlgebraOp::Select);
  return make_tuple(llvm::map_to_vector<8>(indices, [&](auto i) { return t.at(i); }));
}

template <int... I, class T, class Cat = category_of<T>> IntTuple<Cat> select(const T &tuple) {
  auto indices = std::array<int32_t, sizeof...(I)>{I...};
  return select(tuple, llvm::ArrayRef<int32_t>(indices));
}

template <class T, class Cat> IntTuple<Cat> group(const T &tuple, int32_t begin, int32_t end) {
  auto t = tuple.asRef();
  if (begin < 0 || (end != -1 && begin > end) || begin > t.rank() || (end != -1 && end > t.rank()))
    return detail::operandError<Cat>({t}, ErrorCode::IndexOutOfRange, AlgebraOp::Group);
  auto actualEnd = end == -1 ? t.rank() : end;
  if (begin == actualEnd)
    return t.isLeaf() ? wrap(t) : t;
  llvm::SmallVector<IntTupleRef<Cat>, 8> head, inner, tail;
  auto i = int32_t{0};
  for (auto c : t) {
    if (i < begin)
      head.push_back(c);
    else if (end == -1 || i < end)
      inner.push_back(c);
    else
      tail.push_back(c);
    ++i;
  }
  if (inner.empty())
    return t;
  auto mid = make_tuple(inner);
  llvm::SmallVector<IntTupleRef<Cat>, 8> parts;
  parts.append(head.begin(), head.end());
  parts.push_back(mid.asRef());
  parts.append(tail.begin(), tail.end());
  return make_tuple(parts);
}

template <int B, int E, class T, class Cat = category_of<T>> IntTuple<Cat> group(const T &tuple) {
  return group(tuple, B, E);
}

template <class A, class B, class Cat> IntTuple<Cat> append(const A &tuple, const B &sub) {
  auto t = tuple.asRef();
  llvm::SmallVector<IntTupleRef<Cat>, 8> parts;
  for (auto c : t)
    parts.push_back(c);
  parts.push_back(sub.asRef());
  return make_tuple(parts);
}

template <class A, class B, class Cat>
IntTuple<Cat> append(const A &tuple, const B &sub, int32_t n) {
  auto t = tuple.asRef();
  if (n < t.rank())
    return detail::operandError<Cat>({t}, ErrorCode::IndexOutOfRange, AlgebraOp::Append);
  llvm::SmallVector<IntTupleRef<Cat>, 8> parts;
  for (auto c : t)
    parts.push_back(c);
  auto extra = n - static_cast<int32_t>(parts.size());
  if (extra <= 0)
    return t;
  parts.append(extra, sub.asRef());
  return make_tuple(parts);
}

template <int N, class A, class B, class Cat = category_of<A>>
IntTuple<Cat> append(const A &tuple, const B &sub) {
  return append(tuple, sub, N);
}

template <class A, class B, class Cat> IntTuple<Cat> prepend(const A &tuple, const B &sub) {
  auto t = tuple.asRef();
  llvm::SmallVector<IntTupleRef<Cat>, 8> parts;
  parts.push_back(sub.asRef());
  for (auto c : t)
    parts.push_back(c);
  return make_tuple(parts);
}

template <class A, class B, class Cat>
IntTuple<Cat> prepend(const A &tuple, const B &sub, int32_t n) {
  auto t = tuple.asRef();
  if (n < t.rank())
    return detail::operandError<Cat>({t}, ErrorCode::IndexOutOfRange, AlgebraOp::Prepend);
  llvm::SmallVector<IntTupleRef<Cat>, 8> parts;
  for (auto c : t)
    parts.push_back(c);
  auto extra = n - static_cast<int32_t>(parts.size());
  if (extra <= 0)
    return t;
  parts.insert(parts.begin(), extra, sub.asRef());
  return make_tuple(parts);
}

template <int N, class A, class B, class Cat = category_of<A>>
IntTuple<Cat> prepend(const A &tuple, const B &sub) {
  return prepend(tuple, sub, N);
}

template <class A, class B, class Cat>
IntTuple<Cat> replace(const A &tuple, int32_t i, const B &sub) {
  auto t = tuple.asRef();
  auto s = sub.asRef();

  if (t.isLeaf() ? i != 0 : i < 0 || i >= t.rank())
    return detail::operandError<Cat>({t}, ErrorCode::IndexOutOfRange, AlgebraOp::Replace);
  if (t.isLeaf())
    return s;
  llvm::SmallVector<IntTupleRef<Cat>, 8> parts;
  auto idx = int32_t{0};
  for (auto c : t) {
    parts.push_back(idx == i ? s : c);
    ++idx;
  }
  return make_tuple(parts);
}

template <int N, class T, class X, class Cat = category_of<T>>
IntTuple<Cat> replace(const T &tuple, const X &sub) {
  return replace(tuple, N, sub);
}

template <int N, class T, class X, class Cat> IntTuple<Cat> insert(const T &tuple, const X &sub) {
  auto t = tuple.asRef();
  static_assert(N >= 0);
  if (t.isLeaf() || N > t.rank())
    return detail::operandError<Cat>({t}, ErrorCode::IndexOutOfRange, AlgebraOp::Insert);
  auto parts = llvm::SmallVector<IntTupleRef<Cat>, 8>{};
  auto i = int32_t{0};
  for (auto child : t) {
    if (i++ == N)
      parts.push_back(sub.asRef());
    parts.push_back(child);
  }
  if (N == t.rank())
    parts.push_back(sub.asRef());
  return make_tuple(parts);
}

template <int N, class T, class Cat> IntTuple<Cat> remove(const T &tuple) {
  auto t = tuple.asRef();
  static_assert(N >= 0);
  if (t.isLeaf() || N >= t.rank())
    return detail::operandError<Cat>({t}, ErrorCode::IndexOutOfRange, AlgebraOp::Remove);
  auto parts = llvm::SmallVector<IntTupleRef<Cat>, 8>{};
  auto i = int32_t{0};
  for (auto child : t)
    if (i++ != N)
      parts.push_back(child);
  return make_tuple(parts);
}

template <class T, class X, class Cat = category_of<T>>
IntTuple<Cat> replace_front(const T &tuple, const X &sub) {
  return replace<0>(tuple, sub);
}

template <class T, class X, class Cat = category_of<T>>
IntTuple<Cat> replace_back(const T &tuple, const X &sub) {
  auto t = tuple.asRef();
  return replace(tuple, t.rank() - 1, sub);
}

template <int N, class X, class Cat> IntTuple<Cat> repeat(const X &value) {
  static_assert(N >= 0);
  if constexpr (N == 1)
    return value.asRef();
  else {
    auto parts = llvm::SmallVector<IntTupleRef<Cat>, 8>(N, value.asRef());
    return make_tuple(parts);
  }
}

template <class X, class Cat> IntTuple<Cat> repeat(const X &value, int N) {
  if (N == 1)
    return value.asRef();
  else {
    auto parts = llvm::SmallVector<IntTupleRef<Cat>, 8>(N, value.asRef());
    return make_tuple(parts);
  }
}

template <class T, class Cat> IntTuple<Cat> zip(const T &tuple) {
  auto t = tuple.asRef();
  if (t.isLeaf())
    return t;
  if (t.empty())
    return t;

  auto first = t.at(0);
  if (first.isLeaf())
    return make_tuple(t);

  auto innerRank = first.isLeaf() ? int32_t{1} : first.rank();
  auto i = int32_t{0};
  for (auto child : t) {
    if ((child.isLeaf() ? int32_t{1} : child.rank()) != innerRank)
      return detail::operandError<Cat>({t}, ErrorCode::IncompatibleProfile, AlgebraOp::Zip, i);
    ++i;
  }
  llvm::SmallVector<IntTuple<Cat>, 8> columns;
  for (auto j = int32_t{0}; j < innerRank; ++j) {
    llvm::SmallVector<IntTupleRef<Cat>, 8> column;
    for (auto row : t)
      column.push_back(row.isLeaf() ? row : row.at(j));
    columns.push_back(make_tuple(column));
  }
  return make_tuple(columns);
}

template <class Cat> IntTuple<Cat> zip(llvm::ArrayRef<IntTupleRef<Cat>> ts) {
  return zip(make_tuple(ts));
}

template <class T0, class T1, class... Ts, class Cat = category_of<T0>>
IntTuple<Cat> zip(const T0 &t0, const T1 &t1, const Ts &...ts) {
  return zip(make_tuple(t0, t1, ts...));
}

template <class T, class TG, class Cat> IntTuple<Cat> zip2_by(const T &t, const TG &guide) {
  auto tuple = t.asRef();
  auto profile = guide.asRef();

  // A leaf of the guide marks a mode that must already be split in two.
  auto isSplit = [](IntTupleRef<Cat> mode) { return !mode.isLeaf() && mode.rank() == 2; };
  if (profile.isLeaf()) {
    if (!isSplit(tuple))
      return detail::operandError<Cat>({tuple}, ErrorCode::IncompatibleProfile, AlgebraOp::Zip2By);
    return tuple;
  }
  if (tuple.isLeaf() || tuple.rank() < profile.rank())
    return detail::operandError<Cat>({tuple}, ErrorCode::IncompatibleProfile, AlgebraOp::Zip2By);
  auto splits = llvm::SmallVector<IntTuple<Cat>, 8>{};
  auto tupleChild = tuple.begin();
  auto i = int32_t{0};
  for (auto guideChild : profile) {
    if (guideChild.isLeaf() && !isSplit(*tupleChild))
      return detail::operandError<Cat>({*tupleChild}, ErrorCode::IncompatibleProfile,
                                       AlgebraOp::Zip2By, i);
    auto split = zip2_by(*tupleChild, guideChild);
    if (split.isError())
      return split;
    splits.push_back(std::move(split));
    ++tupleChild;
    ++i;
  }

  auto first = llvm::SmallVector<IntTupleRef<Cat>, 8>{};
  auto second = llvm::SmallVector<IntTupleRef<Cat>, 8>{};
  for (const auto &split : splits) {
    first.push_back(split.at(0));
    second.push_back(split.at(1));
  }
  for (auto end = tuple.end(); tupleChild != end; ++tupleChild)
    second.push_back(*tupleChild);

  auto firstTuple = make_tuple(first);
  auto secondTuple = make_tuple(second);
  return make_tuple(firstTuple, secondTuple);
}

template <class T, class Cat> IntTuple<Cat> reverse(const T &tuple) {
  auto t = tuple.asRef();
  if (t.isLeaf())
    return t;
  auto parts = llvm::SmallVector<IntTupleRef<Cat>, 8>{};
  for (auto child : t)
    parts.push_back(child);
  std::reverse(parts.begin(), parts.end());
  return make_tuple(parts);
}

//===----------------------------------------------------------------------===//
// Predicates
//===----------------------------------------------------------------------===//

template <class A, class B, class Cat = category_of<A>> bool congruent(const A &a, const B &b) {
  return Node::congruent(a.asRef().nodes(), b.asRef().nodes());
}

template <class A, class B, class Cat = category_of<A>>
bool weakly_congruent(const A &a, const B &b) {
  return Node::weakly_congruent(a.asRef().nodes(), b.asRef().nodes());
}

template <class A, class B, class Cat> IntTuple<Cat> compatible(const A &a, const B &b) {
  auto lhs = a.asRef();
  auto rhs = b.asRef();

  if (lhs.isLeaf())
    return leaf_eq(lhs, product(rhs));
  if (rhs.isLeaf() || lhs.rank() != rhs.rank())
    return IntTuple<Cat>::getZero();
  auto result = IntTuple<Cat>::getOne();
  for (auto [lhsChild, rhsChild] : llvm::zip(lhs, rhs))
    result = result * compatible(lhsChild, rhsChild);
  return result;
}

namespace detail {

template <class Cat>
IntTuple<Cat> lex_less_impl(IntTupleRef<Cat> a, IntTupleRef<Cat> b, int32_t i) {
  if (a.isLeaf() || b.isLeaf())
    return leaf_lt(a, b);
  if (i == b.rank())
    return IntTuple<Cat>::getZero();
  if (i == a.rank())
    return IntTuple<Cat>::getOne();
  auto lhs = a.at(i);
  auto rhs = b.at(i);
  return leaf_bit_or(lex_less_impl(lhs, rhs, 0),
                     leaf_bit_and(eq(lhs, rhs), lex_less_impl(a, b, i + 1)));
}

template <class Cat>
IntTuple<Cat> colex_less_impl(IntTupleRef<Cat> a, IntTupleRef<Cat> b, int32_t i) {
  if (a.isLeaf() || b.isLeaf())
    return leaf_lt(a, b);
  if (i == b.rank())
    return IntTuple<Cat>::getZero();
  if (i == a.rank())
    return IntTuple<Cat>::getOne();
  auto lhs = a.at(a.rank() - 1 - i);
  auto rhs = b.at(b.rank() - 1 - i);
  return leaf_bit_or(colex_less_impl(lhs, rhs, 0),
                     leaf_bit_and(eq(lhs, rhs), colex_less_impl(a, b, i + 1)));
}

template <class Cat>
IntTuple<Cat> elem_less_impl(IntTupleRef<Cat> a, IntTupleRef<Cat> b, int32_t i) {
  if (a.isLeaf() || b.isLeaf())
    return leaf_lt(a, b);
  if (i == a.rank())
    return IntTuple<Cat>::getOne();
  if (i == b.rank())
    return IntTuple<Cat>::getZero();
  return leaf_bit_and(elem_less_impl(a.at(i), b.at(i), 0), elem_less_impl(a, b, i + 1));
}

} // namespace detail

template <class A, class B, class Cat = category_of<A>>
IntTuple<Cat> lex_less(const A &a, const B &b) {
  return detail::lex_less_impl(a.asRef(), b.asRef(), 0);
}

template <class A, class B, class Cat = category_of<A>>
IntTuple<Cat> lex_leq(const A &a, const B &b) {
  return leaf_eq(lex_less(b, a), IntTuple<Cat>::getZero());
}

template <class A, class B, class Cat = category_of<A>>
IntTuple<Cat> lex_gtr(const A &a, const B &b) {
  return lex_less(b, a);
}

template <class A, class B, class Cat = category_of<A>>
IntTuple<Cat> lex_geq(const A &a, const B &b) {
  return leaf_eq(lex_less(a, b), IntTuple<Cat>::getZero());
}

template <class A, class B, class Cat = category_of<A>>
IntTuple<Cat> colex_less(const A &a, const B &b) {
  return detail::colex_less_impl(a.asRef(), b.asRef(), 0);
}

template <class A, class B, class Cat = category_of<A>>
IntTuple<Cat> colex_leq(const A &a, const B &b) {
  return leaf_eq(colex_less(b, a), IntTuple<Cat>::getZero());
}

template <class A, class B, class Cat = category_of<A>>
IntTuple<Cat> colex_gtr(const A &a, const B &b) {
  return colex_less(b, a);
}

template <class A, class B, class Cat = category_of<A>>
IntTuple<Cat> colex_geq(const A &a, const B &b) {
  return leaf_eq(colex_less(a, b), IntTuple<Cat>::getZero());
}

template <class A, class B, class Cat = category_of<A>>
IntTuple<Cat> elem_less(const A &a, const B &b) {
  return detail::elem_less_impl(a.asRef(), b.asRef(), 0);
}

template <class A, class B, class Cat = category_of<A>>
IntTuple<Cat> elem_leq(const A &a, const B &b) {
  return leaf_eq(elem_less(b, a), IntTuple<Cat>::getZero());
}

template <class A, class B, class Cat = category_of<A>>
IntTuple<Cat> elem_gtr(const A &a, const B &b) {
  return elem_less(b, a);
}

template <class A, class B, class Cat = category_of<A>>
IntTuple<Cat> elem_geq(const A &a, const B &b) {
  return leaf_eq(elem_less(a, b), IntTuple<Cat>::getZero());
}

//===----------------------------------------------------------------------===//
// Arithmetic
//===----------------------------------------------------------------------===//

template <int... Is, class T, class Cat> int32_t rank(const T &tuple) {
  if constexpr (sizeof...(Is) == 0)
    return tuple.asRef().rank();
  else
    return get<Is...>(tuple).rank();
}

template <int... Is, class T, class Cat> int32_t depth(const T &tuple) {
  if constexpr (sizeof...(Is) == 0)
    return tuple.asRef().depth();
  else
    return get<Is...>(tuple).depth();
}

template <int... Is, class T, class Cat> IntTuple<Cat> shape(const T &tuple) {
  if constexpr (sizeof...(Is) == 0)
    return tuple.asRef();
  else
    return get<Is...>(tuple);
}

template <class T, class Cat> IntTuple<Cat> product(const T &t) {
  auto tuple = t.asRef();

  if (tuple.isLeaf())
    return tuple;
  auto result = Leaf<Cat>::getOne();
  for (auto leaf : tuple.leaves())
    result = result * leaf;
  return IntTuple<Cat>::fromLeaf(result);
}

template <int... Is, class T, class Cat> IntTuple<Cat> size(const T &t) {
  if constexpr (sizeof...(Is) == 0)
    return product(t);
  else
    return product(get<Is...>(t));
}

template <class T, class Cat> IntTuple<Cat> sum(const T &t) {
  auto tuple = t.asRef();
  if (tuple.isLeaf())
    return tuple;
  auto result = IntTuple<Cat>::getZero();
  for (auto child : tuple)
    result = result + sum(child);
  return result;
}

template <class T, class Cat = category_of<T>> IntTuple<Cat> product_each(const T &tuple) {
  return transform(wrap(tuple), [](IntTupleRef<Cat> child) { return product(child); });
}

template <class Tuple, class Guide, class Cat>
IntTuple<Cat> product_like(const Tuple &tuple, const Guide &guide) {
  if (tuple.asRef().isError())
    return tuple.asRef();
  if (guide.asRef().isError())
    return guide.asRef();
  auto result = transform_leaf(
      guide, tuple, [](IntTupleRef<Cat>, IntTupleRef<Cat> value) { return product(value); });
  return detail::tagError(std::move(result), AlgebraOp::ProductLike);
}

template <class A, class B, class Cat> IntTuple<Cat> inner_product(const A &a, const B &b) {
  auto lhs = a.asRef();
  auto rhs = b.asRef();

  if (lhs.isLeaf() != rhs.isLeaf() || (!lhs.isLeaf() && lhs.rank() != rhs.rank()))
    return detail::operandError<Cat>({lhs, rhs}, ErrorCode::IncompatibleProfile,
                                     AlgebraOp::InnerProduct);
  if (lhs.isLeaf())
    return lhs * rhs;
  auto result = IntTuple<Cat>::getZero();
  for (auto [lhsChild, rhsChild] : llvm::zip(lhs, rhs))
    result = result + inner_product(lhsChild, rhsChild);
  return result;
}

template <class T, class Cat> IntTuple<Cat> gcd(const T &t) {
  auto value = t.asRef();
  if (value.isLeaf())
    return value;
  auto result = IntTuple<Cat>::getZero();
  for (auto child : value)
    result = leaf_gcd(result, gcd(child));
  return result;
}

template <class T0, class T1, class... Ts, class Cat>
IntTuple<Cat> gcd(const T0 &t0, const T1 &t1, const Ts &...ts) {
  return leaf_gcd(gcd(t0), gcd(t1, ts...));
}

namespace detail {

template <bool PadRhsWithOnes, class Cat, class LeafFn>
IntTuple<Cat> tuple_div(IntTupleRef<Cat> a, IntTupleRef<Cat> b, LeafFn &&leafFn) {
  if (a.isLeaf() && b.isLeaf())
    return leafFn(a, b);
  if (a.isLeaf())
    return leafFn(a, product(b).asRef());
  if (!b.isLeaf()) {
    if (PadRhsWithOnes ? a.rank() < b.rank() : a.rank() != b.rank())
      return operandError<Cat>({a, b}, ErrorCode::IncompatibleShapeDivision, AlgebraOp::TupleDiv);

    llvm::SmallVector<IntTuple<Cat>, 8> result;
    auto ia = a.begin();
    auto ib = b.begin();
    auto eb = b.end();
    for (auto ea = a.end(); ia != ea; ++ia) {
      if (ib != eb) {
        result.push_back(tuple_div<PadRhsWithOnes>(*ia, *ib, leafFn));
        ++ib;
      } else {
        auto one = IntTuple<Cat>::getOne();
        result.push_back(tuple_div<PadRhsWithOnes>(*ia, one.asRef(), leafFn));
      }
    }
    return make_tuple(result);
  }

  IntTuple<Cat> rest = b;
  llvm::SmallVector<IntTuple<Cat>, 8> result;
  for (auto child : a) {
    result.push_back(tuple_div<PadRhsWithOnes>(child, rest.asRef(), leafFn));
    rest = leafFn(rest.asRef(), product(child).asRef());
  }
  return make_tuple(result);
}

template <class Cat>
std::pair<IntTuple<Cat>, IntTuple<Cat>> compact_col_major_impl(IntTupleRef<Cat> shape,
                                                               IntTuple<Cat> current) {
  if (shape.isLeaf()) {
    if (shape.isOne())
      return {IntTuple<Cat>::getZero(), current};
    return {current, current * shape};
  }
  llvm::SmallVector<IntTuple<Cat>, 8> strides;
  for (auto child : shape) {
    auto next = compact_col_major_impl(child, current);
    strides.push_back(std::move(next.first));
    current = std::move(next.second);
  }
  return {make_tuple(strides), current};
}

template <class Cat>
std::pair<IntTuple<Cat>, IntTuple<Cat>> compact_row_major_impl(IntTupleRef<Cat> shape,
                                                               IntTuple<Cat> current) {
  if (shape.isLeaf()) {
    if (shape.isOne())
      return {IntTuple<Cat>::getZero(), current};
    return {current, current * shape};
  }
  // The fold runs right to left, so collect the children once instead of
  // paying at(i)'s O(i) descent on every step.
  auto children = llvm::SmallVector<IntTupleRef<Cat>, 8>{};
  for (auto child : shape)
    children.push_back(child);
  llvm::SmallVector<IntTuple<Cat>, 8> strides;
  for (auto i = children.size(); i-- > 0;) {
    auto next = compact_row_major_impl(children[i], current);
    strides.push_back(std::move(next.first));
    current = std::move(next.second);
  }
  std::reverse(strides.begin(), strides.end());
  return {make_tuple(strides), current};
}

} // namespace detail

template <class A, class B, class Cat> auto ceil_div(const A &a, const B &b) {
  return detail::tuple_div<true>(a.asRef(), b.asRef(), [](IntTupleRef<Cat> x, IntTupleRef<Cat> y) {
    return leaf_ceil_div(x, y);
  });
}

template <class A, class B, class Cat> auto shape_div(const A &a, const B &b) {
  return detail::tuple_div<false>(a.asRef(), b.asRef(), [](IntTupleRef<Cat> x, IntTupleRef<Cat> y) {
    return leaf_shape_div(x, y);
  });
}

template <class A, class B, class Cat = category_of<A>> auto round_up(const A &a, const B &b) {
  return ceil_div(a, b) * b;
}

template <class A, class B, class Cat> IntTuple<Cat> elem_scale(const A &a, const B &b) {
  auto x = a.asRef();
  auto y = b.asRef();

  if (x.isLeaf())
    return x * product(y);
  if (y.isLeaf() || x.rank() != y.rank())
    return detail::operandError<Cat>({x, y}, ErrorCode::IncompatibleProfile, AlgebraOp::ElemScale);
  return transform(x, y,
                   [](IntTupleRef<Cat> lhs, IntTupleRef<Cat> rhs) { return elem_scale(lhs, rhs); });
}

template <class T, class V, class Cat> IntTuple<Cat> repeat_like(const T &guide, const V &value) {
  auto g = guide.asRef();
  if (g.isError())
    return g;
  if (g.isLeaf())
    return value.asRef();
  return transform(g, [&](IntTupleRef<Cat> child) { return repeat_like(child, value); });
}

template <class A, class B, class Cat> IntTuple<Cat> filter_zeros(const A &mask, const B &value) {
  auto a = mask.asRef();
  auto b = value.asRef();
  if (a.isError())
    return a;
  if (a.isLeaf()) {
    if (a.isSInt(0))
      return repeat_like(b, IntTuple<Cat>::getOne());
    return b;
  }
  if (b.isLeaf() || a.rank() != b.rank())
    return detail::operandError<Cat>({a, b}, ErrorCode::IncompatibleProfile,
                                     AlgebraOp::FilterZeros);
  return transform(a, b, [](IntTupleRef<Cat> maskChild, IntTupleRef<Cat> valueChild) {
    return filter_zeros(maskChild, valueChild);
  });
}

template <class T, class Cat = category_of<T>> IntTuple<Cat> filter_zeros(const T &tuple) {
  return filter_zeros(tuple, tuple);
}

namespace detail {

template <bool KeepNone, class Cat>
IntTuple<Cat> liftSliceDice(IntTupleRef<Cat> coord, IntTupleRef<Cat> value) {
  if (coord.isLeaf()) {
    auto keep = coord.isNone() == KeepNone;
    return keep ? make_tuple(value) : IntTuple<Cat>::getEmpty();
  }
  constexpr auto op = KeepNone ? AlgebraOp::Slice : AlgebraOp::Dice;
  if (value.isLeaf() || coord.rank() != value.rank())
    return operandError<Cat>({coord, value}, ErrorCode::IncompatibleProfile, op);
  auto result = IntTuple<Cat>::getEmpty();
  auto i = int32_t{0};
  for (auto [coordChild, valueChild] : llvm::zip(coord, value)) {
    auto part = liftSliceDice<KeepNone>(coordChild, valueChild);
    if (part.isError())
      return tagError(part, op, i);
    result = concat(result, part);
    ++i;
  }
  return result;
}

} // namespace detail

template <class Coord, class T, class Cat> IntTuple<Cat> slice(const Coord &coord, const T &value) {
  auto c = coord.asRef();
  auto v = value.asRef();
  if (c.isLeaf()) {
    if (c.isNone())
      return v;
    return IntTuple<Cat>::getEmpty();
  }
  return detail::liftSliceDice<true>(c, v);
}

template <class Coord, class T, class Cat> IntTuple<Cat> dice(const Coord &coord, const T &value) {
  auto c = coord.asRef();
  auto v = value.asRef();
  if (c.isLeaf()) {
    if (c.isNone())
      return IntTuple<Cat>::getEmpty();
    return v;
  }
  return detail::liftSliceDice<false>(c, v);
}

template <class T, class Current, class Cat>
auto compact_col_major(const T &shape, const Current &current) {
  IntTuple<Cat> ownedCurrent = current.asRef();
  if (shape.asRef().isError())
    return IntTuple<Cat>(shape.asRef());
  if (ownedCurrent.isError())
    return ownedCurrent;
  return detail::compact_col_major_impl(shape.asRef(), std::move(ownedCurrent)).first;
}

template <class T, class Current, class Cat>
auto compact_row_major(const T &shape, const Current &current) {
  IntTuple<Cat> ownedCurrent = current.asRef();
  if (shape.asRef().isError())
    return IntTuple<Cat>(shape.asRef());
  if (ownedCurrent.isError())
    return ownedCurrent;
  return detail::compact_row_major_impl(shape.asRef(), std::move(ownedCurrent)).first;
}

/// Convert a scalar generalized column-major coordinate to the profile of `shape`.
template <class Index, class Shape, class Cat>
IntTuple<Cat> idx2crd(const Index &index, const Shape &shape) {
  auto idx = index.asRef();
  auto s = shape.asRef();
  if (idx.isLeaf()) {
    if (s.isLeaf())
      return idx;
    auto result = llvm::SmallVector<IntTuple<Cat>, 8>{};
    IntTuple<Cat> rest = idx;
    auto rank = s.rank();
    auto i = int32_t{0};
    for (auto child : s) {
      if (++i == rank) {
        result.push_back(idx2crd(rest, child));
        break;
      }
      auto childSize = product(child);
      result.push_back(idx2crd(rest % childSize, child));
      rest = rest / childSize;
    }
    return make_tuple(result);
  }
  if (s.isLeaf() || idx.rank() != s.rank())
    return detail::operandError<Cat>({idx, s}, ErrorCode::IncompatibleProfile, AlgebraOp::Idx2Crd);
  return transform(idx, s, [](IntTupleRef<Cat> indexChild, IntTupleRef<Cat> shapeChild) {
    return idx2crd(indexChild, shapeChild);
  });
}

template <class Index, class Shape, class Stride, class Cat = category_of<Shape>>
IntTuple<Cat> idx2crd(const Index &index, const Shape &shape, const Stride &stride) {
  auto idx = index.asRef();
  auto s = shape.asRef();
  auto d = stride.asRef();
  if (idx.isLeaf()) {
    if (!s.isLeaf()) {
      if (d.isLeaf()) {
        auto compact = compact_col_major(s, d);
        return idx2crd(idx, s, compact);
      }
      if (s.rank() != d.rank())
        return detail::operandError<Cat>({s, d}, ErrorCode::IncompatibleProfile,
                                         AlgebraOp::Idx2Crd);
      return transform(s, d, [&](IntTupleRef<Cat> shapeChild, IntTupleRef<Cat> strideChild) {
        return idx2crd(idx, shapeChild, strideChild);
      });
    }
    if (!d.isLeaf())
      return detail::operandError<Cat>({s, d}, ErrorCode::IncompatibleProfile, AlgebraOp::Idx2Crd);
    if (s.isZero())
      return IntTuple<Cat>::getError(
          ErrorInfo(ErrorCode::InvalidLayout).withFrame(AlgebraOp::Idx2Crd));
    if (idx.isZero() || s.isOne() || d.isZero())
      return IntTuple<Cat>::getZero();
    return (idx / d) % s;
  }
  if (s.isLeaf() || d.isLeaf() || idx.rank() != s.rank() || idx.rank() != d.rank())
    return detail::operandError<Cat>({idx, s, d}, ErrorCode::IncompatibleProfile,
                                     AlgebraOp::Idx2Crd);
  return transform(
      idx, s, d,
      [](IntTupleRef<Cat> indexChild, IntTupleRef<Cat> shapeChild, IntTupleRef<Cat> strideChild) {
        return idx2crd(indexChild, shapeChild, strideChild);
      });
}

template <class Coord, class Shape, class Stride, class Cat = category_of<Shape>>
IntTuple<Cat> crd2idx(const Coord &coord, const Shape &shape, const Stride &stride) {
  auto c = coord.asRef();
  auto s = shape.asRef();
  auto d = stride.asRef();
  if (c.isLeaf() && !s.isLeaf()) {
    if (d.isLeaf() || s.rank() != d.rank())
      return detail::operandError<Cat>({s, d}, ErrorCode::IncompatibleProfile, AlgebraOp::Crd2Idx);
    auto result = IntTuple<Cat>::getZero();
    IntTuple<Cat> rest = c;
    auto is = s.begin();
    auto id = d.begin();
    auto rank_s = s.rank();
    for (auto i = int32_t{0}; i < rank_s; ++i, ++is, ++id) {
      auto childCoord = rest;
      if (i + 1 < rank_s) {
        auto childSize = product(*is);
        childCoord = rest % childSize;
        rest = rest / childSize;
      }
      result = result + crd2idx(childCoord, *is, *id);
    }
    return result;
  }
  if (s.isLeaf()) {
    if (!c.isLeaf() || !d.isLeaf())
      return detail::operandError<Cat>({c, d}, ErrorCode::IncompatibleProfile, AlgebraOp::Crd2Idx);
    return c * d;
  }
  if (d.isLeaf() || c.rank() != s.rank() || s.rank() != d.rank())
    return detail::operandError<Cat>({c, s, d}, ErrorCode::IncompatibleProfile, AlgebraOp::Crd2Idx);
  auto result = IntTuple<Cat>::getZero();
  for (auto [coordChild, shapeChild, strideChild] : llvm::zip(c, s, d))
    result = result + crd2idx(coordChild, shapeChild, strideChild);
  return result;
}

template <class Coord, class Shape, class Cat>
IntTuple<Cat> crd2idx(const Coord &coord, const Shape &shape) {
  auto c = coord.asRef();
  if (c.isLeaf())
    return c;
  auto compact = compact_col_major(shape);
  return crd2idx(c, shape, compact);
}

template <class Coord, class SrcShape, class DstShape, class Cat>
IntTuple<Cat> crd2crd(const Coord &coord, const SrcShape &srcShape, const DstShape &dstShape) {
  auto c = coord.asRef();
  auto src = srcShape.asRef();
  auto dst = dstShape.asRef();
  if (!c.isLeaf() && !src.isLeaf() && !dst.isLeaf()) {
    if (c.rank() != src.rank() || c.rank() != dst.rank())
      return detail::operandError<Cat>({c, src, dst}, ErrorCode::IncompatibleProfile,
                                       AlgebraOp::Crd2Crd);
    return transform(
        c, src, dst,
        [](IntTupleRef<Cat> coordChild, IntTupleRef<Cat> srcChild, IntTupleRef<Cat> dstChild) {
          return crd2crd(coordChild, srcChild, dstChild);
        });
  }
  auto index = crd2idx(coord, srcShape);
  if (!index.isLeaf())
    return detail::operandError<Cat>({index.asRef()}, ErrorCode::ExpectedLeafOperand,
                                     AlgebraOp::Crd2Crd);
  return idx2crd(index, dst);
}

namespace detail {

template <class Cat> IntTuple<Cat> tupleFromLeaves(llvm::ArrayRef<IntTuple<Cat>> leaves) {
  if (leaves.empty())
    return IntTuple<Cat>::getEmpty();
  if (leaves.size() == 1)
    return leaves.front();
  return make_tuple(leaves);
}

template <class Cat>
void collectOrderedModes(IntTupleRef<Cat> shape, IntTupleRef<Cat> order,
                         llvm::SmallVectorImpl<IntTuple<Cat>> &modeShapes,
                         llvm::SmallVectorImpl<IntTuple<Cat>> &modeOrders) {
  if (order.isLeaf()) {
    modeShapes.push_back(product(shape));
    modeOrders.push_back(order);
    return;
  }
  FLYDSL_CORE_ASSERT(!shape.isLeaf() && shape.rank() == order.rank());
  auto is = shape.begin();
  auto io = order.begin();
  for (auto end = order.end(); io != end; ++is, ++io)
    collectOrderedModes(*is, *io, modeShapes, modeOrders);
}

template <class Cat>
IntTuple<Cat> buildOrderedStride(IntTupleRef<Cat> shape, IntTupleRef<Cat> order,
                                 const llvm::SmallVectorImpl<int64_t> &keys,
                                 const llvm::SmallVectorImpl<IntTuple<Cat>> &modeShapes,
                                 size_t &leafIndex) {
  if (order.isLeaf()) {
    auto key = keys[leafIndex++];
    auto current = IntTuple<Cat>::getOne();
    for (auto i = size_t{0}; i < keys.size(); ++i)
      if (keys[i] < key)
        current = current * modeShapes[i];
    return compact_col_major(shape, current);
  }
  llvm::SmallVector<IntTuple<Cat>, 8> result;
  auto is = shape.begin();
  auto io = order.begin();
  for (auto end = order.end(); io != end; ++is, ++io)
    result.push_back(buildOrderedStride(*is, *io, keys, modeShapes, leafIndex));
  return make_tuple(result);
}

} // namespace detail

template <class Shape, class Order, class Cat>
IntTuple<Cat> compact_order(const Shape &shape, const Order &order) {
  if constexpr (std::is_same_v<Order, LayoutLeft>) {
    return compact_col_major(shape);
  } else if constexpr (std::is_same_v<Order, LayoutRight>) {
    return compact_row_major(shape);
  } else {
    auto s = shape.asRef();
    auto o = order.asRef();
    if (!Node::weakly_congruent(o.nodes(), s.nodes()))
      return detail::operandError<Cat>({s, o}, ErrorCode::IncompatibleProfile,
                                       AlgebraOp::CompactOrder);

    llvm::SmallVector<IntTuple<Cat>, 8> modeShapes;
    llvm::SmallVector<IntTuple<Cat>, 8> modeOrders;
    detail::collectOrderedModes(s, o, modeShapes, modeOrders);

    auto maxStatic = int64_t{0};
    for (auto value : modeOrders)
      if (value.isSInt())
        maxStatic = std::max(maxStatic, value.staticValue());

    llvm::SmallVector<int64_t, 8> keys;
    keys.reserve(modeOrders.size());
    auto dynamicKey = maxStatic + 1;
    for (auto value : modeOrders)
      keys.push_back(value.isSInt() ? value.staticValue() : dynamicKey++);

    auto leafIndex = size_t{0};
    return detail::buildOrderedStride(s, o, keys, modeShapes, leafIndex);
  }
}

//===----------------------------------------------------------------------===//
// Printing
//===----------------------------------------------------------------------===//

template <class T, class Cat, std::enable_if_t<is_int_tuple_v<T>, int>>
void print(const T &value, llvm::raw_ostream &os) {
  auto t = value.asRef();
  if (t.isLeaf()) {
    print(t.getLeaf(), os);
    return;
  }
  os << '(';
  auto first = true;
  for (auto c : t) {
    if (!first)
      os << ',';
    print(c, os);
    first = false;
  }
  os << ')';
}

template <class Cat> std::optional<IntTuple<Cat>> IntTuple<Cat>::fromString(llvm::StringRef text) {
  if (!text.starts_with('(')) {
    auto leaf = Leaf::fromString(text);
    return leaf ? std::optional(IntTuple::fromLeaf(*leaf)) : std::nullopt;
  }
  auto close = detail::findMatching(text, 0, '(', ')');
  if (close != text.size() - 1) {
    auto leaf = Leaf::fromString(text);
    return leaf ? std::optional(IntTuple::fromLeaf(*leaf)) : std::nullopt;
  }
  auto body = text.slice(1, close);
  if (body.empty())
    return IntTuple::getEmpty();
  llvm::SmallVector<IntTuple, 8> children;
  while (true) {
    auto comma = detail::findTopLevel(body, ",");
    auto item = comma == llvm::StringRef::npos ? body : body.take_front(comma);
    auto child = IntTuple::fromString(item);
    if (!child)
      return std::nullopt;
    children.push_back(std::move(*child));
    if (comma == llvm::StringRef::npos)
      break;
    body = body.drop_front(comma + 1);
    if (body.empty())
      return std::nullopt;
  }
  return IntTuple::fromChildren(children);
}

} // namespace mlir::fly::core

#endif // FLYDSL_CORE_ALGEBRA_INTTUPLEALGORITHMS_HPP
