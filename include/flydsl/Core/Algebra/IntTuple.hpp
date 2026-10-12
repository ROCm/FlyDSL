// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_CORE_ALGEBRA_INTTUPLE_HPP
#define FLYDSL_CORE_ALGEBRA_INTTUPLE_HPP

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include "flydsl/Core/Algebra/Leaf.hpp"
#include "flydsl/Core/Algebra/Parse.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <type_traits>

namespace mlir::fly::core {

//===----------------------------------------------------------------------===//
// Flattened Tree Node
//===----------------------------------------------------------------------===//

/// One node of the preorder-flattened tree.  Carries no leaf payload, so the
/// node array is identical across stages and can be shared between them, and
/// between congruent tuples such as a layout's shape and stride.
///
/// A node array is always a full subtree: a preorder run followed by one
/// sentinel whose `firstLeaf` marks where the subtree's leaves end.  The
/// sentinel of a child is just its successor in the parent's preorder, so
/// taking a subtree is a pair of slices rather than a copy.
struct Node {
  static constexpr size_t maxSpan = UINT16_MAX;
  uint16_t span;      ///< nodes this subtree occupies, 1 for a leaf
  uint16_t firstLeaf; ///< index of this subtree's first leaf

  ///
  /// Tree Operations
  ///

  /// Builds the node array of a flat tuple over `leafCount` leaves.
  static void make_flat(llvm::SmallVectorImpl<Node> &out, int32_t leafCount);
  /// Closes a node array under construction: fixes the root span, adds the sentinel.
  static void finish(llvm::SmallVectorImpl<Node> &nodes, int32_t leafCount);
  /// Appends `src`'s nodes (its sentinel excluded) rebased onto `leafBase`.
  static void appendSubtree(llvm::SmallVectorImpl<Node> &nodes, llvm::ArrayRef<Node> src,
                            int32_t leafBase);

  /// Properties
  static int32_t rank(llvm::ArrayRef<Node> nodes);
  static int32_t leafCount(llvm::ArrayRef<Node> nodes);
  static int32_t depth(llvm::ArrayRef<Node> nodes);

  /// Node-array offset of child `i`; O(1) on a flat tuple, O(i) otherwise.
  static int32_t childOffset(llvm::ArrayRef<Node> nodes, int32_t i);
  /// Offset of the sentinel, which is also where child iteration ends.
  static int32_t sentinelOffset(llvm::ArrayRef<Node> nodes) {
    return static_cast<int32_t>(nodes.size()) - 1;
  }

  /// Predicates
  static bool is_leaf(llvm::ArrayRef<Node> nodes);
  static bool is_flat(llvm::ArrayRef<Node> nodes);

  static bool congruent(llvm::ArrayRef<Node> a, llvm::ArrayRef<Node> b);
  static bool weakly_congruent(llvm::ArrayRef<Node> a, llvm::ArrayRef<Node> b);
};

//===----------------------------------------------------------------------===//
// IntTuple
//===----------------------------------------------------------------------===//

template <class Cat> struct IntTupleRef {
public:
  using Leaf = core::Leaf<Cat>;

  IntTupleRef() = default;
  IntTupleRef(llvm::ArrayRef<Node> nodes, llvm::ArrayRef<Leaf> leaves)
      : nodes_(nodes), leaves_(leaves) {}

  IntTupleRef asRef() const { return *this; }
  llvm::ArrayRef<Node> nodes() const { return nodes_; }
  llvm::ArrayRef<Leaf> leaves() const { return leaves_; }

  ///
  /// Node Operations
  ///

  int32_t rank() const { return Node::rank(nodes_); }
  int32_t depth() const { return Node::depth(nodes_); }
  int32_t leafCount() const { return Node::leafCount(nodes_); }
  int32_t dynLeafCount() const;

  IntTupleRef at(int32_t i) const;
  IntTupleRef at(llvm::ArrayRef<int32_t> path) const;
  IntTupleRef at(llvm::ArrayRef<uint8_t> path) const;

  ///
  /// Predicates
  ///

  bool isLeaf() const { return Node::is_leaf(nodes_); }
  bool isFlat() const { return Node::is_flat(nodes_); }
  bool isStatic() const;
  bool empty() const { return leafCount() == 0; }

  /// An invalid tuple is the single error leaf it collapsed into when it was
  /// built, so this never scans the leaves.
  bool isError() const { return leaves_.size() == 1 && leaves_.front().isError(); }
  bool isNone() const { return isLeaf() && getLeaf().isNone(); }
  bool isSInt() const { return isLeaf() && getLeaf().isSInt(); }
  bool isSInt(int64_t v) const { return isLeaf() && getLeaf().isSInt(v); }
  bool isDInt() const { return isLeaf() && getLeaf().isDInt(); }
  bool isRatio() const { return isLeaf() && getLeaf().isRatio(); }
  bool isBasis() const { return isLeaf() && getLeaf().isBasis(); }
  bool isBasis(LeafKind scaleLeafKind) const {
    return isLeaf() && getLeaf().isBasis(scaleLeafKind);
  }
  bool isInt() const { return isLeaf() && getLeaf().isInt(); }
  bool isZero() const { return isLeaf() && getLeaf().isZero(); }
  bool isOne() const { return isLeaf() && getLeaf().isOne(); }

  ///
  /// Accessors
  ///

  Leaf getLeaf() const {
    FLYDSL_CORE_ASSERT(isLeaf());
    return leaves_.front();
  }

  int64_t staticValue() const { return getLeaf().staticValue(); }

  ErrorCode errorCode() const { return errorInfo().reason(); }
  /// The first error in leaf order, which is the one the tuple collapsed into.
  ErrorInfo errorInfo() const {
    if (isError())
      return leaves_.front().errorInfo();
    FLYDSL_CORE_ASSERT(false && "tuple has no error");
    return ErrorCode::ExpectedLeafOperand;
  }

  int32_t divisibility() const { return getLeaf().divisibility(); }

  IntTuple<Cat> scale() const;

  uint8_t modeCount() const { return getLeaf().modeCount(); }
  uint8_t mode(unsigned i) const { return getLeaf().mode(i); }
  llvm::ArrayRef<uint8_t> modes() const {
    FLYDSL_CORE_ASSERT(isLeaf());
    return leaves_.front().modes();
  }

  struct Iterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = IntTupleRef;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = IntTupleRef;

    Iterator() = default;
    Iterator(IntTupleRef parent, int32_t k) : parent_(parent), k_(k) {}

    IntTupleRef operator*() const { return parent_.childAt(k_); }
    Iterator &operator++() {
      k_ += parent_.nodes_[k_].span;
      return *this;
    }
    Iterator operator++(int) {
      auto previous = *this;
      ++*this;
      return previous;
    }
    friend bool operator==(const Iterator &a, const Iterator &b) { return a.k_ == b.k_; }
    friend bool operator!=(const Iterator &a, const Iterator &b) { return !(a == b); }

  private:
    IntTupleRef parent_;
    int32_t k_ = 0;
  };

  Iterator begin() const { return Iterator(*this, isLeaf() ? 0 : 1); }
  Iterator end() const { return Iterator(*this, Node::sentinelOffset(nodes_)); }

  friend bool operator==(const IntTupleRef &a, const IntTupleRef &b) {
    return Node::congruent(a.nodes(), b.nodes()) && a.leaves().size() == b.leaves().size() &&
           std::equal(a.leaves().begin(), a.leaves().end(), b.leaves().begin());
  }
  friend bool operator!=(const IntTupleRef &a, const IntTupleRef &b) { return !(a == b); }

private:
  /// The subtree rooted at node offset `k`, as a pair of slices.
  IntTupleRef childAt(int32_t k) const;

  llvm::ArrayRef<Node> nodes_;
  llvm::ArrayRef<Leaf> leaves_;
};

template <class Cat> struct IntTuple {
public:
  using Leaf = core::Leaf<Cat>;
  using Ref = IntTupleRef<Cat>;

  /// Materialize a borrowing view into owning storage.
  IntTuple(Ref view) : IntTuple(view.nodes(), view.leaves()) {}

  static IntTuple fromLeaf(Leaf leaf) {
    const Node nodes[] = {{1, 0}, {0, 1}};
    return IntTuple(nodes, llvm::ArrayRef<Leaf>(&leaf, 1));
  }
  static IntTuple getZero() { return fromLeaf(Leaf::getZero()); }
  static IntTuple getOne() { return fromLeaf(Leaf::getOne()); }
  static IntTuple getNone() { return fromLeaf(Leaf::getNone()); }
  static IntTuple getEmpty() {
    const Node nodes[] = {{1, 0}, {0, 0}};
    return IntTuple(nodes, {});
  }
  static IntTuple getSInt(int64_t value) { return fromLeaf(Leaf::getSInt(value)); }
  static IntTuple getError(ErrorInfo info) { return fromLeaf(Leaf::getError(info)); }
  static IntTuple getDInt(LogWidth logWidth = LogWidth::I32, int32_t div = 1) {
    return fromLeaf(Leaf::getDInt(logWidth, div));
  }
  static IntTuple getRatio(int32_t num, int32_t den) { return fromLeaf(Leaf::getRatio(num, den)); }
  static IntTuple getBasis(const IntTuple &scale, llvm::ArrayRef<int32_t> modes) {
    return fromLeaf(Leaf::getBasis(scale.getLeaf(), modes));
  }
  static IntTuple getBasis(llvm::ArrayRef<int32_t> modes) { return getBasis(getOne(), modes); }

  static std::optional<IntTuple> fromString(llvm::StringRef text);

  template <class Children> static IntTuple fromChildren(const Children &children) {
    return build(children);
  }

  /// Low-level entry that pairs a node array with its leaves, for callers that
  /// change only the structure or only the leaves.  `leaves` must match
  /// `nodes`; the first error among them is the whole result.
  static IntTuple fromParts(llvm::ArrayRef<Node> nodes, llvm::ArrayRef<Leaf> leaves) {
    auto error = llvm::find_if(leaves, [](const Leaf &leaf) { return leaf.isError(); });
    if (error != leaves.end())
      return fromLeaf(*error);
    return IntTuple(nodes, leaves);
  }

  Ref asRef() const FLYDSL_CORE_LIFETIMEBOUND { return Ref(nodes_, leaves_); }

  llvm::ArrayRef<Node> nodes() const { return nodes_; }
  llvm::ArrayRef<Leaf> leaves() const { return leaves_; }

  ///
  /// Node Operations
  ///

  int32_t rank() const { return asRef().rank(); }
  int32_t depth() const { return asRef().depth(); }
  int32_t leafCount() const { return asRef().leafCount(); }
  int32_t dynLeafCount() const { return asRef().dynLeafCount(); }

  Ref at(int32_t i) const FLYDSL_CORE_LIFETIMEBOUND { return asRef().at(i); }
  Ref at(llvm::ArrayRef<int32_t> path) const FLYDSL_CORE_LIFETIMEBOUND { return asRef().at(path); }
  Ref at(llvm::ArrayRef<uint8_t> path) const FLYDSL_CORE_LIFETIMEBOUND { return asRef().at(path); }

  ///
  /// Predicates
  ///
  bool isLeaf() const { return asRef().isLeaf(); }
  bool isFlat() const { return asRef().isFlat(); }
  bool isStatic() const { return asRef().isStatic(); }
  bool empty() const { return asRef().empty(); }

  bool isError() const { return asRef().isError(); }
  bool isNone() const { return asRef().isNone(); }
  bool isSInt() const { return asRef().isSInt(); }
  bool isSInt(int64_t v) const { return asRef().isSInt(v); }
  bool isDInt() const { return asRef().isDInt(); }
  bool isRatio() const { return asRef().isRatio(); }
  bool isBasis() const { return asRef().isBasis(); }
  bool isBasis(LeafKind scaleLeafKind) const { return asRef().isBasis(scaleLeafKind); }
  bool isInt() const { return asRef().isInt(); }
  bool isZero() const { return asRef().isZero(); }
  bool isOne() const { return asRef().isOne(); }

  ///
  /// Accessors
  ///

  Leaf getLeaf() const { return asRef().getLeaf(); }

  int64_t staticValue() const { return asRef().staticValue(); }
  ErrorCode errorCode() const { return asRef().errorCode(); }
  ErrorInfo errorInfo() const { return asRef().errorInfo(); }
  int32_t divisibility() const { return asRef().divisibility(); }
  IntTuple scale() const { return asRef().scale(); }
  uint8_t modeCount() const { return asRef().modeCount(); }
  uint8_t mode(unsigned i) const { return asRef().mode(i); }
  llvm::ArrayRef<uint8_t> modes() const { return asRef().modes(); }

  friend bool operator==(const IntTuple &a, const IntTuple &b) { return a.asRef() == b.asRef(); }
  friend bool operator==(const IntTuple &a, const Ref &b) { return a.asRef() == b; }
  friend bool operator==(const Ref &a, const IntTuple &b) { return a == b.asRef(); }
  friend bool operator!=(const IntTuple &a, const IntTuple &b) { return !(a == b); }
  friend bool operator!=(const IntTuple &a, const Ref &b) { return !(a == b); }
  friend bool operator!=(const Ref &a, const IntTuple &b) { return !(a == b); }

private:
  friend struct Layout<Cat>;

  IntTuple() = default;

  IntTuple(llvm::ArrayRef<Node> nodes, llvm::ArrayRef<Leaf> leaves)
      : nodes_(nodes.begin(), nodes.end()), leaves_(leaves.begin(), leaves.end()) {
    FLYDSL_CORE_ASSERT(!nodes.empty());
    FLYDSL_CORE_ASSERT(Node::leafCount(nodes_) == static_cast<int32_t>(leaves_.size()));
    FLYDSL_CORE_ASSERT(leaves_.size() <= 1 ||
                       llvm::none_of(leaves_, [](const Leaf &leaf) { return leaf.isError(); }));
    rebase(nodes_);
  }

  static Ref asChildRef(const IntTuple &child) { return child.asRef(); }
  static Ref asChildRef(Ref child) { return child; }

  template <class Children> static IntTuple build(const Children &children);

  /// An owning tuple keeps only its own leaves, so a node array taken from a
  /// subtree slice has to be shifted back to start at leaf 0.
  static void rebase(llvm::SmallVectorImpl<Node> &nodes);

  llvm::SmallVector<Node, 10> nodes_;
  llvm::SmallVector<Leaf, 6> leaves_;
};

template <class... Ts>
constexpr bool is_make_tuple_arg_v = ((is_int_tuple_v<Ts> || is_leaf_v<Ts>) && ...);

template <class Cat> auto make_tuple();

template <class T0, class... Ts, std::enable_if_t<is_make_tuple_arg_v<T0, Ts...>, int>>
auto make_tuple(const T0 &t0, const Ts &...ts);

template <int I, int... Is, class T, std::enable_if_t<is_int_tuple_v<T>, int>>
auto get(const T &tuple);

template <class T0, class... Ts,
          std::enable_if_t<is_int_tuple_v<T0> && (is_int_tuple_v<Ts> && ...), int>>
auto concat(const T0 &t0, const Ts &...ts);

template <class Children, class Child, std::enable_if_t<is_int_tuple_v<Child>, int>>
auto concat(const Children &children);

template <class A, class B, class Cat = category_of<A>> auto eq(const A &a, const B &b);

template <class T, class Cat = category_of<T>> IntTuple<Cat> as_arithmetic_tuple(const T &value);

template <class Basis, class T, class Cat = category_of<T>>
IntTuple<Cat> basis_get(const Basis &basis, const T &tuple);

template <class T, class Cat = category_of<T>> IntTuple<Cat> basis_value(const T &value);

template <class T, std::enable_if_t<is_int_tuple_v<T>, int>> llvm::hash_code hash_value(const T &t);

template <class A, class B, std::enable_if_t<is_int_tuple_v<A> && is_int_tuple_v<B>, int>>
auto operator+(const A &left, const B &right);

template <class A, class B, std::enable_if_t<is_int_tuple_v<A> && is_int_tuple_v<B>, int>>
auto operator-(const A &left, const B &right);

//===----------------------------------------------------------------------===//
// Definitions
//===----------------------------------------------------------------------===//

inline bool Node::is_leaf(llvm::ArrayRef<Node> nodes) {
  return nodes.size() == 2 && nodes[0].span == 1 && nodes[1].firstLeaf - nodes[0].firstLeaf == 1;
}

inline bool Node::is_flat(llvm::ArrayRef<Node> nodes) {
  return is_leaf(nodes) || static_cast<int32_t>(nodes.size()) == leafCount(nodes) + 2;
}

inline int32_t Node::leafCount(llvm::ArrayRef<Node> nodes) {
  return nodes.back().firstLeaf - nodes.front().firstLeaf;
}

inline int32_t Node::rank(llvm::ArrayRef<Node> nodes) {
  if (is_leaf(nodes))
    return 1;
  int32_t r = 0;
  for (int32_t k = 1, e = sentinelOffset(nodes); k < e; k += nodes[k].span)
    ++r;
  return r;
}

inline int32_t Node::depth(llvm::ArrayRef<Node> nodes) {
  if (is_leaf(nodes))
    return 0;
  int32_t d = 0;
  for (int32_t k = 1, e = sentinelOffset(nodes); k < e; k += nodes[k].span) {
    int32_t span = nodes[k].span;
    d = std::max(d, depth(nodes.slice(static_cast<size_t>(k), static_cast<size_t>(span + 1))) + 1);
  }
  return d;
}

inline int32_t Node::childOffset(llvm::ArrayRef<Node> nodes, int32_t i) {
  if (is_flat(nodes))
    return 1 + i;
  int32_t k = 1;
  for (int32_t j = 0; j < i; ++j)
    k += nodes[k].span;
  return k;
}

inline void Node::appendSubtree(llvm::SmallVectorImpl<Node> &nodes, llvm::ArrayRef<Node> src,
                                int32_t leafBase) {
  FLYDSL_CORE_ASSERT(!src.empty());
  FLYDSL_CORE_ASSERT(nodes.size() + src.size() - 1 <= maxSpan);
  auto srcLeafBase = src.front().firstLeaf;
  for (size_t i = 0; i + 1 < src.size(); ++i) {
    Node n = src[i];
    auto firstLeaf = n.firstLeaf - srcLeafBase + leafBase;
    FLYDSL_CORE_ASSERT(firstLeaf >= 0 && firstLeaf <= UINT16_MAX);
    n.firstLeaf = static_cast<uint16_t>(firstLeaf);
    nodes.push_back(n);
  }
}

inline void Node::finish(llvm::SmallVectorImpl<Node> &nodes, int32_t leafCount) {
  FLYDSL_CORE_ASSERT(!nodes.empty() && nodes.size() <= maxSpan);
  FLYDSL_CORE_ASSERT(leafCount >= 0 && leafCount <= UINT16_MAX);
  nodes.front().span = static_cast<uint16_t>(nodes.size());
  nodes.push_back({0, static_cast<uint16_t>(leafCount)});
}

inline void Node::make_flat(llvm::SmallVectorImpl<Node> &out, int32_t leafCount) {
  FLYDSL_CORE_ASSERT(leafCount >= 0 && leafCount < UINT16_MAX);
  out.clear();
  out.push_back({static_cast<uint16_t>(leafCount + 1), 0});
  for (int32_t i = 0; i < leafCount; ++i)
    out.push_back({1, static_cast<uint16_t>(i)});
  out.push_back({0, static_cast<uint16_t>(leafCount)});
}

inline bool Node::congruent(llvm::ArrayRef<Node> a, llvm::ArrayRef<Node> b) {
  if (a.size() != b.size())
    return false;
  auto da = a.front().firstLeaf;
  auto db = b.front().firstLeaf;
  for (size_t i = 0; i < a.size(); ++i) {
    if (i + 1 < a.size() && a[i].span != b[i].span)
      return false;
    if (a[i].firstLeaf - da != b[i].firstLeaf - db)
      return false;
  }
  return true;
}

inline bool Node::weakly_congruent(llvm::ArrayRef<Node> a, llvm::ArrayRef<Node> b) {
  if (is_leaf(a))
    return true;
  if (is_leaf(b))
    return false;
  int32_t ea = sentinelOffset(a);
  int32_t eb = sentinelOffset(b);
  int32_t ka = 1;
  int32_t kb = 1;
  while (ka < ea && kb < eb) {
    if (!weakly_congruent(a.slice(ka, a[ka].span + 1), b.slice(kb, b[kb].span + 1)))
      return false;
    ka += a[ka].span;
    kb += b[kb].span;
  }
  return ka == ea && kb == eb;
}

template <class Cat> IntTupleRef<Cat> IntTupleRef<Cat>::at(int32_t i) const {
  if (isLeaf()) {
    FLYDSL_CORE_ASSERT(i == 0);
    return *this;
  }
  FLYDSL_CORE_ASSERT(i >= 0 && i < rank());
  return childAt(Node::childOffset(nodes_, i));
}

template <class Cat> IntTupleRef<Cat> IntTupleRef<Cat>::at(llvm::ArrayRef<int32_t> path) const {
  auto cur = *this;
  for (auto i : path)
    cur = cur.at(i);
  return cur;
}

template <class Cat> IntTupleRef<Cat> IntTupleRef<Cat>::at(llvm::ArrayRef<uint8_t> path) const {
  auto cur = *this;
  for (auto i : path)
    cur = cur.at(i);
  return cur;
}

template <class Cat> bool IntTupleRef<Cat>::isStatic() const {
  for (auto l : leaves_)
    if (!l.isStatic() || l.isError())
      return false;
  return true;
}

template <class Cat> int32_t IntTupleRef<Cat>::dynLeafCount() const {
  auto n = int32_t{0};
  for (auto l : leaves_)
    if (!l.isStatic())
      ++n;
  return n;
}

template <class Cat> IntTupleRef<Cat> IntTupleRef<Cat>::childAt(int32_t k) const {
  auto span = nodes_[k].span;
  auto rootFirst = nodes_.front().firstLeaf;
  auto first = nodes_[k].firstLeaf;
  auto next = nodes_[k + span].firstLeaf;
  return IntTupleRef(nodes_.slice(k, span + 1), leaves_.slice(first - rootFirst, next - first));
}

template <class Cat>
template <class Children>
IntTuple<Cat> IntTuple<Cat>::build(const Children &children) {
  auto t = IntTuple();
  t.nodes_.push_back({0, 0});
  for (const auto &child : children) {
    auto c = asChildRef(child);
    if (c.isError())
      return IntTuple(c);
    if (t.nodes_.size() + c.nodes().size() - 1 > Node::maxSpan ||
        t.leaves_.size() + c.leaves().size() > UINT16_MAX)
      return getError(ErrorCode::TupleCapacityExceeded);
    Node::appendSubtree(t.nodes_, c.nodes(), static_cast<int32_t>(t.leaves_.size()));
    t.leaves_.append(c.leaves().begin(), c.leaves().end());
  }
  Node::finish(t.nodes_, static_cast<int32_t>(t.leaves_.size()));
  return t;
}

template <class Cat> void IntTuple<Cat>::rebase(llvm::SmallVectorImpl<Node> &nodes) {
  auto base = nodes.front().firstLeaf;
  if (base != 0)
    for (auto &n : nodes)
      n.firstLeaf -= base;
  nodes.back().span = 0;
}

template <class Cat> IntTuple<Cat> IntTupleRef<Cat>::scale() const {
  return IntTuple<Cat>::fromLeaf(getLeaf().scale());
}

namespace detail {

template <class T, class Cat = category_of<T>> auto as_tuple_child(const T &t) {
  if constexpr (is_int_tuple_v<T>)
    return t.asRef();
  else
    return IntTuple<Cat>::fromLeaf(t);
}

/// Records `op` in the error of an invalid `value`, at `mode` for a mode-wise
/// step; a valid value is returned unchanged.
template <class T, class Cat = category_of<T>, std::enable_if_t<is_int_tuple_v<T>, int> = 0>
IntTuple<Cat> tagError(T value, AlgebraOp op, int64_t mode = -1) {
  if (!value.isError())
    return value;
  return IntTuple<Cat>::getError(value.errorInfo().withFrame(op, mode));
}

/// The error an operation reports when its operands do not fit together. An
/// operand that already carries an error is the real cause, so the first such
/// error is returned as is; otherwise `op` detected `reason`.
template <class Cat>
IntTuple<Cat> operandError(std::initializer_list<IntTupleRef<Cat>> operands, ErrorCode reason,
                           AlgebraOp op, int64_t mode = -1) {
  for (auto operand : operands)
    if (operand.isError())
      return IntTuple<Cat>::getError(operand.errorInfo());
  return IntTuple<Cat>::getError(ErrorInfo(reason).withFrame(op, mode));
}

} // namespace detail

template <class Cat> auto make_tuple() { return IntTuple<Cat>::getEmpty(); }

template <class T0, class... Ts, std::enable_if_t<is_make_tuple_arg_v<T0, Ts...>, int> = 0>
auto make_tuple(const T0 &t0, const Ts &...ts) {
  using Cat = category_of<T0>;
  auto children = std::array<IntTuple<Cat>, 1 + sizeof...(Ts)>{detail::as_tuple_child(t0),
                                                               detail::as_tuple_child(ts)...};
  return IntTuple<Cat>::fromChildren(children);
}

template <class Children, class Child = typename std::decay_t<Children>::value_type,
          std::enable_if_t<is_int_tuple_v<Child>, int> = 0>
auto make_tuple(const Children &children) {
  return IntTuple<category_of<Child>>::fromChildren(children);
}

template <int I, int... Is, class T, std::enable_if_t<is_int_tuple_v<T>, int> = 0>
auto get(const T &tuple) {
  static_assert(I >= 0);
  auto child = tuple.at(I);
  if constexpr (sizeof...(Is) == 0)
    return child;
  else
    return get<Is...>(child);
}

template <class T, std::enable_if_t<is_int_tuple_v<T>, int> = 0>
auto get(const T &tuple, int32_t i) {
  return tuple.at(i);
}

namespace detail {

template <class Cat>
void append_modes(llvm::SmallVectorImpl<IntTupleRef<Cat>> &parts, IntTupleRef<Cat> tuple) {
  for (auto child : tuple)
    parts.push_back(child);
}

} // namespace detail

template <class T0, class... Ts,
          std::enable_if_t<is_int_tuple_v<T0> && (is_int_tuple_v<Ts> && ...), int> = 0>
auto concat(const T0 &t0, const Ts &...ts) {
  using Cat = category_of<T0>;
  static_assert((std::is_same_v<Cat, category_of<Ts>> && ...),
                "IntTuple children must have the same category");
  auto parts = llvm::SmallVector<IntTupleRef<Cat>, 8>{};
  detail::append_modes(parts, t0.asRef());
  (detail::append_modes(parts, ts.asRef()), ...);
  return make_tuple(parts);
}

template <class Children, class Child = typename std::decay_t<Children>::value_type,
          std::enable_if_t<is_int_tuple_v<Child>, int> = 0>
auto concat(const Children &children) {
  using Cat = category_of<Child>;
  auto parts = llvm::SmallVector<IntTupleRef<Cat>, 8>{};
  for (const auto &child : children)
    detail::append_modes(parts, child.asRef());
  return make_tuple(parts);
}

template <class A, class B, class Cat> auto eq(const A &a, const B &b) {
  auto lhs = a.asRef();
  auto rhs = b.asRef();
  if (lhs.isError())
    return IntTuple<Cat>(lhs);
  if (rhs.isError())
    return IntTuple<Cat>(rhs);
  if (!Node::congruent(lhs.nodes(), rhs.nodes()))
    return IntTuple<Cat>::getZero();

  auto result = Leaf<Cat>::getOne();
  for (auto [left, right] : llvm::zip(lhs.leaves(), rhs.leaves())) {
    auto comparison = eq(left, right);
    if (comparison.isZero())
      return IntTuple<Cat>::getZero();
    // Preserve runtime/Symbol/EmitOp payloads while folding value equality.
    result = result & comparison;
  }
  return IntTuple<Cat>::fromLeaf(result);
}

namespace detail {

template <class Cat> auto basis2tuple(Leaf<Cat> basis) {
  FLYDSL_CORE_ASSERT(basis.isBasis());
  auto cur = IntTuple<Cat>::fromLeaf(basis.scale());
  auto modes = basis.modes();
  for (auto i = static_cast<int>(modes.size()) - 1; i >= 0; --i) {
    llvm::SmallVector<IntTuple<Cat>, 8> children;
    for (auto j = uint8_t{0}; j < modes[i]; ++j)
      children.push_back(IntTuple<Cat>::getZero());
    children.push_back(cur);
    cur = make_tuple(children);
  }
  return cur;
}

template <class A, class B, class LeafFn, class Cat = category_of<A>>
auto leaf_binary(const A &left, const B &right, LeafFn &&leafFn) {
  static_assert(std::is_same_v<Cat, category_of<B>>,
                "IntTuple operands must have the same category");
  auto lhs = left.asRef();
  auto rhs = right.asRef();
  if (!lhs.isLeaf() || !rhs.isLeaf())
    return IntTuple<Cat>::getError(ErrorCode::ExpectedLeafOperand);
  return IntTuple<Cat>::fromLeaf(leafFn(lhs.getLeaf(), rhs.getLeaf()));
}

template <class T, class LeafFn, class Cat = category_of<T>>
auto leaf_unary(const T &value, LeafFn &&leafFn) {
  auto tuple = value.asRef();
  if (!tuple.isLeaf())
    return IntTuple<Cat>::getError(ErrorCode::ExpectedLeafOperand);
  return IntTuple<Cat>::fromLeaf(leafFn(tuple.getLeaf()));
}

template <class Cat> auto neg_tuple(IntTupleRef<Cat> tuple) -> IntTuple<Cat> {
  if (tuple.isLeaf())
    return IntTuple<Cat>::fromLeaf(-tuple.getLeaf());
  llvm::SmallVector<IntTuple<Cat>, 8> result;
  for (auto child : tuple)
    result.push_back(neg_tuple(child));
  return make_tuple(result);
}

template <AlgebraOp Op, class Cat> bool is_tuple_arith_left_identity(IntTupleRef<Cat> scalar) {
  if constexpr (Op == AlgebraOp::Add || Op == AlgebraOp::Sub)
    return scalar.isZero();
  else if constexpr (Op == AlgebraOp::Mul)
    return scalar.isOne();
  else
    return false;
}

template <AlgebraOp Op, class Cat> bool is_tuple_arith_right_identity(IntTupleRef<Cat> scalar) {
  if constexpr (Op == AlgebraOp::Add || Op == AlgebraOp::Sub)
    return scalar.isZero();
  else if constexpr (Op == AlgebraOp::Mul || Op == AlgebraOp::Floor)
    return scalar.isOne();
  else
    return false;
}

/// `lhs op e` for the operator's right identity `e`.
template <AlgebraOp Op, class Cat>
auto tuple_arith_missing_rhs(IntTupleRef<Cat> lhs, IntTupleRef<Cat> rhs) -> IntTuple<Cat> {
  if constexpr (Op == AlgebraOp::Mod)
    return operandError<Cat>({lhs, rhs}, ErrorCode::IncompatibleProfile, Op);
  else
    return lhs;
}

/// `e op rhs` for the operator's left identity `e`; `0 - rhs` for Sub.
template <AlgebraOp Op, class Cat>
auto tuple_arith_missing_lhs(IntTupleRef<Cat> lhs, IntTupleRef<Cat> rhs) -> IntTuple<Cat> {
  if constexpr (Op == AlgebraOp::Add || Op == AlgebraOp::Mul)
    return rhs;
  else if constexpr (Op == AlgebraOp::Sub)
    return neg_tuple(rhs);
  else
    return operandError<Cat>({lhs, rhs}, ErrorCode::IncompatibleProfile, Op);
}

template <AlgebraOp Op, class Cat, class LeafFn>
IntTuple<Cat> arith_tuple_binary(IntTupleRef<Cat> lhs, IntTupleRef<Cat> rhs, LeafFn &leafFn) {
  if constexpr (Op == AlgebraOp::Add) {
    if (lhs.isBasis()) {
      if (!rhs.isLeaf()) {
        auto expanded = basis2tuple(lhs.getLeaf());
        return arith_tuple_binary<Op>(expanded.asRef(), rhs, leafFn);
      }
      if (rhs.isBasis()) {
        auto expandedLhs = basis2tuple(lhs.getLeaf());
        auto expandedRhs = basis2tuple(rhs.getLeaf());
        return arith_tuple_binary<Op>(expandedLhs.asRef(), expandedRhs.asRef(), leafFn);
      }
    }
    if (!lhs.isLeaf() && rhs.isBasis()) {
      auto expanded = basis2tuple(rhs.getLeaf());
      return arith_tuple_binary<Op>(lhs, expanded.asRef(), leafFn);
    }
  }

  if (lhs.isLeaf() && rhs.isLeaf())
    return IntTuple<Cat>::fromLeaf(leafFn(lhs.getLeaf(), rhs.getLeaf()));

  if (!lhs.isLeaf() && !rhs.isLeaf()) {
    llvm::SmallVector<IntTuple<Cat>, 8> result;
    result.reserve(static_cast<size_t>(std::max(lhs.rank(), rhs.rank())));

    auto lhsIt = lhs.begin();
    auto rhsIt = rhs.begin();
    auto lhsEnd = lhs.end();
    auto rhsEnd = rhs.end();
    while (lhsIt != lhsEnd && rhsIt != rhsEnd) {
      result.push_back(arith_tuple_binary<Op>(*lhsIt, *rhsIt, leafFn));
      ++lhsIt;
      ++rhsIt;
    }
    for (; lhsIt != lhsEnd; ++lhsIt) {
      auto mode = tuple_arith_missing_rhs<Op>(*lhsIt, rhs);
      if (mode.isError())
        return mode;
      result.push_back(std::move(mode));
    }
    for (; rhsIt != rhsEnd; ++rhsIt) {
      auto mode = tuple_arith_missing_lhs<Op>(lhs, *rhsIt);
      if (mode.isError())
        return mode;
      result.push_back(std::move(mode));
    }
    return make_tuple(result);
  }

  if (lhs.isLeaf()) {
    if (rhs.empty())
      return tuple_arith_missing_rhs<Op>(lhs, rhs);
    if (is_tuple_arith_left_identity<Op>(lhs))
      return tuple_arith_missing_lhs<Op>(lhs, rhs);
  } else {
    if (lhs.empty())
      return tuple_arith_missing_lhs<Op>(lhs, rhs);
    if (is_tuple_arith_right_identity<Op>(rhs))
      return tuple_arith_missing_rhs<Op>(lhs, rhs);
  }
  return operandError<Cat>({lhs, rhs}, ErrorCode::ExpectedLeafOperand, Op);
}

template <AlgebraOp Op, class A, class B, class LeafFn, class Cat = category_of<A>>
auto arith_tuple(const A &left, const B &right, LeafFn &&leafFn) {
  static_assert(std::is_same_v<Cat, category_of<B>>,
                "IntTuple operands must have the same category");
  return arith_tuple_binary<Op>(left.asRef(), right.asRef(), leafFn);
}

} // namespace detail

/// Converts every basis leaf to the arithmetic tuple denoted by its mode path.
template <class T, class Cat> IntTuple<Cat> as_arithmetic_tuple(const T &value) {
  auto tuple = value.asRef();
  if (tuple.isBasis())
    return detail::basis2tuple(tuple.getLeaf());
  if (tuple.isLeaf())
    return tuple;
  auto children = llvm::SmallVector<IntTuple<Cat>, 8>{};
  for (auto child : tuple)
    children.push_back(as_arithmetic_tuple(child));
  return make_tuple(children);
}

template <class Basis, class T, class Cat>
IntTuple<Cat> basis_get(const Basis &basis, const T &tuple) {
  auto b = basis.asRef();
  if (!b.isBasis())
    return tuple.asRef();
  auto result = tuple.asRef();
  for (auto mode : b.modes())
    result = result.at(mode);
  return result;
}

template <class T, class Cat> IntTuple<Cat> basis_value(const T &value) {
  auto tuple = value.asRef();
  if (!tuple.isBasis())
    return tuple;
  return tuple.scale();
}

namespace detail {

template <class Cat>
auto make_basis_like_impl(IntTupleRef<Cat> shape, llvm::SmallVectorImpl<int32_t> &modes) {
  if (shape.isLeaf())
    return IntTuple<Cat>::getBasis(modes);
  auto result = llvm::SmallVector<IntTuple<Cat>, 8>{};
  auto mode = int32_t{0};
  for (auto child : shape) {
    modes.push_back(mode++);
    result.push_back(make_basis_like_impl(child, modes));
    modes.pop_back();
  }
  return make_tuple(result);
}

} // namespace detail

template <int... Ns, class Shape, class Cat = category_of<Shape>>
auto make_basis_like(const Shape &shape) {
  auto modes = llvm::SmallVector<int32_t, 8>{Ns...};
  return detail::make_basis_like_impl(shape.asRef(), modes);
}

#define FLYDSL_CORE_LEAF_OPERATOR(name, op)                                                        \
  template <class A, class B, class Cat = category_of<A>>                                          \
  auto name(const A &left, const B &right) {                                                       \
    return detail::leaf_binary(left, right,                                                        \
                               [](Leaf<Cat> lhs, Leaf<Cat> rhs) { return lhs op rhs; });           \
  }
#define FLYDSL_CORE_LEAF_FUNCTION(name, fn)                                                        \
  template <class A, class B, class Cat = category_of<A>>                                          \
  auto name(const A &left, const B &right) {                                                       \
    return detail::leaf_binary(left, right,                                                        \
                               [](Leaf<Cat> lhs, Leaf<Cat> rhs) { return fn(lhs, rhs); });         \
  }
#define FLYDSL_CORE_LEAF_UNARY(name, fn)                                                           \
  template <class T, class Cat = category_of<T>> auto name(const T &value) {                       \
    return detail::leaf_unary(value, [](Leaf<Cat> operand) { return fn(operand); });               \
  }
#define FLYDSL_CORE_ARITH_TUPLE_OPERATOR(op, algebraOp)                                            \
  template <class A, class B, std::enable_if_t<is_int_tuple_v<A> && is_int_tuple_v<B>, int> = 0>   \
  auto operator op(const A &left, const B &right) {                                                \
    using Cat = category_of<A>;                                                                    \
    return detail::arith_tuple<algebraOp>(                                                         \
        left, right, [](Leaf<Cat> lhs, Leaf<Cat> rhs) { return lhs op rhs; });                     \
  }

FLYDSL_CORE_LEAF_OPERATOR(leaf_add, +)
FLYDSL_CORE_LEAF_OPERATOR(leaf_sub, -)
FLYDSL_CORE_LEAF_OPERATOR(leaf_mul, *)
FLYDSL_CORE_LEAF_OPERATOR(leaf_div, /)
FLYDSL_CORE_LEAF_OPERATOR(leaf_mod, %)
FLYDSL_CORE_LEAF_OPERATOR(leaf_bit_and, &)
FLYDSL_CORE_LEAF_OPERATOR(leaf_bit_or, |)
FLYDSL_CORE_LEAF_OPERATOR(leaf_bit_xor, ^)
FLYDSL_CORE_LEAF_OPERATOR(leaf_shl, <<)
FLYDSL_CORE_LEAF_OPERATOR(leaf_shr, >>)

FLYDSL_CORE_LEAF_UNARY(leaf_abs, abs)
FLYDSL_CORE_LEAF_UNARY(leaf_signum, signum)

FLYDSL_CORE_LEAF_FUNCTION(leaf_eq, eq)
FLYDSL_CORE_LEAF_FUNCTION(leaf_ne, ne)
FLYDSL_CORE_LEAF_FUNCTION(leaf_lt, lt)
FLYDSL_CORE_LEAF_FUNCTION(leaf_min, min)
FLYDSL_CORE_LEAF_FUNCTION(leaf_max, max)
FLYDSL_CORE_LEAF_FUNCTION(leaf_gcd, gcd)
FLYDSL_CORE_LEAF_FUNCTION(leaf_safe_div, safe_div)
FLYDSL_CORE_LEAF_FUNCTION(leaf_ceil_div, ceil_div)
FLYDSL_CORE_LEAF_FUNCTION(leaf_shape_div, shape_div)
FLYDSL_CORE_LEAF_FUNCTION(leaf_exact_div, exact_div)

FLYDSL_CORE_ARITH_TUPLE_OPERATOR(+, AlgebraOp::Add)
FLYDSL_CORE_ARITH_TUPLE_OPERATOR(-, AlgebraOp::Sub)
FLYDSL_CORE_ARITH_TUPLE_OPERATOR(*, AlgebraOp::Mul)
FLYDSL_CORE_ARITH_TUPLE_OPERATOR(/, AlgebraOp::Floor)
FLYDSL_CORE_ARITH_TUPLE_OPERATOR(%, AlgebraOp::Mod)

#undef FLYDSL_CORE_LEAF_OPERATOR
#undef FLYDSL_CORE_LEAF_FUNCTION
#undef FLYDSL_CORE_LEAF_UNARY
#undef FLYDSL_CORE_ARITH_TUPLE_OPERATOR

template <class T, std::enable_if_t<is_int_tuple_v<T>, int> = 0>
llvm::hash_code hash_value(const T &t) {
  auto tuple = t.asRef();
  auto shapeHash = llvm::hash_value(0);
  auto base = tuple.nodes().front().firstLeaf;
  for (size_t i = 0; i < tuple.nodes().size(); ++i) {
    auto node = tuple.nodes()[i];
    auto span = i + 1 == tuple.nodes().size() ? 0 : node.span;
    shapeHash = llvm::hash_combine(shapeHash, span, node.firstLeaf - base);
  }
  return llvm::hash_combine(shapeHash,
                            llvm::hash_combine_range(tuple.leaves().begin(), tuple.leaves().end()));
}

template <class T, std::enable_if_t<is_int_tuple_v<T>, int> = 0> auto operator-(const T &value) {
  return detail::neg_tuple(value.asRef());
}

} // namespace mlir::fly::core

#endif // FLYDSL_CORE_ALGEBRA_INTTUPLE_HPP
