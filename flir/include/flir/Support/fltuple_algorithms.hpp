#pragma once

#include "flir/Support/fltuple.h"
#include "mlir/IR/Value.h"
#include <functional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace mlir::flir::support {

/// Apply `f` to corresponding leaves of one (or more) FlTuple(s) and then call `g`
/// with the list of per-leaf results.
///
/// This is the FlTuple analog of "transform-apply" (aka `tapply`):
///   g( f(leaf0), f(leaf1), ... )
///
/// Note: this is intentionally a *tree* algorithm, not a `std::tuple` algorithm.
/// It matches the nested type-mode structure that python frontend builds (e.g. (9,(4,8))).
namespace detail {

inline bool sameStructure(const FlTuple &a, const FlTuple &b) {
  if (a.isLeaf != b.isLeaf)
    return false;
  if (a.isLeaf)
    return true;
  if (a.children.size() != b.children.size())
    return false;
  for (size_t i = 0; i < a.children.size(); ++i) {
    if (!sameStructure(a.children[i], b.children[i]))
      return false;
  }
  return true;
}

template <class R>
inline void collect_unary(const FlTuple &t, R &out, const std::function<typename R::value_type(Value)> &fn);

template <class R>
inline void collect_unary(const FlTuple &t, R &out,
                          const std::function<typename R::value_type(Value)> &fn) {
  if (t.isLeaf) {
    out.push_back(fn(t.value));
    return;
  }
  for (auto const &ch : t.children)
    collect_unary(ch, out, fn);
}

} // namespace detail

/// Unary transform over leaves, preserving structure.
template <class Fn>
FlTuple transform(const FlTuple &t, Fn &&fn) {
  if (t.isLeaf)
    return FlTuple(fn(t.value));
  std::vector<FlTuple> out;
  out.reserve(t.children.size());
  for (auto const &ch : t.children)
    out.push_back(transform(ch, fn));
  return FlTuple(std::move(out));
}

/// Unary tapply: collect `f(leaf)` over leaves (left-to-right) and call `g(results...)`.
///
/// Because C++ can't "splat" a runtime-sized vector into a parameter pack, this variant
/// uses `g(std::vector<R>)` by default. That is the MLIR-friendly form (rank is often dynamic).
template <class F, class G>
auto tapply(const FlTuple &t, F &&f, G &&g) {
  using R = std::invoke_result_t<F, Value>;
  std::vector<R> results;
  results.reserve(8);
  std::function<R(Value)> fn = [&](Value v) -> R { return static_cast<F&&>(f)(v); };
  detail::collect_unary(t, results, fn);
  return static_cast<G&&>(g)(results);
}

/// Binary leaf zip: map `f(leafA, leafB)` over leaves, preserving structure.
template <class Fn>
FlTuple zip_transform(const FlTuple &a, const FlTuple &b, Fn &&fn) {
  if (!detail::sameStructure(a, b)) {
    // Prefer hard failure in C++ utility; callers in lowering should validate earlier.
    throw std::invalid_argument("fltuple zip_transform: mismatched structure");
  }
  if (a.isLeaf)
    return FlTuple(fn(a.value, b.value));
  std::vector<FlTuple> out;
  out.reserve(a.children.size());
  for (size_t i = 0; i < a.children.size(); ++i)
    out.push_back(zip_transform(a.children[i], b.children[i], fn));
  return FlTuple(std::move(out));
}

} // namespace mlir::flir::support


