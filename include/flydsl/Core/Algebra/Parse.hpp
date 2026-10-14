// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_CORE_ALGEBRA_PARSE_HPP
#define FLYDSL_CORE_ALGEBRA_PARSE_HPP

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace mlir::fly::core {

namespace detail {

inline std::optional<int64_t> parseInt64(llvm::StringRef &text) {
  int64_t value = 0;
  if (text.consumeInteger(10, value))
    return std::nullopt;
  return value;
}

inline size_t findMatching(llvm::StringRef text, size_t open, char lhs, char rhs) {
  int32_t depth = 0;
  for (auto i = open; i < text.size(); ++i) {
    if (text[i] == lhs)
      ++depth;
    else if (text[i] == rhs && --depth == 0)
      return i;
  }
  return llvm::StringRef::npos;
}

inline size_t findTopLevel(llvm::StringRef text, llvm::StringRef needle) {
  int32_t paren = 0;
  int32_t bracket = 0;
  int32_t brace = 0;
  for (size_t i = 0; i + needle.size() <= text.size(); ++i) {
    auto c = text[i];
    if (c == '(')
      ++paren;
    else if (c == ')')
      --paren;
    else if (c == '[')
      ++bracket;
    else if (c == ']')
      --bracket;
    else if (c == '{')
      ++brace;
    else if (c == '}')
      --brace;
    if (paren == 0 && bracket == 0 && brace == 0 && text.substr(i).starts_with(needle))
      return i;
  }
  return llvm::StringRef::npos;
}

template <class T> struct FromString {
  static std::optional<T> parse(llvm::StringRef text) { return T::fromString(text); }
};

template <class T>
using PrintResult = decltype(print(std::declval<const T &>(), std::declval<llvm::raw_ostream &>()));

} // namespace detail

template <class T> std::optional<T> fromString(llvm::StringRef text) {
  return detail::FromString<T>::parse(text);
}

/// Every algebra type defines `print(value, os)` as its only text formatter,
/// following LLVM's print(raw_ostream &) convention: composites stream into
/// the caller's buffer without temporaries. `toString` and `<<` derive from
/// it, as llvm::to_string does, so they always agree.
template <class T, class = detail::PrintResult<T>> std::string toString(const T &value) {
  std::string result;
  llvm::raw_string_ostream os(result);
  print(value, os);
  return result;
}

template <class T, class = detail::PrintResult<T>>
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const T &value) {
  print(value, os);
  return os;
}

} // namespace mlir::fly::core

#endif // FLYDSL_CORE_ALGEBRA_PARSE_HPP
