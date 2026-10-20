// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_CORE_ALGEBRA_PARSE_HPP
#define FLYDSL_CORE_ALGEBRA_PARSE_HPP

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include "flydsl/Core/Algebra/Leaf.hpp"

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

/// The LogWidth of a `bits`-wide integer; none for a width no leaf has.
inline std::optional<LogWidth> parseLogWidth(int64_t bits) {
  switch (bits) {
  case 1:
    return LogWidth::I1;
  case 8:
    return LogWidth::I8;
  case 16:
    return LogWidth::I16;
  case 32:
    return LogWidth::I32;
  case 64:
    return LogWidth::I64;
  default:
    return std::nullopt;
  }
}

/// The ErrorCode whose message is `text`.
inline std::optional<ErrorCode> parseError(llvm::StringRef text) {
  for (auto raw = 0u; raw <= static_cast<unsigned>(ErrorCode::Last); ++raw) {
    auto code = static_cast<ErrorCode>(raw);
    if (text == get_error_message(code))
      return code;
  }
  return std::nullopt;
}

/// The printed suffixes of a leaf: bit width, `{div= range= fdm}` and basis
/// modes, split from the value text in front of them.
struct LeafSuffix {
  llvm::StringRef value;
  LogWidth logWidth = LogWidth::I32;
  int32_t divisibility = 1;
  std::optional<std::pair<int32_t, int32_t>> range;
  bool fastDivMod = false;
  llvm::SmallVector<int32_t, 4> modes;
};

inline std::optional<LeafSuffix> parseLeafSuffix(llvm::StringRef text) {
  LeafSuffix result;
  auto end = text.size();

  while (end > 0) {
    auto e = text.rfind('E', end - 1);
    if (e == llvm::StringRef::npos || e + 1 == end)
      break;
    auto mode = int32_t{0};
    if (text.slice(e + 1, end).getAsInteger(10, mode) || mode < 0 || mode > 255)
      break;
    result.modes.insert(result.modes.begin(), mode);
    end = e;
  }

  if (end > 0 && text[end - 1] == '}') {
    auto open = text.rfind('{', end - 1);
    if (open == llvm::StringRef::npos)
      return std::nullopt;
    auto fields = text.slice(open + 1, end - 1);
    while (!fields.empty()) {
      fields = fields.ltrim();
      if (fields.consume_front("fdm")) {
        result.fastDivMod = true;
      } else if (fields.consume_front("div=")) {
        auto value = parseInt64(fields);
        if (!value || *value < 1 || *value > INT32_MAX)
          return std::nullopt;
        result.divisibility = static_cast<int32_t>(*value);
      } else if (fields.consume_front("range=[")) {
        auto lo = parseInt64(fields);
        if (!lo || !fields.consume_front(","))
          return std::nullopt;
        auto hi = parseInt64(fields);
        if (!hi || !fields.consume_front("]") || *lo < INT32_MIN || *lo > INT32_MAX ||
            *hi < INT32_MIN || *hi > INT32_MAX)
          return std::nullopt;
        result.range = std::pair(static_cast<int32_t>(*lo), static_cast<int32_t>(*hi));
      } else {
        return std::nullopt;
      }
      fields = fields.ltrim();
    }
    end = open;
  }

  auto widthPos = text.slice(0, end).rfind('i');
  if (widthPos != llvm::StringRef::npos && widthPos != 0) {
    auto bits = int64_t{0};
    if (!text.slice(widthPos + 1, end).getAsInteger(10, bits)) {
      auto logWidth = parseLogWidth(bits);
      if (!logWidth)
        return std::nullopt;
      result.logWidth = *logWidth;
      end = widthPos;
    }
  }
  result.value = text.take_front(end);
  return result;
}

/// Prints the bit width, unless i32, and the `{div= range= fdm}` facts of the
/// dynamic integer `scale`, a leaf without basis modes.
inline void printDynIntSuffix(const LeafProperty &scale, llvm::raw_ostream &os) {
  if (scale.logWidth() != LogWidth::I32)
    os << 'i' << scale.bitWidth();
  if (scale.divisibility() == 1 && !scale.hasRange() && !scale.hasFastDivMod())
    return;
  os << '{';
  auto first = true;
  if (scale.divisibility() != 1) {
    os << "div=" << scale.divisibility();
    first = false;
  }
  if (scale.hasRange()) {
    if (!first)
      os << ' ';
    os << "range=[" << scale.rangeMin() << ',' << scale.rangeMax() << ']';
    first = false;
  }
  if (scale.hasFastDivMod()) {
    if (!first)
      os << ' ';
    os << "fdm";
  }
  os << '}';
}

/// Prints the basis modes of `leaf`, e.g. `E0E1`; nothing for a scalar.
inline void printModes(const LeafProperty &leaf, llvm::raw_ostream &os) {
  if (leaf.isBasis())
    for (auto mode : leaf.modes())
      os << 'E' << static_cast<int>(mode);
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
