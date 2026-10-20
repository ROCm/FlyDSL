// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_TESTS_CORE_COMMON_HPP
#define FLYDSL_TESTS_CORE_COMMON_HPP

#include "flydsl/Core/Runtime.hpp"

#include "gtest/gtest.h"

#include <initializer_list>

/// Reference results for the Core algebra.
///
/// Each case evaluates one operation on Runtime values and compares the
/// Runtime print of the result with the expected result.  The Runtime print
/// shows the values, which leaves are static (`8`) or dynamic (`?8`), the
/// divisibility of the dynamic ones (`?8{div=4}`) and the tuple nesting.
namespace core_test {

using namespace mlir::fly::core;

using RT = Leaf<Runtime>;
using Tup = IntTuple<Runtime>;
using Lay = Layout<Runtime>;
using Til = Tile<Runtime>;
using Swz = Swizzle<Runtime>;

/// A dynamic leaf holding `value`, known to be a multiple of `div`.
inline Tup dyn(int64_t value, int32_t div = 1) {
  return Tup::fromLeaf(RT::getDyn(value, LogWidth::I32, div));
}

inline Tup E(const Tup &scale, std::initializer_list<int32_t> modes) {
  return Tup::getBasis(scale, modes);
}

inline const Tup _ = Tup::getNone();

inline MMAAtom<Runtime> MMAAtom_CDNA3Mfma16x16x16F16() {
  struct Op {};
  struct F16 {};
  struct F32 {};
  auto f16 = OpaqueValue::getType<F16>();
  auto f32 = OpaqueValue::getType<F32>();
  auto layoutAB =
      make_layout(make_tuple(make_tuple(16_c, 4_c), 4_c), make_tuple(make_tuple(1_c, 64_c), 16_c));
  auto layoutC = make_layout(make_tuple(make_tuple(16_c, 4_c), make_tuple(4_c, 1_c)),
                             make_tuple(make_tuple(16_c, 4_c), make_tuple(1_c, 16_c)));
  return MMAAtom<Runtime>(OpaqueValue::getType<Op>(), make_layout(64_ct, 1_ct),
                          make_tuple(16_c, 16_c, 16_c), MMAValueTypeIdentities{f16, f16, f32, f32},
                          layoutAB, layoutAB, layoutC);
}

inline CopyAtom<Runtime> universalCopy(int64_t copyBits, int32_t valueBits) {
  struct Op {};
  struct Value {};
  auto bits = make_layout(make_tuple(1_c, RT::getSInt(copyBits)), make_tuple(1_c, 1_c));
  return CopyAtom<Runtime>(OpaqueValue::getType<Op>(), OpaqueValue::getType<Value>(), valueBits,
                           make_layout(1_ct, 1_ct), bits, bits, bits);
}

} // namespace core_test

#endif // FLYDSL_TESTS_CORE_COMMON_HPP
