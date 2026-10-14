//===- SymbolTest.cpp - The Symbol category --------------------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Leaf<Symbol> and Pointer<Symbol> of lib/Core/Testing/Symbol.cpp: the Trait
// property of each leaf paired with the expression that computes it.  The cases
// check the expressions the leaf operations and the layout algebra build, and
// that each property still holds for its expression.
//
//===----------------------------------------------------------------------===//

#include "flydsl/Core/Symbol.hpp"

#include "flydsl/Core/Algebra/Layout.hpp"

#include "gtest/gtest.h"

#include <string>
#include <thread>

using namespace mlir::fly::core;

namespace {

using Sy = Leaf<Symbol>;
using Tup = IntTuple<Symbol>;
using Lay = Layout<Symbol>;

Sy i(int64_t v) { return Sy::getSInt(v); }
Sy n(llvm::StringRef name, int32_t div = 1) { return Sy::getSym(name, LogWidth::I32, div); }
Tup leaf(Sy v) { return Tup::fromLeaf(v); }
Tup leaf(int64_t v) { return Tup::fromLeaf(i(v)); }

/// The expression of `l` without its `{div= range= fdm}` suffix.
std::string exprStr(Sy l) {
  auto text = toString(l);
  auto brace = text.rfind('{');
  if (brace != std::string::npos && text.back() == '}')
    text.erase(brace);
  return text;
}

template <class T, std::enable_if_t<is_int_tuple_v<T>, int> = 0> bool allSound(const T &t) {
  for (auto value : t.asRef().leaves())
    if (!value.isSound())
      return false;
  return true;
}

bool allSound(const Lay &layout) { return allSound(layout.shape()) && allSound(layout.stride()); }

} // namespace

TEST(SymbolCategory, AlignedDynamicRatioBasis) {
  auto value = n("aligned", 8);
  auto ratio = Sy::getBasis(Sy::getRatio(3, 2), {2, 1});
  auto result = ratio * value;
  ASSERT_FALSE(result.isError());
  EXPECT_TRUE(result.isSound());
  EXPECT_EQ(result.modes(), ratio.modes());
  EXPECT_EQ(toString(result), toString(safe_div(value, i(2)) * i(3)) + "E2E1");
  EXPECT_TRUE((ratio * n("unaligned")).isError());
}

// isSound() of a static leaf checks that it carries no expression.
TEST(SymbolCategory, StaticLeavesCarryNoExpr) {
  EXPECT_TRUE(i(8).isStatic());
  EXPECT_TRUE(i(8).isSound());
  EXPECT_TRUE(Sy::getNone().isSound());
  EXPECT_TRUE(Sy::getRatio(1, 2).isSound());
  EXPECT_TRUE(Sy::getError(ErrorCode::DivisionByZero).isSound());
  EXPECT_TRUE(Sy::getBasis(i(4), {1}).isSound());
  EXPECT_TRUE(Sy::getBasis(i(4), {1}).isStatic());
}

TEST(SymbolCategory, SymbolsWithTheSameNameAreEqual) {
  auto a = n("N");
  auto b = n("N");
  auto c = n("M");
  EXPECT_FALSE(a.isStatic());
  EXPECT_TRUE(a.isDInt());
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
  EXPECT_TRUE(a.isSound());
  EXPECT_EQ(toString(a), "N");
}

TEST(SymbolCategory, StaticArithDropsExpr) {
  auto sum = i(12) + i(4);
  EXPECT_TRUE(sum.isSInt(16));
  EXPECT_TRUE(sum.isSound());
  EXPECT_EQ(toString(sum), "16");
  EXPECT_EQ((i(12) * i(4)).staticValue(), 48);
  EXPECT_TRUE((i(6) & i(3)).isSound());
}

TEST(SymbolCategory, FoldToStaticClearsExpr) {
  auto N = n("N");
  auto zero = N * i(0);
  EXPECT_TRUE(zero.isSInt(0));
  EXPECT_TRUE(zero.isSound());

  auto masked = N & i(0);
  EXPECT_TRUE(masked.isSInt(0));
  EXPECT_TRUE(masked.isSound());
}

TEST(SymbolCategory, DynamicKeepsExpr) {
  auto N = n("N");
  auto sum = N + i(4);
  EXPECT_TRUE(sum.isDInt());
  EXPECT_TRUE(sum.isSound());
  EXPECT_EQ(toString(sum), "N+4");
}

TEST(SymbolCategory, UnitIdentitiesReuseOperand) {
  auto N = n("N");
  EXPECT_EQ(N * i(1), N);
  EXPECT_EQ(i(1) * N, N);
  EXPECT_EQ(N + i(0), N);
  EXPECT_EQ(N / i(1), N);
}

TEST(SymbolCategory, TypeRailMatchesAbstractStage) {
  auto N = n("N", 8);
  auto prod = N * i(3);
  EXPECT_EQ(Leaf<Trait>(prod), Leaf<Trait>::getDInt(LogWidth::I32, 8) * Leaf<Trait>::getSInt(3));
  EXPECT_EQ(prod.divisibility(), 24);
  EXPECT_EQ(toString(prod), "N*3{div=24}");
}

TEST(SymbolCategory, BasisKeepsScaleExpr) {
  auto N = n("N");
  auto basis = Sy::getBasis(N, {1});
  EXPECT_TRUE(basis.isBasis());
  EXPECT_FALSE(basis.isStatic());
  EXPECT_EQ(basis.scale(), N);
  EXPECT_EQ(toString(basis), "NE1");

  auto scaled = basis * i(3);
  EXPECT_TRUE(scaled.isBasis());
  EXPECT_EQ(toString(scaled), "N*3{div=3}E1");
}

TEST(SymbolCategory, UnaryAndComparisons) {
  auto N = n("N");
  auto cmp = eq(N, n("M"));
  EXPECT_FALSE(cmp.isStatic());
  EXPECT_EQ(exprStr(cmp), "N==M");
  EXPECT_EQ(toString(ceil_div(N, i(4))).find("ceil_div(N,4)"), 0u);
}

TEST(SymbolCategory, OperatorsMatchTypeStage) {
  auto N = n("N");
  auto D = n("D", 8);
  EXPECT_EQ(exprStr(N - i(4)), "N-4");
  EXPECT_EQ(exprStr(N % i(3)), "N%3");
  EXPECT_EQ(exprStr(N & i(7)), "N&7");
  EXPECT_EQ(exprStr(N | i(1)), "N|1");
  EXPECT_EQ(exprStr(N ^ i(1)), "N^1");
  EXPECT_EQ(exprStr(N << i(1)), "N<<1");
  EXPECT_EQ(exprStr(N >> i(1)), "N>>1");
  EXPECT_EQ(exprStr(lshr(N, i(1))), "lshr(N,1)");
  EXPECT_EQ(toString(~N), "~N");
  EXPECT_EQ(exprStr(min(N, n("M"))), "min(N,M)");
  EXPECT_EQ(exprStr(max(N, n("M"))), "max(N,M)");
  EXPECT_EQ(exprStr(gcd(N, n("M"))), "gcd(N,M)");
  EXPECT_EQ(toString(abs(N)), "abs(N)");
  EXPECT_EQ(toString(signum(N)),
            FLYDSL_CORE_ENABLE_RANGE_INFERENCE ? "signum(N){range=[-1,1]}" : "signum(N)");
  EXPECT_EQ(exprStr(exact_div(D, i(4))), "exact_div(D,4)");
  EXPECT_EQ(exprStr(safe_div(D, i(4))), "safe_div(D,4)");
  EXPECT_EQ(exprStr(shape_div(N, i(4))), "shape_div(N,4)");
  EXPECT_EQ(exprStr(ne(N, n("M"))), "N!=M");
  EXPECT_EQ(exprStr(lt(N, n("M"))), "N<M");
  EXPECT_TRUE(exact_div(N, i(4)).isError());
  EXPECT_TRUE(exact_div(N, i(4)).isSound());
  EXPECT_EQ(toString(N - i(4)), "N-4");
  EXPECT_EQ(toString(min(N, n("M"))), "min(N,M)");
  EXPECT_EQ(toString(gcd(N, n("M"))), "gcd(N,M)");
  EXPECT_EQ(toString(abs(-N)), "abs(N)");
  EXPECT_EQ(toString(signum(signum(N))),
            FLYDSL_CORE_ENABLE_RANGE_INFERENCE ? "signum(N){range=[-1,1]}" : "signum(N)");
}

// Expressions computed separately are equal by structure.
TEST(SymbolCategory, StructuralEqualityAndTypeErrors) {
  auto N = n("N");
  EXPECT_EQ(N + i(4), N + i(4));
  EXPECT_NE(N + i(4), N + i(5));
  EXPECT_NE(Sy::getDInt(), Sy::getDInt());

  auto err = N / i(0);
  EXPECT_TRUE(err.isError());
  EXPECT_TRUE(err.isSound());
}

TEST(SymbolCategory, Soundness) {
  EXPECT_TRUE(i(8).isSound());
  EXPECT_TRUE(n("N").isSound());
  EXPECT_TRUE(Sy::getDInt().isSound());
  // A dynamic expression folded to a static value drops the expression.
  EXPECT_TRUE((n("N") * i(0)).isSound());
}

TEST(SymbolCategory, LayoutCompactColumnMajor) {
  auto N = n("N");
  auto layout = make_layout(make_tuple(leaf(N), leaf(4)));
  EXPECT_EQ(toString(layout.shape()), "(N,4)");
  EXPECT_EQ(toString(layout.stride()), "(1,N)");
  EXPECT_TRUE(layout.stride().at(0).getLeaf().isSInt(1));
  EXPECT_EQ(layout.stride().at(1).getLeaf(), N);
  EXPECT_FALSE(layout.isStatic());
}

TEST(SymbolCategory, CompactOrderPutsSymbolsInDynamicClass) {
  auto P = n("P");
  auto Q = n("Q");
  auto layout = make_ordered_layout(make_tuple(leaf(2), leaf(3), leaf(5)),
                                    make_tuple(leaf(P), leaf(Q), leaf(1)));
  EXPECT_EQ(toString(layout), "(2,3,5):(5,10,1)");
}

TEST(SymbolCategory, ExplicitSymbolicStride) {
  auto DM = n("DM");
  auto DN = n("DN");
  auto M = n("M");
  auto N = n("N");
  auto layout = make_layout(make_tuple(leaf(M), leaf(N)), make_tuple(leaf(DM), leaf(DN)));
  EXPECT_EQ(toString(layout), "(M,N):(DM,DN)");
}

TEST(SymbolCategory, NestedAffineIndex) {
  auto tileM = n("tileM");
  auto tm = n("tm");
  auto tileN = n("tileN");
  auto tn = n("tn");
  auto N = n("N");

  auto row = tileM * i(64) + tm;
  auto col = tileN * i(32) + tn;
  auto idx = row * N + col;

  EXPECT_TRUE(row.isSound());
  EXPECT_TRUE(col.isSound());
  EXPECT_TRUE(idx.isSound());
  EXPECT_EQ(exprStr(row), "tileM*64+tm");
  EXPECT_EQ(exprStr(idx), "(tileM*64+tm)*N+(tileN*32+tn)");
  EXPECT_EQ(Leaf<Trait>(idx),
            ((Leaf<Trait>(tileM) * Leaf<Trait>(i(64)) + Leaf<Trait>(tm)) * Leaf<Trait>(N)) +
                (Leaf<Trait>(tileN) * Leaf<Trait>(i(32)) + Leaf<Trait>(tn)));
}

TEST(SymbolCategory, SharedSubexprAndReassociation) {
  auto N = n("N");
  auto M = n("M");
  auto K = n("K");

  auto s = N + i(4);
  auto a = s * M;
  auto b = (N + i(4)) * M;
  EXPECT_EQ(a, b);
  EXPECT_EQ(exprStr(a), "(N+4)*M");

  auto addr = a + K;
  EXPECT_EQ(addr, (N + i(4)) * M + K);
  EXPECT_EQ(exprStr(addr), "(N+4)*M+K");

  auto left = (N + M) + K;
  auto right = N + (M + K);
  EXPECT_NE(left, right);
  EXPECT_EQ(exprStr(left), "N+M+K");
  EXPECT_EQ(exprStr(right), "N+(M+K)");
}

TEST(SymbolCategory, MidPipelineFoldsAndAlign) {
  auto N = n("N");
  auto M = n("M");
  auto K = n("K");

  auto acc = (N + M) * i(2);
  EXPECT_FALSE(acc.isStatic());
  EXPECT_TRUE(acc.isSound());
  auto killed = acc * i(0) + K;
  EXPECT_EQ(killed, K);
  EXPECT_TRUE(killed.isSound());

  auto scaled = (N * i(3)) * i(1);
  EXPECT_EQ(scaled, N * i(3));
  EXPECT_EQ(exprStr(scaled), "N*3");

  // Adding zero preserves both the symbolic expression and known divisibility.
  auto plusZero = N * i(3) + i(0);
  EXPECT_EQ(plusZero, N * i(3));
  EXPECT_EQ(plusZero.divisibility(), 3);
  EXPECT_EQ(exprStr(plusZero), "N*3");

  auto aligned = (N / i(8)) * i(8) + (N % i(8));
  EXPECT_FALSE(aligned.isStatic());
  EXPECT_TRUE(aligned.isSound());
  EXPECT_NE(aligned, N);
  EXPECT_EQ(exprStr(aligned), "N/8*8+N%8");
}

TEST(SymbolCategory, TupleScaleBiasAndInnerProduct) {
  auto N = n("N");
  auto M = n("M");
  auto K = n("K");
  auto shape = make_tuple(leaf(N), leaf(M));
  auto scaled = shape * make_tuple(leaf(2), leaf(4));
  auto biased = scaled + shape;

  EXPECT_TRUE(allSound(biased));
  EXPECT_EQ(exprStr(biased.at(0).getLeaf()), "N*2+N");
  EXPECT_EQ(exprStr(biased.at(1).getLeaf()), "M*4+M");

  auto grouped = group(make_tuple(leaf(N), leaf(4), leaf(M)), 1, 3);
  auto each = product_each(grouped);
  EXPECT_EQ(toString(grouped), "(N,(4,M))");
  EXPECT_EQ(exprStr(each.at(1).getLeaf()), "4*M");

  auto dot = inner_product(shape, make_tuple(leaf(1), leaf(K)));
  EXPECT_TRUE(dot.isLeaf());
  EXPECT_EQ(dot.getLeaf(), N + M * K);
  EXPECT_EQ(exprStr(dot.getLeaf()), "N+M*K");
}

TEST(SymbolCategory, CompactLayoutSizeAndCrd2Idx) {
  auto N = n("N");
  auto M = n("M");
  auto iIdx = n("i");
  auto jIdx = n("j");
  auto layout = make_layout(make_tuple(leaf(N), leaf(M), leaf(4)));

  EXPECT_TRUE(allSound(layout));
  EXPECT_EQ(toString(layout.shape()), "(N,M,4)");
  EXPECT_TRUE(layout.stride().at(0).getLeaf().isSInt(1));
  EXPECT_EQ(layout.stride().at(1).getLeaf(), N);
  EXPECT_EQ(exprStr(layout.stride().at(2).getLeaf()), "N*M");
  EXPECT_EQ(exprStr(size(layout).getLeaf()), "N*M*4");

  auto idx = crd2idx(make_tuple(leaf(iIdx), leaf(jIdx), leaf(2)), layout);
  ASSERT_TRUE(idx.isLeaf());
  EXPECT_TRUE(idx.getLeaf().isSound());
  EXPECT_EQ(exprStr(idx.getLeaf()), "i+j*N+2*(N*M)");

  auto nested = make_layout(make_tuple(make_tuple(leaf(N), leaf(4)), leaf(M)));
  EXPECT_EQ(toString(nested.shape()), "((N,4),M)");
  EXPECT_EQ(exprStr(nested.stride().at(1).getLeaf()), "N*4");
  auto flat = flatten(nested);
  EXPECT_EQ(toString(flat.shape()), "(N,4,M)");
  EXPECT_EQ(flat.stride().at(2).getLeaf(), nested.stride().at(1).getLeaf());
}

TEST(SymbolCategory, CoalesceCompositionSliceAndProduct) {
  auto N = n("N");

  auto coalesced = coalesce(make_layout(make_tuple(leaf(4), leaf(8), leaf(N))));
  EXPECT_TRUE(allSound(coalesced));
  EXPECT_EQ(toString(coalesced.shape()), "(32,N)");
  EXPECT_EQ(toString(coalesced.stride()), "(1,32)");
  EXPECT_EQ(toString(coalesced), "(32,N):(1,32)");

  auto lhs = make_layout(make_tuple(leaf(8), leaf(4)), make_tuple(leaf(1), leaf(N)));
  auto composed = composition(lhs, make_layout(leaf(16)));
  EXPECT_TRUE(allSound(composed));
  EXPECT_EQ(toString(composed), "(8,2):(1,N)");
  auto at = crd2idx(leaf(10), composed);
  ASSERT_TRUE(at.isLeaf());
  EXPECT_EQ(at.getLeaf(), i(2) + N);
  EXPECT_EQ(exprStr(at.getLeaf()), "2+N");
  EXPECT_EQ(at, crd2idx(leaf(10), lhs));

  auto layout = make_layout(make_tuple(leaf(N), leaf(4), leaf(n("M"))));
  auto wildcard = Tup::getNone();
  auto coord = make_tuple(wildcard, leaf(2), wildcard);
  auto sliced = slice(coord, layout);
  EXPECT_TRUE(allSound(sliced));
  EXPECT_EQ(toString(sliced.shape()), "(N,M)");
  EXPECT_EQ(exprStr(sliced.stride().at(1).getLeaf()), "N*4");
  auto [sliced2, offset] = slice_and_offset(coord, layout);
  EXPECT_EQ(sliced2, sliced);
  ASSERT_TRUE(offset.isLeaf());
  EXPECT_EQ(exprStr(offset.getLeaf()), "2*N");

  auto product = logical_product(make_layout(leaf(4)), make_layout(leaf(N)));
  EXPECT_TRUE(allSound(product));
  EXPECT_EQ(toString(product), "(4,N):(1,4)");
}

TEST(SymbolCategory, BitwiseAndExactDivPipeline) {
  auto idx = n("idx");
  auto mixed = ((idx >> i(2)) & i(3)) ^ idx;
  EXPECT_TRUE(mixed.isSound());
  EXPECT_EQ(exprStr(mixed), "idx>>2&3^idx");

  auto D = n("D", 16);
  auto q = exact_div(D, i(4));
  auto r = q * i(3) + i(12);
  auto t = exact_div(r, i(6));
  EXPECT_TRUE(q.isSound());
  EXPECT_TRUE(r.isSound());
  EXPECT_TRUE(t.isSound());
  EXPECT_EQ(q.divisibility(), 4);
  EXPECT_EQ(r.divisibility(), 12);
  EXPECT_EQ(t.divisibility(), 2);
  EXPECT_EQ(exprStr(t), "exact_div(exact_div(D,4)*3+12,6)");
}

TEST(SymbolCategory, PointerOffsetKeepsAddressSymbolic) {
  struct F32 {};
  auto pointer = Pointer<Symbol>(OpaqueValue::get(Identity<F32>{}), 32, 16, "base");
  auto shifted = add_offset(pointer, n("index"));
  EXPECT_EQ(shifted.value(), Sy::getSym("base", LogWidth::I64) + n("index"));
  EXPECT_EQ(shifted.byteAlignment(), 4);

  // Pointers compare by metadata and address structure.
  EXPECT_EQ(shifted, add_offset(pointer, n("index")));
  EXPECT_NE(shifted, add_offset(pointer, n("other")));
  EXPECT_EQ(add_offset(pointer, i(0)), pointer);
}

// Both threads build on one shared expression, retaining and releasing it
// concurrently.
TEST(SymbolCategory, SharedSymbolExpressionsSurviveConcurrentUse) {
  auto shared = Leaf<Symbol>::getSym("shared") * Leaf<Symbol>::getSInt(4);
  auto worker = std::thread([shared] {
    for (int i = 0; i != 5000; ++i)
      (void)(shared + Leaf<Symbol>::getSym("worker-" + std::to_string(i)));
  });
  for (int i = 0; i != 5000; ++i)
    EXPECT_EQ(shared + Leaf<Symbol>::getZero(), shared);
  worker.join();
  EXPECT_EQ(toString(shared), "shared*4{div=4}");
}

// A leaf prints to text that parses back to the same leaf.
TEST(SymbolCategory, TextRoundTrip) {
  auto sum = (n("N") + n("M") * i(3)).withRange(0, 128);
  for (auto leaf :
       {Sy::getRatio(3, 7), Sy::getBasis(sum, {1}), ceil_div(n("N"), i(4)), lshr(n("N"), i(2))}) {
    auto parsed = Sy::fromString(toString(leaf));
    ASSERT_TRUE(parsed.has_value()) << toString(leaf);
    EXPECT_EQ(*parsed, leaf) << toString(leaf);
  }
}

TEST(SymbolCategory, PointerKeepsAddressSpace) {
  auto space = OpaqueValue::get(int64_t{17});
  auto type = Pointer<Trait>(OpaqueValue::getType<float>(), 32, 16, space);
  auto pointer = Pointer<Symbol>(type.elementType(), 32, 16, "address-space", {}, 0, space);
  EXPECT_EQ(add_offset(pointer, i(2)).addressSpace(), space);
  EXPECT_EQ(to_trait(pointer), type);
}

TEST(SymbolCategory, FastDivModMakesAStaticDivisorTheConstant) {
  auto seven = i(7).withFastDivMod();
  ASSERT_TRUE(seven.isDInt());
  EXPECT_TRUE(seven.hasFastDivMod());
  EXPECT_TRUE(seven.isSound());
  EXPECT_EQ(exprStr(seven), "7");
  auto divisor = n("d").withFastDivMod();
  EXPECT_EQ(exprStr(divisor), "d");
  EXPECT_EQ(exprStr(n("x") / divisor), "x/d");
}
