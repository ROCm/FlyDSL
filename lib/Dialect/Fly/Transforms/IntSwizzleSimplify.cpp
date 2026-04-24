// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

#include <functional>
#include <numeric>

using namespace mlir;
using namespace mlir::fly;

namespace mlir {
namespace fly {
#define GEN_PASS_DEF_FLYINTSWIZZLESIMPLIFYPASS
#include "flydsl/Dialect/Fly/Transforms/Passes.h.inc"
} // namespace fly
} // namespace mlir

namespace {

//===----------------------------------------------------------------------===//
// Divisibility lattice over SSA values.
//
// For each value we take max() of two evidences:
//   * forward, from the defining arith op (gcd / sat-mul / shl);
//   * backward, from any Make*Op user that carries a static
//     IntAttr.divisibility annotation on its result IntTupleAttr.
//===----------------------------------------------------------------------===//

class Divisibility {
public:
  int get(Value v);

private:
  static int satMul(int a, int b) {
    if (a == 0 || b == 0)
      return 0;
    return (a > INT32_MAX / b) ? INT32_MAX : a * b;
  }

  int fromDef(Value v);
  int fromUses(Value v);
  static int fromMakeOpUser(Operation *user, unsigned operandIdx);

  DenseMap<Value, int> cache;
  llvm::SmallPtrSet<Value, 16> visiting;
};

int Divisibility::get(Value v) {
  if (auto c = getConstantIntValue(v)) {
    int64_t abs_c = *c == INT64_MIN ? INT64_MAX : std::abs(*c);
    return abs_c == 0 ? INT32_MAX : (int)std::min(abs_c, (int64_t)INT32_MAX);
  }
  if (auto it = cache.find(v); it != cache.end())
    return it->second;
  if (!visiting.insert(v).second)
    return 1; // conservative on cycles
  int d = std::max(fromDef(v), fromUses(v));
  visiting.erase(v);
  return cache[v] = d;
}

int Divisibility::fromDef(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return 1;
  return llvm::TypeSwitch<Operation *, int>(def)
      .Case<arith::AddIOp>([&](auto op) { return std::gcd(get(op.getLhs()), get(op.getRhs())); })
      .Case<arith::MulIOp>([&](auto op) { return satMul(get(op.getLhs()), get(op.getRhs())); })
      .Case<arith::ShLIOp>([&](auto op) {
        int lhs = get(op.getLhs());
        auto k = getConstantIntValue(op.getRhs());
        return (k && *k >= 0 && *k < 31) ? satMul(lhs, 1 << *k) : lhs;
      })
      .Case<arith::SelectOp>(
          [&](auto op) { return std::gcd(get(op.getTrueValue()), get(op.getFalseValue())); })
      .Default(1);
}

int Divisibility::fromUses(Value v) {
  int best = 1;
  for (OpOperand &use : v.getUses())
    best = std::max(best, fromMakeOpUser(use.getOwner(), use.getOperandNumber()));
  return best;
}

int Divisibility::fromMakeOpUser(Operation *user, unsigned operandIdx) {
  auto resTy = llvm::TypeSwitch<Operation *, IntTupleType>(user)
                   .Case<MakeIntTupleOp, MakeShapeOp, MakeStrideOp, MakeCoordOp>(
                       [](auto op) { return cast<IntTupleType>(op.getType()); })
                   .Default(IntTupleType());
  if (!resTy)
    return 1;

  // DFS the IntTupleAttr to find the operandIdx-th dynamic leaf.
  unsigned cursor = 0;
  int result = 1;
  std::function<bool(IntTupleAttr)> visit = [&](IntTupleAttr a) -> bool {
    if (a.isLeaf()) {
      auto leaf = a.extractIntFromLeaf();
      if (!leaf || leaf.getStaticFlag())
        return false;
      if (cursor++ == operandIdx) {
        int32_t d = leaf.getDivisibility();
        result = d > 0 ? (int)d : 1;
        return true;
      }
      return false;
    }
    auto arr = dyn_cast<ArrayAttr>(a.getValue());
    if (!arr)
      return false;
    for (int i = 0, e = arr.size(); i < e; ++i)
      if (visit(a.at(i)))
        return true;
    return false;
  };
  visit(resTy.getAttr());
  return result;
}

//===------------------------------------------------------------------------------------------===//
// Peel period-aligned summands out of the swizzle's input:
//
//   swizzle(base + Σ aᵢ)  →  swizzle(base) + Σ aᵢ
//
// when each aᵢ ≡ 0 (mod period).  When `base` drops out entirely the swizzle of 0 folds to 0.
//===------------------------------------------------------------------------------------------===//

struct PeelResult {
  Value base;
  SmallVector<Value> offsets;
};

PeelResult peelByPeriod(Value v, int period, Divisibility &div) {
  if (period <= 0)
    return {v, {}};
  auto period_multiple = [&](Value x) { return div.get(x) % period == 0; };

  if (auto add = v.getDefiningOp<arith::AddIOp>()) {
    if (period_multiple(add.getRhs())) {
      auto rec = peelByPeriod(add.getLhs(), period, div);
      rec.offsets.push_back(add.getRhs());
      return rec;
    }
    if (period_multiple(add.getLhs())) {
      auto rec = peelByPeriod(add.getRhs(), period, div);
      rec.offsets.push_back(add.getLhs());
      return rec;
    }
    return {v, {}};
  }
  if (period_multiple(v))
    return {/*base=*/Value(), {v}};
  return {v, {}};
}

//===----------------------------------------------------------------------===//
// Pass.
//===----------------------------------------------------------------------===//

class FlyIntSwizzleSimplifyPass
    : public mlir::fly::impl::FlyIntSwizzleSimplifyPassBase<FlyIntSwizzleSimplifyPass> {
public:
  using mlir::fly::impl::FlyIntSwizzleSimplifyPassBase<
      FlyIntSwizzleSimplifyPass>::FlyIntSwizzleSimplifyPassBase;

  void runOnOperation() override {
    auto module = getOperation();
    module->walk([&](gpu::GPUFuncOp fn) { simplifyInFunc(fn); });
  }

  void simplifyInFunc(gpu::GPUFuncOp fn) {
    Divisibility div;

    // Collect candidate xori ops (with their period) up-front so we can rewrite
    // without invalidating the walk.
    SmallVector<std::pair<arith::XOrIOp, int>> candidates;

    fn.walk([&](arith::XOrIOp xori) {
      auto shrui = xori.getRhs().getDefiningOp<arith::ShRUIOp>();
      if (!shrui)
        return;
      auto andi = shrui.getLhs().getDefiningOp<arith::AndIOp>();
      if (!andi)
        return;
      Value x = xori.getLhs();
      if (andi.getLhs() != x)
        return;
      auto maskC = getConstantIntValue(andi.getRhs());
      auto shiftC = getConstantIntValue(shrui.getRhs());
      if (!maskC || !shiftC)
        return;
      uint64_t mask = (uint64_t)*maskC;
      if (mask == 0)
        return;
      uint64_t low = mask & -mask;
      uint64_t plus = mask + low;
      if ((mask & plus) != 0)
        return;
      int K = llvm::countr_zero(mask);
      int M = llvm::popcount(mask);
      if ((int)*shiftC > K)
        return;
      candidates.push_back({xori, 1 << (M + K)});
    });

    for (auto [xori, period] : candidates) {
      Value input = xori.getLhs();
      auto peeled = peelByPeriod(input, period, div);
      if (peeled.offsets.empty())
        continue;

      OpBuilder b(xori);
      Location loc = xori.getLoc();
      Type ty = xori.getType();

      auto shrui = cast<arith::ShRUIOp>(xori.getRhs().getDefiningOp());
      auto andi = cast<arith::AndIOp>(shrui.getLhs().getDefiningOp());
      Value mask = andi.getRhs();
      Value shiftC = shrui.getRhs();

      Value cur;
      if (peeled.base) {
        Value masked = arith::AndIOp::create(b, loc, peeled.base, mask);
        Value shifted = arith::ShRUIOp::create(b, loc, masked, shiftC);
        cur = arith::XOrIOp::create(b, loc, peeled.base, shifted);
      } else {
        cur = arith::ConstantIntOp::create(b, loc, ty, 0);
      }
      for (Value v : peeled.offsets)
        cur = arith::AddIOp::create(b, loc, cur, v);

      xori.replaceAllUsesWith(cur);
      xori.erase();
    }
  }
};

} // namespace
