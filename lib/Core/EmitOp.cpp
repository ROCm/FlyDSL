// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#include "flydsl/Core/EmitOp.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"

#include <algorithm>
#include <utility>

namespace mlir::fly::core {

EmitContext::EmitContext(mlir::OpBuilder &builder, mlir::Location location, const EmitHooks *hooks)
    : builder(builder), location(location), hooks(hooks) {}

const EmitHooks &EmitContext::getHooks() const {
  static const EmitHooks defaults;
  return hooks ? *hooks : defaults;
}

namespace emit_op {
namespace {

mlir::Value emitConstant(EmitContext &ctx, int64_t value, unsigned bits) {
  return mlir::arith::ConstantIntOp::create(ctx.builder, ctx.location, value, bits);
}

/// i1 is a predicate (0/1); wider Core integers use signed arithmetic.
mlir::Value castInteger(EmitContext &ctx, mlir::Value value, unsigned bits) {
  auto width = mlir::cast<mlir::IntegerType>(value.getType()).getWidth();
  if (width == bits)
    return value;
  auto type = ctx.builder.getIntegerType(bits);
  if (width > bits)
    return mlir::arith::TruncIOp::create(ctx.builder, ctx.location, type, value);
  if (width == 1)
    return mlir::arith::ExtUIOp::create(ctx.builder, ctx.location, type, value);
  return mlir::arith::ExtSIOp::create(ctx.builder, ctx.location, type, value);
}

/// The binding of `leaf` once its property becomes `type`: a fast-divmod
/// divisor that `type` no longer marks is bound to its divisor.
mlir::Value rebind(const Leaf<EmitOp> &leaf, Leaf<Trait> type) {
  if (!leaf.hasFastDivMod() || type.isStatic() || type.scale().hasFastDivMod())
    return leaf.value();
  return leaf.context()->getHooks().getFastDivModDivisor(*leaf.context(), leaf.value());
}

} // namespace

mlir::Type get_leaf_type(EmitContext &ctx, Leaf<Trait> scale) {
  if (scale.hasFastDivMod())
    return ctx.getHooks().getFastDivModType(ctx, scale.bitWidth());
  return ctx.builder.getIntegerType(scale.bitWidth());
}

} // namespace emit_op

//===----------------------------------------------------------------------===//
// Fast-divmod handles
//===----------------------------------------------------------------------===//

// A divisor d in [1, 2^(N-1)) has the magic numbers of the round-up method of
// Granlund and Montgomery, as CK's MagicDivision uses them: with l = ceil(log2 d),
//
//   m = floor(2^N * (2^l - d) / d) + 1,    q = (umulhi(n, m) + n) >> l
//
// gives q = floor(n / d) for every n in [0, 2^(N-1)), where the add cannot
// carry out. Neither d = 1 nor a power of two needs a special case, so the
// division is branch-free.
//
// The default multiplier divides integers, which is cheap where it usually
// runs: on the host, before a launch. A backend may make the handle another
// way where it runs elsewhere (makeFastDivModF64).

namespace emit_op {
namespace {

/// ceil(log2 d) of the divisor d >= 1, at the width of `d`.
mlir::Value emitCeilLog2(EmitContext &ctx, mlir::Value d) {
  auto &b = ctx.builder;
  auto bits = d.getType().getIntOrFloatBitWidth();
  auto below = mlir::arith::SubIOp::create(b, ctx.location, d, emitConstant(ctx, 1, bits));
  // ctlz(0) is defined, so d = 1 gives 0.
  auto zeros = mlir::LLVM::CountLeadingZerosOp::create(b, ctx.location, d.getType(), below,
                                                       b.getIntegerAttr(b.getI1Type(), 0));
  return mlir::arith::SubIOp::create(b, ctx.location, emitConstant(ctx, bits, bits), zeros);
}

/// The multiplier of the i32 divisor `d` of shift `l`, floor((2^l - d) * 2^32 / d) + 1: one 64-bit
/// division, since 2^l - d < d keeps the quotient within 32 bits.
mlir::Value emitMultiplier32(EmitContext &ctx, mlir::Value d, mlir::Value l) {
  auto &b = ctx.builder;
  auto loc = ctx.location;
  auto i64 = b.getI64Type();
  auto c = [&](int64_t value) { return emitConstant(ctx, value, 64); };
  auto d64 = mlir::arith::ExtUIOp::create(b, loc, i64, d);
  auto power =
      mlir::arith::ShLIOp::create(b, loc, c(1), mlir::arith::ExtUIOp::create(b, loc, i64, l));
  auto rest = mlir::arith::SubIOp::create(b, loc, power, d64);
  auto quotient =
      mlir::arith::DivUIOp::create(b, loc, mlir::arith::ShLIOp::create(b, loc, rest, c(32)), d64);
  return mlir::arith::AddIOp::create(
      b, loc, mlir::arith::TruncIOp::create(b, loc, b.getI32Type(), quotient),
      emitConstant(ctx, 1, 32));
}

/// The same multiplier from an f64 division: the quotient q = floor(2^(32+l) / d), in
/// [2^32, 2^33), has an f64 error below one, so one exact step either way corrects it.
mlir::Value emitMultiplier32F64(EmitContext &ctx, mlir::Value d, mlir::Value l) {
  auto &b = ctx.builder;
  auto loc = ctx.location;
  auto i64 = b.getI64Type();
  auto c = [&](int64_t value) { return emitConstant(ctx, value, 64); };
  auto exponent =
      mlir::arith::AddIOp::create(b, loc, mlir::arith::ExtUIOp::create(b, loc, i64, l), c(32));
  // 2^(32+l) as an integer, and as the f64 of that exponent.
  auto power = mlir::arith::ShLIOp::create(b, loc, c(1), exponent);
  auto powerBits = mlir::arith::ShLIOp::create(
      b, loc, mlir::arith::AddIOp::create(b, loc, exponent, c(1023)), c(52));
  auto powerF = mlir::arith::BitcastOp::create(b, loc, b.getF64Type(), powerBits);
  auto divisorF = mlir::arith::UIToFPOp::create(b, loc, b.getF64Type(), d);
  auto quotientF = mlir::arith::DivFOp::create(b, loc, powerF, divisorF);
  mlir::Value q = mlir::arith::FPToUIOp::create(b, loc, i64, quotientF);
  auto d64 = mlir::arith::ExtUIOp::create(b, loc, i64, d);
  // One step down when q * d > 2^(32+l), then one up when (q + 1) * d <= 2^(32+l).
  auto over = mlir::arith::CmpIOp::create(b, loc, mlir::arith::CmpIPredicate::ugt,
                                          mlir::arith::MulIOp::create(b, loc, q, d64), power);
  q = mlir::arith::SubIOp::create(b, loc, q, mlir::arith::ExtUIOp::create(b, loc, i64, over));
  auto next = mlir::arith::AddIOp::create(b, loc, mlir::arith::MulIOp::create(b, loc, q, d64), d64);
  auto under = mlir::arith::CmpIOp::create(b, loc, mlir::arith::CmpIPredicate::ule, next, power);
  q = mlir::arith::AddIOp::create(b, loc, q, mlir::arith::ExtUIOp::create(b, loc, i64, under));
  // m = q - 2^32 + 1: the low half of q, plus one.
  return mlir::arith::AddIOp::create(
      b, loc, mlir::arith::TruncIOp::create(b, loc, b.getI32Type(), q), emitConstant(ctx, 1, 32));
}

/// The multiplier of the i64 divisor `d` of shift `l`: floor(r * 2^64 / d) + 1
/// with r = 2^l - d < d. That 128-by-64-bit division is done as two
/// 64-by-32-bit quotient digits (Knuth's algorithm D, as divlu in Hacker's
/// Delight), since a 128-bit division is a libcall on the host and a bit
/// loop on the GPU.
mlir::Value emitMultiplier64(EmitContext &ctx, mlir::Value d, mlir::Value l) {
  auto &b = ctx.builder;
  auto loc = ctx.location;
  auto i64 = b.getI64Type();
  auto c = [&](int64_t value) { return emitConstant(ctx, value, 64); };
  auto cmp = [&](mlir::arith::CmpIPredicate predicate, mlir::Value lhs, mlir::Value rhs) {
    return mlir::arith::CmpIOp::create(b, loc, predicate, lhs, rhs);
  };
  auto rest = mlir::arith::SubIOp::create(b, loc, mlir::arith::ShLIOp::create(b, loc, c(1), l), d);
  // Normalize the divisor so that its top bit is set.
  auto norm =
      mlir::LLVM::CountLeadingZerosOp::create(b, loc, i64, d, b.getIntegerAttr(b.getI1Type(), 0));
  auto v = mlir::arith::ShLIOp::create(b, loc, d, norm);
  auto vHigh = mlir::arith::ShRUIOp::create(b, loc, v, c(32));
  auto vLow = mlir::arith::AndIOp::create(b, loc, v, c(0xffffffff));
  auto digitBase = c(int64_t{1} << 32);
  // The quotient digit of `numerator` * 2^32 by v, estimated from the high
  // half of v; the estimate is at most two too large.
  auto tooLarge = [&](mlir::Value q, mlir::Value remainder) {
    auto low = cmp(mlir::arith::CmpIPredicate::ugt, mlir::arith::MulIOp::create(b, loc, q, vLow),
                   mlir::arith::ShLIOp::create(b, loc, remainder, c(32)));
    return mlir::arith::OrIOp::create(b, loc, cmp(mlir::arith::CmpIPredicate::uge, q, digitBase),
                                      low);
  };
  auto digit = [&](mlir::Value numerator) -> mlir::Value {
    mlir::Value q = mlir::arith::DivUIOp::create(b, loc, numerator, vHigh);
    mlir::Value remainder = mlir::arith::SubIOp::create(
        b, loc, numerator, mlir::arith::MulIOp::create(b, loc, q, vHigh));
    auto first = tooLarge(q, remainder);
    q = mlir::arith::SubIOp::create(b, loc, q, mlir::arith::ExtUIOp::create(b, loc, i64, first));
    remainder = mlir::arith::AddIOp::create(
        b, loc, remainder, mlir::arith::SelectOp::create(b, loc, first, vHigh, c(0)));
    auto second = mlir::arith::AndIOp::create(
        b, loc, first,
        mlir::arith::AndIOp::create(b, loc,
                                    cmp(mlir::arith::CmpIPredicate::ult, remainder, digitBase),
                                    tooLarge(q, remainder)));
    return mlir::arith::SubIOp::create(b, loc, q,
                                       mlir::arith::ExtUIOp::create(b, loc, i64, second));
  };
  auto upper = mlir::arith::ShLIOp::create(b, loc, rest, norm);
  auto high = digit(upper);
  auto lower =
      mlir::arith::SubIOp::create(b, loc, mlir::arith::ShLIOp::create(b, loc, upper, c(32)),
                                  mlir::arith::MulIOp::create(b, loc, high, v));
  auto low = digit(lower);
  auto quotient =
      mlir::arith::AddIOp::create(b, loc, mlir::arith::ShLIOp::create(b, loc, high, c(32)), low);
  return mlir::arith::AddIOp::create(b, loc, quotient, c(1));
}

/// The quotient and remainder of `n` in [0, 2^(N-1)) by the divisor `d` of
/// multiplier `m` and shift `s`.
std::pair<mlir::Value, mlir::Value> emitMagicDivMod(EmitContext &ctx, mlir::Value n, mlir::Value d,
                                                    mlir::Value m, mlir::Value s) {
  auto &b = ctx.builder;
  auto loc = ctx.location;
  auto nuw = mlir::arith::IntegerOverflowFlags::nuw;
  auto nsw = mlir::arith::IntegerOverflowFlags::nsw;
  auto high = mlir::arith::MulUIExtendedOp::create(b, loc, n, m).getHigh();
  auto sum = mlir::arith::AddIOp::create(b, loc, high, n);
  sum.setOverflowFlags(nuw);
  auto quotient = mlir::arith::ShRUIOp::create(b, loc, sum, s);
  auto product = mlir::arith::MulIOp::create(b, loc, quotient, d);
  product.setOverflowFlags(nuw | nsw);
  auto remainder = mlir::arith::SubIOp::create(b, loc, n, product);
  remainder.setOverflowFlags(nuw | nsw);
  return {quotient, remainder};
}

mlir::Value extractField(EmitContext &ctx, mlir::Value handle, int64_t index) {
  return mlir::LLVM::ExtractValueOp::create(ctx.builder, ctx.location, handle,
                                            llvm::ArrayRef<int64_t>{index});
}

/// The lowered handle of `divisor`, its multiplier computed by `multiplier`
/// from the divisor and its shift.
mlir::Value buildFastDivMod(EmitContext &ctx, mlir::Value divisor,
                            mlir::Value (*multiplier)(EmitContext &, mlir::Value, mlir::Value)) {
  auto shift = emitCeilLog2(ctx, divisor);
  mlir::Value fields[] = {divisor, multiplier(ctx, divisor, shift), shift};
  mlir::Value handle = mlir::LLVM::UndefOp::create(
      ctx.builder, ctx.location,
      getFastDivModType(*ctx.builder.getContext(), divisor.getType().getIntOrFloatBitWidth()));
  for (auto [index, field] : llvm::enumerate(fields))
    handle = mlir::LLVM::InsertValueOp::create(ctx.builder, ctx.location, handle, field,
                                               llvm::ArrayRef<int64_t>{int64_t(index)});
  return handle;
}

} // namespace

mlir::Value makeFastDivModF64(EmitContext &ctx, mlir::Value divisor) {
  return buildFastDivMod(ctx, divisor, emitMultiplier32F64);
}

} // namespace emit_op

EmitHooks::~EmitHooks() = default;

mlir::Type EmitHooks::getFastDivModType(EmitContext &ctx, unsigned bitWidth) const {
  return emit_op::getFastDivModType(*ctx.builder.getContext(), bitWidth);
}

mlir::Value EmitHooks::makeFastDivMod(EmitContext &ctx, mlir::Value divisor) const {
  if (divisor.getType().getIntOrFloatBitWidth() == 32)
    return emit_op::buildFastDivMod(ctx, divisor, emit_op::emitMultiplier32);
  return emit_op::buildFastDivMod(ctx, divisor, emit_op::emitMultiplier64);
}

mlir::Value EmitHooks::getFastDivModDivisor(EmitContext &ctx, mlir::Value handle) const {
  return emit_op::extractField(ctx, handle, 0);
}

std::pair<mlir::Value, mlir::Value> EmitHooks::fastDivMod(EmitContext &ctx, mlir::Value dividend,
                                                          mlir::Value handle) const {
  return emit_op::emitMagicDivMod(ctx, dividend, emit_op::extractField(ctx, handle, 0),
                                  emit_op::extractField(ctx, handle, 1),
                                  emit_op::extractField(ctx, handle, 2));
}

//===----------------------------------------------------------------------===//
// Leaf<EmitOp>
//===----------------------------------------------------------------------===//

Leaf<EmitOp>::Leaf(LeafProperty property, mlir::Value value, EmitContext *context)
    : LeafProperty(property) {
  auto scale = Leaf<Trait>(property).scale();
  if (scale.isDInt()) {
    if (!value || !context || value.getType() != emit_op::get_leaf_type(*context, scale)) {
      static_cast<LeafProperty &>(*this) = Leaf<Trait>::getError(ErrorCode::InvalidSSAValue);
      return;
    }
    value_ = value;
    context_ = context;
  }
}

Leaf<EmitOp> Leaf<EmitOp>::getBasis(Leaf scale, llvm::ArrayRef<int32_t> modes) {
  auto type = Leaf<Trait>::getBasis(Leaf<Trait>(scale), modes);
  return Leaf(type, emit_op::rebind(scale, type), scale.context_);
}

Leaf<EmitOp> Leaf<EmitOp>::withModes(Leaf proto) const {
  auto type = Leaf<Trait>(*this).withModes(Leaf<Trait>(proto));
  return Leaf(type, emit_op::rebind(*this, type), context_);
}

Leaf<EmitOp> Leaf<EmitOp>::withFastDivMod(EmitContext *context) const {
  if (hasFastDivMod())
    return *this;
  auto type = Leaf<Trait>(*this).withFastDivMod();
  if (type.isError())
    return Leaf(type);
  auto *ctx = context_ ? context_ : context;
  if (!ctx)
    return getError(ErrorCode::MissingEmitContext);
  // A static divisor is materialized to become dynamic.
  auto divisor = isStatic() ? emit_op::emitConstant(*ctx, staticValue(), type.bitWidth()) : value_;
  return Leaf(type, ctx->getHooks().makeFastDivMod(*ctx, divisor), ctx);
}

Leaf<EmitOp> Leaf<EmitOp>::getDyn(mlir::Value value, EmitContext &context, int32_t div) {
  auto type = value ? mlir::dyn_cast<mlir::IntegerType>(value.getType()) : mlir::IntegerType{};
  if (!type || !type.isSignless())
    return getError(ErrorCode::InvalidSSAValue);
  auto width = detail::parseLogWidth(type.getWidth());
  return width ? Leaf(Leaf<Trait>::getDInt(*width, div), value, &context)
               : getError(ErrorCode::InvalidSSAValue);
}

std::optional<Leaf<EmitOp>> Leaf<EmitOp>::fromString(llvm::StringRef text) {
  auto type = Leaf<Trait>::fromString(text);
  if (!type || !type->isStatic())
    return std::nullopt;
  return Leaf(*type);
}

bool Leaf<EmitOp>::isSound() const {
  if (isStatic())
    return !value_ && !context_;
  return context_ && value_ &&
         value_.getType() == emit_op::get_leaf_type(*context_, Leaf<Trait>(*this).scale());
}

Leaf<EmitOp> Leaf<EmitOp>::withRange(int32_t lo, int32_t hi) const {
  if constexpr (!FLYDSL_CORE_ENABLE_RANGE_INFERENCE)
    return *this;
  return Leaf(Leaf<Trait>(*this).withRange(lo, hi), value_, context_);
}

mlir::FailureOr<mlir::Value> Leaf<EmitOp>::materialize(EmitContext &ctx) const {
  if (!isSound())
    return mlir::failure();
  auto type = Leaf<Trait>(*this).scale();
  if (type.isError() || type.isRatio())
    return mlir::failure();
  if (type.isSInt())
    return emit_op::emitConstant(ctx, type.staticValue(), type.bitWidth());
  if (context_ != &ctx || !value_)
    return mlir::failure();
  return value_;
}

//===----------------------------------------------------------------------===//
// Leaf operations
//===----------------------------------------------------------------------===//

namespace emit_op {
namespace {

mlir::Value emitScalar(EmitContext &ctx, Leaf<EmitOp> leaf, unsigned width) {
  auto scale = leaf.scale();
  if (scale.isSInt())
    return emitConstant(ctx, scale.staticValue(), width);
  // A fast-divmod divisor used as an integer is its divisor.
  auto value = scale.hasFastDivMod() ? ctx.getHooks().getFastDivModDivisor(ctx, scale.value())
                                     : scale.value();
  return castInteger(ctx, value, width);
}

/// Whether `a` divides by the handle of `b`: a fast-divmod divisor at least as
/// wide as `a`, which Core leaves in [0, 2^(width-1)).
bool usesFastDivMod(Leaf<EmitOp> a, Leaf<EmitOp> b) {
  auto lhs = a.scale(), rhs = b.scale();
  return rhs.hasFastDivMod() && lhs.isInt() && lhs.bitWidth() <= rhs.bitWidth();
}

/// The quotient and remainder of `a` by the fast-divmod divisor `b`, at the
/// width of `b`.
std::pair<mlir::Value, mlir::Value> emitFastDivMod(EmitContext &ctx, Leaf<EmitOp> a,
                                                   Leaf<EmitOp> b) {
  auto divisor = b.scale();
  return ctx.getHooks().fastDivMod(ctx, emitScalar(ctx, a, divisor.bitWidth()), divisor.value());
}

mlir::Value emitAbs(EmitContext &ctx, mlir::Value value, bool unsignedMagnitude = false) {
  auto zero = emitConstant(ctx, 0, value.getType().getIntOrFloatBitWidth());
  auto neg = mlir::arith::SubIOp::create(ctx.builder, ctx.location, zero, value);
  if (!unsignedMagnitude)
    neg.setOverflowFlags(mlir::arith::IntegerOverflowFlags::nsw);
  return mlir::arith::MaxSIOp::create(ctx.builder, ctx.location, value, neg);
}

/// Euclid's algorithm uses unsigned magnitudes, including abs(INT_MIN).
mlir::Value emitGcd(EmitContext &ctx, mlir::Value a, mlir::Value b) {
  auto &builder = ctx.builder;
  auto loc = ctx.location;
  a = emitAbs(ctx, a, true);
  b = emitAbs(ctx, b, true);
  auto loop = mlir::scf::WhileOp::create(builder, loc, mlir::TypeRange{a.getType(), b.getType()},
                                         mlir::ValueRange{a, b});
  {
    auto guard = mlir::OpBuilder::InsertionGuard(builder);
    auto *before =
        builder.createBlock(&loop.getBefore(), {}, {a.getType(), b.getType()}, {loc, loc});
    auto zero = emitConstant(ctx, 0, a.getType().getIntOrFloatBitWidth());
    auto cond = mlir::arith::CmpIOp::create(builder, loc, mlir::arith::CmpIPredicate::ne,
                                            before->getArgument(1), zero);
    mlir::scf::ConditionOp::create(builder, loc, cond, before->getArguments());
    auto *after = builder.createBlock(&loop.getAfter(), {}, {a.getType(), b.getType()}, {loc, loc});
    auto rem =
        mlir::arith::RemUIOp::create(builder, loc, after->getArgument(0), after->getArgument(1));
    mlir::scf::YieldOp::create(builder, loc, mlir::ValueRange{after->getArgument(1), rem});
  }
  return loop.getResult(0);
}

/// Both operands of an integer operation whose result is `width` bits wide.
/// The operation itself is at least i8 wide.
std::pair<mlir::Value, mlir::Value> emitOperands(EmitContext &ctx, Leaf<EmitOp> a, Leaf<EmitOp> b,
                                                 unsigned width) {
  width = std::max(width, 8u);
  auto lhs = emitScalar(ctx, a, width);
  auto rhs = emitScalar(ctx, b, width);
  return {lhs, rhs};
}

template <class ArithOp>
mlir::Value emitArith(EmitContext &ctx, Leaf<EmitOp> a, Leaf<EmitOp> b, unsigned width) {
  auto [lhs, rhs] = emitOperands(ctx, a, b, width);
  return ArithOp::create(ctx.builder, ctx.location, lhs, rhs);
}

/// Core integer algebra follows signed C++ arithmetic: overflow is outside
/// the valid domain, not wrapping arithmetic with unsound divisibility.
template <class ArithOp>
mlir::Value emitNsw(EmitContext &ctx, Leaf<EmitOp> a, Leaf<EmitOp> b, unsigned width) {
  auto [lhs, rhs] = emitOperands(ctx, a, b, width);
  auto op = ArithOp::create(ctx.builder, ctx.location, lhs, rhs);
  op.setOverflowFlags(mlir::arith::IntegerOverflowFlags::nsw);
  return op;
}

/// Whether `a` divides by `b` without rounding. Trait only lets exact_div
/// through when the quotient is exact; safe_div with a dynamic operand
/// truncates like the rounding divisions, and for all of them a dividend whose
/// divisibility is a multiple of the static divisor divides exactly.
bool dividesExactly(Leaf<EmitOp> a, Leaf<EmitOp> b) {
  auto lhs = a.scale(), rhs = b.scale();
  if (!lhs.isDInt() || !rhs.isSInt() || rhs.staticValue() == 0 || rhs.staticValue() == INT64_MIN)
    return false;
  auto divisor = rhs.staticValue() < 0 ? -rhs.staticValue() : rhs.staticValue();
  return lhs.divisibility() % divisor == 0;
}

/// The ceiling of `a / b`.
mlir::Value emitCeilDiv(EmitContext &ctx, Leaf<EmitOp> a, Leaf<EmitOp> b, unsigned width) {
  if (usesFastDivMod(a, b)) {
    auto [quotient, remainder] = emitFastDivMod(ctx, a, b);
    auto inexact = mlir::arith::CmpIOp::create(
        ctx.builder, ctx.location, mlir::arith::CmpIPredicate::ne, remainder,
        emitConstant(ctx, 0, remainder.getType().getIntOrFloatBitWidth()));
    auto sum = mlir::arith::AddIOp::create(
        ctx.builder, ctx.location, quotient,
        mlir::arith::ExtUIOp::create(ctx.builder, ctx.location, quotient.getType(), inexact));
    sum.setOverflowFlags(mlir::arith::IntegerOverflowFlags::nuw |
                         mlir::arith::IntegerOverflowFlags::nsw);
    return sum;
  }
  auto [lhs, rhs] = emitOperands(ctx, a, b, width);
  // `divsi exact` lowers to a shift or multiply, while ceildivsi expands to
  // a divide plus a compare-and-select fixup.
  if (dividesExactly(a, b))
    return mlir::arith::DivSIOp::create(ctx.builder, ctx.location, lhs, rhs, /*isExact=*/true);
  return mlir::arith::CeilDivSIOp::create(ctx.builder, ctx.location, lhs, rhs);
}

/// `value` shifted left, or arithmetically right, by `amount`.
mlir::Value emitShift(EmitContext &ctx, bool left, mlir::Value value, mlir::Value amount) {
  if (left)
    return mlir::arith::ShLIOp::create(ctx.builder, ctx.location, value, amount);
  return mlir::arith::ShRSIOp::create(ctx.builder, ctx.location, value, amount);
}

/// Comparisons compute at operand width, then zero-extend their i1 result to
/// the integer width prescribed by Trait (currently i32).
mlir::Value emitCompare(EmitContext &ctx, mlir::arith::CmpIPredicate predicate, Leaf<EmitOp> a,
                        Leaf<EmitOp> b) {
  unsigned width = std::max({a.bitWidth(), b.bitWidth(), 8});
  auto lhs = emitScalar(ctx, a, width);
  auto rhs = emitScalar(ctx, b, width);
  return mlir::arith::CmpIOp::create(ctx.builder, ctx.location, predicate, lhs, rhs);
}

//===----------------------------------------------------------------------===//
// Leaf rules
//===----------------------------------------------------------------------===//

// One rule per operation. A rule emits the value of a dynamic result `width`
// bits wide; `apply` casts it to that width. An identity may still change the
// inferred width (e.g. i8 + static 0), so it returns the operand at `width`.

using Rule = mlir::Value (*)(EmitContext &, Leaf<EmitOp>, Leaf<EmitOp>, unsigned);

mlir::Value add(EmitContext &ctx, Leaf<EmitOp> a, Leaf<EmitOp> b, unsigned width) {
  if (b.scale().isZero())
    return emitScalar(ctx, a, width);
  if (a.scale().isZero())
    return emitScalar(ctx, b, width);
  return emitNsw<mlir::arith::AddIOp>(ctx, a, b, width);
}

mlir::Value sub(EmitContext &ctx, Leaf<EmitOp> a, Leaf<EmitOp> b, unsigned width) {
  if (b.scale().isZero())
    return emitScalar(ctx, a, width);
  return emitNsw<mlir::arith::SubIOp>(ctx, a, b, width);
}

mlir::Value mul(EmitContext &ctx, Leaf<EmitOp> a, Leaf<EmitOp> b, unsigned width) {
  auto lhs = a.scale(), rhs = b.scale();
  if (lhs.isRatio() || rhs.isRatio()) {
    // Trait admits a ratio only times a dynamic integer its denominator
    // divides: divide first, as Trait does.
    auto ratio = lhs.isRatio() ? lhs : rhs;
    auto value = lhs.isRatio() ? rhs : lhs;
    auto quotient = safe_div(value, Leaf<EmitOp>::getSInt(ratio.den()));
    return emitScalar(ctx, quotient * Leaf<EmitOp>::getSInt(ratio.num()), width);
  }
  if (rhs.isOne())
    return emitScalar(ctx, a, width);
  if (lhs.isOne())
    return emitScalar(ctx, b, width);
  return emitArith<mlir::arith::MulIOp>(ctx, a, b, width);
}

mlir::Value div(EmitContext &ctx, Leaf<EmitOp> a, Leaf<EmitOp> b, unsigned width) {
  if (b.scale().isOne())
    return emitScalar(ctx, a, width);
  if (usesFastDivMod(a, b))
    return emitFastDivMod(ctx, a, b).first;
  auto [lhs, rhs] = emitOperands(ctx, a, b, width);
  return mlir::arith::DivSIOp::create(ctx.builder, ctx.location, lhs, rhs, dividesExactly(a, b));
}

mlir::Value ceilDiv(EmitContext &ctx, Leaf<EmitOp> a, Leaf<EmitOp> b, unsigned width) {
  if (b.scale().isOne())
    return emitScalar(ctx, a, width);
  return emitCeilDiv(ctx, a, b, width);
}

mlir::Value mod(EmitContext &ctx, Leaf<EmitOp> a, Leaf<EmitOp> b, unsigned width) {
  if (usesFastDivMod(a, b))
    return emitFastDivMod(ctx, a, b).second;
  return emitArith<mlir::arith::RemSIOp>(ctx, a, b, width);
}

mlir::Value exactDiv(EmitContext &ctx, Leaf<EmitOp> a, Leaf<EmitOp> b, unsigned width) {
  if (b.scale().isOne())
    return emitScalar(ctx, a, width);
  auto [lhs, rhs] = emitOperands(ctx, a, b, width);
  return mlir::arith::DivSIOp::create(ctx.builder, ctx.location, lhs, rhs, /*isExact=*/true);
}

mlir::Value shapeDiv(EmitContext &ctx, Leaf<EmitOp> a, Leaf<EmitOp> b, unsigned width) {
  return emitCeilDiv(ctx, a, b, width);
}

mlir::Value minimum(EmitContext &ctx, Leaf<EmitOp> a, Leaf<EmitOp> b, unsigned width) {
  return emitArith<mlir::arith::MinSIOp>(ctx, a, b, width);
}

mlir::Value maximum(EmitContext &ctx, Leaf<EmitOp> a, Leaf<EmitOp> b, unsigned width) {
  return emitArith<mlir::arith::MaxSIOp>(ctx, a, b, width);
}

mlir::Value greatestCommonDivisor(EmitContext &ctx, Leaf<EmitOp> a, Leaf<EmitOp> b,
                                  unsigned width) {
  auto [lhs, rhs] = emitOperands(ctx, a, b, width);
  return emitGcd(ctx, lhs, rhs);
}

mlir::Value bitAnd(EmitContext &ctx, Leaf<EmitOp> a, Leaf<EmitOp> b, unsigned width) {
  return emitArith<mlir::arith::AndIOp>(ctx, a, b, width);
}

mlir::Value bitOr(EmitContext &ctx, Leaf<EmitOp> a, Leaf<EmitOp> b, unsigned width) {
  if (b.scale().isZero())
    return emitScalar(ctx, a, width);
  if (a.scale().isZero())
    return emitScalar(ctx, b, width);
  return emitArith<mlir::arith::OrIOp>(ctx, a, b, width);
}

mlir::Value bitXor(EmitContext &ctx, Leaf<EmitOp> a, Leaf<EmitOp> b, unsigned width) {
  if (b.scale().isZero())
    return emitScalar(ctx, a, width);
  if (a.scale().isZero())
    return emitScalar(ctx, b, width);
  return emitArith<mlir::arith::XOrIOp>(ctx, a, b, width);
}

/// `a` shifted left, or arithmetically right, by `b`; a negative amount
/// shifts the other way.
mlir::Value shift(EmitContext &ctx, bool left, Leaf<EmitOp> a, Leaf<EmitOp> b, unsigned width) {
  auto amount = b.scale();
  if (amount.isZero())
    return emitScalar(ctx, a, width);
  if (amount.isSInt() && amount.staticValue() < 0) {
    left = !left;
    b = Leaf<EmitOp>::getSInt(-amount.staticValue());
  }
  auto [value, bits] = emitOperands(ctx, a, b, width);
  auto result = emitShift(ctx, left, value, bits);
  if (amount.isSInt())
    return result;
  auto zero = emitConstant(ctx, 0, bits.getType().getIntOrFloatBitWidth());
  auto negative = mlir::arith::CmpIOp::create(ctx.builder, ctx.location,
                                              mlir::arith::CmpIPredicate::slt, bits, zero);
  auto magnitude = mlir::arith::SubIOp::create(ctx.builder, ctx.location, zero, bits);
  auto reverse = emitShift(ctx, !left, value, magnitude);
  return mlir::arith::SelectOp::create(ctx.builder, ctx.location, negative, reverse, result);
}

mlir::Value shl(EmitContext &ctx, Leaf<EmitOp> a, Leaf<EmitOp> b, unsigned width) {
  return shift(ctx, /*left=*/true, a, b, width);
}

mlir::Value shr(EmitContext &ctx, Leaf<EmitOp> a, Leaf<EmitOp> b, unsigned width) {
  return shift(ctx, /*left=*/false, a, b, width);
}

/// `a` shifted right logically by `b`, both non-negative.
mlir::Value logicalShr(EmitContext &ctx, Leaf<EmitOp> a, Leaf<EmitOp> b, unsigned width) {
  if (b.scale().isZero())
    return emitScalar(ctx, a, width);
  return emitArith<mlir::arith::ShRUIOp>(ctx, a, b, width);
}

mlir::Value lessThan(EmitContext &ctx, Leaf<EmitOp> a, Leaf<EmitOp> b, unsigned /*width*/) {
  return emitCompare(ctx, mlir::arith::CmpIPredicate::slt, a, b);
}

mlir::Value equal(EmitContext &ctx, Leaf<EmitOp> a, Leaf<EmitOp> b, unsigned /*width*/) {
  return emitCompare(ctx, mlir::arith::CmpIPredicate::eq, a, b);
}

mlir::Value notEqual(EmitContext &ctx, Leaf<EmitOp> a, Leaf<EmitOp> b, unsigned /*width*/) {
  return emitCompare(ctx, mlir::arith::CmpIPredicate::ne, a, b);
}

/// The result `type` of a binary operation on `a` and `b`. Static and error
/// results need no IR; `rule` emits a dynamic one in the context the operands
/// share.
Leaf<EmitOp> apply(Leaf<Trait> type, Rule rule, Leaf<EmitOp> a, Leaf<EmitOp> b) {
  if (type.isError())
    return Leaf<EmitOp>::getError(type.errorInfo());
  if (type.isStatic())
    return Leaf<EmitOp>(type);
  // Trait never computes a fast-divmod divisor: one passes through an
  // identity, keeping its handle.
  if (type.hasFastDivMod())
    return Leaf<Trait>(a) == type ? a : b;
  auto *ctx = a.context() ? a.context() : b.context();
  if (!ctx || (a.context() && b.context() && a.context() != b.context()))
    return Leaf<EmitOp>::getError(ErrorCode::MissingEmitContext);
  auto width = static_cast<unsigned>(type.bitWidth());
  return Leaf<EmitOp>(type, castInteger(*ctx, rule(*ctx, a, b, width), width), ctx);
}

} // namespace
} // namespace emit_op

Leaf<EmitOp> operator+(Leaf<EmitOp> a, Leaf<EmitOp> b) {
  return emit_op::apply(Leaf<Trait>(a) + Leaf<Trait>(b), emit_op::add, a, b);
}

Leaf<EmitOp> operator-(Leaf<EmitOp> a, Leaf<EmitOp> b) {
  return emit_op::apply(Leaf<Trait>(a) - Leaf<Trait>(b), emit_op::sub, a, b);
}

Leaf<EmitOp> operator*(Leaf<EmitOp> a, Leaf<EmitOp> b) {
  return emit_op::apply(Leaf<Trait>(a) * Leaf<Trait>(b), emit_op::mul, a, b);
}

Leaf<EmitOp> operator/(Leaf<EmitOp> a, Leaf<EmitOp> b) {
  return emit_op::apply(Leaf<Trait>(a) / Leaf<Trait>(b), emit_op::div, a, b);
}

Leaf<EmitOp> operator%(Leaf<EmitOp> a, Leaf<EmitOp> b) {
  return emit_op::apply(Leaf<Trait>(a) % Leaf<Trait>(b), emit_op::mod, a, b);
}

Leaf<EmitOp> operator&(Leaf<EmitOp> a, Leaf<EmitOp> b) {
  return emit_op::apply(Leaf<Trait>(a) & Leaf<Trait>(b), emit_op::bitAnd, a, b);
}

Leaf<EmitOp> operator|(Leaf<EmitOp> a, Leaf<EmitOp> b) {
  return emit_op::apply(Leaf<Trait>(a) | Leaf<Trait>(b), emit_op::bitOr, a, b);
}

Leaf<EmitOp> operator^(Leaf<EmitOp> a, Leaf<EmitOp> b) {
  return emit_op::apply(Leaf<Trait>(a) ^ Leaf<Trait>(b), emit_op::bitXor, a, b);
}

Leaf<EmitOp> operator<<(Leaf<EmitOp> a, Leaf<EmitOp> b) {
  return emit_op::apply(Leaf<Trait>(a) << Leaf<Trait>(b), emit_op::shl, a, b);
}

Leaf<EmitOp> operator>>(Leaf<EmitOp> a, Leaf<EmitOp> b) {
  return emit_op::apply(Leaf<Trait>(a) >> Leaf<Trait>(b), emit_op::shr, a, b);
}

Leaf<EmitOp> lshr(Leaf<EmitOp> a, Leaf<EmitOp> b) {
  return emit_op::apply(lshr(Leaf<Trait>(a), Leaf<Trait>(b)), emit_op::logicalShr, a, b);
}

Leaf<EmitOp> operator<(Leaf<EmitOp> a, Leaf<EmitOp> b) {
  return emit_op::apply(Leaf<Trait>(a) < Leaf<Trait>(b), emit_op::lessThan, a, b);
}

Leaf<EmitOp> eq(Leaf<EmitOp> a, Leaf<EmitOp> b) {
  return emit_op::apply(eq(Leaf<Trait>(a), Leaf<Trait>(b)), emit_op::equal, a, b);
}

Leaf<EmitOp> ne(Leaf<EmitOp> a, Leaf<EmitOp> b) {
  return emit_op::apply(ne(Leaf<Trait>(a), Leaf<Trait>(b)), emit_op::notEqual, a, b);
}

Leaf<EmitOp> lt(Leaf<EmitOp> a, Leaf<EmitOp> b) {
  return emit_op::apply(lt(Leaf<Trait>(a), Leaf<Trait>(b)), emit_op::lessThan, a, b);
}

Leaf<EmitOp> min(Leaf<EmitOp> a, Leaf<EmitOp> b) {
  return emit_op::apply(min(Leaf<Trait>(a), Leaf<Trait>(b)), emit_op::minimum, a, b);
}

Leaf<EmitOp> max(Leaf<EmitOp> a, Leaf<EmitOp> b) {
  return emit_op::apply(max(Leaf<Trait>(a), Leaf<Trait>(b)), emit_op::maximum, a, b);
}

Leaf<EmitOp> gcd(Leaf<EmitOp> a, Leaf<EmitOp> b) {
  return emit_op::apply(gcd(Leaf<Trait>(a), Leaf<Trait>(b)), emit_op::greatestCommonDivisor, a, b);
}

Leaf<EmitOp> exact_div(Leaf<EmitOp> a, Leaf<EmitOp> b) {
  return emit_op::apply(exact_div(Leaf<Trait>(a), Leaf<Trait>(b)), emit_op::exactDiv, a, b);
}

// With a dynamic operand, safe_div is the truncating quotient.
Leaf<EmitOp> safe_div(Leaf<EmitOp> a, Leaf<EmitOp> b) {
  return emit_op::apply(safe_div(Leaf<Trait>(a), Leaf<Trait>(b)), emit_op::div, a, b);
}

Leaf<EmitOp> ceil_div(Leaf<EmitOp> a, Leaf<EmitOp> b) {
  return emit_op::apply(ceil_div(Leaf<Trait>(a), Leaf<Trait>(b)), emit_op::ceilDiv, a, b);
}

Leaf<EmitOp> shape_div(Leaf<EmitOp> a, Leaf<EmitOp> b) {
  return emit_op::apply(shape_div(Leaf<Trait>(a), Leaf<Trait>(b)), emit_op::shapeDiv, a, b);
}

Leaf<EmitOp> operator-(Leaf<EmitOp> a) { return a * Leaf<EmitOp>::getSInt(-1); }

Leaf<EmitOp> operator~(Leaf<EmitOp> a) { return a ^ Leaf<EmitOp>::getSInt(-1); }

Leaf<EmitOp> abs(Leaf<EmitOp> a) {
  auto type = abs(Leaf<Trait>(a));
  if (type.isStatic())
    return Leaf<EmitOp>(type);
  // A fast-divmod divisor is positive.
  if (type.hasFastDivMod())
    return a;
  auto *ctx = a.context();
  if (!ctx)
    return Leaf<EmitOp>::getError(ErrorCode::MissingEmitContext);
  if (a.bitWidth() == 1)
    return Leaf<EmitOp>(type, a.value(), ctx);
  return Leaf<EmitOp>(type, emit_op::emitAbs(*ctx, a.value()), ctx);
}

Leaf<EmitOp> signum(Leaf<EmitOp> a) {
  auto type = signum(Leaf<Trait>(a));
  if (type.isStatic())
    return Leaf<EmitOp>(type);
  auto *ctx = a.context();
  if (!ctx)
    return Leaf<EmitOp>::getError(ErrorCode::MissingEmitContext);
  auto &b = ctx->builder;
  if (a.bitWidth() == 1)
    return Leaf<EmitOp>(type, emit_op::castInteger(*ctx, a.value(), type.bitWidth()), ctx);
  auto zero = emit_op::emitConstant(*ctx, 0, a.bitWidth());
  auto positive = mlir::arith::CmpIOp::create(b, ctx->location, mlir::arith::CmpIPredicate::sgt,
                                              a.value(), zero);
  auto negative = mlir::arith::CmpIOp::create(b, ctx->location, mlir::arith::CmpIPredicate::slt,
                                              a.value(), zero);
  auto value = mlir::arith::SubIOp::create(b, ctx->location,
                                           emit_op::castInteger(*ctx, positive, type.bitWidth()),
                                           emit_op::castInteger(*ctx, negative, type.bitWidth()));
  return Leaf<EmitOp>(type, value, ctx);
}

/// The SSA binding is not printed, so only static leaves round-trip.
void print(Leaf<EmitOp> leaf, llvm::raw_ostream &os) { print(Leaf<Trait>(leaf), os); }

//===----------------------------------------------------------------------===//
// Pointer<EmitOp>
//===----------------------------------------------------------------------===//

Pointer<EmitOp>::Pointer(OpaqueValue elemType, int32_t elemBitWidth, int32_t byteAlignment,
                         mlir::Value value, Swizzle<EmitOp> swizzle, int64_t bitAlignment,
                         OpaqueValue addressSpace, int32_t storageElementBitWidth)
    : PointerProperty(std::move(elemType), elemBitWidth, byteAlignment,
                      bitAlignment == 0 ? int64_t{byteAlignment} * 8 : bitAlignment, swizzle,
                      std::move(addressSpace), storageElementBitWidth),
      value_(value) {}

Pointer<EmitOp>::Pointer(PointerProperty property, mlir::Value value, EmitContext &context)
    : PointerProperty(std::move(property)), value_(value), context_(&context) {
  if (value && value.getContext() != context.builder.getContext())
    value_ = {};
}

bool operator==(const Pointer<EmitOp> &lhs, const Pointer<EmitOp> &rhs) {
  return static_cast<const PointerProperty &>(lhs) == static_cast<const PointerProperty &>(rhs) &&
         lhs.value() == rhs.value();
}

mlir::FailureOr<mlir::Value> EmitHooks::addOffset(EmitContext &ctx, const Pointer<EmitOp> &pointer,
                                                  IntTupleRef<EmitOp> offset,
                                                  const Pointer<Trait> &) const {
  if (!mlir::isa<mlir::LLVM::LLVMPointerType>(pointer.value().getType()) ||
      pointer.storageElementBitWidth() % 8 != 0)
    return mlir::failure();
  auto index = emit_op::emitScalar(ctx, offset.getLeaf(), 64);
  auto bytes = emit_op::emitConstant(ctx, pointer.storageElementBitWidth() / 8, 64);
  index = mlir::arith::MulIOp::create(ctx.builder, ctx.location, index, bytes);
  return mlir::LLVM::GEPOp::create(ctx.builder, ctx.location, pointer.value().getType(),
                                   ctx.builder.getI8Type(), pointer.value(),
                                   mlir::ValueRange{index})
      .getResult();
}

Pointer<EmitOp> add_offset(const Pointer<EmitOp> &pointer, IntTupleRef<EmitOp> offsetTuple) {
  auto offset = offsetTuple.isLeaf() ? offsetTuple.getLeaf()
                                     : Leaf<EmitOp>::getError(ErrorCode::ExpectedLeafOperand);
  auto *ctx = pointer.context();
  if (!ctx)
    ctx = offset.context();
  auto fail = [&](llvm::StringRef message) {
    if (ctx)
      mlir::emitError(ctx->location, message);
    return Pointer<EmitOp>(pointer.elementType(), pointer.elementBitWidth(),
                           pointer.byteAlignment(), {}, pointer.swizzle(),
                           pointer.storageBitAlignment(), pointer.addressSpace(),
                           pointer.storageElementBitWidth());
  };
  if (!pointer.isValid() || !offset.isInt() || offset.isNone() ||
      (pointer.context() && offset.context() && pointer.context() != offset.context()))
    return fail("invalid pointer/offset binding");
  if (offset.isZero())
    return pointer;
  if (!ctx)
    return fail("pointer arithmetic requires an EmitContext");
  auto typedOffset = IntTuple<EmitOp>::fromLeaf(offset);
  auto resultProfile = add_offset(Pointer<Trait>(pointer), Leaf<Trait>(offset));
  auto address = ctx->getHooks().addOffset(*ctx, pointer, typedOffset.asRef(), resultProfile);
  if (mlir::failed(address))
    return fail("pointer offset emitter failed");
  if (!*address || (*address).getContext() != ctx->builder.getContext())
    return fail("pointer offset emitter returned an invalid SSA value");
  return Pointer<EmitOp>(resultProfile, *address, *ctx);
}

//===----------------------------------------------------------------------===//
// LLVM representation
//===----------------------------------------------------------------------===//

namespace emit_op {

mlir::LLVM::LLVMStructType getLLVMStructType(mlir::MLIRContext &context,
                                             llvm::ArrayRef<mlir::Type> fields) {
  auto present = llvm::SmallVector<mlir::Type, 8>{};
  for (auto field : fields)
    if (field)
      present.push_back(field);
  if (present.empty())
    return {};
  return mlir::LLVM::LLVMStructType::getLiteral(&context, present, /*isPacked=*/true);
}

mlir::LLVM::LLVMStructType getFastDivModType(mlir::MLIRContext &context, unsigned bitWidth) {
  auto field = mlir::IntegerType::get(&context, bitWidth);
  return mlir::LLVM::LLVMStructType::getLiteral(&context, {field, field, field});
}

mlir::Value buildLLVMStruct(EmitContext &ctx, llvm::ArrayRef<mlir::Value> parts) {
  auto fields = llvm::SmallVector<mlir::Value, 8>{};
  auto types = llvm::SmallVector<mlir::Type, 8>{};
  for (auto part : parts) {
    if (part) {
      fields.push_back(part);
      types.push_back(part.getType());
    }
  }
  if (fields.empty())
    return {};
  auto type = getLLVMStructType(*ctx.builder.getContext(), types);
  mlir::Value aggregate = mlir::LLVM::UndefOp::create(ctx.builder, ctx.location, type);
  for (auto [index, field] : llvm::enumerate(fields))
    aggregate =
        mlir::LLVM::InsertValueOp::create(ctx.builder, ctx.location, aggregate, field,
                                          llvm::ArrayRef<int64_t>{static_cast<int64_t>(index)});
  return aggregate;
}

mlir::Value extractLLVMField(EmitContext &ctx, mlir::Value aggregate,
                             llvm::ArrayRef<int64_t> path) {
  return ctx.builder.createOrFold<mlir::LLVM::ExtractValueOp>(ctx.location, aggregate, path);
}

} // namespace emit_op
} // namespace mlir::fly::core
