// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include "flydsl/Conversion/FlyToROCDL/FlyToROCDL.h"

using namespace mlir;
using amdgpu::Chipset;

namespace {

/// What the #rocdl.target of the gpu.module says about the code being compiled.
struct Target {
  StringRef chip;
  Chipset chipset;
  /// `fast` or `finite_only`: no NaN reaches a conversion.
  bool finiteOnly;
  /// `fast` or `unsafe_math`: approximate results are allowed.
  bool unsafeMath;
};

enum class Fp8Format { None, Fp8, Bf8 };

/// v_cvt_pk_bf16_f32.
static bool hasBF16ConversionInsts(Chipset chipset) {
  return (chipset.majorVersion == 9 && chipset.minorVersion >= 5) ||
         (chipset.majorVersion == 12 && chipset.minorVersion >= 5);
}

/// Which of the chip's fp8 / bf8 conversion instructions handle `type`.
static Fp8Format nativeFp8Format(Type type, Chipset chipset) {
  if (chipset.majorVersion == 9 && chipset.minorVersion == 4) {
    if (isa<Float8E4M3FNUZType>(type))
      return Fp8Format::Fp8;
    if (isa<Float8E5M2FNUZType>(type))
      return Fp8Format::Bf8;
  } else if (amdgpu::hasOcpFp8(chipset)) {
    if (isa<Float8E4M3FNType>(type))
      return Fp8Format::Fp8;
    if (isa<Float8E5M2Type>(type))
      return Fp8Format::Bf8;
  }
  return Fp8Format::None;
}

/// Float types whose conversions convert-gpu-to-rocdl lowers natively.
static bool isLLVMFloat(Type type) {
  return isa<Float16Type, BFloat16Type, Float32Type, Float64Type>(type);
}

/// `x` was converted exactly from a value bf16 holds: an integer of at most 8 bits, or a float of
/// at most 8 bits or bf16.
static bool isExactInBF16(Value x) {
  Operation *def = x.getDefiningOp();
  if (isa_and_nonnull<arith::SIToFPOp, arith::UIToFPOp>(def))
    return getElementTypeOrSelf(def->getOperand(0).getType()).getIntOrFloatBitWidth() <= 8;
  if (auto ext = dyn_cast_or_null<arith::ExtFOp>(def)) {
    Type src = getElementTypeOrSelf(ext.getIn().getType());
    return src.getIntOrFloatBitWidth() <= 8 || src.isBF16();
  }
  return false;
}

/// `x` was converted from an integer.
static bool isNeverNaN(Value x) {
  return isa_and_nonnull<arith::SIToFPOp, arith::UIToFPOp>(x.getDefiningOp());
}

class ConversionBuilder {
public:
  ConversionBuilder(OpBuilder &b, Location loc) : b(b), loc(loc) {}

  Value i32(int64_t value) {
    return createScalarOrSplatConstant(b, loc, b.getI32Type(), APInt(32, value, /*isSigned=*/true));
  }

  /// `shaped` (a scalar or a vector of any rank) with element type `elementType`.
  static Type withElementType(Type shaped, Type elementType) {
    if (auto vectorType = dyn_cast<VectorType>(shaped))
      return VectorType::get(vectorType.getShape(), elementType);
    return elementType;
  }

  /// The elements of a scalar or of a vector of any rank.
  SmallVector<Value> lanes(Value value) {
    auto vectorType = dyn_cast<VectorType>(value.getType());
    if (!vectorType)
      return {value};
    Value flat = flatten(value);
    SmallVector<Value> result;
    for (int64_t i = 0, e = vectorType.getNumElements(); i < e; ++i)
      result.push_back(vector::ExtractOp::create(b, loc, flat, i));
    return result;
  }

  /// The value of `type` (a scalar or a vector of any rank) made of `elements`.
  Value fromLanes(Type type, ValueRange elements) {
    auto vectorType = dyn_cast<VectorType>(type);
    if (!vectorType)
      return elements.front();
    Value flat = vector::FromElementsOp::create(b, loc, flatType(vectorType), elements);
    return unflatten(vectorType, flat);
  }

  /// The i32 whose high half is the bf16 that `x` (f32) rounds to under `mode`. A NaN gives the
  /// canonical NaN 0x7fc0 unless `noNaNs`; without the check, rounding can carry a NaN into an
  /// infinity or a negative zero.
  Value roundToBF16Bits(Value x, arith::RoundingMode mode, bool noNaNs) {
    Value bits = arith::BitcastOp::create(b, loc, b.getI32Type(), x);
    Value rounded = bits;
    switch (mode) {
    case arith::RoundingMode::to_nearest_even: {
      Value lsb =
          arith::AndIOp::create(b, loc, arith::ShRUIOp::create(b, loc, bits, i32(16)), i32(1));
      rounded =
          arith::AddIOp::create(b, loc, arith::AddIOp::create(b, loc, bits, lsb), i32(0x7fff));
      break;
    }
    case arith::RoundingMode::to_nearest_away:
      rounded = arith::AddIOp::create(b, loc, bits, i32(0x8000));
      break;
    case arith::RoundingMode::toward_zero:
      break;
    case arith::RoundingMode::upward:
    case arith::RoundingMode::downward: {
      // Round the magnitude up for positive values under `upward`, negative ones under `downward`.
      Value negative = arith::ShRSIOp::create(b, loc, bits, i32(31));
      Value up = mode == arith::RoundingMode::upward
                     ? arith::XOrIOp::create(b, loc, negative, i32(-1)).getResult()
                     : negative;
      rounded = arith::AddIOp::create(b, loc, bits, arith::AndIOp::create(b, loc, up, i32(0xffff)));
      break;
    }
    }
    if (noNaNs)
      return rounded;
    Value ordered = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::ORD, x, x);
    return arith::SelectOp::create(b, loc, ordered, rounded, i32(0x7fc00000));
  }

  /// bf16 values made of the high halves of the i32 `bits`, two per v_perm_b32.
  SmallVector<Value> highHalvesAsBF16(ArrayRef<Value> bits) {
    Type i32Type = b.getI32Type();
    auto pairType = VectorType::get({2}, b.getBF16Type());
    SmallVector<Value> result;
    size_t i = 0;
    for (; i + 1 < bits.size(); i += 2) {
      // Bytes 2-3 of bits[i] become the low half, bytes 2-3 of bits[i + 1] the high half.
      Value word =
          LLVM::CallIntrinsicOp::create(b, loc, i32Type, b.getStringAttr("llvm.amdgcn.perm"),
                                        ValueRange{bits[i + 1], bits[i], i32(0x07060302)})
              ->getResult(0);
      Value pair = vector::BitCastOp::create(
          b, loc, pairType,
          vector::FromElementsOp::create(b, loc, VectorType::get({1}, i32Type), word));
      result.push_back(vector::ExtractOp::create(b, loc, pair, 0));
      result.push_back(vector::ExtractOp::create(b, loc, pair, 1));
    }
    if (i < bits.size()) {
      Value high = arith::ShRUIOp::create(b, loc, bits[i], i32(16));
      Value half = arith::TruncIOp::create(b, loc, b.getI16Type(), high);
      result.push_back(arith::BitcastOp::create(b, loc, b.getBF16Type(), half));
    }
    return result;
  }

  /// The f32 that `x` (f64) rounds to under round-to-odd: inexact results get an odd last bit.
  /// Rounding it further to at most 22 significand bits gives the correctly rounded result under
  /// any rounding mode (Boldo and Melquiond, "When double rounding is odd").
  Value roundToOddF32(Value x) {
    // The same sequence as LLVM's TargetLowering::expandRoundInexactToOdd.
    Type i32Type = b.getI32Type();
    Value nearest = arith::TruncFOp::create(b, loc, b.getF32Type(), x);
    Value back = arith::ExtFOp::create(b, loc, b.getF64Type(), nearest);
    Value bits = arith::BitcastOp::create(b, loc, i32Type, nearest);
    Value odd = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ne,
                                      arith::AndIOp::create(b, loc, bits, i32(1)), i32(0));
    // Exact, NaN and odd results stay.
    Value keep = arith::OrIOp::create(
        b, loc, arith::CmpFOp::create(b, loc, arith::CmpFPredicate::UEQ, x, back), odd);
    // An even inexact result moves to its odd neighbour on the side of x.
    Value roundedDown =
        arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OGT, math::AbsFOp::create(b, loc, x),
                              math::AbsFOp::create(b, loc, back));
    Value step = arith::SelectOp::create(b, loc, roundedDown, i32(1), i32(-1));
    Value adjusted = arith::AddIOp::create(b, loc, bits, step);
    Value result = arith::SelectOp::create(b, loc, keep, bits, adjusted);
    return arith::BitcastOp::create(b, loc, b.getF32Type(), result);
  }

  /// `x` rounded to the type of `nearest` under `mode` (not to nearest even), from `nearest`, its
  /// rounding to nearest even: one step in magnitude where `mode` rounds the other way. The narrow
  /// values and the midpoints between them are exact in the type of `x`.
  Value fixUpRounding(Value x, Value nearest, arith::RoundingMode mode) {
    Type narrowType = nearest.getType();
    unsigned width = getElementTypeOrSelf(narrowType).getIntOrFloatBitWidth();
    Type intType = withElementType(narrowType, b.getIntegerType(width));
    auto intConst = [&](int64_t value) {
      return createScalarOrSplatConstant(b, loc, intType, APInt(width, value, /*isSigned=*/true));
    };
    auto cmp = [&](arith::CmpFPredicate predicate, Value lhs, Value rhs) -> Value {
      return arith::CmpFOp::create(b, loc, predicate, lhs, rhs);
    };
    Value back = arith::ExtFOp::create(b, loc, x.getType(), nearest);
    Value bits = arith::BitcastOp::create(b, loc, intType, nearest);
    Value negative = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, bits, intConst(0));
    Value absX = math::AbsFOp::create(b, loc, x), absBack = math::AbsFOp::create(b, loc, back);
    // `step` is added to the bits: +1 grows the magnitude, -1 shrinks it.
    Value adjust, step;
    switch (mode) {
    case arith::RoundingMode::upward:
      adjust = cmp(arith::CmpFPredicate::OLT, back, x);
      step = arith::SelectOp::create(b, loc, negative, intConst(-1), intConst(1));
      break;
    case arith::RoundingMode::downward:
      adjust = cmp(arith::CmpFPredicate::OGT, back, x);
      step = arith::SelectOp::create(b, loc, negative, intConst(1), intConst(-1));
      break;
    case arith::RoundingMode::toward_zero:
      adjust = cmp(arith::CmpFPredicate::OGT, absBack, absX);
      step = intConst(-1);
      break;
    case arith::RoundingMode::to_nearest_away: {
      // A tie that went to the even neighbour nearer zero.
      Value nextBits = arith::AddIOp::create(b, loc, bits, intConst(1));
      Value nextNarrow = arith::BitcastOp::create(b, loc, narrowType, nextBits);
      Value next = arith::ExtFOp::create(b, loc, x.getType(), nextNarrow);
      Value sum = arith::AddFOp::create(b, loc, absBack, math::AbsFOp::create(b, loc, next));
      APFloat half(0.5);
      bool losesInfo;
      half.convert(cast<FloatType>(getElementTypeOrSelf(x.getType())).getFloatSemantics(),
                   APFloat::rmNearestTiesToEven, &losesInfo);
      Value mid = arith::MulFOp::create(b, loc, sum,
                                        createScalarOrSplatConstant(b, loc, x.getType(), half));
      adjust = arith::AndIOp::create(b, loc, cmp(arith::CmpFPredicate::OEQ, absX, mid),
                                     cmp(arith::CmpFPredicate::OLT, absBack, absX));
      step = intConst(1);
      break;
    }
    case arith::RoundingMode::to_nearest_even:
      llvm_unreachable("nothing to fix up");
    }
    Value moved = arith::AddIOp::create(b, loc, bits, step);
    Value result = arith::SelectOp::create(b, loc, adjust, moved, bits);
    return arith::BitcastOp::create(b, loc, narrowType, result);
  }

  /// f16 values that the f32 `x` round to toward zero, two per v_cvt_pkrtz_f16_f32.
  SmallVector<Value> f16TowardZero(ArrayRef<Value> x) {
    auto pairType = VectorType::get({2}, b.getF16Type());
    SmallVector<Value> result;
    for (size_t i = 0; i < x.size(); i += 2) {
      Value second = i + 1 < x.size() ? x[i + 1] : x[i];
      Value pair = ROCDL::CvtPkRtz::create(b, loc, pairType, x[i], second);
      result.push_back(vector::ExtractOp::create(b, loc, pair, 0));
      if (i + 1 < x.size())
        result.push_back(vector::ExtractOp::create(b, loc, pair, 1));
    }
    return result;
  }

  /// The value of `type` (fp8 / bf8 elements) that the f32 `x` convert to, four per i32.
  Value encodeFp8(ArrayRef<Value> x, Fp8Format format, Type type) {
    Type i32Type = b.getI32Type();
    auto pack = [&](Value first, Value second, Value old, bool high) -> Value {
      if (format == Fp8Format::Fp8)
        return ROCDL::CvtPkFp8F32Op::create(b, loc, i32Type, first, second, old, high);
      return ROCDL::CvtPkBf8F32Op::create(b, loc, i32Type, first, second, old, high);
    };
    // Lanes and bytes past the end are left undefined, as in AMDGPUToROCDL.
    Value undefF32 = LLVM::UndefOp::create(b, loc, b.getF32Type());
    SmallVector<Value> words;
    for (size_t i = 0; i < x.size(); i += 4) {
      auto at = [&](size_t j) { return j < x.size() ? x[j] : undefF32; };
      Value word = pack(at(i), at(i + 1), LLVM::UndefOp::create(b, loc, i32Type), /*high=*/false);
      if (i + 2 < x.size())
        word = pack(at(i + 2), at(i + 3), word, /*high=*/true);
      words.push_back(word);
    }
    auto wordsType = VectorType::get({static_cast<int64_t>(words.size())}, i32Type);
    Type elementType = getElementTypeOrSelf(type);
    int64_t count = static_cast<int64_t>(x.size());
    Value bytes = vector::BitCastOp::create(
        b, loc, VectorType::get({4 * wordsType.getNumElements()}, elementType),
        vector::FromElementsOp::create(b, loc, wordsType, words));
    if (4 * wordsType.getNumElements() != count)
      bytes = vector::ExtractStridedSliceOp::create(b, loc, bytes, {0}, {count}, {1});
    auto vectorType = dyn_cast<VectorType>(type);
    if (!vectorType)
      return vector::ExtractOp::create(b, loc, bytes, 0);
    return unflatten(vectorType, bytes);
  }

  /// The f32 values of the fp8 / bf8 elements of `value`, four per i32. With `highWordOnly`, the
  /// low word is shifted into the high one rather than converted in place.
  SmallVector<Value> decodeFp8(Value value, Fp8Format format, bool highWordOnly) {
    Type i32Type = b.getI32Type(), i8Type = b.getI8Type();
    SmallVector<Value> words;
    int64_t count = 1;
    if (auto vectorType = dyn_cast<VectorType>(value.getType())) {
      count = vectorType.getNumElements();
      int64_t padded = (count + 3) / 4 * 4;
      Value bytes =
          vector::BitCastOp::create(b, loc, VectorType::get({count}, i8Type), flatten(value));
      if (padded != count) {
        auto paddedType = VectorType::get({padded}, i8Type);
        Value zeros = createScalarOrSplatConstant(b, loc, paddedType, 0);
        bytes = vector::InsertStridedSliceOp::create(b, loc, bytes, zeros, ArrayRef<int64_t>{0},
                                                     ArrayRef<int64_t>{1});
      }
      Value wordVector =
          vector::BitCastOp::create(b, loc, VectorType::get({padded / 4}, i32Type), bytes);
      words = lanes(wordVector);
    } else {
      Value byte = arith::BitcastOp::create(b, loc, i8Type, value);
      words.push_back(arith::ExtUIOp::create(b, loc, i32Type, byte));
    }
    auto pairType = VectorType::get({2}, b.getF32Type());
    auto unpack = [&](Value word, bool high) -> Value {
      if (highWordOnly && !high) {
        word = arith::ShLIOp::create(b, loc, word, i32(16));
        high = true;
      }
      if (format == Fp8Format::Fp8)
        return ROCDL::CvtPkF32Fp8Op::create(b, loc, pairType, word, high);
      return ROCDL::CvtPkF32Bf8Op::create(b, loc, pairType, word, high);
    };
    SmallVector<Value> result;
    for (auto [index, word] : llvm::enumerate(words)) {
      for (bool high : {false, true}) {
        int64_t first = 4 * static_cast<int64_t>(index) + (high ? 2 : 0);
        if (first >= count)
          break;
        Value pair = unpack(word, high);
        for (int64_t j = 0; j < 2 && first + j < count; ++j)
          result.push_back(vector::ExtractOp::create(b, loc, pair, j));
      }
    }
    return result;
  }

private:
  static VectorType flatType(VectorType vectorType) {
    return VectorType::get({vectorType.getNumElements()}, vectorType.getElementType());
  }

  Value flatten(Value value) {
    auto vectorType = cast<VectorType>(value.getType());
    if (vectorType.getRank() == 1)
      return value;
    return vector::ShapeCastOp::create(b, loc, flatType(vectorType), value);
  }

  Value unflatten(VectorType vectorType, Value flat) {
    if (vectorType.getRank() == 1)
      return flat;
    return vector::ShapeCastOp::create(b, loc, vectorType, flat);
  }

  OpBuilder &b;
  Location loc;
};

class FloatConversionLowering {
public:
  FloatConversionLowering(const Target &target, const llvm::DenseSet<Operation *> &exactToBF16)
      : target(target), exactToBF16(exactToBF16) {}

  LogicalResult lower(Operation *op) {
    if (auto truncf = dyn_cast<arith::TruncFOp>(op))
      return lowerTruncF(truncf);
    if (auto extf = dyn_cast<arith::ExtFOp>(op))
      return lowerExtF(extf);
    if (isa<arith::SIToFPOp, arith::UIToFPOp>(op))
      return lowerIntToFP(op);
    return lowerFPToInt(op);
  }

private:
  bool noNaNs(arith::FastMathFlags flags) const {
    return target.finiteOnly || arith::bitEnumContainsAll(flags, arith::FastMathFlags::nnan);
  }

  bool approx(arith::FastMathFlags flags) const {
    return target.unsafeMath || arith::bitEnumContainsAll(flags, arith::FastMathFlags::afn);
  }

  /// On gfx12+ the VOP1 form of v_cvt_pk_f32_fp8 / bf8, which LLVM picks for the low word, prints
  /// its True16 operand as `vN`, which the AMDGPU assembler rejects (it wants `vN.l`); the VOP3
  /// form used for the high word assembles.
  bool decodesHighWordOnly() const { return target.chipset.majorVersion >= 12; }

  LogicalResult unsupported(Operation *op, Type src, Type dst,
                            std::optional<arith::RoundingMode> mode = std::nullopt) const {
    InFlightDiagnostic diag = op->emitOpError() << "from " << src << " to " << dst;
    if (mode)
      diag << " rounding " << arith::stringifyRoundingMode(*mode);
    return diag << " is not supported on " << target.chip;
  }

  /// f32 values of the float `x` for a further rounding to at most 22 significand bits: exact
  /// from narrower formats, round-to-odd from f64 unless `approximate` allows rounding twice.
  SmallVector<Value> toF32ForRounding(ConversionBuilder &cb, OpBuilder &b, Location loc, Value x,
                                      bool approximate) {
    Type src = getElementTypeOrSelf(x.getType());
    Type f32Type = ConversionBuilder::withElementType(x.getType(), b.getF32Type());
    if (src.isF32())
      return cb.lanes(x);
    if (!src.isF64())
      return cb.lanes(arith::ExtFOp::create(b, loc, f32Type, x));
    if (approximate)
      return cb.lanes(arith::TruncFOp::create(b, loc, f32Type, x));
    SmallVector<Value> result;
    for (Value lane : cb.lanes(x))
      result.push_back(cb.roundToOddF32(lane));
    return result;
  }

  LogicalResult lowerTruncF(arith::TruncFOp op) {
    Type src = getElementTypeOrSelf(op.getIn().getType());
    Type dst = getElementTypeOrSelf(op.getType());
    arith::RoundingMode mode = op.getRoundingmode().value_or(arith::RoundingMode::to_nearest_even);
    bool nearestEven = mode == arith::RoundingMode::to_nearest_even;
    arith::FastMathFlags flags = op.getFastmath().value_or(arith::FastMathFlags::none);
    Fp8Format fp8 = nativeFp8Format(dst, target.chipset);
    if (!dst.isF32() && !dst.isF16() && !dst.isBF16() && fp8 == Fp8Format::None)
      return unsupported(op, src, dst);
    if (fp8 != Fp8Format::None && !nearestEven)
      return unsupported(op, src, dst, mode);

    bool exact = exactToBF16.contains(op);
    bool nativeBF16 =
        dst.isBF16() && hasBF16ConversionInsts(target.chipset) && (nearestEven || exact);
    // One native instruction; from f64 to bf16, LLVM rounds to odd first just as below.
    if ((nearestEven && (dst.isF32() || (src.isF32() && dst.isF16()))) ||
        (nativeBF16 && (src.isF32() || !approx(flags))))
      return success();

    OpBuilder b(op);
    Location loc = op.getLoc();
    ConversionBuilder cb(b, loc);
    Type resultType = op.getType();
    Value result;
    if (dst.isF32()) {
      // From f64, in a mode other than to nearest even.
      result = cb.fixUpRounding(op.getIn(), arith::TruncFOp::create(b, loc, resultType, op.getIn()),
                                mode);
      op.replaceAllUsesWith(result);
      op.erase();
      return success();
    }
    SmallVector<Value> x = toF32ForRounding(cb, b, loc, op.getIn(), approx(flags) && nearestEven);
    if (fp8 != Fp8Format::None) {
      result = cb.encodeFp8(x, fp8, resultType);
    } else if (dst.isF16() && mode == arith::RoundingMode::toward_zero) {
      result = cb.fromLanes(resultType, cb.f16TowardZero(x));
    } else if (dst.isF16() || nativeBF16) {
      Value wide = cb.fromLanes(ConversionBuilder::withElementType(resultType, b.getF32Type()), x);
      Value nearest = arith::TruncFOp::create(b, loc, resultType, wide);
      result = nativeBF16 || nearestEven ? nearest : cb.fixUpRounding(wide, nearest, mode);
    } else {
      bool skipNaNs = noNaNs(flags) || isNeverNaN(op.getIn());
      SmallVector<Value> bits;
      for (Value lane : x)
        bits.push_back(exact ? arith::BitcastOp::create(b, loc, b.getI32Type(), lane).getResult()
                             : cb.roundToBF16Bits(lane, mode, skipNaNs));
      result = cb.fromLanes(resultType, cb.highHalvesAsBF16(bits));
    }
    op.replaceAllUsesWith(result);
    op.erase();
    return success();
  }

  LogicalResult lowerExtF(arith::ExtFOp op) {
    Type src = getElementTypeOrSelf(op.getIn().getType());
    Type dst = getElementTypeOrSelf(op.getType());
    if (isLLVMFloat(src))
      return success();
    Fp8Format fp8 = nativeFp8Format(src, target.chipset);
    if (fp8 == Fp8Format::None)
      return unsupported(op, src, dst);

    OpBuilder b(op);
    Location loc = op.getLoc();
    ConversionBuilder cb(b, loc);
    SmallVector<Value> x = cb.decodeFp8(op.getIn(), fp8, decodesHighWordOnly());
    Type resultType = op.getType();
    Value result;
    if (dst.isBF16()) {
      // Every fp8 / bf8 value is a bf16 value, so the high half of its f32 is exact.
      SmallVector<Value> bits;
      for (Value lane : x)
        bits.push_back(arith::BitcastOp::create(b, loc, b.getI32Type(), lane));
      result = cb.fromLanes(resultType, cb.highHalvesAsBF16(bits));
    } else {
      result = cb.fromLanes(ConversionBuilder::withElementType(resultType, b.getF32Type()), x);
      if (dst.isF64())
        result = arith::ExtFOp::create(b, loc, resultType, result);
      else if (dst.isF16())
        result = arith::TruncFOp::create(b, loc, resultType, result);
    }
    op.replaceAllUsesWith(result);
    op.erase();
    return success();
  }

  LogicalResult lowerIntToFP(Operation *op) {
    Value in = op->getOperand(0);
    Type src = getElementTypeOrSelf(in.getType());
    Type resultType = op->getResult(0).getType();
    Type dst = getElementTypeOrSelf(resultType);
    Fp8Format fp8 = nativeFp8Format(dst, target.chipset);
    bool emulatedBF16 = dst.isBF16() && !hasBF16ConversionInsts(target.chipset);
    if (isLLVMFloat(dst) && !emulatedBF16)
      return success();
    if (!emulatedBF16 && fp8 == Fp8Format::None)
      return unsupported(op, src, dst);

    OpBuilder b(op);
    Location loc = op->getLoc();
    ConversionBuilder cb(b, loc);
    Type f32Type = ConversionBuilder::withElementType(resultType, b.getF32Type());
    Value wide = isa<arith::SIToFPOp>(op)
                     ? arith::SIToFPOp::create(b, loc, f32Type, in).getResult()
                     : arith::UIToFPOp::create(b, loc, f32Type, in).getResult();
    SmallVector<Value> x = cb.lanes(wide);
    Value result;
    if (fp8 != Fp8Format::None) {
      result = cb.encodeFp8(x, fp8, resultType);
    } else {
      bool exact = src.getIntOrFloatBitWidth() <= 8;
      SmallVector<Value> bits;
      for (Value lane : x)
        bits.push_back(exact ? arith::BitcastOp::create(b, loc, b.getI32Type(), lane).getResult()
                             : cb.roundToBF16Bits(lane, arith::RoundingMode::to_nearest_even,
                                                  /*noNaNs=*/true));
      result = cb.fromLanes(resultType, cb.highHalvesAsBF16(bits));
    }
    op->getResult(0).replaceAllUsesWith(result);
    op->erase();
    return success();
  }

  LogicalResult lowerFPToInt(Operation *op) {
    Value in = op->getOperand(0);
    Type src = getElementTypeOrSelf(in.getType());
    Type resultType = op->getResult(0).getType();
    if (isLLVMFloat(src))
      return success();
    Fp8Format fp8 = nativeFp8Format(src, target.chipset);
    if (fp8 == Fp8Format::None)
      return unsupported(op, src, getElementTypeOrSelf(resultType));

    OpBuilder b(op);
    Location loc = op->getLoc();
    ConversionBuilder cb(b, loc);
    Value wide = cb.fromLanes(ConversionBuilder::withElementType(in.getType(), b.getF32Type()),
                              cb.decodeFp8(in, fp8, decodesHighWordOnly()));
    Value result = isa<arith::FPToSIOp>(op)
                       ? arith::FPToSIOp::create(b, loc, resultType, wide).getResult()
                       : arith::FPToUIOp::create(b, loc, resultType, wide).getResult();
    op->getResult(0).replaceAllUsesWith(result);
    op->erase();
    return success();
  }

  const Target &target;
  /// truncf ops to bf16 whose operand bf16 holds exactly.
  const llvm::DenseSet<Operation *> &exactToBF16;
};

static std::optional<Target> getTarget(gpu::GPUModuleOp module) {
  ArrayAttr targets = module.getTargetsAttr();
  if (!targets || targets.size() != 1)
    return std::nullopt;
  auto rocdlTarget = dyn_cast<ROCDL::ROCDLTargetAttr>(targets[0]);
  if (!rocdlTarget)
    return std::nullopt;
  FailureOr<Chipset> chipset = Chipset::parse(rocdlTarget.getChip());
  if (failed(chipset))
    return std::nullopt;
  return Target{rocdlTarget.getChip(), *chipset,
                rocdlTarget.hasFastMath() || rocdlTarget.hasFiniteOnly(),
                rocdlTarget.hasFastMath() || rocdlTarget.hasUnsafeMath()};
}

} // namespace

LogicalResult mlir::fly_rocdl::lowerFloatConversions(gpu::GPUModuleOp module) {
  std::optional<Target> target = getTarget(module);
  if (!target)
    return success();

  SmallVector<Operation *> conversions;
  llvm::DenseSet<Operation *> exactToBF16;
  module.walk([&](Operation *op) {
    if (!isa<arith::TruncFOp, arith::ExtFOp, arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp,
             arith::FPToUIOp>(op))
      return;
    conversions.push_back(op);
    // Read before the defining conversion is rewritten.
    auto truncf = dyn_cast<arith::TruncFOp>(op);
    if (truncf && getElementTypeOrSelf(truncf.getType()).isBF16() && isExactInBF16(truncf.getIn()))
      exactToBF16.insert(op);
  });

  FloatConversionLowering lowering(*target, exactToBF16);
  for (Operation *op : conversions)
    if (failed(lowering.lower(op)))
      return failure();
  return success();
}
