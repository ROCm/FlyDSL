// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

#include <optional>

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Transforms/Passes.h"
#include "flydsl/Dialect/Fly/Utils/AtomUtils.h"
#include "flydsl/Dialect/Fly/Utils/CoreBridge.h"
#include "flydsl/Dialect/Fly/Utils/LegacyLayout.h"

#include <string>

using namespace mlir;
using namespace mlir::fly;

namespace mlir {
namespace fly {
#define GEN_PASS_DEF_FLYLAYOUTLOWERINGPASS
#include "flydsl/Dialect/Fly/Transforms/Passes.h.inc"
} // namespace fly
} // namespace mlir

namespace {

using IntTupleTrait = core::IntTuple<core::Trait>;

/// Static value of a single-leaf tuple.
int64_t staticLeafValue(const IntTupleTrait &tuple) { return tuple.getLeaf().staticValue(); }

/// The modes of flat `flatShape` other than `skip`, as a tuple (or that mode alone).
IntTupleTrait restFlatShape(const IntTupleTrait &flatShape, int32_t flatRank, int32_t skip) {
  SmallVector<IntTupleTrait> elems;
  for (int32_t i = 0; i < flatRank; ++i)
    if (i != skip)
      elems.push_back(flatShape.isLeaf() ? flatShape : IntTupleTrait(flatShape.at(i)));
  return elems.size() == 1 ? elems.front() : core::make_tuple(elems);
}

/// The hierarchical coordinate of chunk `chunk`: the flat coordinate over the
/// non-contiguous modes, with 0 in the contiguous mode `contigIdx`.
IntTupleAttr chunkCoordAttr(MLIRContext *ctx, const IntTupleTrait &shape,
                            const IntTupleTrait &restShape, int32_t flatRank, int32_t contigIdx,
                            int64_t chunk) {
  IntTupleTrait restCoord = core::idx2crd(IntTupleTrait::getSInt(chunk), restShape);
  SmallVector<IntTupleTrait> flatCoord;
  int32_t restIdx = 0;
  for (int32_t j = 0; j < flatRank; ++j) {
    if (j == contigIdx) {
      flatCoord.push_back(IntTupleTrait::getZero());
      continue;
    }
    flatCoord.push_back(flatRank == 2 || restCoord.isLeaf() ? restCoord
                                                            : IntTupleTrait(restCoord.at(restIdx)));
    ++restIdx;
  }
  auto coord = flatCoord.size() == 1 ? flatCoord.front() : core::make_tuple(flatCoord);
  return IntTupleAttr::get(ctx, core::unflatten(coord, shape));
}

/// Depth limit for the narrow-value proof walk; layout index chains are short.
constexpr unsigned kMaxNarrowProofDepth = 16;

Value castPrintfArg(PatternRewriter &rewriter, Location loc, Value value, std::string &format) {
  Type type = value.getType();
  if (isa<IndexType>(type)) {
    format += "%ld";
    return arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), value);
  }
  if (auto intTy = dyn_cast<IntegerType>(type)) {
    if (intTy.getWidth() <= 32) {
      format += "%d";
      if (intTy.getWidth() < 32) {
        return arith::ExtSIOp::create(rewriter, loc, rewriter.getI32Type(), value);
      }
      return value;
    }
    format += "%ld";
    if (intTy.getWidth() != 64) {
      return arith::ExtSIOp::create(rewriter, loc, rewriter.getI64Type(), value);
    }
    return value;
  }
  if (auto floatTy = dyn_cast<FloatType>(type)) {
    if (floatTy.getWidth() <= 32) {
      format += "%.2f";
      if (floatTy.getWidth() < 32) {
        return arith::ExtFOp::create(rewriter, loc, rewriter.getF32Type(), value);
      }
      return value;
    }
    format += "%.2lf";
    if (floatTy.getWidth() != 64) {
      return arith::ExtFOp::create(rewriter, loc, rewriter.getF64Type(), value);
    }
    return value;
  }
  return nullptr;
}

Value castVectorElementPrintfArg(PatternRewriter &rewriter, Location loc, Value value,
                                 std::string &format) {
  Type type = value.getType();
  if (auto intTy = dyn_cast<IntegerType>(type)) {
    format += "%d";
    if (intTy.getWidth() < 32)
      return arith::ExtSIOp::create(rewriter, loc, rewriter.getI32Type(), value);
    if (intTy.getWidth() > 32)
      return arith::TruncIOp::create(rewriter, loc, rewriter.getI32Type(), value);
    return value;
  }
  if (auto floatTy = dyn_cast<FloatType>(type)) {
    format += "%.2f";
    if (floatTy.getWidth() < 32)
      return arith::ExtFOp::create(rewriter, loc, rewriter.getF32Type(), value);
    if (floatTy.getWidth() > 32)
      return arith::TruncFOp::create(rewriter, loc, rewriter.getF32Type(), value);
    return value;
  }
  return nullptr;
}

bool appendScalarPrintfArg(PatternRewriter &rewriter, Location loc, Value value,
                           std::string &format, SmallVectorImpl<Value> &args) {
  Value casted = castPrintfArg(rewriter, loc, value, format);
  if (!casted) {
    return false;
  }
  args.push_back(casted);
  return true;
}

bool appendVectorPrintf(PatternRewriter &rewriter, Location loc, Value value, std::string &format,
                        SmallVectorImpl<Value> &args) {
  auto vectorTy = dyn_cast<VectorType>(value.getType());
  if (!vectorTy || vectorTy.getRank() != 1 || vectorTy.isScalable() || !vectorTy.hasStaticShape())
    return false;

  format += "[";
  for (int64_t i = 0, e = vectorTy.getDimSize(0); i < e; ++i) {
    if (i > 0)
      format += ", ";
    Value element = vector::ExtractOp::create(rewriter, loc, value, i);
    Value casted = castVectorElementPrintfArg(rewriter, loc, element, format);
    if (!casted)
      return false;
    args.push_back(casted);
  }
  format += "]";
  return true;
}

bool appendIntTuplePrintf(PatternRewriter &rewriter, core::IntTupleRef<core::EmitOp> tuple,
                          core::EmitContext &ctx, std::string &format,
                          SmallVectorImpl<Value> &args) {
  if (tuple.isLeaf()) {
    auto leaf = tuple.getLeaf();
    auto scale = leaf.scale();
    if (leaf.isNone())
      format += "*";
    else if (scale.isRatio())
      format += std::to_string(scale.num()) + "/" + std::to_string(scale.den());
    else if (scale.isStatic())
      format += std::to_string(scale.staticValue());
    else {
      auto value = scale.materialize(ctx);
      if (failed(value) || !appendScalarPrintfArg(rewriter, ctx.location, *value, format, args))
        return false;
    }
    if (leaf.isBasis())
      for (unsigned i = 0; i < leaf.modeCount(); ++i)
        format += "E" + std::to_string(leaf.mode(i));
    return !leaf.isError();
  }
  format += "(";
  for (int i = 0; i < tuple.rank(); ++i) {
    if (i)
      format += ",";
    if (!appendIntTuplePrintf(rewriter, tuple.at(i), ctx, format, args))
      return false;
  }
  format += ")";
  return true;
}

bool appendLayoutPrintf(PatternRewriter &rewriter, const core::AnyLayout<core::EmitOp> &layout,
                        core::EmitContext &ctx, std::string &format, SmallVectorImpl<Value> &args);

/// A swizzle has no dynamic part: it prints as in a composed layout's type.
template <class AttrT> void appendAttrPrintf(AttrT attr, std::string &format) {
  std::string text;
  llvm::raw_string_ostream os(text);
  Attribute(attr).print(os);
  auto prefix = ("#fly." + AttrT::getMnemonic()).str();
  format += StringRef(text).drop_front(prefix.size()).str();
}

/// A composed layout prints as CuTe does: `inner o offset o outer`.
bool appendComposedPrintf(PatternRewriter &rewriter,
                          const core::ComposedLayout<core::EmitOp> &value, core::EmitContext &ctx,
                          std::string &format, SmallVectorImpl<Value> &args) {
  auto *context = rewriter.getContext();
  bool ok = std::visit(
      [&](const auto &inner) {
        using Inner = std::decay_t<decltype(inner)>;
        if constexpr (std::is_same_v<Inner, core::Swizzle<core::EmitOp>>) {
          appendAttrPrintf(SwizzleAttr::get(context, core::to_trait(inner)), format);
          return true;
        } else if constexpr (std::is_same_v<Inner, core::CoordSwizzle<core::EmitOp>>) {
          appendAttrPrintf(CoordSwizzleAttr::get(context, core::to_trait(inner)), format);
          return true;
        } else {
          return appendLayoutPrintf(rewriter, inner, ctx, format, args);
        }
      },
      value.inner());
  format += " o ";
  ok = ok && appendIntTuplePrintf(rewriter, value.offset(), ctx, format, args);
  format += " o ";
  return ok && appendLayoutPrintf(rewriter, value.outer(), ctx, format, args);
}

bool appendLayoutPrintf(PatternRewriter &rewriter, const core::AnyLayout<core::EmitOp> &layout,
                        core::EmitContext &ctx, std::string &format, SmallVectorImpl<Value> &args) {
  if (auto *composed = std::get_if<core::ComposedLayout<core::EmitOp>>(&layout))
    return appendComposedPrintf(rewriter, *composed, ctx, format, args);
  const auto &plain = std::get<core::Layout<core::EmitOp>>(layout);
  if (!appendIntTuplePrintf(rewriter, plain.shape(), ctx, format, args))
    return false;
  format += ":";
  return appendIntTuplePrintf(rewriter, plain.stride(), ctx, format, args);
}

struct ContigSegment {
  int32_t idx;
  int64_t vecWidth;
};

enum class ContigResult { Vector, Scalar, Invalid };

std::pair<ContigResult, ContigSegment> findContigSegment(IntTupleAttr shapeAttr,
                                                         IntTupleAttr strideAttr) {
  // Inspect the original flattened modes so vector permutation stays column-major.
  // A singleton has no varying coordinate and cannot compete for the unit-stride
  // segment, regardless of its stored stride.
  auto layout = core::flatten(core::Layout<core::Trait>(shapeAttr.getRef(), strideAttr.getRef()));
  if (layout.isError())
    return {ContigResult::Invalid, {}};
  int count = 0;
  auto result = ContigSegment{0, 0};
  for (int i = 0; i < layout.rank(); ++i) {
    auto mode = core::get(layout, i);
    if (mode.shape().isOne())
      continue;
    if (mode.stride().isOne()) {
      if (++count > 1)
        return {ContigResult::Invalid, {}};
      if (!mode.shape().isSInt())
        return {ContigResult::Invalid, {}};
      result = {i, mode.shape().staticValue()};
    }
  }
  if (count == 0)
    return {ContigResult::Scalar, {}};
  return {ContigResult::Vector, result};
}

Value permuteLoadedVec(PatternRewriter &rewriter, Location loc, Value vec, IntTupleAttr flatShape,
                       int32_t flatRank, int32_t contigIdx, int64_t vecWidth, int64_t numChunks) {
  if (contigIdx == 0)
    return vec;

  auto elemTy = cast<VectorType>(vec.getType()).getElementType();
  int32_t numPre = contigIdx;
  int32_t numPost = flatRank - contigIdx - 1;

  SmallVector<int64_t> intermediateShape;
  for (int32_t i = flatRank - 1; i > contigIdx; --i)
    intermediateShape.push_back(flatShape.at(i).getLeaf().getValue());
  for (int32_t i = contigIdx - 1; i >= 0; --i)
    intermediateShape.push_back(flatShape.at(i).getLeaf().getValue());
  intermediateShape.push_back(vecWidth);

  auto intermediateTy = VectorType::get(intermediateShape, elemTy);
  Value shaped = vector::ShapeCastOp::create(rewriter, loc, intermediateTy, vec);

  SmallVector<int64_t> perm;
  for (int32_t i = 0; i < numPost; ++i)
    perm.push_back(i);
  perm.push_back(numPost + numPre);
  for (int32_t i = 0; i < numPre; ++i)
    perm.push_back(numPost + i);

  SmallVector<int64_t> transposedShape;
  for (auto p : perm)
    transposedShape.push_back(intermediateShape[p]);
  auto transposedTy = VectorType::get(transposedShape, elemTy);
  Value transposed = vector::TransposeOp::create(rewriter, loc, transposedTy, shaped, perm);

  auto flatTy = VectorType::get({numChunks * vecWidth}, elemTy);
  return vector::ShapeCastOp::create(rewriter, loc, flatTy, transposed);
}

Value permuteForStore(PatternRewriter &rewriter, Location loc, Value vec, IntTupleAttr flatShape,
                      int32_t flatRank, int32_t contigIdx, int64_t vecWidth, int64_t numChunks) {
  if (contigIdx == 0)
    return vec;

  auto elemTy = cast<VectorType>(vec.getType()).getElementType();
  int32_t numPre = contigIdx;
  int32_t numPost = flatRank - contigIdx - 1;

  SmallVector<int64_t> targetShape;
  for (int32_t i = flatRank - 1; i > contigIdx; --i)
    targetShape.push_back(flatShape.at(i).getLeaf().getValue());
  targetShape.push_back(vecWidth);
  for (int32_t i = contigIdx - 1; i >= 0; --i)
    targetShape.push_back(flatShape.at(i).getLeaf().getValue());

  auto targetTy = VectorType::get(targetShape, elemTy);
  Value shaped = vector::ShapeCastOp::create(rewriter, loc, targetTy, vec);

  SmallVector<int64_t> perm;
  for (int32_t i = 0; i < numPost; ++i)
    perm.push_back(i);
  for (int32_t i = 0; i < numPre; ++i)
    perm.push_back(numPost + 1 + i);
  perm.push_back(numPost);

  SmallVector<int64_t> transposedShape;
  for (auto p : perm)
    transposedShape.push_back(targetShape[p]);
  auto transposedTy = VectorType::get(transposedShape, elemTy);
  Value transposed = vector::TransposeOp::create(rewriter, loc, transposedTy, shaped, perm);

  auto flatTy = VectorType::get({numChunks * vecWidth}, elemTy);
  return vector::ShapeCastOp::create(rewriter, loc, flatTy, transposed);
}

class MakeFragmentLikeOpLowering : public OpRewritePattern<MakeFragmentLikeOp> {
public:
  using OpRewritePattern<MakeFragmentLikeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(MakeFragmentLikeOp op, PatternRewriter &rewriter) const override {
    auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());
    auto type = cast<fly::MemRefType>(op.getType());
    auto layout =
        core::bind(cast<LayoutAttr>(type.getLayout()).getRef(), core::EmitBindings{}, ctx);
    if (failed(layout))
      return failure();
    auto result = materializeValue(*layout, ctx);
    if (failed(result))
      return failure();
    rewriter.replaceOpWithNewOp<MemRefAllocaOp>(op, type, *result);
    return success();
  }
};

class GetIterLowering : public OpRewritePattern<GetIterOp> {
public:
  using OpRewritePattern<GetIterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GetIterOp op, PatternRewriter &rewriter) const override {
    Value memref = op.getMemref();

    if (auto makeViewOp = memref.getDefiningOp<MakeViewOp>()) {
      rewriter.replaceOp(op, makeViewOp.getIter());
      return success();
    }
    return failure();
  }
};

/// The leaves of a constructed int tuple or layout are its constructor's
/// operands. The algebra conversion imports values before its replacements
/// commit, so it can leave get_leaves of a constructor behind; reading the
/// operands keeps their arithmetic visible to the passes that follow.
class GetLeavesOfConstructor : public OpRewritePattern<GetLeavesOp> {
public:
  using OpRewritePattern<GetLeavesOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GetLeavesOp op, PatternRewriter &rewriter) const override {
    Value input = op.getInput();
    Operation *producer = input.getDefiningOp();
    if (!producer || !isa<MakeIntTupleOp, MakeShapeOp, MakeStrideOp, MakeCoordOp, MakeLayoutOp,
                          MakeComposedLayoutOp>(producer))
      return failure();
    auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());
    FailureOr<core::EmitBindings> bindings = failure();
    auto collect = [&](const auto &value) {
      if (succeeded(value))
        bindings = core::get_bindings(*value, ctx);
    };
    if (isa<IntTupleType>(input.getType()))
      collect(importValue<IntTupleType>(input, ctx));
    else if (isa<LayoutType>(input.getType()))
      collect(importValue<LayoutType>(input, ctx));
    else if (isa<ComposedLayoutType>(input.getType()))
      collect(importValue<ComposedLayoutType>(input, ctx));
    if (failed(bindings) || !bindings->pointers.empty() ||
        TypeRange(ValueRange(bindings->scalars)) != op.getResultTypes())
      return failure();
    rewriter.replaceOp(op, bindings->scalars);
    return success();
  }
};

class PrintOpLowering : public OpRewritePattern<PrintOp> {
public:
  using OpRewritePattern<PrintOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PrintOp op, PatternRewriter &rewriter) const override {
    bool isGpuContext = op->getParentOfType<gpu::GPUFuncOp>() != nullptr;

    auto loc = op.getLoc();
    std::string userFormat = op.getFormat().str();
    std::string format;
    SmallVector<Value> args;

    auto emit = makeDefaultEmitContext(rewriter, loc);
    bool formatFailed = false;
    auto formatValueToString = [&](Value val) -> std::string {
      std::string valFormat;
      if (isa<IntTupleType>(val.getType())) {
        auto tuple = importValue<IntTupleType>(val, emit);
        formatFailed |=
            failed(tuple) || !appendIntTuplePrintf(rewriter, tuple->asRef(), emit, valFormat, args);
      } else if (isa<LayoutType>(val.getType())) {
        auto layout = importValue<LayoutType>(val, emit);
        formatFailed |=
            failed(layout) || !appendLayoutPrintf(rewriter, *layout, emit, valFormat, args);
      } else if (isa<ComposedLayoutType>(val.getType())) {
        auto layout = importValue<ComposedLayoutType>(val, emit);
        formatFailed |=
            failed(layout) || !appendComposedPrintf(rewriter, *layout, emit, valFormat, args);
      } else if (isa<fly::MemRefType>(val.getType())) {
        // A memref prints its layout; its iterator is not a printable value.
        auto memref = importValue<fly::MemRefType>(val, emit);
        formatFailed |= failed(memref) ||
                        !appendLayoutPrintf(rewriter, memref->layout(), emit, valFormat, args);
      } else if (isa<CoordTensorType>(val.getType())) {
        auto tensor = importValue<CoordTensorType>(val, emit);
        if (failed(tensor)) {
          formatFailed = true;
        } else {
          formatFailed |=
              !appendIntTuplePrintf(rewriter, tensor->data().asRef(), emit, valFormat, args);
          valFormat += " o ";
          formatFailed |= !appendLayoutPrintf(rewriter, tensor->layout(), emit, valFormat, args);
        }
      } else if (isa<VectorType>(val.getType())) {
        formatFailed |= !appendVectorPrintf(rewriter, loc, val, valFormat, args);
      } else if (isa<IntegerType, FloatType, IndexType>(val.getType())) {
        formatFailed |= !appendScalarPrintfArg(rewriter, loc, val, valFormat, args);
      } else {
        // Other values (pointers, tiles, atoms) print their static type.
        llvm::raw_string_ostream os(valFormat);
        val.getType().print(os);
      }
      return valFormat;
    };

    // For CPU context, we need to interleave text and values
    // Collect text segments and argument indices
    struct PrintSegment {
      std::string text;
      int argIndex = -1; // -1 means text only
    };
    SmallVector<PrintSegment> segments;

    auto expandFormatToSegments = [&](const std::string &fmtStr, size_t argBase) {
      size_t fpos = 0;
      size_t argCur = argBase;
      while (fpos < fmtStr.size()) {
        size_t ph = fmtStr.find("%", fpos);
        if (ph == std::string::npos) {
          segments.push_back({fmtStr.substr(fpos), -1});
          break;
        }
        if (ph > fpos) {
          segments.push_back({fmtStr.substr(fpos, ph - fpos), -1});
        }
        size_t specEnd = ph + 1;
        while (specEnd < fmtStr.size() &&
               std::string_view("diuoxXfFeEgGaAcspn").find(fmtStr[specEnd]) ==
                   std::string_view::npos) {
          ++specEnd;
        }
        if (specEnd < fmtStr.size()) {
          ++specEnd;
        }
        segments.push_back({"", static_cast<int>(argCur++)});
        fpos = specEnd;
      }
    };

    if (!userFormat.empty()) {
      size_t valueIdx = 0;
      size_t pos = 0;
      while (pos < userFormat.size()) {
        size_t placeholderPos = userFormat.find("{}", pos);
        if (placeholderPos == std::string::npos) {
          segments.push_back({userFormat.substr(pos), -1});
          break;
        }
        if (placeholderPos > pos) {
          segments.push_back({userFormat.substr(pos, placeholderPos - pos), -1});
        }
        if (valueIdx < op.getValues().size()) {
          size_t argStartIdx = args.size();
          std::string staticFormat = formatValueToString(op.getValues()[valueIdx]);
          size_t numArgsAdded = args.size() - argStartIdx;
          if (numArgsAdded == 0 && !staticFormat.empty()) {
            segments.push_back({staticFormat, -1});
          } else {
            expandFormatToSegments(staticFormat, argStartIdx);
          }
          valueIdx++;
        }
        pos = placeholderPos + 2;
      }
    } else {
      bool first = true;
      for (Value val : op.getValues()) {
        if (!first) {
          segments.push_back({" ", -1});
        }
        first = false;
        size_t argStartIdx = args.size();
        std::string staticFormat = formatValueToString(val);
        size_t numArgsAdded = args.size() - argStartIdx;
        if (numArgsAdded == 0 && !staticFormat.empty()) {
          segments.push_back({staticFormat, -1});
        } else {
          expandFormatToSegments(staticFormat, argStartIdx);
        }
      }
    }

    if (formatFailed)
      return op.emitOpError("unsupported print leaf or scalar");
    if (isGpuContext) {
      // For GPU, build printf format string
      for (const auto &seg : segments) {
        if (seg.argIndex >= 0) {
          castPrintfArg(rewriter, loc, args[seg.argIndex], format);
        } else {
          format += seg.text;
        }
      }
      format += "\n";
      gpu::PrintfOp::create(rewriter, loc, rewriter.getStringAttr(format), args);
    } else {
      // For CPU, print segments in order
      for (size_t i = 0; i < segments.size(); ++i) {
        const auto &seg = segments[i];
        if (seg.argIndex >= 0) {
          bool isLast = (i == segments.size() - 1);
          auto punctuation =
              isLast ? vector::PrintPunctuation::NewLine : vector::PrintPunctuation::NoPunctuation;
          vector::PrintOp::create(rewriter, loc, args[seg.argIndex], punctuation);
        } else if (!seg.text.empty()) {
          vector::PrintOp::create(rewriter, loc, seg.text);
        }
      }
      if (segments.empty() || segments.back().argIndex < 0) {
        vector::PrintOp::create(rewriter, loc, vector::PrintPunctuation::NewLine);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

class MmaMakeFragmentOpLowering : public OpRewritePattern<MmaMakeFragmentOp> {
public:
  using OpRewritePattern<MmaMakeFragmentOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MmaMakeFragmentOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto tiledMmaTy = dyn_cast<TiledMmaType>(op.getTiledMma().getType());
    if (!tiledMmaTy)
      return failure();

    auto mmaAtom = dyn_cast<MmaAtomType>(tiledMmaTy.getMmaAtom());
    if (!mmaAtom)
      return failure();

    Type elemTy;
    switch (op.getOperandId()) {
    case MmaOperand::A:
      elemTy = mmaAtom.getValTypeA();
      break;
    case MmaOperand::B:
      elemTy = mmaAtom.getValTypeB();
      break;
    case MmaOperand::C:
      elemTy = mmaAtom.getValTypeC();
      break;
    case MmaOperand::D:
      elemTy = mmaAtom.getValTypeD();
      break;
    }

    auto resultTy = cast<fly::MemRefType>(op.getType());
    auto layoutAttr = dyn_cast<LayoutAttr>(resultTy.getLayout());
    if (!layoutAttr || !layoutAttr.isStatic())
      return rewriter.notifyMatchFailure(op, "fragment layout must be fully static to alloca");

    auto emit = makeDefaultEmitContext(rewriter, loc);
    auto layout = core::bind(layoutAttr.getRef(), core::EmitBindings{}, emit);
    if (failed(layout))
      return failure();
    auto fragmentLayout = materializeValue(*layout, emit);
    if (failed(fragmentLayout))
      return failure();
    rewriter.replaceOpWithNewOp<MemRefAllocaOp>(op, resultTy, *fragmentLayout);
    return success();
  }
};

/// Expands fly.copy into atom calls through the Core EmitOp category.
struct ExpandCopyOpLowering : OpConversionPattern<CopyOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(CopyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value copyAtomVal = adaptor.getCopyAtom();
    if (isa<TiledCopyType>(copyAtomVal.getType()))
      copyAtomVal = GetCopyAtomOp::create(rewriter, loc, copyAtomVal);
    auto emitAtom = [&](Value src, Value dst, Value pred) {
      CopyAtomCall::create(rewriter, loc, copyAtomVal, src, dst, pred);
    };

    std::optional<int64_t> atomValsSrc, atomValsDst;
    if (auto atom = dyn_cast<CopyAtomType>(copyAtomVal.getType())) {
      // Whole-tile atoms consume the original layouts themselves.
      if (atom.getCopyOp().hasTrait<WholeTileCopy>()) {
        emitAtom(adaptor.getSrc(), adaptor.getDst(), adaptor.getPred());
        rewriter.eraseOp(op);
        return success();
      }
      auto numVals = [](Attribute thrVal) -> std::optional<int64_t> {
        auto size = core::size(core::get(cast<LayoutAttr>(thrVal).getRef(), 1));
        return size.isStatic() ? std::optional(size.staticValue()) : std::nullopt;
      };
      atomValsSrc = numVals(atom.getThrValLayoutSrc());
      atomValsDst = numVals(atom.getThrValLayoutDst());
    }
    if (failed(expandCopy(rewriter, loc, adaptor.getSrc(), adaptor.getDst(), adaptor.getPred(),
                          atomValsSrc, atomValsDst, emitAtom)))
      return rewriter.notifyMatchFailure(op, "cannot expand the copy into atom calls");
    rewriter.eraseOp(op);
    return success();
  }
};

/// Expands fly.gemm into atom calls through the Core EmitOp category.
struct ExpandGemmOpLowering : OpConversionPattern<GemmOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(GemmOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value mmaAtomVal = adaptor.getMmaAtom();
    if (isa<TiledMmaType>(mmaAtomVal.getType()))
      mmaAtomVal = GetMmaAtomOp::create(rewriter, loc, mmaAtomVal);
    LayoutAttr traversalLayout;
    if (Value layout = adaptor.getTraversalLayout())
      traversalLayout = cast<LayoutType>(layout.getType()).getAttr();
    auto emitAtom = [&](Value d, ValueRange a, ValueRange b, Value c) {
      MmaAtomCall::create(rewriter, loc, mmaAtomVal, d, a, b, c);
    };
    if (failed(expandGemm(rewriter, loc, adaptor.getD(), adaptor.getA(), adaptor.getB(),
                          adaptor.getC(), op.getTraversalOrder(), traversalLayout, emitAtom)))
      return rewriter.notifyMatchFailure(op, "cannot expand the gemm into atom calls");
    rewriter.eraseOp(op);
    return success();
  }
};

class MemRefLoadVecOpLowering : public OpRewritePattern<MemRefLoadVecOp> {
public:
  using OpRewritePattern<MemRefLoadVecOp>::OpRewritePattern;

  LogicalResult lowerCoordTensor1D(MemRefLoadVecOp op, CoordTensorType coordTensorTy,
                                   PatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    auto resVecTy = dyn_cast<VectorType>(op.getResult().getType());
    if (!resVecTy)
      return failure();

    // Only a plain LinearLayout is supported, matching the MemRefType path; ComposedLayout
    // (swizzle/offset) is not handled yet.
    auto linearLayout = dyn_cast<LayoutAttr>(coordTensorTy.getLayout());
    if (!linearLayout)
      return failure();

    IntTupleAttr shapeAttr = linearLayout.getShape();
    if (!shapeAttr.isStatic())
      return failure();
    int64_t size = resVecTy.getNumElements();

    auto elemTy = dyn_cast<IntegerType>(resVecTy.getElementType());
    if (!elemTy)
      return failure();

    Value result = arith::ConstantOp::create(rewriter, loc, rewriter.getZeroAttr(resVecTy));
    for (int64_t i = 0; i < size; ++i) {
      Value coord = MakeIntTupleOp::create(
          rewriter, loc, IntTupleType::get(IntTupleAttr::getLeafStatic(ctx, i)), ValueRange{});
      Value idx = MemRefLoadOp::create(rewriter, loc, op.getMemref(), coord);
      Value scalar = GetScalarOp::create(rewriter, loc, idx);
      auto scalarTy = cast<IntegerType>(scalar.getType());
      if (scalarTy.getWidth() < elemTy.getWidth())
        scalar = arith::ExtSIOp::create(rewriter, loc, elemTy, scalar).getResult();
      else if (scalarTy.getWidth() > elemTy.getWidth())
        scalar = arith::TruncIOp::create(rewriter, loc, elemTy, scalar).getResult();
      result = vector::InsertOp::create(rewriter, loc, scalar, result, i);
    }
    rewriter.replaceOp(op, result);
    return success();
  }

  LogicalResult matchAndRewrite(MemRefLoadVecOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    if (auto coordTensorTy = dyn_cast<CoordTensorType>(op.getMemref().getType()))
      return lowerCoordTensor1D(op, coordTensorTy, rewriter);

    auto memrefTy = dyn_cast<fly::MemRefType>(op.getMemref().getType());
    if (!memrefTy)
      return failure();

    auto layoutAttr = dyn_cast<LayoutAttr>(memrefTy.getLayout());
    if (!layoutAttr)
      return failure();
    if (!layoutAttr.isStaticShape())
      return failure();
    IntTupleAttr shapeAttr = layoutAttr.getShape();
    IntTupleAttr strideAttr = layoutAttr.getStride();

    auto resVecTy = dyn_cast<VectorType>(op.getResult().getType());
    if (!resVecTy)
      return failure();

    Value memref = op.getMemref();

    IntTupleAttr flatShape = IntTupleAttr::get(ctx, core::flatten(shapeAttr.getRef()));
    int32_t flatRank = flatShape.rank();

    auto [contigResult, contigSeg] = findContigSegment(shapeAttr, strideAttr);
    if (contigResult == ContigResult::Invalid)
      return failure();

    Value iter = GetIterOp::create(rewriter, loc, memref);
    Value layout = GetLayoutOp::create(rewriter, loc, memref);

    Value result = arith::ConstantOp::create(rewriter, loc, rewriter.getZeroAttr(resVecTy));

    if (contigResult == ContigResult::Scalar) {
      int64_t totalElems = staticLeafValue(core::product(shapeAttr.getRef()));

      Type scalarTy = resVecTy.getElementType();
      for (int64_t i = 0; i < totalElems; ++i) {
        auto coordAttr =
            IntTupleAttr::get(ctx, core::idx2crd(IntTupleTrait::getSInt(i), shapeAttr.getRef()));

        Value coord =
            MakeIntTupleOp::create(rewriter, loc, IntTupleType::get(coordAttr), ValueRange{});
        Value offset = Crd2IdxOp::create(rewriter, loc, coord, layout);
        Value ptr = AddOffsetOp::create(rewriter, loc, iter, offset);
        Value scalar = PtrLoadOp::create(rewriter, loc, scalarTy, ptr);
        result = vector::InsertOp::create(rewriter, loc, scalar, result, i);
      }
      rewriter.replaceOp(op, result);
      return success();
    }

    int64_t vecWidth = contigSeg.vecWidth;
    int32_t contigIdx = contigSeg.idx;

    if (flatRank == 1) {
      Value loaded = PtrLoadOp::create(rewriter, loc, resVecTy, iter);
      rewriter.replaceOp(op, loaded);
      return success();
    }

    auto restShape = restFlatShape(flatShape.getRef(), flatRank, contigIdx);

    int64_t numChunks = staticLeafValue(core::product(restShape));
    VectorType chunkVecTy = VectorType::get({vecWidth}, resVecTy.getElementType());

    for (int64_t i = 0; i < numChunks; ++i) {
      auto coordAttr = chunkCoordAttr(ctx, shapeAttr.getRef(), restShape, flatRank, contigIdx, i);

      Value coord =
          MakeIntTupleOp::create(rewriter, loc, IntTupleType::get(coordAttr), ValueRange{});
      Value offset = Crd2IdxOp::create(rewriter, loc, coord, layout);
      Value ptr = AddOffsetOp::create(rewriter, loc, iter, offset);
      Value chunkVec = PtrLoadOp::create(rewriter, loc, chunkVecTy, ptr);
      result = vector::InsertStridedSliceOp::create(rewriter, loc, chunkVec, result,
                                                    ArrayRef<int64_t>{i * vecWidth},
                                                    ArrayRef<int64_t>{1})
                   .getResult();
    }

    result = permuteLoadedVec(rewriter, loc, result, flatShape, flatRank, contigIdx, vecWidth,
                              numChunks);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class MemRefStoreVecOpLowering : public OpRewritePattern<MemRefStoreVecOp> {
public:
  using OpRewritePattern<MemRefStoreVecOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MemRefStoreVecOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto memrefTy = dyn_cast<fly::MemRefType>(op.getMemref().getType());
    if (!memrefTy)
      return failure();

    auto layoutAttr = dyn_cast<LayoutAttr>(memrefTy.getLayout());
    if (!layoutAttr)
      return failure();
    if (!layoutAttr.isStaticShape())
      return failure();
    IntTupleAttr shapeAttr = layoutAttr.getShape();
    IntTupleAttr strideAttr = layoutAttr.getStride();

    Value vec = op.getVector();
    Value memref = op.getMemref();

    IntTupleAttr flatShape = IntTupleAttr::get(ctx, core::flatten(shapeAttr.getRef()));
    int32_t flatRank = flatShape.isLeaf() ? 1 : flatShape.rank();

    auto [contigResult, contigSeg] = findContigSegment(shapeAttr, strideAttr);
    if (contigResult == ContigResult::Invalid)
      return failure();

    Value iter = GetIterOp::create(rewriter, loc, memref);
    Value layout = GetLayoutOp::create(rewriter, loc, memref);

    if (contigResult == ContigResult::Scalar) {
      int64_t totalElems = staticLeafValue(core::product(shapeAttr.getRef()));
      for (int64_t i = 0; i < totalElems; ++i) {
        auto coordAttr =
            IntTupleAttr::get(ctx, core::idx2crd(IntTupleTrait::getSInt(i), shapeAttr.getRef()));
        Value coord =
            MakeIntTupleOp::create(rewriter, loc, IntTupleType::get(coordAttr), ValueRange{});
        Value offset = Crd2IdxOp::create(rewriter, loc, coord, layout);
        Value ptr = AddOffsetOp::create(rewriter, loc, iter, offset);
        // A single-element memref may store a scalar rather than a vector.
        Value scalar =
            isa<VectorType>(vec.getType()) ? vector::ExtractOp::create(rewriter, loc, vec, i) : vec;
        PtrStoreOp::create(rewriter, loc, scalar, ptr);
      }
      rewriter.eraseOp(op);
      return success();
    }

    int64_t vecWidth = contigSeg.vecWidth;
    int32_t contigIdx = contigSeg.idx;

    if (flatRank == 1) {
      PtrStoreOp::create(rewriter, loc, vec, iter);
      rewriter.eraseOp(op);
      return success();
    }

    auto restShape = restFlatShape(flatShape.getRef(), flatRank, contigIdx);
    int64_t numChunks = staticLeafValue(core::product(restShape));

    vec = permuteForStore(rewriter, loc, vec, flatShape, flatRank, contigIdx, vecWidth, numChunks);

    for (int64_t i = 0; i < numChunks; ++i) {
      auto coordAttr = chunkCoordAttr(ctx, shapeAttr.getRef(), restShape, flatRank, contigIdx, i);

      Value coord =
          MakeIntTupleOp::create(rewriter, loc, IntTupleType::get(coordAttr), ValueRange{});
      Value offset = Crd2IdxOp::create(rewriter, loc, coord, layout);
      Value ptr = AddOffsetOp::create(rewriter, loc, iter, offset);
      Value chunkVec =
          vector::ExtractStridedSliceOp::create(rewriter, loc, vec, ArrayRef<int64_t>{i * vecWidth},
                                                ArrayRef<int64_t>{vecWidth}, ArrayRef<int64_t>{1})
              .getResult();
      PtrStoreOp::create(rewriter, loc, chunkVec, ptr);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Canonicalize an add_offset chain to `add_offset(add_offset(ptr, dyn), static)`
/// -- at most one dynamic layer plus one static tail.
///
/// Fusing a runtime offset with a compile-time constant makes every access
/// compute its own address off the runtime base, so the backend materializes
/// one base register per access instead of sharing one.  Keeping the static
/// part as a separate outer offset preserves the shared base.
///
/// Convergence: each rewrite either drops a level from the chain (fuse) or
/// moves the static offset strictly outward (swap), and the resulting
/// `dyn -> static` form is rejected by the already-canonical check, so the
/// pattern cannot loop.
/// Conservative upper bound on |v|, or nullopt when it cannot be established.
///
/// The fold below is only sound when the value survives a trip through i32, and
/// the type system cannot say that: an `index` is 64-bit.  Estimating a bound
/// instead of answering a yes/no question matters, because "built from bounded
/// roots" is not the same as "stays small": `tid * tid * tid * tid` is made
/// entirely of launch coordinates and still reaches 2^40.
///
/// Roots are launch coordinates (bounded by the block/grid geometry) and
/// constants (bounded by their own value).  Every operator combines its
/// operands' bounds the way the operation combines values, and any step that
/// would overflow the estimate itself gives up.  Anything unrecognised -- a
/// kernel argument, a load, an op not on the list -- is unknown.
///
/// This is the value-range knowledge upstream deliberately does not assume, and
/// the only reason FlyDSL may re-fold what `arith` no longer does.
static std::optional<uint64_t> narrowUpperBound(Value v, unsigned depth,
                                                llvm::SmallPtrSetImpl<Operation *> &visited) {
  if (depth > kMaxNarrowProofDepth)
    return std::nullopt;

  // A non-negative constant bounds itself.  Negative values are rejected
  // outright: every rule below reasons about unsigned magnitudes, and a
  // negative operand reads as a huge unsigned value there, so an
  // absolute-value bound would understate it.
  APInt cst;
  if (matchPattern(v, m_ConstantInt(&cst))) {
    if (cst.isNegative())
      return std::nullopt;
    return cst.getLimitedValue();
  }

  Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt; // block/function argument: unknown range

  // A value reached twice on one walk (cyclic chain) cannot be bounded here.
  if (!visited.insert(def).second)
    return std::nullopt;
  auto pop = llvm::scope_exit([&] { visited.erase(def); });

  // Launch coordinates: ask the op for its own range.  Every gpu index op
  // implements InferIntRangeInterface, and that implementation already prefers
  // an explicit `upper_bound` attribute, then the launch geometry declared by an
  // enclosing gpu.launch / gpu.func / `gpu.known_*_size`, and only then its own
  // implicit maximum (kMaxDim, i.e. uint32_t::max).  Deferring to it keeps this
  // analysis in step with upstream instead of restating the ordering here.
  if (auto inferrable = dyn_cast<InferIntRangeInterface>(def)) {
    if (def->getNumOperands() == 0 && def->getNumResults() == 1) {
      std::optional<uint64_t> inferred;
      inferrable.inferResultRanges({}, [&](Value v, const ConstantIntRanges &range) {
        if (v == def->getResult(0))
          inferred = range.umax().getZExtValue();
      });
      if (inferred)
        return inferred;
    }
  }

  auto operandBound = [&](unsigned i) {
    return narrowUpperBound(def->getOperand(i), depth + 1, visited);
  };

  auto checkedAdd = [](uint64_t a, uint64_t b) -> std::optional<uint64_t> {
    uint64_t r;
    return __builtin_add_overflow(a, b, &r) ? std::nullopt : std::optional<uint64_t>(r);
  };
  auto checkedMul = [](uint64_t a, uint64_t b) -> std::optional<uint64_t> {
    uint64_t r;
    return __builtin_mul_overflow(a, b, &r) ? std::nullopt : std::optional<uint64_t>(r);
  };

  // Binary ops: combine the operand bounds the way the operation combines values.
  if (def->getNumOperands() == 2) {
    auto lhs = operandBound(0);
    auto rhs = operandBound(1);

    // `and`, `rem` and `min` are bounded by one operand alone, so an unknown
    // other side is tolerable.
    if (isa<arith::AndIOp, arith::RemUIOp, arith::MinUIOp>(def)) {
      if (!lhs && !rhs)
        return std::nullopt;
      if (!lhs)
        return rhs;
      if (!rhs)
        return lhs;
      return std::min(*lhs, *rhs);
    }
    // Shifting right and dividing only shrink the left operand.  Both read
    // their operands as unsigned, which is sound here because every value this
    // function can bound is non-negative: the sources are non-negative
    // constants and launch coordinates, and no rule below can turn those into a
    // negative result.
    if (isa<arith::ShRUIOp, arith::DivUIOp>(def))
      return lhs;

    if (!lhs || !rhs)
      return std::nullopt;

    if (isa<arith::AddIOp>(def))
      return checkedAdd(*lhs, *rhs);
    // Subtraction is deliberately absent: with unsigned reasoning `a - b` wraps
    // to a huge value whenever b > a, which no bound on a and b rules out.
    if (isa<arith::MulIOp>(def))
      return checkedMul(*lhs, *rhs);
    if (isa<arith::ShLIOp>(def)) {
      if (*rhs >= 64)
        return std::nullopt;
      return checkedMul(*lhs, uint64_t(1) << *rhs);
    }
    if (isa<arith::OrIOp, arith::XOrIOp, arith::MaxUIOp>(def)) {
      // Bit-wise or/xor cannot exceed the next power of two above either side.
      uint64_t m = std::max(*lhs, *rhs);
      return m == UINT64_MAX ? std::nullopt
                             : std::optional<uint64_t>(m == 0 ? 0 : llvm::NextPowerOf2(m) - 1);
    }
    return std::nullopt;
  }

  // Casts narrow or widen, and for the signed form either direction can turn a
  // small non-negative value into a negative one that reads as huge unsigned.
  // Narrowing does it by truncating (index -> i8 keeps 128 as -128); widening
  // does it by sign-extending whatever the narrow type already held (i8 -> index
  // takes that -128 to 2^64-128).  Both are governed by the *narrower* of the
  // two types, so check the bound against that one: it must fit with the sign
  // bit clear.  An unsigned cast has no sign bit to flip and only has to fit.
  if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(def)) {
    auto srcBound = operandBound(0);
    if (!srcBound)
      return std::nullopt;
    auto intWidth = [](Type t) -> unsigned {
      auto intTy = dyn_cast<IntegerType>(getElementTypeOrSelf(t));
      return intTy ? intTy.getWidth() : 64; // `index` occupies its full storage
    };
    unsigned narrowBits =
        std::min(intWidth(def->getOperand(0).getType()), intWidth(def->getResult(0).getType()));
    if (narrowBits >= 64)
      return srcBound;
    uint64_t limit = isa<arith::IndexCastUIOp>(def) ? (uint64_t(1) << narrowBits)
                                                    : (uint64_t(1) << (narrowBits - 1));
    return *srcBound < limit ? srcBound : std::nullopt;
  }

  // select: either arm may be taken.
  if (isa<arith::SelectOp>(def)) {
    auto t = operandBound(1);
    auto f = operandBound(2);
    if (!t || !f)
      return std::nullopt;
    return std::max(*t, *f);
  }

  return std::nullopt;
}

/// Is \p v provably small enough to survive a round trip through \p bits?
static bool isProvablyNarrow(Value v, unsigned bits) {
  llvm::SmallPtrSet<Operation *, 16> visited;
  std::optional<uint64_t> bound = narrowUpperBound(v, 0, visited);
  if (!bound)
    return false;
  // An intermediate at least as wide as `index` cannot lose anything, and the
  // shift below would be undefined for it.
  if (bits >= 64)
    return true;
  // Signed casts, so the usable range is one bit smaller.
  return *bound < (uint64_t(1) << (bits - 1));
}

/// Fold `cast(cast(x : index -> iN) : iN -> index)` back to `x`.
///
/// Only the signed form (`arith.index_cast`) is handled.  The pattern is a
/// template so the unsigned form could be added, but nothing in tree emits
/// `arith.index_castui`, and `isProvablyNarrow` reserves the sign bit, so
/// instantiating it would add an untested path for no benefit.
///
/// Upstream MLIR used to fold this in `arith`'s canonicalizer, but the pattern
/// was unsound in general and was removed in llvm/llvm-project@8c81064169c5: for
/// a narrow intermediate such as i8 the inner cast truncates, so the round trip
/// loses bits and must be preserved.  The check upstream now applies is purely
/// type-based -- an `index` is assumed to occupy its full 64-bit internal
/// storage -- so `index -> i32 -> index` is rejected as lossy even where it is
/// not.
///
/// FlyDSL emits exactly that shape when layout arithmetic crosses between
/// `index` and i32, and `isProvablyNarrow` supplies the value-range knowledge
/// upstream deliberately does not assume.  Re-folding removes 42 redundant cast
/// pairs from a MoE stage1 kernel, which after lowering would otherwise become
/// `trunc`/`sext` pairs inside the MFMA loop (measured: vgpr_spill_count
/// 147 -> 19).
///
/// Both casts must be the same op: mixing signed and unsigned changes the
/// meaning of the round trip, and the template parameter enforces that.
/// Scalars and vectors are both handled -- for a vector the element types carry
/// the index/integer distinction, and requiring the outer result type to equal
/// the innermost source type keeps the shapes in step.
///
/// Convergence: each rewrite replaces one op with an existing value and adds
/// nothing, so the pattern cannot loop.
template <typename CastOp> class RedundantIndexCastPairImpl : public OpRewritePattern<CastOp> {
public:
  using OpRewritePattern<CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CastOp op, PatternRewriter &rewriter) const override {
    // Outer cast must land back on `index` (element type, so vectors work too).
    if (!isa<IndexType>(getElementTypeOrSelf(op.getType())))
      return failure();

    auto inner = op.getIn().template getDefiningOp<CastOp>();
    if (!inner)
      return failure();

    // Innermost source must be exactly what we are returning to; for vectors
    // this also pins the shape.
    Value orig = inner.getIn();
    if (orig.getType() != op.getType())
      return failure();

    // The intermediate must be wide enough to hold the value.  Anything
    // narrower than 32 bits is genuinely lossy and is left alone.
    auto midTy = dyn_cast<IntegerType>(getElementTypeOrSelf(inner.getType()));
    if (!midTy || midTy.getWidth() < 32)
      return failure();

    // Type width alone does not make the round trip lossless: an `index` is
    // 64-bit, so a value above 2^31 would not survive it.  Only fold when the
    // value is provably narrow.
    if (!isProvablyNarrow(orig, midTy.getWidth()))
      return failure();

    rewriter.replaceOp(op, orig);
    return success();
  }
};

using RedundantIndexCastPair = RedundantIndexCastPairImpl<arith::IndexCastOp>;

class AddOffsetCanonicalization : public OpRewritePattern<AddOffsetOp> {
public:
  using OpRewritePattern<AddOffsetOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOffsetOp op, PatternRewriter &rewriter) const override {
    auto inner = op.getPtr().getDefiningOp<AddOffsetOp>();
    if (!inner)
      return failure();
    // Rewrite pointer based add_offset only. IntTuple bases are handled by the int_tuple_add rule.
    if (!isa<PointerType>(inner.getPtr().getType()))
      return failure();
    auto innerTy = dyn_cast<IntTupleType>(inner.getOffset().getType());
    auto outerTy = dyn_cast<IntTupleType>(op.getOffset().getType());
    if (!innerTy || !outerTy)
      return failure();
    if (!innerTy.getAttr().isLeafInt() || !outerTy.getAttr().isLeafInt())
      return failure();

    bool innerStatic = innerTy.getAttr().getLeaf().isStatic();
    bool outerStatic = outerTy.getAttr().getLeaf().isStatic();

    // Already canonical.
    if (!innerStatic && outerStatic)
      return failure();

    Location loc = op.getLoc();

    // static -> dynamic: swap so the static offset becomes the outer tail.
    if (innerStatic && !outerStatic) {
      // The swap rebuilds the inner op, so with other users the old one
      // survives and we would add an add_offset instead of replacing one.
      // Fusion below has no such problem: it rewrites only the outer op, so
      // the inner one stays valid for its other users.
      if (!inner->hasOneUse())
        return failure();
      Value dynBase = AddOffsetOp::create(rewriter, loc, inner.getPtr(), op.getOffset());
      rewriter.replaceOpWithNewOp<AddOffsetOp>(op, dynBase, inner.getOffset());
      return success();
    }

    // static -> static and dynamic -> dynamic both fuse: no runtime value is
    // merged with a constant.
    auto ctx = makeDefaultEmitContext(rewriter, loc);
    auto innerOffset = importValue<IntTupleType>(inner.getOffset(), ctx);
    auto outerOffset = importValue<IntTupleType>(op.getOffset(), ctx);
    if (failed(innerOffset) || failed(outerOffset))
      return failure();
    auto sum = materializeValue(*innerOffset + *outerOffset, ctx);
    if (failed(sum))
      return failure();
    rewriter.replaceOpWithNewOp<AddOffsetOp>(op, inner.getPtr(), *sum);
    return success();
  }
};

class MemRefAllocaOpLowering : public OpRewritePattern<MemRefAllocaOp> {
public:
  using OpRewritePattern<MemRefAllocaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MemRefAllocaOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto memrefTy = cast<fly::MemRefType>(op.getResult().getType());
    assert(isa<LayoutAttr>(memrefTy.getLayout()) &&
           "MemRefAllocaOp: doesn't support ComposedLayout");
    LayoutAttr layoutAttr = cast<LayoutAttr>(memrefTy.getLayout());
    IntTupleAttr totalSize =
        IntTupleAttr::get(layoutAttr.getContext(), core::cosize(layoutAttr.getRef()));

    assert(totalSize.isStatic() && totalSize.isLeaf());

    auto ptrAttrs = rewriter.getDictionaryAttr({rewriter.getNamedAttr(
        "allocSize", rewriter.getI64IntegerAttr(totalSize.getLeaf().getValue()))});
    Value flyPtr =
        MakePtrOp::create(rewriter, loc, memrefTy.getPointerType(), ValueRange{}, ptrAttrs);

    rewriter.replaceOpWithNewOp<MakeViewOp>(op, flyPtr, op.getLayout());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Algebra expansion. Every pattern binds its operands to Core EmitOp values,
// evaluates the Core function that the operation's type inference evaluates on
// the Trait category, and materializes the result.
//===----------------------------------------------------------------------===//

using IntTupleEmit = core::IntTuple<core::EmitOp>;

/// Replaces `op` by the materialized Core `value`.
template <class T>
LogicalResult replaceWithMaterialized(Operation *op, const T &value, core::EmitContext &ctx,
                                      ConversionPatternRewriter &rewriter) {
  auto result = materializeValue(value, ctx);
  if (failed(result))
    return rewriter.notifyMatchFailure(op, "cannot materialize the algebra result");
  rewriter.replaceOp(op, *result);
  return success();
}

LogicalResult bindFailure(Operation *op, ConversionPatternRewriter &rewriter) {
  return rewriter.notifyMatchFailure(op, "cannot import the operands");
}

//===----------------------------------------------------------------------===//
// Constructors and extractors
//===----------------------------------------------------------------------===//

struct MakeLayoutLikeOpLowering : OpConversionPattern<MakeLayoutLikeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(MakeLayoutLikeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());
    auto ref = importValue<LayoutType>(adaptor.getRef(), ctx);
    if (failed(ref))
      return bindFailure(op, rewriter);
    return replaceWithMaterialized(op, core::make_layout_like(*ref), ctx, rewriter);
  }
};

struct MakeOrderedLayoutOpLowering : OpConversionPattern<MakeOrderedLayoutOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(MakeOrderedLayoutOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());
    auto shape = importValue<IntTupleType>(adaptor.getShape(), ctx);
    auto order = importValue<IntTupleType>(adaptor.getOrder(), ctx);
    if (failed(shape) || failed(order))
      return bindFailure(op, rewriter);
    return replaceWithMaterialized(op, core::make_ordered_layout(*shape, *order), ctx, rewriter);
  }
};

struct MakeIdentityLayoutOpLowering : OpConversionPattern<MakeIdentityLayoutOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(MakeIdentityLayoutOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());
    auto shape = importValue<IntTupleType>(adaptor.getShape(), ctx);
    if (failed(shape))
      return bindFailure(op, rewriter);
    return replaceWithMaterialized(op, core::make_identity_layout(*shape), ctx, rewriter);
  }
};

/// A fragment layout is fully static: materialize it from the result type.
struct MakeFragmentLayoutLikeOpLowering : OpConversionPattern<MakeFragmentLayoutLikeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(MakeFragmentLayoutLikeOp op, OpAdaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());
    auto layout = core::bind(cast<LayoutType>(op.getType()).getRef(), core::EmitBindings{}, ctx);
    if (failed(layout))
      return bindFailure(op, rewriter);
    return replaceWithMaterialized(op, *layout, ctx, rewriter);
  }
};

struct GetShapeOpLowering : OpConversionPattern<GetShapeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(GetShapeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());
    return llvm::TypeSwitch<Type, LogicalResult>(op.getLayout().getType())
        .Case<LayoutType, ComposedLayoutType, fly::MemRefType, CoordTensorType>([&](auto type) {
          auto value = importValue<decltype(type)>(adaptor.getLayout(), ctx);
          if (failed(value))
            return bindFailure(op, rewriter);
          return replaceWithMaterialized(op, core::shape(*value), ctx, rewriter);
        })
        .Default([&](Type) { return bindFailure(op, rewriter); });
  }
};

struct GetStrideOpLowering : OpConversionPattern<GetStrideOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(GetStrideOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());
    return llvm::TypeSwitch<Type, LogicalResult>(op.getLayout().getType())
        .Case<LayoutType, fly::MemRefType, CoordTensorType>([&](auto type) {
          auto value = importValue<decltype(type)>(adaptor.getLayout(), ctx);
          if (failed(value))
            return bindFailure(op, rewriter);
          return replaceWithMaterialized(op, core::stride(*value), ctx, rewriter);
        })
        .Default([&](Type) { return bindFailure(op, rewriter); });
  }
};

struct GetLayoutOpLowering : OpConversionPattern<GetLayoutOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(GetLayoutOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());
    return llvm::TypeSwitch<Type, LogicalResult>(op.getMemref().getType())
        .Case<fly::MemRefType, CoordTensorType>([&](auto type) {
          auto tensor = importValue<decltype(type)>(adaptor.getMemref(), ctx);
          if (failed(tensor))
            return bindFailure(op, rewriter);
          return replaceWithMaterialized(op, tensor->layout(), ctx, rewriter);
        })
        .Default([&](Type) { return bindFailure(op, rewriter); });
  }
};

/// A coordinate tensor's iterator is its base coordinate. A memref's pointer is
/// not algebra: GetIterLowering folds it or the final conversion extracts it.
struct CoordTensorGetIterOpLowering : OpConversionPattern<GetIterOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(GetIterOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());
    auto tensor = importValue<CoordTensorType>(adaptor.getMemref(), ctx);
    if (failed(tensor))
      return bindFailure(op, rewriter);
    return replaceWithMaterialized(op, tensor->data(), ctx, rewriter);
  }
};

/// composed_get_inner/offset/outer read one part of a bound composed layout.
template <class OpTy> struct ComposedGetLowering : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());
    auto composed = importValue<ComposedLayoutType>(adaptor.getInput(), ctx);
    if (failed(composed))
      return bindFailure(op, rewriter);
    if constexpr (std::is_same_v<OpTy, ComposedGetOffsetOp>) {
      return replaceWithMaterialized(op, composed->offset(), ctx, rewriter);
    } else {
      const auto &part = [&]() -> decltype(auto) {
        if constexpr (std::is_same_v<OpTy, ComposedGetInnerOp>)
          return composed->inner();
        else
          return composed->outer();
      }();
      return std::visit(
          [&](const auto &value) { return replaceWithMaterialized(op, value, ctx, rewriter); },
          part);
    }
  }
};

/// fly.get_leaves of a constructed value reads the constructor operands.
bool hasScalarConstructor(Value value) {
  Operation *producer = value.getDefiningOp();
  return producer && isa<MakeIntTupleOp, MakeShapeOp, MakeStrideOp, MakeCoordOp, MakeLayoutOp,
                         MakeComposedLayoutOp, MakeTileOp, MakeViewOp>(producer);
}

struct GetLeavesOpLowering : OpConversionPattern<GetLeavesOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(GetLeavesOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getOperands()[0];
    if (!hasScalarConstructor(input))
      return failure();
    auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());
    auto bindings =
        llvm::TypeSwitch<Type, FailureOr<core::EmitBindings>>(input.getType())
            .Case<IntTupleType, LayoutType, ComposedLayoutType, TileType, fly::MemRefType,
                  CoordTensorType>([&](auto type) -> FailureOr<core::EmitBindings> {
              auto value = importValue<decltype(type)>(input, ctx);
              if (failed(value))
                return failure();
              return core::get_bindings(*value, ctx);
            })
            .Default([](Type) { return failure(); });
    if (failed(bindings) || TypeRange(bindings->scalars) != op.getResultTypes())
      return failure();
    rewriter.replaceOp(op, bindings->scalars);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// IntTuple operations
//===----------------------------------------------------------------------===//

#define FLY_INT_TUPLE_BINARY_LOWERING(OP, EXPR)                                                    \
  struct OP##Lowering : OpConversionPattern<OP> {                                                  \
    using OpConversionPattern::OpConversionPattern;                                                \
    LogicalResult matchAndRewrite(OP op, OpAdaptor adaptor,                                        \
                                  ConversionPatternRewriter &rewriter) const override {            \
      auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());                                    \
      auto lhs = importValue<IntTupleType>(adaptor.getLhs(), ctx);                                 \
      auto rhs = importValue<IntTupleType>(adaptor.getRhs(), ctx);                                 \
      if (failed(lhs) || failed(rhs))                                                              \
        return bindFailure(op, rewriter);                                                          \
      return replaceWithMaterialized(op, IntTupleEmit(EXPR), ctx, rewriter);                       \
    }                                                                                              \
  };

FLY_INT_TUPLE_BINARY_LOWERING(IntTupleAddOp, *lhs + *rhs)
FLY_INT_TUPLE_BINARY_LOWERING(IntTupleSubOp, *lhs - *rhs)
FLY_INT_TUPLE_BINARY_LOWERING(IntTupleMulOp, *lhs **rhs)
FLY_INT_TUPLE_BINARY_LOWERING(IntTupleDivOp, *lhs / *rhs)
FLY_INT_TUPLE_BINARY_LOWERING(IntTupleModOp, *lhs % *rhs)
FLY_INT_TUPLE_BINARY_LOWERING(IntTupleProductLikeOp, core::product_like(*lhs, *rhs))
FLY_INT_TUPLE_BINARY_LOWERING(ShapeDivOp, core::shape_div(*lhs, *rhs))
FLY_INT_TUPLE_BINARY_LOWERING(CeilDivOp, core::ceil_div(*lhs, *rhs))

#undef FLY_INT_TUPLE_BINARY_LOWERING

#define FLY_INT_TUPLE_UNARY_LOWERING(OP, FN)                                                       \
  struct OP##Lowering : OpConversionPattern<OP> {                                                  \
    using OpConversionPattern::OpConversionPattern;                                                \
    LogicalResult matchAndRewrite(OP op, OpAdaptor adaptor,                                        \
                                  ConversionPatternRewriter &rewriter) const override {            \
      auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());                                    \
      auto input = importValue<IntTupleType>(adaptor.getInput(), ctx);                             \
      if (failed(input))                                                                           \
        return bindFailure(op, rewriter);                                                          \
      return replaceWithMaterialized(op, IntTupleEmit(core::FN(*input)), ctx, rewriter);           \
    }                                                                                              \
  };

FLY_INT_TUPLE_UNARY_LOWERING(IntTupleProductOp, product)
FLY_INT_TUPLE_UNARY_LOWERING(IntTupleProductEachOp, product_each)

#undef FLY_INT_TUPLE_UNARY_LOWERING

/// Comparisons produce an i1 predicate rather than a Fly int tuple.
template <class OpTy, bool ElementLess> struct ComparisonLowering : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());
    auto lhs = importValue<IntTupleType>(adaptor.getLhs(), ctx);
    auto rhs = importValue<IntTupleType>(adaptor.getRhs(), ctx);
    if (failed(lhs) || failed(rhs))
      return bindFailure(op, rewriter);
    auto result = ElementLess ? core::elem_less(*lhs, *rhs) : core::eq(*lhs, *rhs);
    if (!result.isLeaf() || result.isError())
      return rewriter.notifyMatchFailure(op, "algebra comparison failed");
    auto predicate = result.getLeaf().materialize(ctx);
    if (failed(predicate))
      return rewriter.notifyMatchFailure(op, "cannot materialize the predicate");
    if (!predicate->getType().isInteger(1)) {
      auto type = cast<IntegerType>(predicate->getType());
      auto zero = arith::ConstantIntOp::create(rewriter, op.getLoc(), 0, type.getWidth());
      predicate =
          arith::CmpIOp::create(rewriter, op.getLoc(), arith::CmpIPredicate::ne, *predicate, zero)
              .getResult();
    }
    rewriter.replaceOp(op, *predicate);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// IntTupleLike operations
//===----------------------------------------------------------------------===//

/// Operations on one int-tuple-like operand, evaluated by `apply(value)`.
#define FLY_INT_TUPLE_LIKE_LOWERING(OP, GETTER, APPLY)                                             \
  struct OP##Lowering : OpConversionPattern<OP> {                                                  \
    using OpConversionPattern::OpConversionPattern;                                                \
    LogicalResult matchAndRewrite(OP op, OpAdaptor adaptor,                                        \
                                  ConversionPatternRewriter &rewriter) const override {            \
      auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());                                    \
      auto apply = APPLY;                                                                          \
      return llvm::TypeSwitch<Type, LogicalResult>(op.GETTER().getType())                          \
          .Case<IntTupleType, LayoutType, ComposedLayoutType, fly::MemRefType, CoordTensorType>(   \
              [&](auto type) {                                                                     \
                auto value = importValue<decltype(type)>(adaptor.GETTER(), ctx);                   \
                if (failed(value))                                                                 \
                  return bindFailure(op, rewriter);                                                \
                return replaceWithMaterialized(op, apply(*value), ctx, rewriter);                  \
              })                                                                                   \
          .Default([&](Type) { return bindFailure(op, rewriter); });                               \
    }                                                                                              \
  };

FLY_INT_TUPLE_LIKE_LOWERING(TakeOp, getTuple, [&](const auto &value) {
  return core::legacy_take(value, op.getBegin(), op.getEnd());
})
FLY_INT_TUPLE_LIKE_LOWERING(SelectOp, getTuple, [&](const auto &value) {
  return core::legacy_select(value, op.getIndices());
})
FLY_INT_TUPLE_LIKE_LOWERING(GroupOp, getTuple, [&](const auto &value) {
  return core::group(value, op.getBegin(), op.getEnd());
})
FLY_INT_TUPLE_LIKE_LOWERING(SizeOp, getIntTuple,
                            [&](const auto &value) { return IntTupleEmit(core::size(value)); })
FLY_INT_TUPLE_LIKE_LOWERING(GetOp, getInput, [&](const auto &input) {
  using Value = std::decay_t<decltype(input)>;
  Value value = input;
  for (int32_t index : op.getMode()) {
    Value child = core::get(value, index);
    value = std::move(child);
  }
  return value;
})

#undef FLY_INT_TUPLE_LIKE_LOWERING

/// Operations with an int-tuple coordinate and an int-tuple-like operand.
template <class OpTy, bool Slice> struct SliceOrDiceLowering : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());
    auto coord = importValue<IntTupleType>(adaptor.getCoord(), ctx);
    if (failed(coord))
      return bindFailure(op, rewriter);
    return llvm::TypeSwitch<Type, LogicalResult>(op.getSrc().getType())
        .template Case<IntTupleType, LayoutType, ComposedLayoutType, fly::MemRefType,
                       CoordTensorType>([&](auto type) {
          auto value = importValue<decltype(type)>(adaptor.getSrc(), ctx);
          if (failed(value))
            return bindFailure(op, rewriter);
          if constexpr (Slice)
            return replaceWithMaterialized(op, core::legacy_slice(*coord, *value), ctx, rewriter);
          else
            return replaceWithMaterialized(op, core::legacy_dice(*coord, *value), ctx, rewriter);
        })
        .Default([&](Type) { return bindFailure(op, rewriter); });
  }
};

template <class OpTy, bool Append> struct AppendOrPrependLowering : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());
    int32_t n = op.getN() ? static_cast<int32_t>(*op.getN()) : -1;
    auto apply = [&](const auto &tuple, const auto &elem) {
      if constexpr (Append)
        return n == -1 ? core::append(tuple, elem) : core::append(tuple, elem, n);
      else
        return n == -1 ? core::prepend(tuple, elem) : core::prepend(tuple, elem, n);
    };
    return llvm::TypeSwitch<Type, LogicalResult>(op.getTuple().getType())
        .Case([&](IntTupleType) {
          auto tuple = importValue<IntTupleType>(adaptor.getTuple(), ctx);
          auto elem = importValue<IntTupleType>(adaptor.getElem(), ctx);
          if (failed(tuple) || failed(elem))
            return bindFailure(op, rewriter);
          return replaceWithMaterialized(op, apply(*tuple, *elem), ctx, rewriter);
        })
        .template Case<LayoutType, ComposedLayoutType>([&](auto type) {
          auto tuple = importValue<decltype(type)>(adaptor.getTuple(), ctx);
          auto elem = importValue<LayoutType>(adaptor.getElem(), ctx);
          if (failed(tuple) || failed(elem))
            return bindFailure(op, rewriter);
          return replaceWithMaterialized(op, apply(*tuple, *elem), ctx, rewriter);
        })
        .Default([&](Type) { return bindFailure(op, rewriter); });
  }
};

//===----------------------------------------------------------------------===//
// LayoutLike operations
//===----------------------------------------------------------------------===//

/// cosize/coshape/coprofile of a layout, or of the layout of a tensor.
#define FLY_LAYOUT_QUERY_LOWERING(OP, FN)                                                          \
  struct OP##Lowering : OpConversionPattern<OP> {                                                  \
    using OpConversionPattern::OpConversionPattern;                                                \
    LogicalResult matchAndRewrite(OP op, OpAdaptor adaptor,                                        \
                                  ConversionPatternRewriter &rewriter) const override {            \
      auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());                                    \
      auto query = [&](const auto &layout) {                                                       \
        return replaceWithMaterialized(op, IntTupleEmit(core::FN(layout)), ctx, rewriter);         \
      };                                                                                           \
      return llvm::TypeSwitch<Type, LogicalResult>(op.getLayout().getType())                       \
          .Case<LayoutType, ComposedLayoutType>([&](auto type) {                                   \
            auto value = importValue<decltype(type)>(adaptor.getLayout(), ctx);                    \
            return failed(value) ? bindFailure(op, rewriter) : query(*value);                      \
          })                                                                                       \
          .Case<fly::MemRefType, CoordTensorType>([&](auto type) {                                 \
            auto value = importValue<decltype(type)>(adaptor.getLayout(), ctx);                    \
            return failed(value) ? bindFailure(op, rewriter) : std::visit(query, value->layout()); \
          })                                                                                       \
          .Default([&](Type) { return bindFailure(op, rewriter); });                               \
    }                                                                                              \
  };

FLY_LAYOUT_QUERY_LOWERING(CoprofileOp, coprofile)
FLY_LAYOUT_QUERY_LOWERING(CoshapeOp, coshape)
FLY_LAYOUT_QUERY_LOWERING(CosizeOp, cosize)

#undef FLY_LAYOUT_QUERY_LOWERING

struct Crd2IdxOpLowering : OpConversionPattern<Crd2IdxOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(Crd2IdxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());
    auto coord = importValue<IntTupleType>(adaptor.getCoord(), ctx);
    if (failed(coord))
      return bindFailure(op, rewriter);
    return llvm::TypeSwitch<Type, LogicalResult>(op.getLayout().getType())
        .Case<LayoutType, ComposedLayoutType>([&](auto type) {
          auto layout = importValue<decltype(type)>(adaptor.getLayout(), ctx);
          if (failed(layout))
            return bindFailure(op, rewriter);
          return replaceWithMaterialized(op, IntTupleEmit(core::crd2idx(*coord, *layout)), ctx,
                                         rewriter);
        })
        .Case<SwizzleType, CoordSwizzleType>([&](auto type) {
          auto swizzle = importValue<decltype(type)>(adaptor.getLayout(), ctx);
          if (failed(swizzle))
            return bindFailure(op, rewriter);
          return replaceWithMaterialized(op, core::swizzle_apply(*swizzle, *coord), ctx, rewriter);
        })
        .Default([&](Type) { return bindFailure(op, rewriter); });
  }
};

/// idx2crd/get_flat_coord/get_1d_coord of an index through a plain layout.
#define FLY_LAYOUT_COORD_LOWERING(OP, MEMBER)                                                      \
  struct OP##Lowering : OpConversionPattern<OP> {                                                  \
    using OpConversionPattern::OpConversionPattern;                                                \
    LogicalResult matchAndRewrite(OP op, OpAdaptor adaptor,                                        \
                                  ConversionPatternRewriter &rewriter) const override {            \
      auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());                                    \
      auto index = importValue<IntTupleType>(adaptor.getOperands()[0], ctx);                       \
      auto layout = importValue<LayoutType>(adaptor.getOperands()[1], ctx);                        \
      if (failed(index) || failed(layout))                                                         \
        return bindFailure(op, rewriter);                                                          \
      return replaceWithMaterialized(op, IntTupleEmit(layout->MEMBER(*index)), ctx, rewriter);     \
    }                                                                                              \
  };

FLY_LAYOUT_COORD_LOWERING(Idx2CrdOp, get_hier_coord)
FLY_LAYOUT_COORD_LOWERING(GetFlatCoordOp, get_flat_coord)
FLY_LAYOUT_COORD_LOWERING(Get1DCoordOp, get_1d_coord)

#undef FLY_LAYOUT_COORD_LOWERING

struct CoalesceOpLowering : OpConversionPattern<CoalesceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(CoalesceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());
    FailureOr<IntTupleEmit> profile = failure();
    if (adaptor.getPattern()) {
      profile = importValue<IntTupleType>(adaptor.getPattern(), ctx);
      if (failed(profile))
        return bindFailure(op, rewriter);
    }
    return llvm::TypeSwitch<Type, LogicalResult>(op.getLayout().getType())
        .Case<LayoutType, ComposedLayoutType, fly::MemRefType, CoordTensorType>([&](auto type) {
          auto value = importValue<decltype(type)>(adaptor.getLayout(), ctx);
          if (failed(value))
            return bindFailure(op, rewriter);
          if (succeeded(profile))
            return replaceWithMaterialized(op, core::coalesce(*value, *profile), ctx, rewriter);
          return replaceWithMaterialized(op, core::coalesce(*value), ctx, rewriter);
        })
        .Default([&](Type) { return bindFailure(op, rewriter); });
  }
};

/// A layout-like operand combined with a layout or tile through `FN`.
#define FLY_WITH_TILER_LOWERING(OP, LHS, RHS, FN)                                                  \
  struct OP##Lowering : OpConversionPattern<OP> {                                                  \
    using OpConversionPattern::OpConversionPattern;                                                \
    LogicalResult matchAndRewrite(OP op, OpAdaptor adaptor,                                        \
                                  ConversionPatternRewriter &rewriter) const override {            \
      auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());                                    \
      Type rhsType = op.RHS().getType();                                                           \
      return llvm::TypeSwitch<Type, LogicalResult>(op.LHS().getType())                             \
          .Case<LayoutType, ComposedLayoutType, fly::MemRefType, CoordTensorType>(                 \
              [&](auto lhsType) {                                                                  \
                auto lhs = importValue<decltype(lhsType)>(adaptor.LHS(), ctx);                     \
                if (failed(lhs))                                                                   \
                  return bindFailure(op, rewriter);                                                \
                return llvm::TypeSwitch<Type, LogicalResult>(rhsType)                              \
                    .Case<LayoutType, TileType>([&](auto type) {                                   \
                      auto rhs = importValue<decltype(type)>(adaptor.RHS(), ctx);                  \
                      if (failed(rhs))                                                             \
                        return bindFailure(op, rewriter);                                          \
                      return replaceWithMaterialized(op, core::FN(*lhs, *rhs), ctx, rewriter);     \
                    })                                                                             \
                    .Default([&](Type) { return bindFailure(op, rewriter); });                     \
              })                                                                                   \
          .Default([&](Type) { return bindFailure(op, rewriter); });                               \
    }                                                                                              \
  };

FLY_WITH_TILER_LOWERING(CompositionOp, getOuter, getInner, composition)
FLY_WITH_TILER_LOWERING(LogicalDivideOp, getLayout, getDivisor, logical_divide)
FLY_WITH_TILER_LOWERING(ZippedDivideOp, getLayout, getDivisor, zipped_divide)
FLY_WITH_TILER_LOWERING(TiledDivideOp, getLayout, getDivisor, tiled_divide)
FLY_WITH_TILER_LOWERING(FlatDivideOp, getLayout, getDivisor, flat_divide)

#undef FLY_WITH_TILER_LOWERING

struct ComplementOpLowering : OpConversionPattern<ComplementOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(ComplementOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());
    FailureOr<IntTupleEmit> codomain = failure();
    if (adaptor.getCodomainSize()) {
      codomain = importValue<IntTupleType>(adaptor.getCodomainSize(), ctx);
      if (failed(codomain))
        return bindFailure(op, rewriter);
    }
    return llvm::TypeSwitch<Type, LogicalResult>(op.getLayout().getType())
        .Case<LayoutType, ComposedLayoutType>([&](auto type) {
          auto layout = importValue<decltype(type)>(adaptor.getLayout(), ctx);
          if (failed(layout))
            return bindFailure(op, rewriter);
          if (succeeded(codomain))
            return replaceWithMaterialized(op, core::complement(*layout, *codomain), ctx, rewriter);
          return replaceWithMaterialized(op, core::complement(*layout), ctx, rewriter);
        })
        .Default([&](Type) { return bindFailure(op, rewriter); });
  }
};

/// Operations on one plain layout operand, evaluated by `APPLY`.
#define FLY_LAYOUT_LOWERING(OP, APPLY)                                                             \
  struct OP##Lowering : OpConversionPattern<OP> {                                                  \
    using OpConversionPattern::OpConversionPattern;                                                \
    LogicalResult matchAndRewrite(OP op, OpAdaptor adaptor,                                        \
                                  ConversionPatternRewriter &rewriter) const override {            \
      auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());                                    \
      auto layout = importValue<LayoutType>(adaptor.getLayout(), ctx);                             \
      if (failed(layout))                                                                          \
        return bindFailure(op, rewriter);                                                          \
      return replaceWithMaterialized(op, APPLY(*layout), ctx, rewriter);                           \
    }                                                                                              \
  };

FLY_LAYOUT_LOWERING(RightInverseOp, core::right_inverse)
FLY_LAYOUT_LOWERING(LeftInverseOp, core::left_inverse)

/// Operations on one (composed) layout operand, evaluated by `APPLY`.
#define FLY_NARROW_LAYOUT_LOWERING(OP, GETTER, APPLY)                                              \
  struct OP##Lowering : OpConversionPattern<OP> {                                                  \
    using OpConversionPattern::OpConversionPattern;                                                \
    LogicalResult matchAndRewrite(OP op, OpAdaptor adaptor,                                        \
                                  ConversionPatternRewriter &rewriter) const override {            \
      auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());                                    \
      auto apply = APPLY;                                                                          \
      return llvm::TypeSwitch<Type, LogicalResult>(op.GETTER().getType())                          \
          .Case<LayoutType, ComposedLayoutType>([&](auto type) {                                   \
            auto layout = importValue<decltype(type)>(adaptor.GETTER(), ctx);                      \
            if (failed(layout))                                                                    \
              return bindFailure(op, rewriter);                                                    \
            return apply(*layout);                                                                 \
          })                                                                                       \
          .Default([&](Type) { return bindFailure(op, rewriter); });                               \
    }                                                                                              \
  };

FLY_NARROW_LAYOUT_LOWERING(RecastLayoutOp, getSrc, [&](const auto &layout) {
  return replaceWithMaterialized(
      op, core::recast_layout(op.getNewTypeBits(), op.getOldTypeBits(), layout), ctx, rewriter);
})
FLY_NARROW_LAYOUT_LOWERING(TileToShapeOp, getBlock, [&](const auto &layout) {
  auto shape = importValue<IntTupleType>(adaptor.getTrgShape(), ctx);
  auto order = importValue<IntTupleType>(adaptor.getOrdShape(), ctx);
  if (failed(shape) || failed(order))
    return bindFailure(op, rewriter);
  return replaceWithMaterialized(op, core::tile_to_shape(layout, *shape, *order), ctx, rewriter);
})

#define FLY_PRODUCT_LOWERING(OP, FN)                                                               \
  FLY_NARROW_LAYOUT_LOWERING(OP, getLayout, [&](const auto &layout) {                              \
    auto tile = importValue<LayoutType>(adaptor.getTile(), ctx);                                   \
    if (failed(tile))                                                                              \
      return bindFailure(op, rewriter);                                                            \
    return replaceWithMaterialized(op, core::FN(layout, *tile), ctx, rewriter);                    \
  })

FLY_PRODUCT_LOWERING(LogicalProductOp, logical_product)
FLY_PRODUCT_LOWERING(ZippedProductOp, zipped_product)
FLY_PRODUCT_LOWERING(TiledProductOp, tiled_product)
FLY_PRODUCT_LOWERING(FlatProductOp, flat_product)
FLY_PRODUCT_LOWERING(BlockedProductOp, blocked_product)
FLY_PRODUCT_LOWERING(RakedProductOp, raked_product)

#undef FLY_PRODUCT_LOWERING
#undef FLY_NARROW_LAYOUT_LOWERING
#undef FLY_LAYOUT_LOWERING

//===----------------------------------------------------------------------===//
// Tiled copy and MMA partitions
//===----------------------------------------------------------------------===//

template <class OpTy, bool Source> struct TiledCopyPartitionLowering : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());
    auto tiled = importValue<TiledCopyType>(op.getTiledCopy(), ctx);
    auto thr = importValue<IntTupleType>(adaptor.getCoord(), ctx);
    if (failed(tiled) || failed(thr))
      return bindFailure(op, rewriter);
    Value input = adaptor.getOperands()[1];
    return llvm::TypeSwitch<Type, LogicalResult>(op->getOperand(1).getType())
        .template Case<fly::MemRefType, CoordTensorType>([&](auto type) {
          auto tensor = importValue<decltype(type)>(input, ctx);
          if (failed(tensor))
            return bindFailure(op, rewriter);
          if constexpr (Source)
            return replaceWithMaterialized(op, tiled->partition_S(*tensor, *thr), ctx, rewriter);
          else
            return replaceWithMaterialized(op, tiled->partition_D(*tensor, *thr), ctx, rewriter);
        })
        .Default([&](Type) { return bindFailure(op, rewriter); });
  }
};

struct TiledCopyRetileOpLowering : OpConversionPattern<TiledCopyRetileOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(TiledCopyRetileOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());
    auto tiled = importValue<TiledCopyType>(op.getTiledCopy(), ctx);
    auto tensor = importValue<fly::MemRefType>(adaptor.getInput(), ctx);
    if (failed(tiled) || failed(tensor))
      return bindFailure(op, rewriter);
    return replaceWithMaterialized(op, tiled->retile(*tensor), ctx, rewriter);
  }
};

/// partition_A/B/C of `tensor` for one thread of a tiled MMA; D uses C.
template <class Engine>
auto partitionMma(const core::ThrMMA<core::EmitOp> &thr, MmaOperand operand,
                  const core::Tensor<core::EmitOp, Engine> &tensor) {
  switch (operand) {
  case MmaOperand::A:
    return thr.partition_A(tensor);
  case MmaOperand::B:
    return thr.partition_B(tensor);
  case MmaOperand::C:
  case MmaOperand::D:
    return thr.partition_C(tensor);
  }
  llvm_unreachable("unknown MMA operand");
}

struct TiledMmaPartitionOpLowering : OpConversionPattern<TiledMmaPartitionOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(TiledMmaPartitionOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());
    auto tiled = importValue<TiledMmaType>(op.getTiledMma(), ctx);
    auto thr = importValue<IntTupleType>(adaptor.getCoord(), ctx);
    if (failed(tiled) || failed(thr))
      return bindFailure(op, rewriter);
    auto slice = tiled->get_slice(*thr);
    return llvm::TypeSwitch<Type, LogicalResult>(op.getInput().getType())
        .Case<fly::MemRefType, CoordTensorType>([&](auto type) {
          auto tensor = importValue<decltype(type)>(adaptor.getInput(), ctx);
          if (failed(tensor))
            return bindFailure(op, rewriter);
          return replaceWithMaterialized(op, partitionMma(slice, op.getOperandId(), *tensor), ctx,
                                         rewriter);
        })
        .Default([&](Type) { return bindFailure(op, rewriter); });
  }
};

struct TiledMmaPartitionShapeOpLowering : OpConversionPattern<TiledMmaPartitionShapeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(TiledMmaPartitionShapeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());
    auto tiled = importValue<TiledMmaType>(op.getTiledMma(), ctx);
    auto shape = importValue<IntTupleType>(adaptor.getShape(), ctx);
    if (failed(tiled) || failed(shape))
      return bindFailure(op, rewriter);
    switch (op.getOperandId()) {
    case MmaOperand::A:
      return replaceWithMaterialized(op, core::partition_shape_A(*tiled, *shape), ctx, rewriter);
    case MmaOperand::B:
      return replaceWithMaterialized(op, core::partition_shape_B(*tiled, *shape), ctx, rewriter);
    case MmaOperand::C:
    case MmaOperand::D:
      return replaceWithMaterialized(op, core::partition_shape_C(*tiled, *shape), ctx, rewriter);
    }
    llvm_unreachable("unknown MMA operand");
  }
};

//===----------------------------------------------------------------------===//
// Tensor and iterator operations
//===----------------------------------------------------------------------===//

struct DecompositionOpLowering : OpConversionPattern<DecompositionOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(DecompositionOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());
    return llvm::TypeSwitch<Type, LogicalResult>(op.getTensor().getType())
        .Case<fly::MemRefType, CoordTensorType>([&](auto type) {
          auto tensor = importValue<decltype(type)>(adaptor.getTensor(), ctx);
          if (failed(tensor))
            return bindFailure(op, rewriter);
          return replaceWithMaterialized(op, core::decompose(*tensor), ctx, rewriter);
        })
        .Default([&](Type) { return bindFailure(op, rewriter); });
  }
};

/// The iterator of `tensor` advanced to the element at `indices`.
template <class Engine>
auto elementIterator(const core::Tensor<core::EmitOp, Engine> &tensor,
                     const IntTupleEmit &indices) {
  auto offset = std::visit([&](const auto &layout) { return core::crd2idx(indices, layout); },
                           tensor.layout());
  return core::add_offset(tensor.data(), offset);
}

struct MemRefLoadOpLowering : OpConversionPattern<MemRefLoadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(MemRefLoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());
    auto indices = importValue<IntTupleType>(adaptor.getIndices(), ctx);
    if (failed(indices))
      return bindFailure(op, rewriter);
    return llvm::TypeSwitch<Type, LogicalResult>(op.getMemref().getType())
        .Case([&](fly::MemRefType) -> LogicalResult {
          auto tensor = importValue<fly::MemRefType>(adaptor.getMemref(), ctx);
          if (failed(tensor))
            return bindFailure(op, rewriter);
          auto pointer = materializeValue(elementIterator(*tensor, *indices), ctx);
          if (failed(pointer))
            return rewriter.notifyMatchFailure(op, "cannot materialize the element pointer");
          rewriter.replaceOpWithNewOp<PtrLoadOp>(op, op.getType(), *pointer);
          return success();
        })
        .Case([&](CoordTensorType) {
          // Loading a coordinate tensor yields the coordinate at `indices`.
          auto tensor = importValue<CoordTensorType>(adaptor.getMemref(), ctx);
          if (failed(tensor))
            return bindFailure(op, rewriter);
          return replaceWithMaterialized(op, elementIterator(*tensor, *indices), ctx, rewriter);
        })
        .Default([&](Type) { return bindFailure(op, rewriter); });
  }
};

struct MemRefStoreOpLowering : OpConversionPattern<MemRefStoreOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(MemRefStoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());
    auto tensor = importValue<fly::MemRefType>(adaptor.getMemref(), ctx);
    auto indices = importValue<IntTupleType>(adaptor.getIndices(), ctx);
    if (failed(tensor) || failed(indices))
      return bindFailure(op, rewriter);
    auto pointer = materializeValue(elementIterator(*tensor, *indices), ctx);
    if (failed(pointer))
      return rewriter.notifyMatchFailure(op, "cannot materialize the element pointer");
    rewriter.replaceOpWithNewOp<PtrStoreOp>(op, adaptor.getValue(), *pointer);
    return success();
  }
};

/// An int-tuple iterator offsets by addition and loads as itself.
struct AddOffsetOpLowering : OpConversionPattern<AddOffsetOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(AddOffsetOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto ctx = makeDefaultEmitContext(rewriter, op.getLoc());
    auto base = importValue<IntTupleType>(adaptor.getPtr(), ctx);
    auto offset = importValue<IntTupleType>(adaptor.getOffset(), ctx);
    if (failed(base) || failed(offset))
      return bindFailure(op, rewriter);
    return replaceWithMaterialized(op, IntTupleEmit(*base + *offset), ctx, rewriter);
  }
};

struct PtrLoadOpLowering : OpConversionPattern<PtrLoadOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(PtrLoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getPtr());
    return success();
  }
};

LogicalResult lowerAlgebraOperations(Operation *root) {
  MLIRContext *ctx = root->getContext();
  ConversionTarget target(*ctx);
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  target.addIllegalOp<
      MakeLayoutLikeOp, MakeOrderedLayoutOp, MakeIdentityLayoutOp, MakeFragmentLayoutLikeOp,
      GetShapeOp, GetStrideOp, GetLayoutOp, ComposedGetInnerOp, ComposedGetOffsetOp,
      ComposedGetOuterOp, IntTupleAddOp, IntTupleSubOp, IntTupleMulOp, IntTupleDivOp, IntTupleModOp,
      IntTupleProductLikeOp, ShapeDivOp, CeilDivOp, IntTupleProductOp, IntTupleProductEachOp,
      ElemLessOp, EqualOp, GetOp, TakeOp, SelectOp, GroupOp, AppendOp, PrependOp, SliceOp, DiceOp,
      SizeOp, CoprofileOp, CoshapeOp, CosizeOp, Crd2IdxOp, Idx2CrdOp, GetFlatCoordOp, Get1DCoordOp,
      CoalesceOp, CompositionOp, ComplementOp, RightInverseOp, LeftInverseOp, LogicalDivideOp,
      ZippedDivideOp, TiledDivideOp, FlatDivideOp, LogicalProductOp, ZippedProductOp,
      TiledProductOp, FlatProductOp, BlockedProductOp, RakedProductOp, RecastLayoutOp,
      TileToShapeOp, TiledCopyPartitionSrcOp, TiledCopyPartitionDstOp, TiledCopyRetileOp,
      TiledMmaPartitionOp, TiledMmaPartitionShapeOp, DecompositionOp, MemRefLoadOp, MemRefStoreOp,
      CopyOp, GemmOp>();

  target.addDynamicallyLegalOp<AddOffsetOp>(
      [](AddOffsetOp op) { return !isa<IntTupleType>(op.getPtr().getType()); });
  target.addDynamicallyLegalOp<PtrLoadOp>(
      [](PtrLoadOp op) { return !isa<IntTupleType>(op.getPtr().getType()); });
  target.addDynamicallyLegalOp<GetLeavesOp>(
      [](GetLeavesOp op) { return !hasScalarConstructor(op.getInput()); });
  target.addDynamicallyLegalOp<GetIterOp>(
      [](GetIterOp op) { return !isa<CoordTensorType>(op.getMemref().getType()); });

  RewritePatternSet patterns(ctx);
  patterns.add<MakeLayoutLikeOpLowering, MakeOrderedLayoutOpLowering, MakeIdentityLayoutOpLowering,
               MakeFragmentLayoutLikeOpLowering, GetShapeOpLowering, GetStrideOpLowering,
               GetLayoutOpLowering, CoordTensorGetIterOpLowering,
               ComposedGetLowering<ComposedGetInnerOp>, ComposedGetLowering<ComposedGetOffsetOp>,
               ComposedGetLowering<ComposedGetOuterOp>, GetLeavesOpLowering>(ctx);
  patterns.add<IntTupleAddOpLowering, IntTupleSubOpLowering, IntTupleMulOpLowering,
               IntTupleDivOpLowering, IntTupleModOpLowering, IntTupleProductLikeOpLowering,
               ShapeDivOpLowering, CeilDivOpLowering, IntTupleProductOpLowering,
               IntTupleProductEachOpLowering, ComparisonLowering<ElemLessOp, true>,
               ComparisonLowering<EqualOp, false>>(ctx);
  patterns.add<GetOpLowering, TakeOpLowering, SelectOpLowering, GroupOpLowering,
               AppendOrPrependLowering<AppendOp, true>, AppendOrPrependLowering<PrependOp, false>,
               SliceOrDiceLowering<SliceOp, true>, SliceOrDiceLowering<DiceOp, false>>(ctx);
  patterns.add<SizeOpLowering, CoprofileOpLowering, CoshapeOpLowering, CosizeOpLowering,
               Crd2IdxOpLowering, Idx2CrdOpLowering, GetFlatCoordOpLowering, Get1DCoordOpLowering>(
      ctx);
  patterns.add<CoalesceOpLowering, CompositionOpLowering, ComplementOpLowering,
               RightInverseOpLowering, LeftInverseOpLowering, LogicalDivideOpLowering,
               ZippedDivideOpLowering, TiledDivideOpLowering, FlatDivideOpLowering,
               LogicalProductOpLowering, ZippedProductOpLowering, TiledProductOpLowering,
               FlatProductOpLowering, BlockedProductOpLowering, RakedProductOpLowering,
               RecastLayoutOpLowering, TileToShapeOpLowering>(ctx);
  patterns
      .add<TiledCopyPartitionLowering<TiledCopyPartitionSrcOp, true>,
           TiledCopyPartitionLowering<TiledCopyPartitionDstOp, false>, TiledCopyRetileOpLowering,
           TiledMmaPartitionOpLowering, TiledMmaPartitionShapeOpLowering>(ctx);
  patterns.add<DecompositionOpLowering, MemRefLoadOpLowering, MemRefStoreOpLowering,
               AddOffsetOpLowering, PtrLoadOpLowering>(ctx);

  patterns.add<ExpandCopyOpLowering, ExpandGemmOpLowering>(ctx);

  return applyPartialConversion(root, target, std::move(patterns));
}

class FlyLayoutLoweringPass
    : public mlir::fly::impl::FlyLayoutLoweringPassBase<FlyLayoutLoweringPass> {
public:
  using mlir::fly::impl::FlyLayoutLoweringPassBase<
      FlyLayoutLoweringPass>::FlyLayoutLoweringPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true).setRegionSimplificationLevel(
        GreedySimplifyRegionLevel::Disabled);

    // Expand the structured operations. The algebra they produce is lowered by
    // the Core conversion below.
    RewritePatternSet patterns(context);
    patterns.add<MakeFragmentLikeOpLowering, MmaMakeFragmentOpLowering>(context);
    patterns.add<MemRefLoadVecOpLowering, MemRefStoreVecOpLowering, MemRefAllocaOpLowering>(
        context);
    patterns.add<AddOffsetCanonicalization, RedundantIndexCastPair>(context);
    patterns.add<PrintOpLowering>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      return signalPassFailure();

    if (failed(lowerAlgebraOperations(getOperation())))
      return signalPassFailure();

    // Fold the iterator extractions and offsets exposed by the Core conversion.
    RewritePatternSet cleanup(context);
    cleanup.add<GetIterLowering, GetLeavesOfConstructor, AddOffsetCanonicalization,
                RedundantIndexCastPair>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(cleanup), config)))
      return signalPassFailure();
    IRRewriter rewriter(context);
    DominanceInfo domInfo(getOperation());
    eliminateCommonSubExpressions(rewriter, domInfo, getOperation());
  }
};

} // namespace
