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
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallPtrSet.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Transforms/Passes.h"
#include "flydsl/Dialect/Fly/Utils/IntTupleUtils.h"
#include "flydsl/Dialect/Fly/Utils/LayoutUtils.h"
#include "flydsl/Dialect/Fly/Utils/NormalForm.h"
#include "flydsl/Dialect/Fly/Utils/TiledOpUtils.h"

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
      format += "%f";
      if (floatTy.getWidth() < 32) {
        return arith::ExtFOp::create(rewriter, loc, rewriter.getF32Type(), value);
      }
      return value;
    }
    format += "%lf";
    if (floatTy.getWidth() != 64) {
      return arith::ExtFOp::create(rewriter, loc, rewriter.getF64Type(), value);
    }
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

bool appendIntTuplePrintf(PatternRewriter &rewriter, Location loc,
                          const IntTupleValueAdaptor &tuple, std::string &format,
                          SmallVectorImpl<Value> &args) {
  if (tuple.isLeaf()) {
    Value leafValue = tuple.getValue();
    return appendScalarPrintfArg(rewriter, loc, leafValue, format, args);
  }

  IntTupleBuilder<IntTupleValueAdaptor> builder(rewriter, loc);
  format += "(";
  for (int i = 0; i < tuple.rank(); ++i) {
    if (i > 0) {
      format += ",";
    }
    if (!appendIntTuplePrintf(rewriter, loc, builder.at(tuple, i), format, args)) {
      return false;
    }
  }
  format += ")";
  return true;
}

bool appendIntTuplePrintfStatic(IntTupleAttr attr, std::string &format) {
  if (attr.isLeaf()) {
    if (attr.getLeafAsInt().isStatic()) {
      format += std::to_string(attr.getLeafAsInt().getValue());
    } else {
      format += "?";
    }
    return true;
  }

  format += "(";
  for (int i = 0; i < attr.rank(); ++i) {
    if (i > 0) {
      format += ",";
    }
    if (!appendIntTuplePrintfStatic(attr.at(i), format)) {
      return false;
    }
  }
  format += ")";
  return true;
}

//===----------------------------------------------------------------------===//
// Constructors
//===----------------------------------------------------------------------===//

class MakeLayoutLikeOpLowering : public OpRewritePattern<MakeLayoutLikeOp> {
public:
  using OpRewritePattern<MakeLayoutLikeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MakeLayoutLikeOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value layoutValue = op.getRef();
    auto layoutTy = dyn_cast<LayoutType>(layoutValue.getType());
    if (!layoutTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(layoutValue)))
      return failure();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor layoutAdaptor(layoutValue, layoutTy.getAttr());
    LayoutValueAdaptor result = layoutMakeLayoutLike(layoutBuilder, layoutAdaptor);
    rewriter.replaceOp(op, result.getValue());
    return success();
  }
};

class MakeOrderedLayoutOpLowering : public OpRewritePattern<MakeOrderedLayoutOp> {
public:
  using OpRewritePattern<MakeOrderedLayoutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MakeOrderedLayoutOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value shapeValue = op.getShape();
    Value orderValue = op.getOrder();

    auto shapeTy = dyn_cast<IntTupleType>(shapeValue.getType());
    auto orderTy = dyn_cast<IntTupleType>(orderValue.getType());
    if (!shapeTy || !orderTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<IntTupleType>>(shapeValue)))
      return failure();

    IntTupleAttr orderAttr = orderTy.getAttr();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    IntTupleValueAdaptor shapeAdaptor =
        IntTupleValueAdaptor::create(layoutBuilder, shapeValue, shapeTy.getAttr());

    LayoutValueAdaptor result = layoutMakeOrderedLayout(layoutBuilder, shapeAdaptor, orderAttr);
    rewriter.replaceOp(op, result.getValue());
    return success();
  }
};

class MakeIdentityLayoutOpLowering : public OpRewritePattern<MakeIdentityLayoutOp> {
public:
  using OpRewritePattern<MakeIdentityLayoutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MakeIdentityLayoutOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value shapeValue = op.getShape();
    auto shapeTy = dyn_cast<IntTupleType>(shapeValue.getType());
    if (!shapeTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<IntTupleType>>(shapeValue)))
      return failure();

    IntTupleAttr shapeAttr = shapeTy.getAttr();
    IntTupleAttr strideAttr = intTupleMakeBasisTupleLike(shapeAttr);

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    IntTupleValueAdaptor shapeAdaptor =
        IntTupleValueAdaptor::create(layoutBuilder, shapeValue, shapeAttr);
    IntTupleValueAdaptor strideAdaptor = layoutBuilder.materializeConstantTuple(strideAttr);
    LayoutValueAdaptor result = layoutBuilder.makeLayout(shapeAdaptor, strideAdaptor);
    rewriter.replaceOp(op, result.getValue());
    return success();
  }
};

class MakeFragmentLikeOpLowering : public OpRewritePattern<MakeFragmentLikeOp> {
public:
  using OpRewritePattern<MakeFragmentLikeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MakeFragmentLikeOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultTy = cast<fly::MemRefType>(op.getType());

    LayoutAttr layoutAttr = cast<LayoutAttr>(resultTy.getLayout());
    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    Value fragmentLayout = layoutBuilder.materializeConstantLayout(layoutAttr).getValue();
    rewriter.replaceOpWithNewOp<MemRefAllocaOp>(op, resultTy, fragmentLayout);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Extractors
//===----------------------------------------------------------------------===//

class GetScalarLowering : public OpRewritePattern<GetScalarOp> {
public:
  using OpRewritePattern<GetScalarOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GetScalarOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value intTuple = op.getIntTuple();

    auto intTupleTy = dyn_cast<IntTupleType>(intTuple.getType());
    if (!intTupleTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<IntTupleType>>(intTuple)))
      return failure();

    IntTupleAttr attr = intTupleTy.getAttr();
    assert(attr.isLeaf() && "IntTuple must be a leaf");

    Type resultTy = op.getResult().getType();
    auto intAttr = attr.extractIntFromLeaf();
    if (intAttr.isStatic()) {
      rewriter.replaceOp(op,
                         arith::ConstantIntOp::create(rewriter, loc, resultTy, intAttr.getValue()));
      return success();
    } else {
      auto defOp = intTuple.getDefiningOp<MakeIntTupleOp>();
      if (!defOp)
        return failure();
      rewriter.replaceOp(op, defOp->getOperand(0));
      return success();
    }
  }
};

class GetShapeLowering : public OpRewritePattern<GetShapeOp> {
public:
  using OpRewritePattern<GetShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GetShapeOp op, PatternRewriter &rewriter) const override {
    auto layout = op.getLayout();

    auto layoutTy = dyn_cast<LayoutType>(layout.getType());
    if (!layoutTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(layout)))
      return failure();

    auto defOp = layout.getDefiningOp<MakeLayoutOp>();
    rewriter.replaceOp(op, defOp.getShape());
    return success();
  }
};

class GetStrideLowering : public OpRewritePattern<GetStrideOp> {
public:
  using OpRewritePattern<GetStrideOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GetStrideOp op, PatternRewriter &rewriter) const override {
    auto layout = op.getLayout();

    auto layoutTy = dyn_cast<LayoutType>(layout.getType());
    if (!layoutTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(layout)))
      return failure();

    auto defOp = layout.getDefiningOp<MakeLayoutOp>();
    rewriter.replaceOp(op, defOp.getStride());
    return success();
  }
};

class GetLayoutLowering : public OpRewritePattern<GetLayoutOp> {
public:
  using OpRewritePattern<GetLayoutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GetLayoutOp op, PatternRewriter &rewriter) const override {
    Value memref = op.getMemref();

    if (auto makeViewOp = memref.getDefiningOp<MakeViewOp>()) {
      rewriter.replaceOp(op, makeViewOp.getLayout());
      return success();
    }
    return failure();
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

class ComposedGetInnerLowering : public OpRewritePattern<ComposedGetInnerOp> {
public:
  using OpRewritePattern<ComposedGetInnerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ComposedGetInnerOp op, PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    if (!isNormalForm(cast<TypedValue<ComposedLayoutType>>(input)))
      return failure();

    auto defOp = input.getDefiningOp<MakeComposedLayoutOp>();
    rewriter.replaceOp(op, defOp.getInner());
    return success();
  }
};

class ComposedGetOffsetLowering : public OpRewritePattern<ComposedGetOffsetOp> {
public:
  using OpRewritePattern<ComposedGetOffsetOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ComposedGetOffsetOp op, PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    if (!isNormalForm(cast<TypedValue<ComposedLayoutType>>(input)))
      return failure();

    auto defOp = input.getDefiningOp<MakeComposedLayoutOp>();
    rewriter.replaceOp(op, defOp.getOffset());
    return success();
  }
};

class ComposedGetOuterLowering : public OpRewritePattern<ComposedGetOuterOp> {
public:
  using OpRewritePattern<ComposedGetOuterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ComposedGetOuterOp op, PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    if (!isNormalForm(cast<TypedValue<ComposedLayoutType>>(input)))
      return failure();

    auto defOp = input.getDefiningOp<MakeComposedLayoutOp>();
    rewriter.replaceOp(op, defOp.getOuter());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// IntTuple operations
//===----------------------------------------------------------------------===//

template <typename OpTy, typename UnaryOpFn>
class IntTupleUnaryOpLowering : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = op.getInput();

    auto inputTy = dyn_cast<IntTupleType>(input.getType());
    if (!inputTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<IntTupleType>>(input)))
      return failure();

    IntTupleBuilder<IntTupleValueAdaptor> builder(rewriter, loc);
    IntTupleValueAdaptor inputAdaptor =
        IntTupleValueAdaptor::create(builder, input, inputTy.getAttr());

    auto result = UnaryOpFn{}(builder, inputAdaptor);
    rewriter.replaceOp(op, builder.finalize(result));
    return success();
  }
};

template <typename OpTy, typename BinaryOpFn>
class IntTupleBinaryOpLowering : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    auto lhsTy = dyn_cast<IntTupleType>(lhs.getType());
    auto rhsTy = dyn_cast<IntTupleType>(rhs.getType());
    if (!lhsTy || !rhsTy)
      return failure();

    if (!isNormalForm(cast<TypedValue<IntTupleType>>(lhs)) ||
        !isNormalForm(cast<TypedValue<IntTupleType>>(rhs)))
      return failure();

    IntTupleBuilder<IntTupleValueAdaptor> builder(rewriter, loc);
    IntTupleValueAdaptor lhsAdaptor = IntTupleValueAdaptor::create(builder, lhs, lhsTy.getAttr());
    IntTupleValueAdaptor rhsAdaptor = IntTupleValueAdaptor::create(builder, rhs, rhsTy.getAttr());

    auto result = BinaryOpFn{}(builder, lhsAdaptor, rhsAdaptor);
    rewriter.replaceOp(op, builder.finalize(result));
    return success();
  }
};

struct IntTupleAddFn {
  IntTupleValueAdaptor operator()(IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                  IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const {
    return intTupleAdd(builder, lhs, rhs);
  }
};
struct IntTupleSubFn {
  IntTupleValueAdaptor operator()(IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                  IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const {
    return intTupleSub(builder, lhs, rhs);
  }
};
struct IntTupleMulFn {
  IntTupleValueAdaptor operator()(IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                  IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const {
    return intTupleMul(builder, lhs, rhs);
  }
};
struct IntTupleDivFn {
  IntTupleValueAdaptor operator()(IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                  IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const {
    return intTupleDiv(builder, lhs, rhs);
  }
};
struct IntTupleModFn {
  IntTupleValueAdaptor operator()(IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                  IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const {
    return intTupleMod(builder, lhs, rhs);
  }
};

struct IntTupleProductFn {
  IntTupleValueAdaptor operator()(IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                  IntTupleValueAdaptor input) const {
    return intTupleProduct(builder, input);
  }
};
struct IntTupleProductEachFn {
  IntTupleValueAdaptor operator()(IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                  IntTupleValueAdaptor input) const {
    return intTupleProductEach(builder, input);
  }
};
struct IntTupleProductLikeFn {
  IntTupleValueAdaptor operator()(IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                  IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const {
    return intTupleProductLike(builder, lhs, rhs);
  }
};

struct IntTupleShapeDivFn {
  IntTupleValueAdaptor operator()(IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                  IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const {
    return intTupleShapeDiv(builder, lhs, rhs);
  }
};
struct IntTupleCeilDivFn {
  IntTupleValueAdaptor operator()(IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                  IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const {
    return intTupleCeilDiv(builder, lhs, rhs);
  }
};
struct IntTupleElemLessFn {
  IntTupleValueAdaptor operator()(IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                  IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const {
    return intTupleElemLess(builder, lhs, rhs);
  }
};
struct IntTupleEqualFn {
  IntTupleValueAdaptor operator()(IntTupleBuilder<IntTupleValueAdaptor> &builder,
                                  IntTupleValueAdaptor lhs, IntTupleValueAdaptor rhs) const {
    return intTupleEqual(builder, lhs, rhs);
  }
};

using IntTupleAddOpLowering = IntTupleBinaryOpLowering<IntTupleAddOp, IntTupleAddFn>;
using IntTupleSubOpLowering = IntTupleBinaryOpLowering<IntTupleSubOp, IntTupleSubFn>;
using IntTupleMulOpLowering = IntTupleBinaryOpLowering<IntTupleMulOp, IntTupleMulFn>;
using IntTupleDivOpLowering = IntTupleBinaryOpLowering<IntTupleDivOp, IntTupleDivFn>;
using IntTupleModOpLowering = IntTupleBinaryOpLowering<IntTupleModOp, IntTupleModFn>;

using IntTupleProductOpLowering = IntTupleUnaryOpLowering<IntTupleProductOp, IntTupleProductFn>;
using IntTupleProductEachOpLowering =
    IntTupleUnaryOpLowering<IntTupleProductEachOp, IntTupleProductEachFn>;
using IntTupleProductLikeOpLowering =
    IntTupleBinaryOpLowering<IntTupleProductLikeOp, IntTupleProductLikeFn>;

using ShapeDivOpLowering = IntTupleBinaryOpLowering<ShapeDivOp, IntTupleShapeDivFn>;
using CeilDivOpLowering = IntTupleBinaryOpLowering<CeilDivOp, IntTupleCeilDivFn>;
using ElemLessOpLowering = IntTupleBinaryOpLowering<ElemLessOp, IntTupleElemLessFn>;
using EqualOpLowering = IntTupleBinaryOpLowering<EqualOp, IntTupleEqualFn>;

//===----------------------------------------------------------------------===//
// IntTupleLike operations
//===----------------------------------------------------------------------===//

class GetOpLowering : public OpRewritePattern<GetOp> {
public:
  using OpRewritePattern<GetOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GetOp op, PatternRewriter &rewriter) const override {
    Value input = op.getInput();
    auto intTupleTy = dyn_cast<IntTupleType>(input.getType());
    if (!intTupleTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<IntTupleType>>(input)))
      return failure();

    IntTupleBuilder<IntTupleValueAdaptor> tupleBuilder(rewriter, op.getLoc());
    IntTupleValueAdaptor adaptor =
        IntTupleValueAdaptor::create(tupleBuilder, input, intTupleTy.getAttr());

    for (int32_t idx : op.getMode()) {
      adaptor = tupleBuilder.at(adaptor, idx);
    }
    rewriter.replaceOp(op, tupleBuilder.finalize(adaptor));
    return success();
  }
};

class TakeOpLowering : public OpRewritePattern<TakeOp> {
public:
  using OpRewritePattern<TakeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TakeOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value tuple = op.getTuple();
    int32_t begin = op.getBegin();
    int32_t end = op.getEnd();

    auto intTupleTy = dyn_cast<IntTupleType>(tuple.getType());
    if (!intTupleTy)
      return failure();

    if (!isNormalForm(cast<TypedValue<IntTupleType>>(tuple)))
      return failure();

    IntTupleBuilder<IntTupleValueAdaptor> builder(rewriter, loc);
    IntTupleValueAdaptor adaptor =
        IntTupleValueAdaptor::create(builder, tuple, intTupleTy.getAttr());

    IntTupleValueAdaptor result = intTupleTake(builder, adaptor, begin, end);
    rewriter.replaceOp(op, builder.finalize(result));
    return success();
  }
};

class SelectOpLowering : public OpRewritePattern<SelectOp> {
public:
  using OpRewritePattern<SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value tuple = op.getTuple();

    auto intTupleTy = dyn_cast<IntTupleType>(tuple.getType());
    if (!intTupleTy)
      return failure();

    if (!isNormalForm(cast<TypedValue<IntTupleType>>(tuple)))
      return failure();

    IntTupleBuilder<IntTupleValueAdaptor> builder(rewriter, loc);
    IntTupleValueAdaptor adaptor =
        IntTupleValueAdaptor::create(builder, tuple, intTupleTy.getAttr());

    ArrayRef<int32_t> indices = op.getIndices();
    IntTupleValueAdaptor result = intTupleSelect(builder, adaptor, indices);
    rewriter.replaceOp(op, builder.finalize(result));
    return success();
  }
};

class GroupOpLowering : public OpRewritePattern<GroupOp> {
public:
  using OpRewritePattern<GroupOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GroupOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value tuple = op.getTuple();
    int32_t begin = op.getBegin();
    int32_t end = op.getEnd();

    auto intTupleTy = dyn_cast<IntTupleType>(tuple.getType());
    if (!intTupleTy)
      return failure();

    if (!isNormalForm(cast<TypedValue<IntTupleType>>(tuple)))
      return failure();

    IntTupleBuilder<IntTupleValueAdaptor> builder(rewriter, loc);
    IntTupleValueAdaptor adaptor =
        IntTupleValueAdaptor::create(builder, tuple, intTupleTy.getAttr());

    IntTupleValueAdaptor result = intTupleGroup(builder, adaptor, begin, end);
    rewriter.replaceOp(op, builder.finalize(result));
    return success();
  }
};

class AppendOpLowering : public OpRewritePattern<AppendOp> {
public:
  using OpRewritePattern<AppendOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AppendOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value tuple = op.getTuple();
    Value elem = op.getElem();
    int32_t n = op.getN().value_or(-1);

    auto intTupleTy = dyn_cast<IntTupleType>(tuple.getType());
    auto elemTy = dyn_cast<IntTupleType>(elem.getType());
    if (!intTupleTy || !elemTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<IntTupleType>>(tuple)) ||
        !isNormalForm(cast<TypedValue<IntTupleType>>(elem)))
      return failure();

    IntTupleBuilder<IntTupleValueAdaptor> builder(rewriter, loc);
    auto tupleAdaptor = IntTupleValueAdaptor::create(builder, tuple, intTupleTy.getAttr());
    auto elemAdaptor = IntTupleValueAdaptor::create(builder, elem, elemTy.getAttr());
    auto result = intTupleAppend(builder, tupleAdaptor, elemAdaptor, n);
    rewriter.replaceOp(op, builder.finalize(result));
    return success();
  }
};

class PrependOpLowering : public OpRewritePattern<PrependOp> {
public:
  using OpRewritePattern<PrependOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PrependOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value tuple = op.getTuple();
    Value elem = op.getElem();
    int32_t n = op.getN().value_or(-1);

    auto intTupleTy = dyn_cast<IntTupleType>(tuple.getType());
    auto elemTy = dyn_cast<IntTupleType>(elem.getType());
    if (!intTupleTy || !elemTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<IntTupleType>>(tuple)) ||
        !isNormalForm(cast<TypedValue<IntTupleType>>(elem)))
      return failure();

    IntTupleBuilder<IntTupleValueAdaptor> builder(rewriter, loc);
    auto tupleAdaptor = IntTupleValueAdaptor::create(builder, tuple, intTupleTy.getAttr());
    auto elemAdaptor = IntTupleValueAdaptor::create(builder, elem, elemTy.getAttr());
    auto result = intTuplePrepend(builder, tupleAdaptor, elemAdaptor, n);
    rewriter.replaceOp(op, builder.finalize(result));
    return success();
  }
};

class SliceLowering : public OpRewritePattern<SliceOp> {
public:
  using OpRewritePattern<SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SliceOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value coord = op.getCoord();

    auto srcTy = dyn_cast<IntTupleType>(src.getType());
    auto coordTy = dyn_cast<IntTupleType>(coord.getType());

    if (!srcTy || !coordTy)
      return failure();

    if (!isNormalForm(cast<TypedValue<IntTupleType>>(src)))
      return failure();

    IntTupleBuilder<IntTupleValueAdaptor> builder(rewriter, loc);
    IntTupleValueAdaptor srcAdaptor = IntTupleValueAdaptor::create(builder, src, srcTy.getAttr());

    IntTupleValueAdaptor result = intTupleSlice(builder, srcAdaptor, coordTy.getAttr());

    rewriter.replaceOp(op, builder.finalize(result));
    return success();
  }
};

class DiceOpLowering : public OpRewritePattern<DiceOp> {
public:
  using OpRewritePattern<DiceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DiceOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value coord = op.getCoord();

    auto intTupleTy = dyn_cast<IntTupleType>(src.getType());
    auto coordTy = dyn_cast<IntTupleType>(coord.getType());
    if (!intTupleTy || !coordTy)
      return failure();

    if (!isNormalForm(cast<TypedValue<IntTupleType>>(src)))
      return failure();

    IntTupleBuilder<IntTupleValueAdaptor> builder(rewriter, loc);
    IntTupleValueAdaptor srcAdaptor =
        IntTupleValueAdaptor::create(builder, src, intTupleTy.getAttr());

    IntTupleValueAdaptor result = intTupleDice(builder, srcAdaptor, coordTy.getAttr());
    rewriter.replaceOp(op, builder.finalize(result));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// LayoutLike operations
//===----------------------------------------------------------------------===//

class CoprofileOpLowering : public OpRewritePattern<CoprofileOp> {
public:
  using OpRewritePattern<CoprofileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CoprofileOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value layoutValue = op.getLayout();
    auto layoutTy = dyn_cast<LayoutType>(layoutValue.getType());
    if (!layoutTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(layoutValue)))
      return failure();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor layoutAdaptor(layoutValue, layoutTy.getAttr());
    auto result = layoutCoprofile(layoutBuilder, layoutAdaptor);
    rewriter.replaceOp(op, layoutBuilder.finalize(result));
    return success();
  }
};

class CoshapeOpLowering : public OpRewritePattern<CoshapeOp> {
public:
  using OpRewritePattern<CoshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CoshapeOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value layoutValue = op.getLayout();
    auto layoutTy = dyn_cast<LayoutType>(layoutValue.getType());
    if (!layoutTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(layoutValue)))
      return failure();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor layoutAdaptor(layoutValue, layoutTy.getAttr());
    auto result = layoutCoshape(layoutBuilder, layoutAdaptor);
    rewriter.replaceOp(op, layoutBuilder.finalize(result));
    return success();
  }
};

class CosizeOpLowering : public OpRewritePattern<CosizeOp> {
public:
  using OpRewritePattern<CosizeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CosizeOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value layoutValue = op.getLayout();
    auto layoutTy = dyn_cast<LayoutType>(layoutValue.getType());
    if (!layoutTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(layoutValue)))
      return failure();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor layoutAdaptor(layoutValue, layoutTy.getAttr());
    auto result = layoutCosize(layoutBuilder, layoutAdaptor);
    rewriter.replaceOp(op, layoutBuilder.finalize(result));
    return success();
  }
};

class Crd2IdxLowering : public OpRewritePattern<Crd2IdxOp> {
public:
  using OpRewritePattern<Crd2IdxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Crd2IdxOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto coord = op.getCoord();
    auto layout = op.getLayout();

    auto coordTy = dyn_cast<IntTupleType>(coord.getType());
    if (!coordTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<IntTupleType>>(coord)))
      return failure();

    LayoutValueAdaptor layoutAdaptor;

    if (auto layoutTy = dyn_cast<LayoutType>(layout.getType())) {
      if (!isNormalForm(cast<TypedValue<LayoutType>>(layout)))
        return failure();
      layoutAdaptor = LayoutValueAdaptor(layout, layoutTy.getAttr());
    } else if (auto composedLayoutTy = dyn_cast<ComposedLayoutType>(layout.getType())) {
      if (!isNormalForm(cast<TypedValue<ComposedLayoutType>>(layout)))
        return failure();
      layoutAdaptor = LayoutValueAdaptor(layout, composedLayoutTy.getAttr());
    } else {
      return failure();
    }

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    IntTupleValueAdaptor coordAdaptor =
        IntTupleValueAdaptor::create(layoutBuilder, coord, coordTy.getAttr());
    IntTupleValueAdaptor result = layoutCrd2Idx(layoutBuilder, coordAdaptor, layoutAdaptor);

    rewriter.replaceOp(op, layoutBuilder.finalize(result));
    return success();
  }
};

class Idx2CrdLowering : public OpRewritePattern<Idx2CrdOp> {
public:
  using OpRewritePattern<Idx2CrdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Idx2CrdOp op, PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto index = op.getCoord();
    auto layout = op.getLayout();

    auto indexTy = dyn_cast<IntTupleType>(index.getType());
    auto layoutTy = dyn_cast<LayoutType>(layout.getType());
    if (!indexTy || !layoutTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<IntTupleType>>(index)))
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(layout)))
      return failure();

    IntTupleBuilder<IntTupleValueAdaptor> builder(rewriter, loc);
    IntTupleValueAdaptor indexAdaptor =
        IntTupleValueAdaptor::create(builder, index, indexTy.getAttr());
    IntTupleValueAdaptor shapeAdaptor = IntTupleValueAdaptor::create(
        builder, layout.getDefiningOp()->getOperand(0), layoutTy.getAttr().getShape());
    IntTupleValueAdaptor strideAdaptor = IntTupleValueAdaptor::create(
        builder, layout.getDefiningOp()->getOperand(1), layoutTy.getAttr().getStride());

    IntTupleValueAdaptor result = layoutIdx2Crd(builder, indexAdaptor, shapeAdaptor, strideAdaptor);
    rewriter.replaceOp(op, builder.finalize(result));
    return success();
  }
};

class GetFlatCoordOpLowering : public OpRewritePattern<GetFlatCoordOp> {
public:
  using OpRewritePattern<GetFlatCoordOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GetFlatCoordOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value index = op.getIndex();
    Value layout = op.getLayout();

    auto indexTy = dyn_cast<IntTupleType>(index.getType());
    auto layoutTy = dyn_cast<LayoutType>(layout.getType());
    if (!indexTy || !layoutTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<IntTupleType>>(index)))
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(layout)))
      return failure();

    LayoutAttr layoutAttr = layoutTy.getAttr();
    IntTupleBuilder<IntTupleValueAdaptor> builder(rewriter, loc);
    IntTupleValueAdaptor indexAdaptor =
        IntTupleValueAdaptor::create(builder, index, indexTy.getAttr());
    IntTupleValueAdaptor shapeAdaptor = IntTupleValueAdaptor::create(
        builder, layout.getDefiningOp()->getOperand(0), layoutAttr.getShape());
    IntTupleValueAdaptor strideAdaptor = IntTupleValueAdaptor::create(
        builder, layout.getDefiningOp()->getOperand(1), layoutAttr.getStride());

    IntTupleValueAdaptor hierCoord =
        layoutIdx2Crd(builder, indexAdaptor, shapeAdaptor, strideAdaptor);
    IntTupleAttr flatShapeAttr = intTupleTransform(
        builder.getAttrBuilder(),
        [&](IntTupleAttr mode) { return builder.getAttrBuilder().materializeConstantLeaf(1); },
        layoutAttr.getShape());
    IntTupleValueAdaptor flatShapeAdaptor = builder.materializeConstantTuple(flatShapeAttr);
    IntTupleValueAdaptor result = layoutCrd2Crd(builder, hierCoord, shapeAdaptor, flatShapeAdaptor);
    rewriter.replaceOp(op, builder.finalize(result));
    return success();
  }
};

class Get1DCoordOpLowering : public OpRewritePattern<Get1DCoordOp> {
public:
  using OpRewritePattern<Get1DCoordOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(Get1DCoordOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value index = op.getIndex();
    Value layout = op.getLayout();

    auto indexTy = dyn_cast<IntTupleType>(index.getType());
    auto layoutTy = dyn_cast<LayoutType>(layout.getType());
    if (!indexTy || !layoutTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<IntTupleType>>(index)))
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(layout)))
      return failure();

    LayoutAttr layoutAttr = layoutTy.getAttr();
    IntTupleBuilder<IntTupleValueAdaptor> builder(rewriter, loc);
    IntTupleValueAdaptor indexAdaptor =
        IntTupleValueAdaptor::create(builder, index, indexTy.getAttr());
    IntTupleValueAdaptor shapeAdaptor = IntTupleValueAdaptor::create(
        builder, layout.getDefiningOp()->getOperand(0), layoutAttr.getShape());
    IntTupleValueAdaptor strideAdaptor = IntTupleValueAdaptor::create(
        builder, layout.getDefiningOp()->getOperand(1), layoutAttr.getStride());

    IntTupleValueAdaptor hierCoord =
        layoutIdx2Crd(builder, indexAdaptor, shapeAdaptor, strideAdaptor);
    IntTupleValueAdaptor result =
        mlir::fly::detail::layoutCrd2IdxColMajor(builder, hierCoord, shapeAdaptor);
    rewriter.replaceOp(op, builder.finalize(result));
    return success();
  }
};

class CoalesceOpLowering : public OpRewritePattern<CoalesceOp> {
public:
  using OpRewritePattern<CoalesceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CoalesceOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value layoutValue = op.getLayout();
    auto layoutTy = dyn_cast<LayoutType>(layoutValue.getType());
    if (!layoutTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(layoutValue)))
      return failure();

    std::optional<IntTupleAttr> profileAttr;
    if (op.getAttr()) {
      auto attrTy = dyn_cast<IntTupleType>(op.getAttr().getType());
      if (attrTy)
        profileAttr = attrTy.getAttr();
    }

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor layoutAdaptor(layoutValue, layoutTy.getAttr());
    LayoutValueAdaptor result = layoutCoalesce(layoutBuilder, layoutAdaptor, profileAttr);
    rewriter.replaceOp(op, result.getValue());
    return success();
  }
};

class CompositionOpLowering : public OpRewritePattern<CompositionOp> {
public:
  using OpRewritePattern<CompositionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CompositionOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value outerValue = op.getOuter();
    Value innerValue = op.getInner();

    auto outerTy = dyn_cast<LayoutType>(outerValue.getType());
    if (!outerTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(outerValue)))
      return failure();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor outerAdaptor(outerValue, outerTy.getAttr());

    if (auto innerLayoutTy = dyn_cast<LayoutType>(innerValue.getType())) {
      if (!isNormalForm(cast<TypedValue<LayoutType>>(innerValue)))
        return failure();
      LayoutValueAdaptor innerAdaptor(innerValue, innerLayoutTy.getAttr());
      LayoutValueAdaptor result = layoutComposition(layoutBuilder, outerAdaptor, innerAdaptor);
      rewriter.replaceOp(op, result.getValue());
      return success();
    }

    if (auto innerTileTy = dyn_cast<TileType>(innerValue.getType())) {
      TileAttr tileAttr = innerTileTy.getAttr();
      LayoutValueAdaptor result = layoutComposition(layoutBuilder, outerAdaptor, tileAttr);
      rewriter.replaceOp(op, result.getValue());
      return success();
    }

    return failure();
  }
};

class ComplementOpLowering : public OpRewritePattern<ComplementOp> {
public:
  using OpRewritePattern<ComplementOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ComplementOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value layoutValue = op.getLayout();
    auto layoutTy = dyn_cast<LayoutType>(layoutValue.getType());
    if (!layoutTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(layoutValue)))
      return failure();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor layoutAdaptor(layoutValue, layoutTy.getAttr());

    std::optional<IntTupleValueAdaptor> codomainSize;
    if (op.getCodomainSize()) {
      auto codomainTy = dyn_cast<IntTupleType>(op.getCodomainSize().getType());
      if (!codomainTy)
        return failure();
      if (!isNormalForm(cast<TypedValue<IntTupleType>>(op.getCodomainSize())))
        return failure();
      codomainSize =
          IntTupleValueAdaptor::create(layoutBuilder, op.getCodomainSize(), codomainTy.getAttr());
    }

    LayoutValueAdaptor result = layoutComplement(layoutBuilder, layoutAdaptor, codomainSize);
    rewriter.replaceOp(op, result.getValue());
    return success();
  }
};

class RightInverseOpLowering : public OpRewritePattern<RightInverseOp> {
public:
  using OpRewritePattern<RightInverseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RightInverseOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value layoutValue = op.getLayout();
    auto layoutTy = dyn_cast<LayoutType>(layoutValue.getType());
    if (!layoutTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(layoutValue)))
      return failure();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor layoutAdaptor(layoutValue, layoutTy.getAttr());
    LayoutValueAdaptor result = layoutRightInverse(layoutBuilder, layoutAdaptor);
    rewriter.replaceOp(op, result.getValue());
    return success();
  }
};

class LeftInverseOpLowering : public OpRewritePattern<LeftInverseOp> {
public:
  using OpRewritePattern<LeftInverseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LeftInverseOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value layoutValue = op.getLayout();
    auto layoutTy = dyn_cast<LayoutType>(layoutValue.getType());
    if (!layoutTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(layoutValue)))
      return failure();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor layoutAdaptor(layoutValue, layoutTy.getAttr());
    LayoutValueAdaptor result = layoutLeftInverse(layoutBuilder, layoutAdaptor);
    rewriter.replaceOp(op, result.getValue());
    return success();
  }
};

template <typename OpTy,
          LayoutValueAdaptor (*DivideFunc)(LayoutBuilder<LayoutValueAdaptor> &, LayoutValueAdaptor,
                                           LayoutValueAdaptor),
          LayoutValueAdaptor (*DivideTileFunc)(LayoutBuilder<LayoutValueAdaptor> &,
                                               LayoutValueAdaptor, TileAttr)>
class LayoutDivideOpLowering : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value layoutValue = op.getLayout();
    Value divisorValue = op.getDivisor();

    auto layoutTy = dyn_cast<LayoutType>(layoutValue.getType());

    if (!layoutTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(layoutValue)))
      return failure();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor layoutAdaptor(layoutValue, layoutTy.getAttr());

    if (auto divisorLayoutTy = dyn_cast<LayoutType>(divisorValue.getType())) {
      if (!isNormalForm(cast<TypedValue<LayoutType>>(divisorValue)))
        return failure();

      LayoutValueAdaptor divisorAdaptor(divisorValue, divisorLayoutTy.getAttr());
      LayoutValueAdaptor result = DivideFunc(layoutBuilder, layoutAdaptor, divisorAdaptor);

      rewriter.replaceOp(op, result.getValue());
      return success();
    }

    if (auto divisorTileTy = dyn_cast<TileType>(divisorValue.getType())) {
      TileAttr tileAttr = divisorTileTy.getAttr();
      LayoutValueAdaptor result = DivideTileFunc(layoutBuilder, layoutAdaptor, tileAttr);

      rewriter.replaceOp(op, result.getValue());
      return success();
    }

    return failure();
  }
};

using LogicalDivideOpLowering =
    LayoutDivideOpLowering<LogicalDivideOp, layoutLogicalDivide<LayoutValueAdaptor>,
                           layoutLogicalDivide<LayoutValueAdaptor>>;
using ZippedDivideOpLowering =
    LayoutDivideOpLowering<ZippedDivideOp, layoutZippedDivide<LayoutValueAdaptor>,
                           layoutZippedDivide<LayoutValueAdaptor>>;
using TiledDivideOpLowering =
    LayoutDivideOpLowering<TiledDivideOp, layoutTiledDivide<LayoutValueAdaptor>,
                           layoutTiledDivide<LayoutValueAdaptor>>;
using FlatDivideOpLowering =
    LayoutDivideOpLowering<FlatDivideOp, layoutFlatDivide<LayoutValueAdaptor>,
                           layoutFlatDivide<LayoutValueAdaptor>>;

template <typename OpTy, LayoutValueAdaptor (*ProductFunc)(LayoutBuilder<LayoutValueAdaptor> &,
                                                           LayoutValueAdaptor, LayoutValueAdaptor)>
class LayoutProductOpLowering : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value layoutValue = op.getLayout();
    Value tileValue = op.getTile();

    auto layoutTy = dyn_cast<LayoutType>(layoutValue.getType());
    if (!layoutTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(layoutValue)))
      return failure();

    auto tileTy = dyn_cast<LayoutType>(tileValue.getType());
    if (!tileTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(tileValue)))
      return failure();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor layoutAdaptor(layoutValue, layoutTy.getAttr());
    LayoutValueAdaptor tileAdaptor(tileValue, tileTy.getAttr());
    LayoutValueAdaptor result = ProductFunc(layoutBuilder, layoutAdaptor, tileAdaptor);

    rewriter.replaceOp(op, result.getValue());
    return success();
  }
};

using LogicalProductOpLowering =
    LayoutProductOpLowering<LogicalProductOp, layoutLogicalProduct<LayoutValueAdaptor>>;
using ZippedProductOpLowering =
    LayoutProductOpLowering<ZippedProductOp, layoutZippedProduct<LayoutValueAdaptor>>;
using TiledProductOpLowering =
    LayoutProductOpLowering<TiledProductOp, layoutTiledProduct<LayoutValueAdaptor>>;
using FlatProductOpLowering =
    LayoutProductOpLowering<FlatProductOp, layoutFlatProduct<LayoutValueAdaptor>>;
using BlockedProductOpLowering =
    LayoutProductOpLowering<BlockedProductOp, layoutBlockedProduct<LayoutValueAdaptor>>;
using RakedProductOpLowering =
    LayoutProductOpLowering<RakedProductOp, layoutRakedProduct<LayoutValueAdaptor>>;

class RecastLayoutOpLowering : public OpRewritePattern<RecastLayoutOp> {
public:
  using OpRewritePattern<RecastLayoutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RecastLayoutOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value layoutValue = op.getSrc();
    auto layoutTy = dyn_cast<LayoutType>(layoutValue.getType());
    if (!layoutTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(layoutValue)))
      return failure();

    int32_t newTypeBits = op.getNewTypeBits();
    int32_t oldTypeBits = op.getOldTypeBits();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor layoutAdaptor(layoutValue, layoutTy.getAttr());
    LayoutValueAdaptor result =
        layoutRecast(layoutBuilder, layoutAdaptor, oldTypeBits, newTypeBits);
    rewriter.replaceOp(op, result.getValue());
    return success();
  }
};

class TileToShapeOpLowering : public OpRewritePattern<TileToShapeOp> {
public:
  using OpRewritePattern<TileToShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TileToShapeOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value blockValue = op.getBlock();
    Value trgShapeValue = op.getTrgShape();
    Value ordShapeValue = op.getOrdShape();

    auto layoutTy = dyn_cast<LayoutType>(blockValue.getType());
    if (!layoutTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<LayoutType>>(blockValue)))
      return failure();

    auto trgShapeTy = dyn_cast<IntTupleType>(trgShapeValue.getType());
    auto ordShapeTy = dyn_cast<IntTupleType>(ordShapeValue.getType());
    if (!trgShapeTy || !ordShapeTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<IntTupleType>>(trgShapeValue)))
      return failure();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor blockAdaptor(blockValue, layoutTy.getAttr());
    IntTupleValueAdaptor trgShapeAdaptor =
        IntTupleValueAdaptor::create(layoutBuilder, trgShapeValue, trgShapeTy.getAttr());

    LayoutValueAdaptor result =
        layoutTileToShape(layoutBuilder, blockAdaptor, trgShapeAdaptor, ordShapeTy.getAttr());
    rewriter.replaceOp(op, result.getValue());
    return success();
  }
};

class PrintOpLowering : public OpRewritePattern<PrintOp> {
public:
  using OpRewritePattern<PrintOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PrintOp op, PatternRewriter &rewriter) const override {
    bool isGpuContext = op->getParentOfType<gpu::GPUFuncOp>() != nullptr;

    for (Value val : op.getValues()) {
      if (auto intTupleVal = dyn_cast<TypedValue<IntTupleType>>(val)) {
        if (!isNormalForm(intTupleVal)) {
          return failure();
        }
      } else if (auto layoutVal = dyn_cast<TypedValue<LayoutType>>(val)) {
        if (!isNormalForm(layoutVal)) {
          return failure();
        }
      } else {
        continue;
      }
    }

    auto loc = op.getLoc();
    std::string userFormat = op.getFormat().str();
    std::string format;
    SmallVector<Value> args;

    auto formatValueToString = [&](Value val) -> std::string {
      std::string valFormat;
      if (auto tupleTy = dyn_cast<IntTupleType>(val.getType())) {
        if (tupleTy.getAttr().isStatic()) {
          appendIntTuplePrintfStatic(tupleTy.getAttr(), valFormat);
        } else {
          IntTupleBuilder<IntTupleValueAdaptor> builder(rewriter, loc);
          IntTupleValueAdaptor tuple =
              IntTupleValueAdaptor::create(builder, val, tupleTy.getAttr());
          appendIntTuplePrintf(rewriter, loc, tuple, valFormat, args);
        }
      } else if (auto layoutTy = dyn_cast<LayoutType>(val.getType())) {
        if (layoutTy.getAttr().isStatic()) {
          appendIntTuplePrintfStatic(layoutTy.getAttr().getShape(), valFormat);
          valFormat += ":";
          appendIntTuplePrintfStatic(layoutTy.getAttr().getStride(), valFormat);
        } else {
          LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
          LayoutValueAdaptor layout(val, layoutTy.getAttr());
          appendIntTuplePrintf(rewriter, loc, layoutBuilder.getShape(layout), valFormat, args);
          valFormat += ":";
          appendIntTuplePrintf(rewriter, loc, layoutBuilder.getStride(layout), valFormat, args);
        }
      } else {
        appendScalarPrintfArg(rewriter, loc, val, valFormat, args);
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
        while (specEnd < fmtStr.size() && !std::isalpha(fmtStr[specEnd])) {
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

//===----------------------------------------------------------------------===//
// TiledCopy/TiledMma Partition Lowering
//===----------------------------------------------------------------------===//

template <typename OpTy,
          LayoutValueAdaptor (*ThrValViewFunc)(LayoutBuilder<LayoutValueAdaptor> &, CopyAtomType,
                                               LayoutAttr, TileAttr, LayoutValueAdaptor)>
class TiledCopyPartitionOpLowering : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value input = op->getOperand(1);
    Value coord = op.getCoord();

    auto tiledCopyTy = dyn_cast<TiledCopyType>(op.getTiledCopy().getType());
    auto coordTy = dyn_cast<IntTupleType>(coord.getType());
    if (!tiledCopyTy || !coordTy)
      return failure();
    if (!isNormalForm(cast<TypedValue<IntTupleType>>(coord)))
      return failure();

    if (auto memrefTyped = dyn_cast<TypedValue<fly::MemRefType>>(input)) {
      if (!isWeaklyNormalForm(memrefTyped))
        return failure();
    } else if (auto coordTensorTyped = dyn_cast<TypedValue<CoordTensorType>>(input)) {
      if (!isWeaklyNormalForm(coordTensorTyped))
        return failure();
    } else {
      return failure();
    }

    auto makeViewOp = input.getDefiningOp<MakeViewOp>();
    Value iter = makeViewOp.getIter();
    Value layoutValue = makeViewOp.getLayout();

    auto copyAtom = dyn_cast<CopyAtomType>(tiledCopyTy.getCopyAtom());
    if (!copyAtom)
      return failure();

    LayoutAttr tiledLayoutThrVal = tiledCopyTy.getLayoutThrVal().getAttr();
    TileAttr tileMN = tiledCopyTy.getTileMN().getAttr();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    Attribute layoutAttr;
    if (auto layoutTy = dyn_cast<LayoutType>(layoutValue.getType()))
      layoutAttr = layoutTy.getAttr();
    else
      layoutAttr = cast<ComposedLayoutType>(layoutValue.getType()).getAttr();
    LayoutValueAdaptor fullLayoutAdaptor(layoutValue, layoutAttr);

    LayoutValueAdaptor outerAdaptor = layoutBuilder.isComposedLayout(fullLayoutAdaptor)
                                          ? layoutBuilder.getOuter(fullLayoutAdaptor)
                                          : fullLayoutAdaptor;
    LayoutAttr outerLayout = layoutBuilder.getLayoutAttr(outerAdaptor);

    LayoutValueAdaptor thrValView =
        ThrValViewFunc(layoutBuilder, copyAtom, tiledLayoutThrVal, tileMN, outerAdaptor);

    auto thrValShape = layoutBuilder.getShape(thrValView);
    auto thrValStride = layoutBuilder.getStride(thrValView);
    auto expandedShape = intTupleExpand(layoutBuilder, thrValShape, {2});
    auto expandedStride = intTupleExpand(layoutBuilder, thrValStride, {2});
    LayoutValueAdaptor expandedLayout = layoutBuilder.makeLayout(expandedShape, expandedStride);

    LayoutValueAdaptor expandedFullLayout =
        layoutBuilder.isComposedLayout(fullLayoutAdaptor)
            ? layoutBuilder.makeComposedLayout(layoutBuilder.getInner(fullLayoutAdaptor),
                                               layoutBuilder.getOffset(fullLayoutAdaptor),
                                               expandedLayout)
            : expandedLayout;

    Value expandedView = MakeViewOp::create(rewriter, loc, iter, expandedFullLayout.getValue());

    SmallVector<Value> dynElems(coord.getDefiningOp()->getOperands());
    SmallVector<Attribute> sliceCoordElems;
    sliceCoordElems.push_back(coordTy.getAttr());
    sliceCoordElems.push_back(IntTupleAttr::getLeafNone(ctx));
    for (int i = 0; i < outerLayout.rank(); ++i)
      sliceCoordElems.push_back(IntTupleAttr::getLeafNone(ctx));
    IntTupleAttr sliceCoordAttr = IntTupleAttr::get(ArrayAttr::get(ctx, sliceCoordElems));

    Value sliceCoord =
        MakeIntTupleOp::create(rewriter, loc, IntTupleType::get(sliceCoordAttr), dynElems);

    Value result = SliceOp::create(rewriter, loc, expandedView, sliceCoord);

    rewriter.replaceOp(op, result);
    return success();
  }
};

using TiledCopyPartitionSrcOpLowering =
    TiledCopyPartitionOpLowering<TiledCopyPartitionSrcOp,
                                 layoutTiledCopyThrValViewSrc<LayoutValueAdaptor>>;
using TiledCopyPartitionDstOpLowering =
    TiledCopyPartitionOpLowering<TiledCopyPartitionDstOp,
                                 layoutTiledCopyThrValViewDst<LayoutValueAdaptor>>;

class TiledCopyRetileOpLowering : public OpRewritePattern<TiledCopyRetileOp> {
public:
  using OpRewritePattern<TiledCopyRetileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TiledCopyRetileOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value input = op.getInput();
    auto tiledCopyTy = dyn_cast<TiledCopyType>(op.getTiledCopy().getType());
    if (!tiledCopyTy)
      return failure();

    if (!isWeaklyNormalForm(cast<TypedValue<fly::MemRefType>>(input)))
      return failure();

    auto makeViewOp = input.getDefiningOp<MakeViewOp>();
    Value inputIter = makeViewOp.getIter();
    Value inputLayoutValue = makeViewOp.getLayout();

    auto copyAtom = dyn_cast<CopyAtomType>(tiledCopyTy.getCopyAtom());
    if (!copyAtom)
      return failure();

    LayoutAttr tiledLayoutThrVal = tiledCopyTy.getLayoutThrVal().getAttr();
    TileAttr tileMN = tiledCopyTy.getTileMN().getAttr();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    LayoutValueAdaptor inputLayoutAdaptor(inputLayoutValue,
                                          cast<LayoutType>(inputLayoutValue.getType()).getAttr());
    LayoutValueAdaptor retiled = layoutTiledCopyRetile(layoutBuilder, copyAtom, tiledLayoutThrVal,
                                                       tileMN, inputLayoutAdaptor);

    Value result = MakeViewOp::create(rewriter, loc, inputIter, retiled.getValue());
    rewriter.replaceOp(op, result);
    return success();
  }
};

class TiledMmaPartitionOpLowering : public OpRewritePattern<TiledMmaPartitionOp> {
public:
  using OpRewritePattern<TiledMmaPartitionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TiledMmaPartitionOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    auto operandId = op.getOperandId();
    Value input = op.getInput();
    Value coord = op.getCoord();

    auto tiledMmaTy = dyn_cast<TiledMmaType>(op.getTiledMma().getType());
    auto coordTy = dyn_cast<IntTupleType>(coord.getType());
    if (!tiledMmaTy || !coordTy)
      return failure();

    if (isa<fly::MemRefType>(input.getType())) {
      if (!isWeaklyNormalForm(cast<TypedValue<fly::MemRefType>>(input)))
        return failure();
    } else if (isa<CoordTensorType>(input.getType())) {
      if (!isWeaklyNormalForm(cast<TypedValue<CoordTensorType>>(input)))
        return failure();
    } else {
      return failure();
    }

    if (!isNormalForm(cast<TypedValue<IntTupleType>>(coord)))
      return failure();

    auto makeViewOp = input.getDefiningOp<MakeViewOp>();
    Value inputIter = makeViewOp.getIter();
    Value inputLayoutValue = makeViewOp.getLayout();

    auto mmaAtom = dyn_cast<MmaAtomTypeInterface>(tiledMmaTy.getMmaAtom());
    if (!mmaAtom)
      return failure();

    LayoutAttr atomLayoutMNK = tiledMmaTy.getAtomLayout().getAttr();
    TileAttr permutationMNK = tiledMmaTy.getPermutation().getAttr();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    Attribute inputLayoutAttr;
    if (auto layoutTy = dyn_cast<LayoutType>(inputLayoutValue.getType()))
      inputLayoutAttr = layoutTy.getAttr();
    else
      inputLayoutAttr = cast<ComposedLayoutType>(inputLayoutValue.getType()).getAttr();
    LayoutValueAdaptor fullLayoutAdaptor(inputLayoutValue, inputLayoutAttr);

    LayoutValueAdaptor outerAdaptor = layoutBuilder.isComposedLayout(fullLayoutAdaptor)
                                          ? layoutBuilder.getOuter(fullLayoutAdaptor)
                                          : fullLayoutAdaptor;

    LayoutValueAdaptor thrValView = layoutTiledMmaThrValOperandView(
        layoutBuilder, mmaAtom, atomLayoutMNK, permutationMNK, operandId, outerAdaptor);

    LayoutValueAdaptor thrValFullLayout =
        layoutBuilder.isComposedLayout(fullLayoutAdaptor)
            ? layoutBuilder.makeComposedLayout(layoutBuilder.getInner(fullLayoutAdaptor),
                                               layoutBuilder.getOffset(fullLayoutAdaptor),
                                               thrValView)
            : thrValView;

    Value thrValMemref = MakeViewOp::create(rewriter, loc, inputIter, thrValFullLayout.getValue());

    LayoutBuilder<LayoutAttr> attrBuilder(ctx);
    LayoutAttr atomThrIDLayout = cast<LayoutAttr>(mmaAtom.getThrLayout());
    LayoutAttr thrLayoutVMNK = layoutTiledProduct(
        attrBuilder, atomThrIDLayout, attrBuilder.materializeConstantLayout(atomLayoutMNK));

    IntTupleAttr vmnkShape = thrLayoutVMNK.getShape();
    IntTupleAttr vmnkStride = thrLayoutVMNK.getStride();

    IntTupleBuilder<IntTupleValueAdaptor> tupleBuilder(rewriter, loc);
    IntTupleValueAdaptor coordAdaptor =
        IntTupleValueAdaptor::create(tupleBuilder, coord, coordTy.getAttr());

    IntTupleValueAdaptor hierCoord =
        layoutIdx2Crd(tupleBuilder, coordAdaptor, tupleBuilder.materializeConstantTuple(vmnkShape),
                      tupleBuilder.materializeConstantTuple(vmnkStride));

    int32_t vmnkRank = vmnkShape.rank();
    IntTupleAttr flatShape = IntTupleAttr::get(ArrayAttr::get(
        ctx, SmallVector<Attribute>(vmnkRank, IntTupleAttr::get(IntAttr::getStatic(ctx, 1)))));
    IntTupleValueAdaptor flatCoord =
        layoutCrd2Crd(tupleBuilder, hierCoord, tupleBuilder.materializeConstantTuple(vmnkShape),
                      tupleBuilder.materializeConstantTuple(flatShape));

    int thrIdx0, thrIdx1;
    switch (operandId) {
    case MmaOperand::C:
      [[fallthrough]];
    case MmaOperand::D:
      thrIdx0 = 1;
      thrIdx1 = 2;
      break;
    case MmaOperand::A:
      thrIdx0 = 1;
      thrIdx1 = 3;
      break;
    case MmaOperand::B:
      thrIdx0 = 2;
      thrIdx1 = 3;
      break;
    }

    IntTupleValueAdaptor thrV = tupleBuilder.at(flatCoord, 0);
    IntTupleValueAdaptor thrDim0 = tupleBuilder.at(flatCoord, thrIdx0);
    IntTupleValueAdaptor thrDim1 = tupleBuilder.at(flatCoord, thrIdx1);

    IntTupleBuilder<IntTupleValueAdaptor>::ElemCollector innerCollector;
    innerCollector.push_back(thrDim0);
    innerCollector.push_back(thrDim1);
    IntTupleValueAdaptor thrDims = tupleBuilder.makeTuple(innerCollector);

    IntTupleBuilder<IntTupleValueAdaptor>::ElemCollector thrCollector;
    thrCollector.push_back(thrV);
    thrCollector.push_back(thrDims);
    IntTupleValueAdaptor thrCoord = tupleBuilder.makeTuple(thrCollector);

    LayoutAttr thrValViewAttr = layoutBuilder.getLayoutAttr(thrValView);
    IntTupleAttr valModeShapeAttr = thrValViewAttr.getShape().at(1);
    auto buildNoneCoord = [&](auto self, IntTupleAttr shapeAttr) -> IntTupleAttr {
      if (shapeAttr.isLeaf()) {
        return IntTupleAttr::getLeafNone(ctx);
      }
      SmallVector<Attribute> elems;
      for (int i = 0; i < shapeAttr.rank(); ++i) {
        elems.push_back(self(self, shapeAttr.at(i)));
      }
      return IntTupleAttr::get(ArrayAttr::get(ctx, elems));
    };
    IntTupleAttr valNoneAttr = buildNoneCoord(buildNoneCoord, valModeShapeAttr);

    SmallVector<Attribute> sliceCoordElems;
    sliceCoordElems.push_back(tupleBuilder.getAttr(thrCoord));
    sliceCoordElems.push_back(valNoneAttr);
    IntTupleAttr sliceCoordAttr = IntTupleAttr::get(ArrayAttr::get(ctx, sliceCoordElems));

    Value thrCoordValue = tupleBuilder.finalize(thrCoord);
    SmallVector<Value> sliceDynElems(thrCoordValue.getDefiningOp()->getOperands());
    Value sliceCoord =
        MakeIntTupleOp::create(rewriter, loc, IntTupleType::get(sliceCoordAttr), sliceDynElems);

    Value result = SliceOp::create(rewriter, loc, thrValMemref, sliceCoord);

    rewriter.replaceOp(op, result);
    return success();
  }
};

class TiledMmaPartitionShapeOpLowering : public OpRewritePattern<TiledMmaPartitionShapeOp> {
public:
  using OpRewritePattern<TiledMmaPartitionShapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TiledMmaPartitionShapeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto operandId = op.getOperandId();
    Value shape = op.getShape();

    auto tiledMmaTy = dyn_cast<TiledMmaType>(op.getTiledMma().getType());
    auto shapeTy = dyn_cast<IntTupleType>(shape.getType());
    if (!tiledMmaTy || !shapeTy)
      return failure();

    if (!isNormalForm(cast<TypedValue<IntTupleType>>(shape)))
      return failure();

    auto mmaAtom = dyn_cast<MmaAtomTypeInterface>(tiledMmaTy.getMmaAtom());
    if (!mmaAtom)
      return failure();

    LayoutAttr atomLayoutMNK = tiledMmaTy.getAtomLayout().getAttr();
    TileAttr permutationMNK = tiledMmaTy.getPermutation().getAttr();

    LayoutBuilder<LayoutValueAdaptor> layoutBuilder(rewriter, loc);
    IntTupleValueAdaptor shapeAdaptor =
        IntTupleValueAdaptor::create(layoutBuilder, shape, shapeTy.getAttr());
    IntTupleValueAdaptor compactStride = intTupleCompactColMajor(layoutBuilder, shapeAdaptor);
    LayoutValueAdaptor dummyLayout = layoutBuilder.makeLayout(shapeAdaptor, compactStride);

    LayoutValueAdaptor thrValView = layoutTiledMmaThrValOperandView(
        layoutBuilder, mmaAtom, atomLayoutMNK, permutationMNK, operandId, dummyLayout);

    auto valShape = layoutBuilder.at(layoutBuilder.getShape(thrValView), 1);
    auto expandedShape = intTupleExpand(layoutBuilder, valShape, {1});

    rewriter.replaceOp(op, layoutBuilder.finalize(expandedShape));
    return success();
  }
};

class MmaMakeFragmentOpLowering : public OpRewritePattern<MmaMakeFragmentOp> {
public:
  using OpRewritePattern<MmaMakeFragmentOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MmaMakeFragmentOp op, PatternRewriter &rewriter) const override {
    auto tiledMmaTy = dyn_cast<TiledMmaType>(op.getTiledMma().getType());
    if (!tiledMmaTy)
      return failure();

    auto mmaAtom = dyn_cast<MmaAtomTypeInterface>(tiledMmaTy.getMmaAtom());
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

    rewriter.replaceOpWithNewOp<MakeFragmentLikeOp>(op, op.getInput(), TypeAttr::get(elemTy));
    return success();
  }
};

class ExpandCopyOpLowering : public OpRewritePattern<CopyOp> {
public:
  using OpRewritePattern<CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CopyOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value copyAtomVal = op.getCopyAtom();
    if (auto tiledCopyOp = copyAtomVal.getDefiningOp<MakeTiledCopyOp>())
      copyAtomVal = tiledCopyOp.getCopyAtom();

    Value src = op.getSrc();
    Value dst = op.getDst();
    Value pred = op.getPred();

    auto srcMemRefTy = cast<fly::MemRefType>(src.getType());
    auto dstMemRefTy = cast<fly::MemRefType>(dst.getType());
    auto predMemRefTy = pred ? cast<fly::MemRefType>(pred.getType()) : nullptr;

    auto getLayoutAttr = [&](Attribute attr) -> LayoutAttr {
      if (auto layout = dyn_cast<LayoutAttr>(attr))
        return layout;
      else
        return cast<ComposedLayoutAttr>(attr).getOuter();
    };

    LayoutAttr srcLayoutAttr = getLayoutAttr(srcMemRefTy.getLayout());
    LayoutAttr dstLayoutAttr = getLayoutAttr(dstMemRefTy.getLayout());
    LayoutAttr predLayoutAttr = nullptr;
    if (pred)
      predLayoutAttr = getLayoutAttr(predMemRefTy.getLayout());

    int32_t srcRank = srcLayoutAttr.rank();
    int32_t dstRank = dstLayoutAttr.rank();

    if (srcRank != dstRank)
      return rewriter.notifyMatchFailure(op, "src/dst ranks mismatch");

    if (srcRank == 1) {
      if (srcLayoutAttr.getShape().isLeaf()) {
        CopyAtomCall::create(rewriter, loc, copyAtomVal, src, dst, pred);
        rewriter.eraseOp(op);
        return success();
      }
      Value srcUnwrapped = GetOp::create(rewriter, loc, src, ArrayRef<int32_t>{0});
      Value dstUnwrapped = GetOp::create(rewriter, loc, dst, ArrayRef<int32_t>{0});
      Value predUnwrapped =
          pred ? GetOp::create(rewriter, loc, pred, ArrayRef<int32_t>{0}) : nullptr;
      CopyOp::create(rewriter, loc, copyAtomVal, srcUnwrapped, dstUnwrapped, predUnwrapped);
      rewriter.eraseOp(op);
      return success();
    }

    IntTupleBuilder<IntTupleAttr> attrBuilder(ctx);
    IntTupleAttr groupedShape = intTupleGroup(attrBuilder, srcLayoutAttr.getShape(), 1, srcRank);
    IntAttr restSize = intTupleProduct(attrBuilder, groupedShape.at(1)).getLeafAsInt();
    if (!restSize.isStatic())
      return rewriter.notifyMatchFailure(op, "restSize is not static");
    int32_t numIter = restSize.getValue();

    Value srcGrouped = GroupOp::create(rewriter, loc, src, 1, srcRank);
    Value dstGrouped = GroupOp::create(rewriter, loc, dst, 1, dstRank);
    Value predGrouped = nullptr;
    if (pred) {
      int32_t predRank = predLayoutAttr.rank();
      if (predRank != srcRank)
        return rewriter.notifyMatchFailure(op, "pred rank mismatch");
      predGrouped = GroupOp::create(rewriter, loc, pred, 1, predRank);
    }

    for (int32_t i = 0; i < numIter; ++i) {
      SmallVector<Attribute> coordElems = {IntTupleAttr::getLeafNone(ctx),
                                           IntTupleAttr::getLeafStatic(ctx, i)};
      IntTupleAttr coordAttr = IntTupleAttr::get(ArrayAttr::get(ctx, coordElems));
      Value coord = MakeIntTupleOp::create(rewriter, loc, IntTupleType::get(coordAttr), {});

      Value srcSlice = SliceOp::create(rewriter, loc, srcGrouped, coord);
      Value dstSlice = SliceOp::create(rewriter, loc, dstGrouped, coord);
      Value predSlice = nullptr;
      if (pred)
        predSlice = SliceOp::create(rewriter, loc, predGrouped, coord);

      CopyOp::create(rewriter, loc, copyAtomVal, srcSlice, dstSlice, predSlice);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

class ExpandGemmOpLowering : public OpRewritePattern<GemmOp> {
public:
  using OpRewritePattern<GemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GemmOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value mmaAtomVal = op.getMmaAtom();
    MmaAtomTypeInterface mmaAtomTy;
    if (auto tiledMmaOp = mmaAtomVal.getDefiningOp<MakeTiledMmaOp>()) {
      mmaAtomVal = tiledMmaOp.getMmaAtom();
    }
    mmaAtomTy = dyn_cast<MmaAtomTypeInterface>(mmaAtomVal.getType());
    if (!mmaAtomTy)
      return failure();

    Value d = op.getD();
    Value a = op.getA();
    Value b = op.getB();
    Value c = op.getC();

    LayoutAttr dLayoutAttr = cast<LayoutAttr>(cast<fly::MemRefType>(d.getType()).getLayout());
    LayoutAttr aLayoutAttr = cast<LayoutAttr>(cast<fly::MemRefType>(a.getType()).getLayout());
    LayoutAttr bLayoutAttr = cast<LayoutAttr>(cast<fly::MemRefType>(b.getType()).getLayout());
    LayoutAttr cLayoutAttr = cast<LayoutAttr>(cast<fly::MemRefType>(c.getType()).getLayout());

    int32_t dRank = dLayoutAttr.rank();
    int32_t aRank = aLayoutAttr.rank();
    int32_t bRank = bLayoutAttr.rank();
    int32_t cRank = cLayoutAttr.rank();

    int32_t loop_m = dLayoutAttr.getShape().at(1).getLeafAsInt().getValue();
    int32_t loop_n = dLayoutAttr.getShape().at(2).getLeafAsInt().getValue();

    assert(loop_m == aLayoutAttr.getShape().at(1).getLeafAsInt().getValue() &&
           "Mismatch in loop_m");
    assert(loop_n == bLayoutAttr.getShape().at(1).getLeafAsInt().getValue() &&
           "Mismatch in loop_n");
    assert(loop_m == cLayoutAttr.getShape().at(1).getLeafAsInt().getValue() &&
           "Mismatch in loop_m");
    assert(loop_n == cLayoutAttr.getShape().at(2).getLeafAsInt().getValue() &&
           "Mismatch in loop_n");

    if (dRank == 1 && aRank == 1 && bRank == 1 && cRank == 1) {
      MmaAtomCall::create(rewriter, loc, mmaAtomVal, d, a, b, c);
      rewriter.eraseOp(op);
      return success();
    }

    if (dRank != 3 || cRank != 3 || aRank < 2 || bRank < 2)
      return failure();

    auto getSliceCoord = [&](ArrayRef<int32_t> idx) {
      SmallVector<Attribute> coordElems;
      // Keep mode-0 unchanged for all operands.
      coordElems.push_back(IntTupleAttr::getLeafNone(ctx));
      for (int32_t i : idx)
        coordElems.push_back(IntTupleAttr::getLeafStatic(ctx, i));
      return MakeIntTupleOp::create(
          rewriter, loc, IntTupleType::get(IntTupleAttr::get(ArrayAttr::get(ctx, coordElems))), {});
    };

    if (aRank == 2 && bRank == 2) {
      for (int32_t m = 0; m < loop_m; ++m) {
        Value aSlice = SliceOp::create(rewriter, loc, a, getSliceCoord({m}));
        for (int32_t n = 0; n < loop_n; ++n) {
          Value bSlice = SliceOp::create(rewriter, loc, b, getSliceCoord({n}));
          Value cSlice = SliceOp::create(rewriter, loc, c, getSliceCoord({m, n}));
          Value dSlice = SliceOp::create(rewriter, loc, d, getSliceCoord({m, n}));
          MmaAtomCall::create(rewriter, loc, mmaAtomVal, dSlice, aSlice, bSlice, cSlice);
        }
      }
      rewriter.eraseOp(op);
      return success();
    } else if (aRank == 3 && bRank == 3) {
      int32_t loop_k = aLayoutAttr.getShape().at(2).getLeafAsInt().getValue();
      assert(loop_k == bLayoutAttr.getShape().at(2).getLeafAsInt().getValue() &&
             "Mismatch in loop_k");

      for (int32_t k = 0; k < loop_k; ++k) {
        Value cSrc = (k == 0) ? c : d;
        for (int32_t m = 0; m < loop_m; ++m) {
          Value aSlice = SliceOp::create(rewriter, loc, a, getSliceCoord({m, k}));
          for (int32_t n = 0; n < loop_n; ++n) {
            Value bSlice = SliceOp::create(rewriter, loc, b, getSliceCoord({n, k}));
            Value cSlice = SliceOp::create(rewriter, loc, cSrc, getSliceCoord({m, n}));
            Value dSlice = SliceOp::create(rewriter, loc, d, getSliceCoord({m, n}));
            MmaAtomCall::create(rewriter, loc, mmaAtomVal, dSlice, aSlice, bSlice, cSlice);
          }
        }
      }
      rewriter.eraseOp(op);
      return success();
    } else {
      return failure();
    }
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

    LayoutBuilder<LayoutAttr> attrBuilder(rewriter.getContext());
    IntTupleAttr totalSize = layoutCosize(attrBuilder, layoutAttr);

    assert(totalSize.isStatic() && totalSize.isLeaf());

    auto ptrAttrs = rewriter.getDictionaryAttr({rewriter.getNamedAttr(
        "allocaSize", rewriter.getI64IntegerAttr(totalSize.getLeafAsInt().getValue()))});
    Value flyPtr =
        MakePtrOp::create(rewriter, loc, memrefTy.getPointerType(), ValueRange{}, ptrAttrs);

    rewriter.replaceOpWithNewOp<MakeViewOp>(op, flyPtr, op.getLayout());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Generated patterns
//===----------------------------------------------------------------------===//

namespace int_tuple_rewrite {
#include "flydsl/Dialect/Fly/Transforms/IntTupleLowering.cpp.inc"
} // namespace int_tuple_rewrite

namespace layout_rewrite {
#include "flydsl/Dialect/Fly/Transforms/LayoutLowering.cpp.inc"
} // namespace layout_rewrite

namespace memref_rewrite {
#include "flydsl/Dialect/Fly/Transforms/MemrefLowering.cpp.inc"
} // namespace memref_rewrite

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

class FlyLayoutLoweringPass
    : public mlir::fly::impl::FlyLayoutLoweringPassBase<FlyLayoutLoweringPass> {
public:
  using mlir::fly::impl::FlyLayoutLoweringPassBase<
      FlyLayoutLoweringPass>::FlyLayoutLoweringPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);

    // Constructors
    patterns.add<MakeOrderedLayoutOpLowering, MakeIdentityLayoutOpLowering,
                 MakeLayoutLikeOpLowering, MakeFragmentLikeOpLowering>(context);

    // Extractors
    patterns.add<GetScalarLowering, GetShapeLowering, GetStrideLowering, GetLayoutLowering,
                 GetIterLowering, ComposedGetInnerLowering, ComposedGetOffsetLowering,
                 ComposedGetOuterLowering>(context);

    // IntTuple operations
    patterns.add<IntTupleAddOpLowering, IntTupleSubOpLowering, IntTupleMulOpLowering,
                 IntTupleDivOpLowering, IntTupleModOpLowering>(context);
    patterns.add<IntTupleProductOpLowering, IntTupleProductEachOpLowering,
                 IntTupleProductLikeOpLowering>(context);
    patterns.add<ShapeDivOpLowering, CeilDivOpLowering, ElemLessOpLowering, EqualOpLowering>(
        context);

    // IntTupleLike operations
    patterns.add<GetOpLowering, TakeOpLowering, SelectOpLowering, GroupOpLowering>(context);
    patterns.add<AppendOpLowering, PrependOpLowering>(context);
    patterns.add<SliceLowering, DiceOpLowering>(context);

    // LayoutLike operations
    patterns.add<CoprofileOpLowering, CoshapeOpLowering, CosizeOpLowering>(context);
    patterns.add<Crd2IdxLowering, Idx2CrdLowering>(context);
    patterns.add<GetFlatCoordOpLowering, Get1DCoordOpLowering>(context);
    patterns.add<CoalesceOpLowering, CompositionOpLowering, ComplementOpLowering>(context);
    patterns.add<RightInverseOpLowering, LeftInverseOpLowering>(context);
    patterns.add<LogicalDivideOpLowering, ZippedDivideOpLowering, TiledDivideOpLowering,
                 FlatDivideOpLowering>(context);
    patterns.add<LogicalProductOpLowering, ZippedProductOpLowering, TiledProductOpLowering,
                 FlatProductOpLowering, BlockedProductOpLowering, RakedProductOpLowering>(context);
    patterns.add<RecastLayoutOpLowering>(context);
    patterns.add<TileToShapeOpLowering>(context);

    // Atom and Tiled Mma/Copy ops
    patterns.add<TiledCopyPartitionSrcOpLowering, TiledCopyPartitionDstOpLowering>(context);
    patterns.add<TiledCopyRetileOpLowering>(context);
    patterns.add<MmaMakeFragmentOpLowering, TiledMmaPartitionOpLowering,
                 TiledMmaPartitionShapeOpLowering>(context);
    patterns.add<ExpandCopyOpLowering, ExpandGemmOpLowering>(context);

    // MemRef/Ptr operations
    patterns.add<MemRefAllocaOpLowering>(context);

    // Utility ops
    patterns.add<PrintOpLowering>(context);

    int_tuple_rewrite::populateWithGenerated(patterns);
    layout_rewrite::populateWithGenerated(patterns);
    memref_rewrite::populateWithGenerated(patterns);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace impl {

std::unique_ptr<::mlir::Pass> createFlyLayoutLoweringPass() {
  return std::make_unique<FlyLayoutLoweringPass>();
}

} // namespace impl
