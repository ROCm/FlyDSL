
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/IntTupleUtils.h"
#include "flydsl/Dialect/Fly/Utils/LayoutUtils.h"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>

#define GET_OP_CLASSES
#include "flydsl/Dialect/Fly/IR/FlyOps.cpp.inc"

#include <algorithm>
#include <tuple>

using namespace mlir;
using namespace mlir::fly;

namespace {

IntTupleAttr makeDynamicLike(IntTupleAttr guide) {
  auto *ctx = guide.getContext();
  IntTupleBuilder<IntTupleAttr> builder(ctx);
  return intTupleTransformLeaf(
      builder, [ctx](IntTupleAttr) { return IntTupleAttr::get(IntAttr::getDynamic(ctx)); }, guide);
}

IntTupleAttr makeCompactStride(IntTupleAttr shapeAttr) {
  auto *ctx = shapeAttr.getContext();
  IntAttr running = IntAttr::getStatic(ctx, 1);

  std::function<IntTupleAttr(IntTupleAttr)> visit = [&](IntTupleAttr shape) -> IntTupleAttr {
    if (shape.isLeaf()) {
      IntTupleAttr stride = IntTupleAttr::get(running);
      running = running * shape.getLeafAsInt();
      return stride;
    }
    SmallVector<Attribute> elements;
    elements.reserve(shape.rank());
    for (int i = 0; i < shape.rank(); ++i) {
      elements.push_back(visit(shape.at(i)));
    }
    return IntTupleAttr::get(ArrayAttr::get(ctx, elements));
  };

  return visit(shapeAttr);
}

LayoutAttr makeOrderedLayoutAttr(IntTupleAttr shapeAttr, IntTupleAttr orderAttr) {
  auto *ctx = shapeAttr.getContext();
  IntTupleBuilder<IntTupleAttr> builder(ctx);
  IntTupleAttr flatShape = intTupleFlatten(builder, shapeAttr);
  IntTupleAttr flatOrder = intTupleFlatten(builder, orderAttr);

  if (flatShape.isLeaf() || flatOrder.isLeaf() || flatShape.rank() != flatOrder.rank()) {
    return LayoutAttr::get(ctx, shapeAttr, makeCompactStride(shapeAttr));
  }

  int32_t rank = flatShape.rank();
  SmallVector<Attribute> strideElems(rank);
  IntAttr running = IntAttr::getStatic(ctx, 1);

  for (int i = 0; i < rank; ++i) {
    IntAttr orderVal = flatOrder.at(i).getLeafAsInt();
    if (!orderVal.isStatic()) {
      return LayoutAttr::get(ctx, shapeAttr, makeCompactStride(shapeAttr));
    }
    int64_t idx = orderVal.getValue();
    if (idx < 0 || idx >= rank || strideElems[idx]) {
      return LayoutAttr::get(ctx, shapeAttr, makeCompactStride(shapeAttr));
    }
    strideElems[idx] = IntTupleAttr::get(running);
    running = running * flatShape.at(idx).getLeafAsInt();
  }

  for (auto elem : strideElems) {
    if (!elem) {
      return LayoutAttr::get(ctx, shapeAttr, makeCompactStride(shapeAttr));
    }
  }

  IntTupleAttr flatStride = IntTupleAttr::get(ArrayAttr::get(ctx, strideElems));
  IntTupleAttr strideAttr = intTupleUnflatten(builder, flatStride, shapeAttr);
  return LayoutAttr::get(ctx, shapeAttr, strideAttr);
}

} // namespace

#define FLY_INFER_RETURN_TYPES(OP)                                                                 \
  llvm::LogicalResult OP::inferReturnTypes(                                                        \
      mlir::MLIRContext *context, std::optional<::mlir::Location> location,                        \
      mlir::ValueRange operands, mlir::DictionaryAttr attributes,                                  \
      mlir::OpaqueProperties properties, mlir::RegionRange regions,                                \
      llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes)

FLY_INFER_RETURN_TYPES(MakeLayoutOp) {
  auto shapeType = dyn_cast<IntTupleType>(operands[0].getType());
  IntTupleAttr shapeAttr = shapeType.getAttr();
  IntTupleAttr strideAttr;

  if (operands.size() > 1) {
    strideAttr = dyn_cast<IntTupleType>(operands[1].getType()).getAttr();
  } else {
    strideAttr = makeCompactStride(shapeAttr);
  }
  auto layoutAttr = LayoutAttr::get(context, shapeAttr, strideAttr);
  inferredReturnTypes.assign({LayoutType::get(context, layoutAttr)});
  return success();
}

FLY_INFER_RETURN_TYPES(MakeTileOp) {
  SmallVector<Attribute> layouts;
  for (auto op : operands) {
    auto layoutType = dyn_cast<LayoutType>(op.getType());
    if (!layoutType)
      return failure();
    layouts.push_back(layoutType.getAttr());
  }
  auto tileAttr = TileAttr::get(ArrayAttr::get(context, layouts));
  inferredReturnTypes.assign({TileType::get(context, tileAttr)});
  return success();
}

FLY_INFER_RETURN_TYPES(MakeViewOp) {
  auto ptrTy = dyn_cast<PointerType>(operands[0].getType());
  auto layoutTy = dyn_cast<LayoutType>(operands[1].getType());
  if (!ptrTy || !layoutTy)
    return failure();
  inferredReturnTypes.assign(
      {MemRefType::get(ptrTy.getElemTy(), ptrTy.getAddressSpace(), layoutTy.getAttr())});
  return success();
}

FLY_INFER_RETURN_TYPES(MakeLayoutLikeOp) {
  if (auto layoutTy = dyn_cast<LayoutType>(operands[0].getType())) {
    LayoutAttr inferred = layoutTy.getAttr();
    inferredReturnTypes.assign({LayoutType::get(context, inferred)});
    return success();
  }
  if (auto memrefTy = dyn_cast<MemRefType>(operands[0].getType())) {
    LayoutAttr inferred = memrefTy.getLayout();
    inferredReturnTypes.assign({LayoutType::get(context, inferred)});
    return success();
  }
  return failure();
}

FLY_INFER_RETURN_TYPES(MakeOrderedLayoutOp) {
  auto shapeTy = dyn_cast<IntTupleType>(operands[0].getType());
  auto orderTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!shapeTy || !orderTy)
    return failure();
  IntTupleAttr shapeAttr = shapeTy.getAttr();
  LayoutAttr layoutAttr = makeOrderedLayoutAttr(shapeAttr, orderTy.getAttr());
  inferredReturnTypes.assign({LayoutType::get(context, layoutAttr)});
  return success();
}

FLY_INFER_RETURN_TYPES(MakeComposedLayoutOp) {
  auto offsetTy = dyn_cast<IntTupleType>(operands[1].getType());
  auto outerTy = dyn_cast<LayoutType>(operands[2].getType());
  if (!offsetTy || !outerTy)
    return failure();
  Attribute innerAttr = nullptr;
  if (auto innerLayoutTy = dyn_cast<LayoutType>(operands[0].getType())) {
    innerAttr = innerLayoutTy.getAttr();
  } else if (auto innerComposedTy = dyn_cast<ComposedLayoutType>(operands[0].getType())) {
    innerAttr = innerComposedTy.getAttr();
  } else {
    return failure();
  }
  auto composedAttr =
      ComposedLayoutAttr::get(context, innerAttr, offsetTy.getAttr(), outerTy.getAttr());
  inferredReturnTypes.assign({ComposedLayoutType::get(context, composedAttr)});
  return success();
}

FLY_INFER_RETURN_TYPES(MakeIdentityLayoutOp) {
  auto shapeTy = dyn_cast<IntTupleType>(operands[0].getType());

  IntTupleAttr shapeAttr = shapeTy.getAttr();
  IntTupleAttr strideAttr = intTupleMakeBasisLike(shapeAttr);
  LayoutAttr layoutAttr = LayoutAttr::get(context, shapeAttr, strideAttr);
  inferredReturnTypes.assign({LayoutType::get(context, layoutAttr)});
  return success();
}

FLY_INFER_RETURN_TYPES(MakeIdentityTensorOp) {
  auto shapeTy = dyn_cast<IntTupleType>(operands[0].getType());
  if (!shapeTy)
    return failure();

  IntTupleAttr shapeAttr = shapeTy.getAttr();
  IntTupleAttr strideAttr = intTupleMakeBasisLike(shapeAttr);
  LayoutAttr layoutAttr = LayoutAttr::get(context, shapeAttr, strideAttr);

  IntTupleBuilder<IntTupleAttr> builder(context);
  IntTupleAttr zeroBaseAttr = intTupleTransformLeaf(
      builder, [](IntTupleAttr attr) { return IntTupleAttr::getLeafStatic(attr.getContext(), 0); },
      shapeAttr);
  inferredReturnTypes.assign({CoordTensorType::get(context, zeroBaseAttr, layoutAttr)});
  return success();
}

FLY_INFER_RETURN_TYPES(MakeFragmentLikeOp) {
  inferredReturnTypes.assign({operands[0].getType()});
  return success();
}

FLY_INFER_RETURN_TYPES(GetScalarOp) {
  auto intTupleType = dyn_cast<IntTupleType>(operands[0].getType());
  if (!intTupleType)
    return failure();
  // Must be a leaf IntTuple
  if (!intTupleType.getAttr().isLeaf())
    return failure();
  inferredReturnTypes.assign({IntegerType::get(context, 32)});
  return success();
}

FLY_INFER_RETURN_TYPES(GetLeavesOp) {
  auto inputTupleTy = dyn_cast<IntTupleType>(operands[0].getType());
  if (inputTupleTy) {
    IntTupleBuilder<IntTupleAttr> builder(context);
    IntTupleAttr flat = intTupleFlatten(builder, inputTupleTy.getAttr());
    inferredReturnTypes.assign({IntTupleType::get(flat)});
    return success();
  }
  auto inputLayoutTy = dyn_cast<LayoutType>(operands[0].getType());
  if (!inputLayoutTy)
    return failure();
  IntTupleBuilder<IntTupleAttr> builder(context);
  IntTupleAttr flat = intTupleFlatten(builder, inputLayoutTy.getAttr().getShape());
  inferredReturnTypes.assign({IntTupleType::get(flat)});
  return success();
}

FLY_INFER_RETURN_TYPES(GetShapeOp) {
  auto layoutType = dyn_cast<LayoutType>(operands[0].getType());
  if (!layoutType)
    return failure();
  LayoutAttr profile = layoutType.getAttr();
  inferredReturnTypes.assign({IntTupleType::get(profile.getShape())});
  return success();
}

FLY_INFER_RETURN_TYPES(GetStrideOp) {
  auto layoutType = dyn_cast<LayoutType>(operands[0].getType());
  if (!layoutType)
    return failure();
  LayoutAttr profile = layoutType.getAttr();
  inferredReturnTypes.assign({IntTupleType::get(profile.getStride())});
  return success();
}

FLY_INFER_RETURN_TYPES(GetLayoutOp) {
  auto memrefTy = dyn_cast<MemRefType>(operands[0].getType());
  if (!memrefTy)
    return failure();
  inferredReturnTypes.assign({LayoutType::get(context, memrefTy.getLayout())});
  return success();
}

FLY_INFER_RETURN_TYPES(GetIterOp) {
  auto memrefTy = dyn_cast<MemRefType>(operands[0].getType());
  if (!memrefTy)
    return failure();
  inferredReturnTypes.assign({PointerType::get(memrefTy.getElemTy(), memrefTy.getAddressSpace())});
  return success();
}

FLY_INFER_RETURN_TYPES(GetLeafOp) {
  int32_t leafIdx = properties.as<Properties *>()->leaf_idx.getInt();

  if (auto layoutType = dyn_cast<LayoutType>(operands[0].getType())) {
    LayoutAttr profile = layoutType.getAttr();
    LayoutAttr leafProfile = profile.at(leafIdx);
    inferredReturnTypes.assign({LayoutType::get(context, leafProfile)});
    return success();
  }

  if (auto intTupleType = dyn_cast<IntTupleType>(operands[0].getType())) {
    IntTupleAttr profile = intTupleType.getAttr();
    IntTupleAttr leafProfile = profile.at(leafIdx);
    inferredReturnTypes.assign({IntTupleType::get(leafProfile)});
    return success();
  }

  return failure();
}

FLY_INFER_RETURN_TYPES(ComposedGetInnerOp) {
  auto inputTy = dyn_cast<ComposedLayoutType>(operands[0].getType());
  if (!inputTy)
    return failure();
  auto innerAttr = inputTy.getAttr().getInner();
  if (auto swizzleAttr = dyn_cast<SwizzleAttr>(innerAttr)) {
    inferredReturnTypes.assign({SwizzleType::get(context, swizzleAttr)});
    return success();
  } else if (auto layoutAttr = dyn_cast<LayoutAttr>(innerAttr)) {
    inferredReturnTypes.assign({LayoutType::get(context, layoutAttr)});
    return success();
  } else if (auto composedLayoutAttr = dyn_cast<ComposedLayoutAttr>(innerAttr)) {
    inferredReturnTypes.assign({ComposedLayoutType::get(context, composedLayoutAttr)});
    return success();
  }
  return failure();
}

FLY_INFER_RETURN_TYPES(ComposedGetOffsetOp) {
  auto inputTy = dyn_cast<ComposedLayoutType>(operands[0].getType());
  if (!inputTy)
    return failure();
  inferredReturnTypes.assign({IntTupleType::get(inputTy.getAttr().getOffset())});
  return success();
}

FLY_INFER_RETURN_TYPES(ComposedGetOuterOp) {
  auto inputTy = dyn_cast<ComposedLayoutType>(operands[0].getType());
  if (!inputTy)
    return failure();
  inferredReturnTypes.assign({LayoutType::get(context, inputTy.getAttr().getOuter())});
  return success();
}

FLY_INFER_RETURN_TYPES(IntTupleAddOp) {
  auto lhsTy = dyn_cast<IntTupleType>(operands[0].getType());
  auto rhsTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!lhsTy || !rhsTy)
    return failure();
  IntTupleBuilder<IntTupleAttr> builder(context);
  inferredReturnTypes.assign(
      {IntTupleType::get(intTupleAdd(builder, lhsTy.getAttr(), rhsTy.getAttr()))});
  return success();
}

FLY_INFER_RETURN_TYPES(IntTupleSubOp) {
  auto lhsTy = dyn_cast<IntTupleType>(operands[0].getType());
  auto rhsTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!lhsTy || !rhsTy)
    return failure();
  IntTupleBuilder<IntTupleAttr> builder(context);
  inferredReturnTypes.assign(
      {IntTupleType::get(intTupleSub(builder, lhsTy.getAttr(), rhsTy.getAttr()))});
  return success();
}

FLY_INFER_RETURN_TYPES(IntTupleMulOp) {
  auto lhsTy = dyn_cast<IntTupleType>(operands[0].getType());
  auto rhsTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!lhsTy || !rhsTy)
    return failure();
  IntTupleBuilder<IntTupleAttr> builder(context);
  inferredReturnTypes.assign(
      {IntTupleType::get(intTupleMul(builder, lhsTy.getAttr(), rhsTy.getAttr()))});
  return success();
}

FLY_INFER_RETURN_TYPES(IntTupleDivOp) {
  auto lhsTy = dyn_cast<IntTupleType>(operands[0].getType());
  auto rhsTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!lhsTy || !rhsTy)
    return failure();
  IntTupleBuilder<IntTupleAttr> builder(context);
  inferredReturnTypes.assign(
      {IntTupleType::get(intTupleDiv(builder, lhsTy.getAttr(), rhsTy.getAttr()))});
  return success();
}

FLY_INFER_RETURN_TYPES(IntTupleModOp) {
  auto lhsTy = dyn_cast<IntTupleType>(operands[0].getType());
  auto rhsTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!lhsTy || !rhsTy)
    return failure();
  inferredReturnTypes.assign({IntTupleType::get(makeDynamicLike(lhsTy.getAttr()))});
  return success();
}

FLY_INFER_RETURN_TYPES(IntTupleProductEachOp) {
  auto inputTy = dyn_cast<IntTupleType>(operands[0].getType());
  if (!inputTy)
    return failure();
  IntTupleBuilder<IntTupleAttr> builder(context);
  inferredReturnTypes.assign({IntTupleType::get(intTupleProductEach(builder, inputTy.getAttr()))});
  return success();
}

FLY_INFER_RETURN_TYPES(IntTupleProductOp) {
  auto inputTy = dyn_cast<IntTupleType>(operands[0].getType());
  if (!inputTy)
    return failure();
  IntTupleBuilder<IntTupleAttr> builder(context);
  IntTupleAttr size = intTupleProduct(builder, inputTy.getAttr());
  inferredReturnTypes.assign({IntTupleType::get(size)});
  return success();
}

FLY_INFER_RETURN_TYPES(ShapeDivOp) {
  auto lhsTy = dyn_cast<IntTupleType>(operands[0].getType());
  auto rhsTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!lhsTy || !rhsTy)
    return failure();
  IntTupleBuilder<IntTupleAttr> builder(context);
  inferredReturnTypes.assign(
      {IntTupleType::get(intTupleShapeDiv(builder, lhsTy.getAttr(), rhsTy.getAttr()))});
  return success();
}

FLY_INFER_RETURN_TYPES(CeilDivOp) {
  auto lhsTy = dyn_cast<IntTupleType>(operands[0].getType());
  auto rhsTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!lhsTy || !rhsTy)
    return failure();
  IntTupleBuilder<IntTupleAttr> builder(context);
  inferredReturnTypes.assign(
      {IntTupleType::get(intTupleCeilDiv(builder, lhsTy.getAttr(), rhsTy.getAttr()))});
  return success();
}

FLY_INFER_RETURN_TYPES(ElemLessOp) {
  auto lhsTy = dyn_cast<IntTupleType>(operands[0].getType());
  auto rhsTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!lhsTy || !rhsTy)
    return failure();
  IntTupleBuilder<IntTupleAttr> builder(context);
  IntTupleAttr result = intTupleElemLess(builder, lhsTy.getAttr(), rhsTy.getAttr());
  inferredReturnTypes.assign({IntTupleType::get(result)});
  return success();
}

FLY_INFER_RETURN_TYPES(EqualOp) {
  auto lhsTy = dyn_cast<IntTupleType>(operands[0].getType());
  auto rhsTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!lhsTy || !rhsTy)
    return failure();
  bool isCongruent = intTupleIsCongruent(lhsTy.getAttr(), rhsTy.getAttr());
  IntTupleAttr result = IntTupleAttr::getLeafStatic(context, isCongruent ? 1 : 0);
  inferredReturnTypes.assign({IntTupleType::get(result)});
  return success();
}

FLY_INFER_RETURN_TYPES(AppendOp) {
  auto baseLayout = dyn_cast<LayoutType>(operands[0].getType());
  auto elemLayout = dyn_cast<LayoutType>(operands[1].getType());
  if (!baseLayout || !elemLayout)
    return failure();

  int32_t n = -1;
  if (properties) {
    auto nAttr = properties.as<Properties *>()->n;
    if (nAttr)
      n = static_cast<int32_t>(nAttr.getInt());
  }

  LayoutAttr baseAttr = baseLayout.getAttr();
  LayoutAttr elemAttr = elemLayout.getAttr();

  IntTupleBuilder<IntTupleAttr> builder(context);
  IntTupleAttr newShape = intTupleAppend(builder, baseAttr.getShape(), elemAttr.getShape(), n);
  IntTupleAttr newStride = intTupleAppend(builder, baseAttr.getStride(), elemAttr.getStride(), n);

  inferredReturnTypes.assign(
      {LayoutType::get(context, LayoutAttr::get(context, newShape, newStride))});
  return success();
}

FLY_INFER_RETURN_TYPES(PrependOp) {
  auto baseLayout = dyn_cast<LayoutType>(operands[0].getType());
  auto elemLayout = dyn_cast<LayoutType>(operands[1].getType());
  if (!baseLayout || !elemLayout)
    return failure();

  int32_t n = -1;
  if (properties) {
    auto nAttr = properties.as<Properties *>()->n;
    if (nAttr)
      n = static_cast<int32_t>(nAttr.getInt());
  }

  LayoutAttr baseAttr = baseLayout.getAttr();
  LayoutAttr elemAttr = elemLayout.getAttr();

  IntTupleBuilder<IntTupleAttr> builder(context);
  IntTupleAttr newShape = intTuplePrepend(builder, baseAttr.getShape(), elemAttr.getShape(), n);
  IntTupleAttr newStride = intTuplePrepend(builder, baseAttr.getStride(), elemAttr.getStride(), n);

  inferredReturnTypes.assign(
      {LayoutType::get(context, LayoutAttr::get(context, newShape, newStride))});
  return success();
}

FLY_INFER_RETURN_TYPES(SelectOp) {
  auto idxArr = properties.as<Properties *>()->indices.asArrayRef();
  SmallVector<int32_t> indices(idxArr.begin(), idxArr.end());

  Type inputTy = operands[0].getType();
  IntTupleBuilder<IntTupleAttr> builder(context);

  if (auto tupleTy = dyn_cast<IntTupleType>(inputTy)) {
    IntTupleAttr profile = tupleTy.getAttr();
    IntTupleAttr selected = intTupleSelect(builder, profile, indices);
    inferredReturnTypes.assign({IntTupleType::get(selected)});
    return success();
  }

  if (auto layoutTy = dyn_cast<LayoutType>(inputTy)) {
    LayoutAttr profile = layoutTy.getAttr();
    IntTupleAttr newShape = intTupleSelect(builder, profile.getShape(), indices);
    IntTupleAttr newStride = intTupleSelect(builder, profile.getStride(), indices);
    inferredReturnTypes.assign(
        {LayoutType::get(context, LayoutAttr::get(context, newShape, newStride))});
    return success();
  }

  return failure();
}

FLY_INFER_RETURN_TYPES(GroupOp) {
  int32_t begin = properties.as<Properties *>()->begin.getInt();
  int32_t end = properties.as<Properties *>()->end.getInt();

  Type inputTy = operands[0].getType();
  IntTupleBuilder<IntTupleAttr> builder(context);

  if (auto tupleTy = dyn_cast<IntTupleType>(inputTy)) {
    IntTupleAttr profile = tupleTy.getAttr();
    IntTupleAttr grouped = intTupleGroup(builder, profile, begin, end);
    inferredReturnTypes.assign({IntTupleType::get(grouped)});
    return success();
  }

  if (auto layoutTy = dyn_cast<LayoutType>(inputTy)) {
    LayoutAttr profile = layoutTy.getAttr();
    IntTupleAttr newShape = intTupleGroup(builder, profile.getShape(), begin, end);
    IntTupleAttr newStride = intTupleGroup(builder, profile.getStride(), begin, end);
    inferredReturnTypes.assign(
        {LayoutType::get(context, LayoutAttr::get(context, newShape, newStride))});
    return success();
  }

  return failure();
}

FLY_INFER_RETURN_TYPES(SliceOp) {
  Type srcTy = operands[0].getType();
  auto coordTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!coordTy)
    return failure();

  IntTupleAttr coordAttr = coordTy.getAttr();
  IntTupleBuilder<IntTupleAttr> builder(context);

  if (auto srcTupleTy = dyn_cast<IntTupleType>(srcTy)) {
    IntTupleAttr result = intTupleSlice(builder, srcTupleTy.getAttr(), coordAttr);
    inferredReturnTypes.assign({IntTupleType::get(result)});
    return success();
  }
  if (auto srcLayoutTy = dyn_cast<LayoutType>(srcTy)) {
    LayoutAttr profile = srcLayoutTy.getAttr();
    IntTupleAttr newShape = intTupleSlice(builder, profile.getShape(), coordAttr);
    IntTupleAttr newStride = intTupleSlice(builder, profile.getStride(), coordAttr);
    inferredReturnTypes.assign(
        {LayoutType::get(context, LayoutAttr::get(context, newShape, newStride))});
    return success();
  }
  if (auto srcMemRefTy = dyn_cast<MemRefType>(srcTy)) {
    LayoutAttr layoutAttr = srcMemRefTy.getLayout();
    IntTupleAttr newShape = intTupleSlice(builder, layoutAttr.getShape(), coordAttr);
    IntTupleAttr newStride = intTupleSlice(builder, layoutAttr.getStride(), coordAttr);
    auto newLayoutAttr = LayoutAttr::get(context, newShape, newStride);
    inferredReturnTypes.assign(
        {MemRefType::get(srcMemRefTy.getElemTy(), srcMemRefTy.getAddressSpace(), newLayoutAttr)});
    return success();
  }

  return failure();
}

FLY_INFER_RETURN_TYPES(DiceOp) {
  Type srcTy = operands[0].getType();
  if (isa<IntTupleType, LayoutType, MemRefType>(srcTy)) {
    inferredReturnTypes.assign({srcTy});
    return success();
  }
  return failure();
}

FLY_INFER_RETURN_TYPES(SizeOp) {
  if (auto intTupleTy = dyn_cast<IntTupleType>(operands[0].getType())) {
    IntTupleBuilder<IntTupleAttr> builder(context);
    IntTupleAttr size = intTupleProduct(builder, intTupleTy.getAttr());
    inferredReturnTypes.assign({IntTupleType::get(size)});
    return success();
  }
  if (auto layoutTy = dyn_cast<LayoutType>(operands[0].getType())) {
    LayoutBuilder<LayoutAttr> layoutBuilder(context);
    IntTupleAttr size = layoutSize(layoutBuilder, layoutTy.getAttr());
    inferredReturnTypes.assign({IntTupleType::get(size)});
    return success();
  }
  if (auto memrefTy = dyn_cast<MemRefType>(operands[0].getType())) {
    LayoutBuilder<LayoutAttr> layoutBuilder(context);
    IntTupleAttr size = layoutSize(layoutBuilder, memrefTy.getLayout());
    inferredReturnTypes.assign({IntTupleType::get(size)});
    return success();
  }
  return failure();
}

FLY_INFER_RETURN_TYPES(CosizeOp) {
  auto layoutTy = dyn_cast<LayoutType>(operands[0].getType());
  if (!layoutTy)
    return failure();

  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  IntTupleAttr cosize = layoutCosize(layoutBuilder, layoutTy.getAttr());
  inferredReturnTypes.assign({IntTupleType::get(cosize)});
  return success();
}

FLY_INFER_RETURN_TYPES(Crd2IdxOp) {
  auto coordTy = dyn_cast<IntTupleType>(operands[0].getType());
  auto layoutTy = dyn_cast<LayoutType>(operands[1].getType());
  if (!coordTy || !layoutTy)
    return failure();

  IntTupleAttr coordAttr = coordTy.getAttr();
  LayoutAttr layoutAttr = layoutTy.getAttr();
  IntTupleBuilder<IntTupleAttr> builder(context);

  IntTupleAttr result =
      layoutCrd2idx(builder, coordAttr, layoutAttr.getShape(), layoutAttr.getStride());
  inferredReturnTypes.assign({IntTupleType::get(result)});
  return success();
}

FLY_INFER_RETURN_TYPES(Idx2CrdOp) {
  auto layoutTy = dyn_cast<LayoutType>(operands[1].getType());
  if (!layoutTy)
    return failure();
  inferredReturnTypes.assign({IntTupleType::get(layoutTy.getAttr().getShape())});
  return success();
}

FLY_INFER_RETURN_TYPES(GetFlatCoordOp) {
  auto inputTupleTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (inputTupleTy) {
    inferredReturnTypes.assign({inputTupleTy});
    return success();
  }
  auto inputLayoutTy = dyn_cast<LayoutType>(operands[1].getType());
  if (!inputLayoutTy)
    return failure();
  inferredReturnTypes.assign({IntTupleType::get(inputLayoutTy.getAttr().getShape())});
  return success();
}

FLY_INFER_RETURN_TYPES(GetHierCoordOp) {
  auto inputTupleTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (inputTupleTy) {
    inferredReturnTypes.assign({inputTupleTy});
    return success();
  }
  auto inputLayoutTy = dyn_cast<LayoutType>(operands[1].getType());
  if (!inputLayoutTy)
    return failure();
  inferredReturnTypes.assign({IntTupleType::get(inputLayoutTy.getAttr().getShape())});
  return success();
}

FLY_INFER_RETURN_TYPES(CoalesceOp) {
  auto layoutTy = dyn_cast<LayoutType>(operands[0].getType());
  if (!layoutTy)
    return failure();

  std::optional<IntTupleAttr> profileAttr;
  if (operands.size() > 1 && operands[1]) {
    auto profileTy = dyn_cast<IntTupleType>(operands[1].getType());
    if (!profileTy)
      return failure();
    profileAttr = profileTy.getAttr();
  }

  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  LayoutAttr inferred = layoutCoalesce(layoutBuilder, layoutTy.getAttr(), profileAttr);
  inferredReturnTypes.assign({LayoutType::get(context, inferred)});
  return success();
}

FLY_INFER_RETURN_TYPES(CompositionOp) {
  auto outerLayoutTy = dyn_cast<LayoutType>(operands[0].getType());
  if (!outerLayoutTy)
    return failure();

  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  Type innerTy = operands[1].getType();
  if (auto tileTy = dyn_cast<TileType>(innerTy)) {
    LayoutAttr inferred =
        layoutComposition(layoutBuilder, outerLayoutTy.getAttr(), tileTy.getAttr());
    inferredReturnTypes.assign({LayoutType::get(context, inferred)});
    return success();
  }
  if (auto innerLayoutTy = dyn_cast<LayoutType>(innerTy)) {
    LayoutAttr inferred =
        layoutComposition(layoutBuilder, outerLayoutTy.getAttr(), innerLayoutTy.getAttr());
    inferredReturnTypes.assign({LayoutType::get(context, inferred)});
    return success();
  }
  return failure();
}

FLY_INFER_RETURN_TYPES(ComplementOp) {
  auto layoutTy = dyn_cast<LayoutType>(operands[0].getType());
  if (!layoutTy)
    return failure();

  std::optional<IntTupleAttr> codomainSizeAttr;
  if (operands.size() > 1 && operands[1]) {
    codomainSizeAttr = cast<IntTupleType>(operands[1].getType()).getAttr();
  }

  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  LayoutAttr inferred = layoutComplement(layoutBuilder, layoutTy.getAttr(), codomainSizeAttr);
  inferredReturnTypes.assign({LayoutType::get(context, inferred)});
  return success();
}

FLY_INFER_RETURN_TYPES(RightInverseOp) {
  auto layoutTy = dyn_cast<LayoutType>(operands[0].getType());
  if (!layoutTy)
    return failure();
  inferredReturnTypes.assign({layoutTy});
  return success();
}

FLY_INFER_RETURN_TYPES(LeftInverseOp) {
  auto layoutTy = dyn_cast<LayoutType>(operands[0].getType());
  if (!layoutTy)
    return failure();
  inferredReturnTypes.assign({layoutTy});
  return success();
}

FLY_INFER_RETURN_TYPES(RecastLayoutOp) {
  auto layoutTy = dyn_cast<LayoutType>(operands[2].getType());
  if (!layoutTy)
    return failure();
  inferredReturnTypes.assign({layoutTy});
  return success();
}

FLY_INFER_RETURN_TYPES(LogicalDivideOp) {
  LayoutAttr layoutAttr = nullptr;
  MemRefType memrefTy = nullptr;
  Type lhsTy = operands[0].getType();

  if (auto layoutTy = dyn_cast<LayoutType>(lhsTy)) {
    layoutAttr = layoutTy.getAttr();
  } else if ((memrefTy = dyn_cast<MemRefType>(lhsTy))) {
    layoutAttr = memrefTy.getLayout();
  } else {
    return failure();
  }

  Type divisorTy = operands[1].getType();
  LayoutAttr inferred;

  if (auto divisorLayoutTy = dyn_cast<LayoutType>(divisorTy)) {
    LayoutBuilder<LayoutAttr> layoutBuilder(context);
    inferred = layoutLogicalDivide(layoutBuilder, layoutAttr, divisorLayoutTy.getAttr());
  } else if (auto divisorTileTy = dyn_cast<TileType>(divisorTy)) {
    LayoutBuilder<LayoutAttr> layoutBuilder(context);
    inferred = layoutLogicalDivide(layoutBuilder, layoutAttr, divisorTileTy.getAttr());
  } else {
    return failure();
  }

  if (memrefTy) {
    inferredReturnTypes.assign(
        {MemRefType::get(memrefTy.getElemTy(), memrefTy.getAddressSpace(), inferred)});
  } else {
    inferredReturnTypes.assign({LayoutType::get(context, inferred)});
  }
  return success();
}

FLY_INFER_RETURN_TYPES(ZippedDivideOp) {
  LayoutAttr layoutAttr = nullptr;
  MemRefType memrefTy = nullptr;
  Type lhsTy = operands[0].getType();

  if (auto layoutTy = dyn_cast<LayoutType>(lhsTy)) {
    layoutAttr = layoutTy.getAttr();
  } else if ((memrefTy = dyn_cast<MemRefType>(lhsTy))) {
    layoutAttr = memrefTy.getLayout();
  } else {
    return failure();
  }

  Type divisorTy = operands[1].getType();
  LayoutAttr inferred;

  if (auto divisorLayoutTy = dyn_cast<LayoutType>(divisorTy)) {
    LayoutBuilder<LayoutAttr> layoutBuilder(context);
    inferred = layoutZippedDivide(layoutBuilder, layoutAttr, divisorLayoutTy.getAttr());
  } else if (auto divisorTileTy = dyn_cast<TileType>(divisorTy)) {
    LayoutBuilder<LayoutAttr> layoutBuilder(context);
    inferred = layoutZippedDivide(layoutBuilder, layoutAttr, divisorTileTy.getAttr());
  } else {
    return failure();
  }

  if (memrefTy) {
    inferredReturnTypes.assign(
        {MemRefType::get(memrefTy.getElemTy(), memrefTy.getAddressSpace(), inferred)});
  } else {
    inferredReturnTypes.assign({LayoutType::get(context, inferred)});
  }
  return success();
}

FLY_INFER_RETURN_TYPES(TiledDivideOp) {
  LayoutAttr layoutAttr = nullptr;
  MemRefType memrefTy = nullptr;
  Type lhsTy = operands[0].getType();

  if (auto layoutTy = dyn_cast<LayoutType>(lhsTy)) {
    layoutAttr = layoutTy.getAttr();
  } else if ((memrefTy = dyn_cast<MemRefType>(lhsTy))) {
    layoutAttr = memrefTy.getLayout();
  } else {
    return failure();
  }

  Type divisorTy = operands[1].getType();
  LayoutAttr inferred;

  if (auto divisorLayoutTy = dyn_cast<LayoutType>(divisorTy)) {
    LayoutBuilder<LayoutAttr> layoutBuilder(context);
    inferred = layoutTiledDivide(layoutBuilder, layoutAttr, divisorLayoutTy.getAttr());
  } else if (auto divisorTileTy = dyn_cast<TileType>(divisorTy)) {
    LayoutBuilder<LayoutAttr> layoutBuilder(context);
    inferred = layoutTiledDivide(layoutBuilder, layoutAttr, divisorTileTy.getAttr());
  } else {
    return failure();
  }

  if (memrefTy) {
    inferredReturnTypes.assign(
        {MemRefType::get(memrefTy.getElemTy(), memrefTy.getAddressSpace(), inferred)});
  } else {
    inferredReturnTypes.assign({LayoutType::get(context, inferred)});
  }
  return success();
}

FLY_INFER_RETURN_TYPES(FlatDivideOp) {
  LayoutAttr layoutAttr = nullptr;
  MemRefType memrefTy = nullptr;
  Type lhsTy = operands[0].getType();

  if (auto layoutTy = dyn_cast<LayoutType>(lhsTy)) {
    layoutAttr = layoutTy.getAttr();
  } else if ((memrefTy = dyn_cast<MemRefType>(lhsTy))) {
    layoutAttr = memrefTy.getLayout();
  } else {
    return failure();
  }

  Type divisorTy = operands[1].getType();
  LayoutAttr inferred;

  if (auto divisorLayoutTy = dyn_cast<LayoutType>(divisorTy)) {
    LayoutBuilder<LayoutAttr> layoutBuilder(context);
    inferred = layoutFlatDivide(layoutBuilder, layoutAttr, divisorLayoutTy.getAttr());
  } else if (auto divisorTileTy = dyn_cast<TileType>(divisorTy)) {
    LayoutBuilder<LayoutAttr> layoutBuilder(context);
    inferred = layoutFlatDivide(layoutBuilder, layoutAttr, divisorTileTy.getAttr());
  } else {
    return failure();
  }

  if (memrefTy) {
    inferredReturnTypes.assign(
        {MemRefType::get(memrefTy.getElemTy(), memrefTy.getAddressSpace(), inferred)});
  } else {
    inferredReturnTypes.assign({LayoutType::get(context, inferred)});
  }
  return success();
}

FLY_INFER_RETURN_TYPES(LogicalProductOp) {
  LayoutAttr layoutAttr = nullptr;
  MemRefType memrefTy = nullptr;
  Type lhsTy = operands[0].getType();

  if (auto layoutTy = dyn_cast<LayoutType>(lhsTy)) {
    layoutAttr = layoutTy.getAttr();
  } else if ((memrefTy = dyn_cast<MemRefType>(lhsTy))) {
    layoutAttr = memrefTy.getLayout();
  } else {
    return failure();
  }

  auto tilerTy = dyn_cast<LayoutType>(operands[1].getType());
  if (!tilerTy)
    return failure();

  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  LayoutAttr inferred = layoutLogicalProduct(layoutBuilder, layoutAttr, tilerTy.getAttr());

  if (memrefTy) {
    inferredReturnTypes.assign(
        {MemRefType::get(memrefTy.getElemTy(), memrefTy.getAddressSpace(), inferred)});
  } else {
    inferredReturnTypes.assign({LayoutType::get(context, inferred)});
  }
  return success();
}

FLY_INFER_RETURN_TYPES(ZippedProductOp) {
  LayoutAttr layoutAttr = nullptr;
  MemRefType memrefTy = nullptr;
  Type lhsTy = operands[0].getType();

  if (auto layoutTy = dyn_cast<LayoutType>(lhsTy)) {
    layoutAttr = layoutTy.getAttr();
  } else if ((memrefTy = dyn_cast<MemRefType>(lhsTy))) {
    layoutAttr = memrefTy.getLayout();
  } else {
    return failure();
  }

  auto tilerTy = dyn_cast<LayoutType>(operands[1].getType());
  if (!tilerTy)
    return failure();

  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  LayoutAttr logicalProd = layoutLogicalProduct(layoutBuilder, layoutAttr, tilerTy.getAttr());

  // zip2_by with tiler shape as guide
  IntTupleBuilder<IntTupleAttr> builder(context);
  IntTupleAttr guide = tilerTy.getAttr().getShape();
  IntTupleAttr newShape = intTupleZip2By(builder, logicalProd.getShape(), guide);
  IntTupleAttr newStride = intTupleZip2By(builder, logicalProd.getStride(), guide);
  LayoutAttr inferred = LayoutAttr::get(context, newShape, newStride);

  if (memrefTy) {
    inferredReturnTypes.assign(
        {MemRefType::get(memrefTy.getElemTy(), memrefTy.getAddressSpace(), inferred)});
  } else {
    inferredReturnTypes.assign({LayoutType::get(context, inferred)});
  }
  return success();
}

FLY_INFER_RETURN_TYPES(TiledProductOp) {
  LayoutAttr layoutAttr = nullptr;
  MemRefType memrefTy = nullptr;
  Type lhsTy = operands[0].getType();

  if (auto layoutTy = dyn_cast<LayoutType>(lhsTy)) {
    layoutAttr = layoutTy.getAttr();
  } else if ((memrefTy = dyn_cast<MemRefType>(lhsTy))) {
    layoutAttr = memrefTy.getLayout();
  } else {
    return failure();
  }

  auto tilerTy = dyn_cast<LayoutType>(operands[1].getType());
  if (!tilerTy)
    return failure();

  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  LayoutAttr logicalProd = layoutLogicalProduct(layoutBuilder, layoutAttr, tilerTy.getAttr());

  IntTupleBuilder<IntTupleAttr> builder(context);
  IntTupleAttr guide = tilerTy.getAttr().getShape();
  IntTupleAttr zippedShape = intTupleZip2By(builder, logicalProd.getShape(), guide);
  IntTupleAttr zippedStride = intTupleZip2By(builder, logicalProd.getStride(), guide);

  // Expand index 1
  // TODO: Implement proper expand logic
  LayoutAttr inferred = LayoutAttr::get(context, zippedShape, zippedStride);

  if (memrefTy) {
    inferredReturnTypes.assign(
        {MemRefType::get(memrefTy.getElemTy(), memrefTy.getAddressSpace(), inferred)});
  } else {
    inferredReturnTypes.assign({LayoutType::get(context, inferred)});
  }
  return success();
}

FLY_INFER_RETURN_TYPES(FlatProductOp) {
  LayoutAttr layoutAttr = nullptr;
  MemRefType memrefTy = nullptr;
  Type lhsTy = operands[0].getType();

  if (auto layoutTy = dyn_cast<LayoutType>(lhsTy)) {
    layoutAttr = layoutTy.getAttr();
  } else if ((memrefTy = dyn_cast<MemRefType>(lhsTy))) {
    layoutAttr = memrefTy.getLayout();
  } else {
    return failure();
  }

  auto tilerTy = dyn_cast<LayoutType>(operands[1].getType());
  if (!tilerTy)
    return failure();

  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  LayoutAttr logicalProd = layoutLogicalProduct(layoutBuilder, layoutAttr, tilerTy.getAttr());

  IntTupleBuilder<IntTupleAttr> builder(context);
  IntTupleAttr guide = tilerTy.getAttr().getShape();
  IntTupleAttr zippedShape = intTupleZip2By(builder, logicalProd.getShape(), guide);
  IntTupleAttr zippedStride = intTupleZip2By(builder, logicalProd.getStride(), guide);

  // Expand indices 0 and 1
  // TODO: Implement proper expand logic
  LayoutAttr inferred = LayoutAttr::get(context, zippedShape, zippedStride);

  if (memrefTy) {
    inferredReturnTypes.assign(
        {MemRefType::get(memrefTy.getElemTy(), memrefTy.getAddressSpace(), inferred)});
  } else {
    inferredReturnTypes.assign({LayoutType::get(context, inferred)});
  }
  return success();
}

FLY_INFER_RETURN_TYPES(BlockedProductOp) {
  LayoutAttr layoutAttr = nullptr;
  MemRefType memrefTy = nullptr;
  Type lhsTy = operands[0].getType();

  if (auto layoutTy = dyn_cast<LayoutType>(lhsTy)) {
    layoutAttr = layoutTy.getAttr();
  } else if ((memrefTy = dyn_cast<MemRefType>(lhsTy))) {
    layoutAttr = memrefTy.getLayout();
  } else {
    return failure();
  }

  auto tilerTy = dyn_cast<LayoutType>(operands[1].getType());
  if (!tilerTy)
    return failure();

  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  LayoutAttr inferred = layoutBlockedProduct(layoutBuilder, layoutAttr, tilerTy.getAttr());

  if (memrefTy) {
    inferredReturnTypes.assign(
        {MemRefType::get(memrefTy.getElemTy(), memrefTy.getAddressSpace(), inferred)});
  } else {
    inferredReturnTypes.assign({LayoutType::get(context, inferred)});
  }
  return success();
}

FLY_INFER_RETURN_TYPES(RakedProductOp) {
  LayoutAttr layoutAttr = nullptr;
  MemRefType memrefTy = nullptr;
  Type lhsTy = operands[0].getType();

  if (auto layoutTy = dyn_cast<LayoutType>(lhsTy)) {
    layoutAttr = layoutTy.getAttr();
  } else if ((memrefTy = dyn_cast<MemRefType>(lhsTy))) {
    layoutAttr = memrefTy.getLayout();
  } else {
    return failure();
  }

  auto tilerTy = dyn_cast<LayoutType>(operands[1].getType());
  if (!tilerTy)
    return failure();

  LayoutBuilder<LayoutAttr> layoutBuilder(context);
  LayoutAttr inferred = layoutRakedProduct(layoutBuilder, layoutAttr, tilerTy.getAttr());

  if (memrefTy) {
    inferredReturnTypes.assign(
        {MemRefType::get(memrefTy.getElemTy(), memrefTy.getAddressSpace(), inferred)});
  } else {
    inferredReturnTypes.assign({LayoutType::get(context, inferred)});
  }
  return success();
}

FLY_INFER_RETURN_TYPES(TileToShapeOp) {
  auto shapeTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!shapeTy)
    return failure();
  IntTupleAttr shapeAttr = shapeTy.getAttr();
  LayoutAttr layoutAttr = LayoutAttr::get(context, shapeAttr, makeDynamicLike(shapeAttr));
  inferredReturnTypes.assign({LayoutType::get(context, layoutAttr)});
  return success();
}

FLY_INFER_RETURN_TYPES(MakeTiledCopyOp) {
  auto copyAtomTy = operands[0].getType();
  auto layoutTy = dyn_cast<LayoutType>(operands[1].getType());
  auto tileTy = dyn_cast<TileType>(operands[2].getType());
  if (!layoutTy || !tileTy)
    return failure();

  auto tiledCopyTy = TiledCopyType::get(context, copyAtomTy, layoutTy, tileTy);
  inferredReturnTypes.assign({tiledCopyTy});
  return success();
}

FLY_INFER_RETURN_TYPES(TiledCopyPartitionSrcOp) {
  auto memrefTy = dyn_cast<MemRefType>(operands[1].getType());
  if (!memrefTy)
    return failure();
  inferredReturnTypes.assign({memrefTy});
  return success();
}

FLY_INFER_RETURN_TYPES(TiledCopyPartitionDstOp) {
  auto memrefTy = dyn_cast<MemRefType>(operands[1].getType());
  if (!memrefTy)
    return failure();
  inferredReturnTypes.assign({memrefTy});
  return success();
}

FLY_INFER_RETURN_TYPES(TiledCopyPartitionDOp) {
  auto memrefTy = dyn_cast<MemRefType>(operands[1].getType());
  if (!memrefTy)
    return failure();
  inferredReturnTypes.assign({memrefTy});
  return success();
}

FLY_INFER_RETURN_TYPES(TiledCopyPartitionSOp) {
  auto memrefTy = dyn_cast<MemRefType>(operands[1].getType());
  if (!memrefTy)
    return failure();
  inferredReturnTypes.assign({memrefTy});
  return success();
}

FLY_INFER_RETURN_TYPES(TiledCopyRetileOp) {
  auto memrefTy = dyn_cast<MemRefType>(operands[1].getType());
  if (!memrefTy)
    return failure();
  inferredReturnTypes.assign({memrefTy});
  return success();
}

FLY_INFER_RETURN_TYPES(TiledMmaPartitionOp) {
  auto memrefTy = dyn_cast<MemRefType>(operands[2].getType());
  if (!memrefTy)
    return failure();
  inferredReturnTypes.assign({memrefTy});
  return success();
}

FLY_INFER_RETURN_TYPES(TiledMmaPartitionShapeOp) {
  auto memrefTy = dyn_cast<MemRefType>(operands[2].getType());
  if (!memrefTy)
    return failure();
  inferredReturnTypes.assign({IntTupleType::get(memrefTy.getLayout().getShape())});
  return success();
}

FLY_INFER_RETURN_TYPES(MemRefAllocSharedOp) {
  auto memrefTy = dyn_cast<MemRefType>(operands[0].getType());
  if (!memrefTy)
    return failure();
  auto sharedSpace = AddressSpaceAttr::get(context, AddressSpace::Shared);
  inferredReturnTypes.assign(
      {MemRefType::get(memrefTy.getElemTy(), sharedSpace, memrefTy.getLayout())});
  return success();
}

FLY_INFER_RETURN_TYPES(MemRefLoadOp) {
  auto memrefTy = dyn_cast<MemRefType>(operands[0].getType());
  if (!memrefTy)
    return failure();
  inferredReturnTypes.push_back(memrefTy.getElemTy());
  return success();
}

FLY_INFER_RETURN_TYPES(MemRefLoadVecOp) {
  auto memrefTy = dyn_cast<MemRefType>(operands[0].getType());
  if (!memrefTy)
    return failure();

  LayoutAttr layoutAttr = memrefTy.getLayout();
  IntTupleBuilder<IntTupleAttr> builder(context);
  IntAttr size = cast<IntAttr>(intTupleProduct(builder, layoutAttr.getShape()).getValue());

  if (!size.isStatic())
    return failure();

  inferredReturnTypes.push_back(VectorType::get({size.getValue()}, memrefTy.getElemTy()));
  return success();
}

FLY_INFER_RETURN_TYPES(RecastIterOp) {
  auto ptrTy = dyn_cast<PointerType>(operands[0].getType());
  if (!ptrTy)
    return failure();
  inferredReturnTypes.assign({ptrTy});
  return success();
}

FLY_INFER_RETURN_TYPES(AddOffsetOp) {
  auto ptrTy = dyn_cast<PointerType>(operands[0].getType());
  auto offsetTy = dyn_cast<IntTupleType>(operands[1].getType());
  if (!ptrTy || !offsetTy)
    return failure();
  // Offset must be a scalar (leaf) int_tuple
  if (!offsetTy.getAttr().isLeaf())
    return failure();
  inferredReturnTypes.assign({ptrTy});
  return success();
}

FLY_INFER_RETURN_TYPES(ApplySwizzleOp) {
  auto ptrTy = dyn_cast<PointerType>(operands[0].getType());
  if (!ptrTy)
    return failure();
  inferredReturnTypes.assign({ptrTy});
  return success();
}

#undef FLY_INFER_RETURN_TYPES
