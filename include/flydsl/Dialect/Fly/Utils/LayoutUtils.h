#ifndef FLYDSL_DIALECT_UTILS_LAYOUTATTR_H
#define FLYDSL_DIALECT_UTILS_LAYOUTATTR_H

#include <algorithm>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/IntTupleUtils.h"
#include "flydsl/Dialect/Fly/Utils/IntUtils.h"

namespace mlir::fly {

namespace detail {

template <class IntTuple>
typename IntTupleBuilder<IntTuple>::ArithValue layoutCrd2idxTTT(IntTupleBuilder<IntTuple> &builder,
                                                                IntTuple coord, IntTuple shape,
                                                                IntTuple stride);

template <class IntTuple>
typename IntTupleBuilder<IntTuple>::ArithValue
layoutCrd2idxITT(IntTupleBuilder<IntTuple> &builder,
                 typename IntTupleBuilder<IntTuple>::ArithValue coord, IntTuple shape,
                 IntTuple stride) {
  using ArithValue = typename IntTupleBuilder<IntTuple>::ArithValue;
  int32_t rank = shape.rank();
  if (rank == 1) {
    return layoutCrd2idxTTT(builder, builder.makeInt(coord), builder.at(shape, 0),
                            builder.at(stride, 0));
  }
  IntTuple si = builder.at(shape, 0);
  IntTuple di = builder.at(stride, 0);

  ArithValue siProduct = intTupleProductImpl(builder, si);
  ArithValue ci = builder.mod(coord, siProduct);
  ArithValue remaining = builder.div(coord, siProduct);

  ArithValue result;
  if (si.isLeaf()) {
    result = builder.mul(ci, builder.getArithValue(di));
  } else {
    result = layoutCrd2idxITT(builder, ci, si, di);
  }

  for (int i = 1; i < rank; ++i) {
    si = builder.at(shape, i);
    di = builder.at(stride, i);

    if (i == rank - 1) {
      ci = remaining;
    } else {
      siProduct = intTupleProductImpl(builder, si);
      ci = builder.mod(remaining, siProduct);
      remaining = builder.div(remaining, siProduct);
    }
    if (si.isLeaf()) {
      result = builder.add(result, builder.mul(ci, builder.getArithValue(di)));
    } else {
      result = builder.add(result, layoutCrd2idxITT(builder, ci, si, di));
    }
  }
  return result;
}

template <class IntTuple>
typename IntTupleBuilder<IntTuple>::ArithValue layoutCrd2idxTTT(IntTupleBuilder<IntTuple> &builder,
                                                                IntTuple coord, IntTuple shape,
                                                                IntTuple stride) {
  using ArithValue = typename IntTupleBuilder<IntTuple>::ArithValue;
  if (coord.isLeaf()) {
    if (shape.isLeaf()) {
      return builder.mul(builder.getArithValue(coord), builder.getArithValue(stride));
    } else {
      return layoutCrd2idxITT(builder, builder.getArithValue(coord), shape, stride);
    }
  } else {
    assert(coord.rank() == shape.rank() && "Mismatched ranks");
    ArithValue result = layoutCrd2idxTTT(builder, builder.at(coord, 0), builder.at(shape, 0),
                                         builder.at(stride, 0));
    for (int i = 1; i < coord.rank(); ++i) {
      result = builder.add(result, layoutCrd2idxTTT(builder, builder.at(coord, i),
                                                    builder.at(shape, i), builder.at(stride, i)));
    }
    return result;
  }
}

} // namespace detail

template <class IntTuple>
IntTuple layoutCrd2idx(IntTupleBuilder<IntTuple> &builder, IntTuple coord, IntTuple shape,
                       IntTuple stride) {
  return builder.makeInt(detail::layoutCrd2idxTTT(builder, coord, shape, stride));
}

template <class Layout> class LayoutBuilder;

class LayoutValueAdaptor {
private:
  Value value;
  LayoutAttr attr;

public:
  LayoutValueAdaptor(Value value, LayoutAttr attr) : value(value), attr(attr) {}

  bool isLeaf() const { return attr.isLeaf(); }
  int32_t rank() const { return attr.rank(); }

  friend class LayoutBuilder<LayoutValueAdaptor>;
};

template <> class LayoutBuilder<LayoutAttr> : public IntTupleBuilder<IntTupleAttr> {
public:
  using IntTupleBuilder<IntTupleAttr>::IntTupleBuilder;
  using IntTuple = IntTupleAttr;

  LayoutAttr getLayoutAttr(LayoutAttr attr) const { return attr; }
  IntTuple getShape(LayoutAttr attr) const { return attr.getShape(); }
  IntTuple getStride(LayoutAttr attr) const { return attr.getStride(); }

  LayoutAttr materializeConstantLayout(IntTupleAttr shape, IntTupleAttr stride) const {
    return LayoutAttr::get(materializeConstantTuple(shape), materializeConstantTuple(stride));
  }
  LayoutAttr materializeConstantLayout(LayoutAttr attr) const {
    assert(attr.isStatic() && "Layout must be static");
    return attr;
  }
  LayoutAttr makeLayout(IntTupleAttr shape, IntTupleAttr stride) const {
    return LayoutAttr::get(shape, stride);
  }
};

template <> class LayoutBuilder<LayoutValueAdaptor> : public IntTupleBuilder<IntTupleValueAdaptor> {
public:
  using IntTupleBuilder<IntTupleValueAdaptor>::IntTupleBuilder;
  using IntTuple = IntTupleValueAdaptor;

  LayoutAttr getLayoutAttr(LayoutValueAdaptor adaptor) const { return adaptor.attr; }
  IntTuple getShape(LayoutValueAdaptor adaptor) const {
    return IntTupleValueAdaptor::create(*this, adaptor.value.getDefiningOp()->getOperand(0),
                                        adaptor.attr.getShape());
  }
  IntTuple getStride(LayoutValueAdaptor adaptor) const {
    return IntTupleValueAdaptor::create(*this, adaptor.value.getDefiningOp()->getOperand(1),
                                        adaptor.attr.getStride());
  }

  LayoutValueAdaptor materializeConstantLayout(IntTupleAttr shape, IntTupleAttr stride) const {
    return makeLayout(materializeConstantTuple(shape), materializeConstantTuple(stride));
  }
  LayoutValueAdaptor materializeConstantLayout(LayoutAttr attr) const {
    return materializeConstantLayout(attr.getShape(), attr.getStride());
  }
  LayoutValueAdaptor makeLayout(IntTuple shape, IntTuple stride) const {
    auto value = MakeLayoutOp::create(this->builder, this->loc, this->finalize(shape),
                                      this->finalize(stride))
                     .getResult();
    return LayoutValueAdaptor(value, LayoutAttr::get(this->getAttr(shape), this->getAttr(stride)));
  }
  Value getValue(LayoutValueAdaptor adaptor) const { return adaptor.value; }
};

//===----------------------------------------------------------------------===//
// Layout operations
//===----------------------------------------------------------------------===//

template <class Layout>
typename LayoutBuilder<Layout>::IntTuple layoutSize(LayoutBuilder<Layout> &builder, Layout layout) {
  return intTupleProduct(builder, builder.getShape(layout));
}

template <class Layout>
typename LayoutBuilder<Layout>::IntTuple layoutCosize(LayoutBuilder<Layout> &builder,
                                                      Layout layout) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;
  using ArithValue = typename LayoutBuilder<Layout>::ArithValue;

  auto shape = builder.getShape(layout);
  auto stride = builder.getStride(layout);

  SmallVector<IntTuple> flatShapeLeaves;
  SmallVector<IntTuple> flatStrideLeaves;
  intTupleFlattenToVector(builder, shape, flatShapeLeaves);
  intTupleFlattenToVector(builder, stride, flatStrideLeaves);

  ArithValue one = builder.materializeConstantArith(1);
  ArithValue s = builder.getArithValue(flatShapeLeaves[0]);
  ArithValue d = builder.getArithValue(flatStrideLeaves[0]);
  ArithValue cosize = builder.add(one, builder.mul(builder.sub(s, one), d));

  for (size_t i = 1; i < flatShapeLeaves.size(); ++i) {
    ArithValue s = builder.getArithValue(flatShapeLeaves[i]);
    ArithValue d = builder.getArithValue(flatStrideLeaves[i]);
    cosize = builder.add(cosize, builder.mul(builder.sub(s, one), d));
  }
  return builder.makeInt(cosize);
}

namespace detail {

template <class IntTuple>
std::pair<IntTuple, IntTuple> coalesceImpl(const IntTupleBuilder<IntTuple> &builder, IntTuple shape,
                                           IntTuple stride) {
  using ArithValue = typename IntTupleBuilder<IntTuple>::ArithValue;

  SmallVector<IntTuple> flatShapeLeaves;
  SmallVector<IntTuple> flatStrideLeaves;
  intTupleFlattenToVector(builder, shape, flatShapeLeaves);
  intTupleFlattenToVector(builder, stride, flatStrideLeaves);

  const int flatRank = flatShapeLeaves.size();
  ArithValue currShapeInt = builder.getArithValue(flatShapeLeaves[flatRank - 1]);
  ArithValue currStrideInt = builder.getArithValue(flatStrideLeaves[flatRank - 1]);

  if (flatRank == 1) {
    if (builder.isStaticValue(currShapeInt, 1)) {
      return {builder.makeInt(builder.materializeConstantArith(1)),
              builder.makeInt(builder.materializeConstantArith(0))};
    } else {
      return {shape, stride};
    }
  }

  typename IntTupleBuilder<IntTuple>::ElemCollector resultShape;
  typename IntTupleBuilder<IntTuple>::ElemCollector resultStride;
  for (int i = flatRank - 2; i >= 0; --i) {
    ArithValue nextShapeInt = builder.getArithValue(flatShapeLeaves[i]);
    ArithValue nextStrideInt = builder.getArithValue(flatStrideLeaves[i]);

    if (builder.isStaticValue(nextShapeInt, 1)) {
      continue;
    }
    if (builder.isStaticValue(currShapeInt, 1)) {
      currShapeInt = nextShapeInt;
      currStrideInt = nextStrideInt;
      continue;
    }

    bool merged = false;
    if (builder.isStatic(nextShapeInt) && builder.isStatic(nextStrideInt) &&
        builder.isStatic(currShapeInt) && builder.isStatic(currStrideInt)) {
      if (builder.getStaticValue(nextShapeInt) * builder.getStaticValue(nextStrideInt) ==
          builder.getStaticValue(currStrideInt)) {
        currShapeInt = builder.mul(nextShapeInt, currShapeInt);
        currStrideInt = nextStrideInt;
        merged = true;
      }
    }
    if (!merged) {
      resultShape.push_back(builder.makeInt(currShapeInt));
      resultStride.push_back(builder.makeInt(currStrideInt));
      currShapeInt = nextShapeInt;
      currStrideInt = nextStrideInt;
    }
  }

  if (resultShape.empty()) {
    return {builder.makeInt(currShapeInt), builder.makeInt(currStrideInt)};
  }
  resultShape.push_back(builder.makeInt(currShapeInt));
  resultStride.push_back(builder.makeInt(currStrideInt));
  resultShape.reverse();
  resultStride.reverse();
  return {builder.makeTuple(resultShape), builder.makeTuple(resultStride)};
}

template <class IntTuple>
std::pair<IntTuple, IntTuple> coalesceWithProfile(const IntTupleBuilder<IntTuple> &builder,
                                                  IntTuple shape, IntTuple stride,
                                                  IntTupleAttr profile) {
  if (profile.isLeaf()) {
    return coalesceImpl(builder, shape, stride);
  }

  typename IntTupleBuilder<IntTuple>::ElemCollector newShapeElems;
  typename IntTupleBuilder<IntTuple>::ElemCollector newStrideElems;

  int32_t profileRank = profile.rank();
  for (int i = 0; i < shape.rank(); ++i) {
    if (i < profileRank) {
      auto [cs, cd] =
          coalesceWithProfile(builder, builder.at(shape, i), builder.at(stride, i), profile.at(i));
      newShapeElems.push_back(cs);
      newStrideElems.push_back(cd);
    } else {
      newShapeElems.push_back(builder.at(shape, i));
      newStrideElems.push_back(builder.at(stride, i));
    }
  }
  return {builder.makeTuple(newShapeElems), builder.makeTuple(newStrideElems)};
}

template <class IntTuple>
std::pair<IntTuple, IntTuple> compositionImpl(const IntTupleBuilder<IntTuple> &builder,
                                              IntTuple lhsShape, IntTuple lhsStride,
                                              IntTuple rhsShape, IntTuple rhsStride) {
  using ArithValue = typename IntTupleBuilder<IntTuple>::ArithValue;

  if (!rhsShape.isLeaf()) {
    typename IntTupleBuilder<IntTuple>::ElemCollector resultShape;
    typename IntTupleBuilder<IntTuple>::ElemCollector resultStride;
    for (int i = 0; i < rhsShape.rank(); ++i) {
      auto [elemShape, elemStride] = compositionImpl(
          builder, lhsShape, lhsStride, builder.at(rhsShape, i), builder.at(rhsStride, i));
      resultShape.push_back(elemShape);
      resultStride.push_back(elemStride);
    }
    return {builder.makeTuple(resultShape), builder.makeTuple(resultStride)};
  }

  ArithValue rhsStrideVal = builder.getArithValue(rhsStride);
  if (builder.isStaticValue(rhsStrideVal, 0)) {
    return {rhsShape, rhsStride};
  }
  if (lhsShape.isLeaf()) {
    return {rhsShape, builder.makeInt(builder.mul(builder.getArithValue(lhsStride), rhsStrideVal))};
  }

  ArithValue restShape = builder.getArithValue(rhsShape);
  ArithValue restStride = rhsStrideVal;

  typename IntTupleBuilder<IntTuple>::ElemCollector resultShape;
  typename IntTupleBuilder<IntTuple>::ElemCollector resultStride;
  int32_t resultCount = 0;
  IntTuple lastShapeElem = rhsShape;
  IntTuple lastStrideElem = rhsStride;

  int R = lhsShape.rank();
  for (int i = 0; i < R - 1; ++i) {
    ArithValue currShape = builder.getArithValue(builder.at(lhsShape, i));
    ArithValue currStride = builder.getArithValue(builder.at(lhsStride, i));

    if (builder.isStatic(currShape) && builder.isStatic(restStride)) {
      int64_t restStrideVal = builder.getStaticValue(restStride);
      int64_t currShapeVal = builder.getStaticValue(currShape);
      assert(restStrideVal % currShapeVal == 0 || restStrideVal < currShapeVal);
    }

    ArithValue nextShape = builder.ceilDiv(currShape, restStride);
    ArithValue nextStride = builder.ceilDiv(restStride, currShape);

    if (builder.isStaticValue(nextShape, 1) || builder.isStaticValue(restShape, 1)) {
      restStride = nextStride;
      continue;
    }

    ArithValue newShape = builder.min(nextShape, restShape);
    ArithValue newStride = builder.mul(restStride, currStride);

    if (builder.isStatic(newShape) && builder.isStatic(restShape)) {
      int64_t restShapeVal = builder.getStaticValue(restShape);
      int64_t newShapeVal = builder.getStaticValue(newShape);
      assert(restShapeVal % newShapeVal == 0);
    }

    IntTuple lastShapeElem = builder.makeInt(newShape);
    IntTuple lastStrideElem = builder.makeInt(newStride);
    resultShape.push_back(lastShapeElem);
    resultStride.push_back(lastStrideElem);
    restShape = builder.div(restShape, newShape);
    restStride = nextStride;

    ++resultCount;
  }

  ArithValue lhsLastStride = builder.getArithValue(builder.at(lhsStride, R - 1));
  if (resultCount == 0) {
    return {builder.makeInt(restShape), builder.makeInt(builder.mul(restStride, lhsLastStride))};
  }
  if (builder.isStaticValue(restShape, 1)) {
    if (resultCount == 1) {
      return {lastShapeElem, lastStrideElem};
    }
    return {builder.makeTuple(resultShape), builder.makeTuple(resultStride)};
  }

  resultShape.push_back(builder.makeInt(restShape));
  resultStride.push_back(builder.makeInt(builder.mul(restStride, lhsLastStride)));
  return {builder.makeTuple(resultShape), builder.makeTuple(resultStride)};
}

template <class IntTuple>
std::pair<IntTuple, IntTuple> complementImpl(const IntTupleBuilder<IntTuple> &builder,
                                             IntTuple filteredShape, IntTuple filteredStride,
                                             IntTuple codomainSize) {
  using ArithValue = typename IntTupleBuilder<IntTuple>::ArithValue;

  if (!codomainSize.isLeaf()) {
    assert(false && "this is for basis-strided layout, maybe support this later");
    return {filteredShape, filteredStride};
  }

  auto flatShape = intTupleFlatten(builder, filteredShape);
  auto flatStride = intTupleFlatten(builder, filteredStride);

  if (flatStride.isLeaf()) {
    if (builder.isStaticValue(builder.getArithValue(flatStride), 0)) {
      return {codomainSize, builder.makeInt(builder.materializeConstantArith(1))};
    }
  }

  const int R = flatStride.rank();
  assert(R == 1 ||
         builder.getAttr(filteredStride).isStatic() && "stride must be static for complement");

  struct ShapeStridePair {
    ArithValue shapeVal;
    ArithValue strideVal;
    int64_t strideStatic;
  };
  SmallVector<ShapeStridePair> modes;
  modes.reserve(R);

  if (!flatStride.isLeaf()) {
    for (int i = 0; i < R; ++i) {
      ArithValue s = builder.getArithValue(builder.at(flatShape, i));
      ArithValue d = builder.getArithValue(builder.at(flatStride, i));
      modes.push_back({s, d, builder.getStaticValue(d)});
    }
    std::sort(modes.begin(), modes.end(), [](const ShapeStridePair &a, const ShapeStridePair &b) {
      return a.strideStatic < b.strideStatic;
    });
  } else {
    modes.push_back({builder.getArithValue(flatShape), builder.getArithValue(flatStride), 0});
  }

  ArithValue lastStride = builder.materializeConstantArith(1);
  typename IntTupleBuilder<IntTuple>::ElemCollector resultShapeVals;
  typename IntTupleBuilder<IntTuple>::ElemCollector resultStrideVals;

  resultStrideVals.push_back(builder.makeInt(lastStride));
  for (int64_t i = 0; i < R - 1; ++i) {
    ArithValue minStride = modes[i].strideVal;
    ArithValue newShape = builder.div(minStride, lastStride);
    ArithValue newStride = builder.mul(minStride, modes[i].shapeVal);

    resultShapeVals.push_back(builder.makeInt(newShape));
    resultStrideVals.push_back(builder.makeInt(newStride));
    lastStride = newStride;
  }

  auto lastMode = modes.back();
  ArithValue newShape = builder.div(lastMode.strideVal, lastStride);
  resultShapeVals.push_back(builder.makeInt(newShape));

  ArithValue newStrideForRest = builder.mul(lastMode.strideVal, lastMode.shapeVal);
  ArithValue restShape = builder.ceilDiv(builder.getArithValue(codomainSize), newStrideForRest);
  ArithValue restStride = newStrideForRest;

  resultShapeVals.push_back(builder.makeInt(restShape));
  resultStrideVals.push_back(builder.makeInt(restStride));

  return coalesceImpl(builder, builder.makeTuple(resultShapeVals),
                      builder.makeTuple(resultStrideVals));
}

} // namespace detail

template <class Layout>
Layout layoutCoalesce(LayoutBuilder<Layout> &builder, Layout layout,
                      std::optional<IntTupleAttr> profileAttr = std::nullopt) {
  auto shape = builder.getShape(layout);
  auto stride = builder.getStride(layout);

  if (profileAttr) {
    auto [cs, cd] = detail::coalesceWithProfile(builder, shape, stride, *profileAttr);
    return builder.makeLayout(cs, cd);
  }
  auto [cs, cd] = detail::coalesceImpl(builder, shape, stride);
  return builder.makeLayout(cs, cd);
}

template <class Layout>
Layout layoutComposition(LayoutBuilder<Layout> &builder, Layout outerLayout, Layout innerLayout) {
  auto [coalShape, coalStride] =
      detail::coalesceImpl(builder, builder.getShape(outerLayout), builder.getStride(outerLayout));
  auto [retShape, retStride] =
      detail::compositionImpl(builder, coalShape, coalStride, builder.getShape(innerLayout),
                              builder.getStride(innerLayout));
  return builder.makeLayout(retShape, retStride);
}
template <class Layout>
Layout layoutComposition(LayoutBuilder<Layout> &builder, Layout outerLayout,
                         TileAttr innerTileAttr) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;

  auto lhsShape = builder.getShape(outerLayout);
  auto lhsStride = builder.getStride(outerLayout);

  typename LayoutBuilder<Layout>::ElemCollector retShape;
  typename LayoutBuilder<Layout>::ElemCollector retStride;

  int32_t tileRank = innerTileAttr.rank();
  for (int i = 0; i < lhsShape.rank(); ++i) {
    if (i < tileRank && !innerTileAttr.isNoneMode(i)) {
      auto [coalShape, coalStride] =
          detail::coalesceImpl(builder, builder.at(lhsShape, i), builder.at(lhsStride, i));

      IntTuple rhsShape, rhsStride;
      if (auto attr = dyn_cast<LayoutAttr>(innerTileAttr.at(i))) {
        rhsShape = builder.materializeConstantTuple(attr.getShape());
        rhsStride = builder.materializeConstantTuple(attr.getStride());
      } else {
        rhsShape = builder.makeInt(
            builder.materializeConstantArith(cast<IntAttr>(innerTileAttr.at(i)).getValue()));
        rhsStride = builder.makeInt(builder.materializeConstantArith(1));
      }
      auto [elemShape, elemStride] =
          detail::compositionImpl(builder, coalShape, coalStride, rhsShape, rhsStride);
      retShape.push_back(elemShape);
      retStride.push_back(elemStride);
    } else {
      retShape.push_back(builder.at(lhsShape, i));
      retStride.push_back(builder.at(lhsStride, i));
    }
  }
  return builder.makeLayout(builder.makeTuple(retShape), builder.makeTuple(retStride));
}

template <class Layout>
Layout layoutComplement(
    LayoutBuilder<Layout> &builder, Layout layout,
    std::optional<typename LayoutBuilder<Layout>::IntTuple> codomainSize = std::nullopt) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;

  auto filteredShape = intTupleFilterZero(builder, builder.getLayoutAttr(layout).getStride(),
                                          builder.getShape(layout));
  auto filteredStride = builder.getStride(layout);

  IntTuple codomain =
      codomainSize ? *codomainSize
                   : layoutCosize(builder, builder.makeLayout(filteredShape, filteredStride));
  auto [retShape, retStride] =
      detail::complementImpl(builder, filteredShape, filteredStride, codomain);
  return builder.makeLayout(retShape, retStride);
}

template <class Layout> Layout layoutRightInverse(LayoutBuilder<Layout> &builder, Layout layout);
template <class Layout> Layout layoutLeftInverse(LayoutBuilder<Layout> &builder, Layout layout);

template <class Layout>
Layout layoutLogicalDivide(LayoutBuilder<Layout> &builder, Layout layout, Layout divisorLayout) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;

  auto coalesced = layoutCoalesce(builder, layout);
  IntTuple codomainSize = layoutSize(builder, coalesced);

  auto complement = layoutComplement(builder, divisorLayout, codomainSize);

  typename LayoutBuilder<Layout>::ElemCollector rhsShapeElems;
  typename LayoutBuilder<Layout>::ElemCollector rhsStrideElems;
  rhsShapeElems.push_back(builder.getShape(divisorLayout));
  rhsShapeElems.push_back(builder.getShape(complement));
  rhsStrideElems.push_back(builder.getStride(divisorLayout));
  rhsStrideElems.push_back(builder.getStride(complement));

  IntTuple rhsShape = builder.makeTuple(rhsShapeElems);
  IntTuple rhsStride = builder.makeTuple(rhsStrideElems);
  Layout rhsLayout = builder.makeLayout(rhsShape, rhsStride);
  return layoutComposition(builder, layout, rhsLayout);
}

template <class Layout>
Layout layoutLogicalDivide(LayoutBuilder<Layout> &builder, Layout layout, TileAttr divisorTile) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;

  auto leafDivide = [&](Layout currentLayout, Attribute divisor) -> Layout {
    if (auto attr = dyn_cast<LayoutAttr>(divisor)) {
      return layoutLogicalDivide(builder, currentLayout, builder.materializeConstantLayout(attr));
    } else if (auto intDivisor = dyn_cast<IntAttr>(divisor)) {
      IntTuple divisorShape = builder.materializeConstantTuple(IntTupleAttr::get(intDivisor));
      IntTuple divisorStride = builder.makeInt(builder.materializeConstantArith(1));
      Layout divisorLayout = builder.makeLayout(divisorShape, divisorStride);
      return layoutLogicalDivide(builder, currentLayout, divisorLayout);
    }
    llvm_unreachable("invalid divisor type");
  };

  if (divisorTile.isLeaf()) {
    return leafDivide(layout, divisorTile.getValue());
  }

  auto shape = builder.getShape(layout);
  auto stride = builder.getStride(layout);
  int32_t layoutRank = shape.rank();
  int32_t tileRank = divisorTile.rank();

  typename LayoutBuilder<Layout>::ElemCollector outShape;
  typename LayoutBuilder<Layout>::ElemCollector outStride;
  for (int i = 0; i < layoutRank; ++i) {
    IntTuple shapeElem = builder.at(shape, i);
    IntTuple strideElem = builder.at(stride, i);
    if (i < tileRank && !divisorTile.isNoneMode(i)) {
      Layout subLayout = builder.makeLayout(shapeElem, strideElem);
      Layout divided = leafDivide(subLayout, divisorTile.at(i));
      outShape.push_back(builder.getShape(divided));
      outStride.push_back(builder.getStride(divided));
    } else {
      outShape.push_back(shapeElem);
      outStride.push_back(strideElem);
    }
  }
  return builder.makeLayout(builder.makeTuple(outShape), builder.makeTuple(outStride));
}

template <class Layout>
Layout layoutZippedDivide(LayoutBuilder<Layout> &builder, Layout layout, Layout divisorLayout) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;

  Layout logicalDiv = layoutLogicalDivide(builder, layout, divisorLayout);

  auto *ctx = builder.getLayoutAttr(layout).getContext();
  IntTupleAttr guide = IntTupleAttr::getLeafStatic(ctx, 1);
  IntTuple retShape = intTupleZip2By(builder, builder.getShape(logicalDiv), guide);
  IntTuple retStride = intTupleZip2By(builder, builder.getStride(logicalDiv), guide);
  return builder.makeLayout(retShape, retStride);
}

template <class Layout>
Layout layoutZippedDivide(LayoutBuilder<Layout> &builder, Layout layout, TileAttr divisorTile) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;

  Layout logicalDiv = layoutLogicalDivide(builder, layout, divisorTile);
  auto *ctx = builder.getLayoutAttr(layout).getContext();

  SmallVector<Attribute> guideElems;
  for (int i = 0; i < divisorTile.rank(); ++i) {
    guideElems.push_back(IntTupleAttr::getLeafNone(ctx));
  }
  IntTupleAttr guide = IntTupleAttr::get(ArrayAttr::get(ctx, guideElems));
  IntTuple retShape = intTupleZip2By(builder, builder.getShape(logicalDiv), guide);
  IntTuple retStride = intTupleZip2By(builder, builder.getStride(logicalDiv), guide);
  return builder.makeLayout(retShape, retStride);
}

template <class Layout>
Layout layoutTiledDivide(LayoutBuilder<Layout> &builder, Layout layout, Layout divisorLayout) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;

  Layout zipped = layoutZippedDivide(builder, layout, divisorLayout);
  IntTuple retShape = intTupleExpand(builder, builder.getShape(zipped), {1});
  IntTuple retStride = intTupleExpand(builder, builder.getStride(zipped), {1});
  return builder.makeLayout(retShape, retStride);
}
template <class Layout>
Layout layoutTiledDivide(LayoutBuilder<Layout> &builder, Layout layout, TileAttr divisorTile) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;
  Layout zipped = layoutZippedDivide(builder, layout, divisorTile);
  IntTuple retShape = intTupleExpand(builder, builder.getShape(zipped), {1});
  IntTuple retStride = intTupleExpand(builder, builder.getStride(zipped), {1});
  return builder.makeLayout(retShape, retStride);
}

template <class Layout>
Layout layoutFlatDivide(LayoutBuilder<Layout> &builder, Layout layout, Layout divisorLayout) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;
  Layout zipped = layoutZippedDivide(builder, layout, divisorLayout);
  IntTuple retShape = intTupleExpand(builder, builder.getShape(zipped), {0, 1});
  IntTuple retStride = intTupleExpand(builder, builder.getStride(zipped), {0, 1});
  return builder.makeLayout(retShape, retStride);
}
template <class Layout>
Layout layoutFlatDivide(LayoutBuilder<Layout> &builder, Layout layout, TileAttr divisorTile) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;
  Layout zipped = layoutZippedDivide(builder, layout, divisorTile);
  IntTuple retShape = intTupleExpand(builder, builder.getShape(zipped), {0, 1});
  IntTuple retStride = intTupleExpand(builder, builder.getStride(zipped), {0, 1});
  return builder.makeLayout(retShape, retStride);
}

template <class Layout>
Layout layoutAppendToRank(LayoutBuilder<Layout> &builder, Layout layout, int32_t targetRank) {
  auto shape = builder.getShape(layout);
  auto stride = builder.getStride(layout);
  int32_t currentRank = shape.rank();
  if (targetRank <= currentRank) {
    return layout;
  }

  typename LayoutBuilder<Layout>::ElemCollector shapeElems;
  typename LayoutBuilder<Layout>::ElemCollector strideElems;
  if (shape.isLeaf()) {
    shapeElems.push_back(shape);
    strideElems.push_back(stride);
  } else {
    for (int i = 0; i < shape.rank(); ++i) {
      shapeElems.push_back(builder.at(shape, i));
      strideElems.push_back(builder.at(stride, i));
    }
  }

  for (int32_t i = currentRank; i < targetRank; ++i) {
    shapeElems.push_back(builder.makeInt(builder.materializeConstantArith(1)));
    strideElems.push_back(builder.makeInt(builder.materializeConstantArith(0)));
  }
  return builder.makeLayout(builder.makeTuple(shapeElems), builder.makeTuple(strideElems));
}

template <class Layout>
Layout layoutLogicalProduct(LayoutBuilder<Layout> &builder, Layout blockLayout,
                            Layout tilerLayout) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;

  IntTuple blockSize = layoutSize(builder, blockLayout);
  IntTuple tilerCosize = layoutCosize(builder, tilerLayout);
  auto blockSizeVal = builder.getArithValue(blockSize);
  auto tilerCosizeVal = builder.getArithValue(tilerCosize);

  if (!builder.isStatic(blockSizeVal) || !builder.isStatic(tilerCosizeVal)) {
    return blockLayout;
  }

  IntTuple codomainSize = builder.makeInt(builder.mul(blockSizeVal, tilerCosizeVal));
  Layout complement = layoutComplement(builder, blockLayout, codomainSize);
  Layout composed = layoutComposition(builder, complement, tilerLayout);

  typename LayoutBuilder<Layout>::ElemCollector retShapeElems;
  typename LayoutBuilder<Layout>::ElemCollector retStrideElems;
  retShapeElems.push_back(builder.getShape(blockLayout));
  retShapeElems.push_back(builder.getShape(composed));
  retStrideElems.push_back(builder.getStride(blockLayout));
  retStrideElems.push_back(builder.getStride(composed));

  return builder.makeLayout(builder.makeTuple(retShapeElems), builder.makeTuple(retStrideElems));
}

template <class Layout>
Layout layoutBlockedProduct(LayoutBuilder<Layout> &builder, Layout blockLayout,
                            Layout tilerLayout) {
  auto blockShape = builder.getShape(blockLayout);
  auto tilerShape = builder.getShape(tilerLayout);
  int32_t blockRank = blockShape.isLeaf() ? 1 : blockShape.rank();
  int32_t tilerRank = tilerShape.isLeaf() ? 1 : tilerShape.rank();
  int32_t targetRank = std::max(blockRank, tilerRank);

  Layout paddedBlock = layoutAppendToRank(builder, blockLayout, targetRank);
  Layout paddedTiler = layoutAppendToRank(builder, tilerLayout, targetRank);
  Layout logicalProd = layoutLogicalProduct(builder, paddedBlock, paddedTiler);

  auto outShape = intTupleZip(builder, builder.at(builder.getShape(logicalProd), 0),
                              builder.at(builder.getShape(logicalProd), 1));
  auto outStride = intTupleZip(builder, builder.at(builder.getStride(logicalProd), 0),
                               builder.at(builder.getStride(logicalProd), 1));
  return builder.makeLayout(outShape, outStride);
}

template <class Layout>
Layout layoutRakedProduct(LayoutBuilder<Layout> &builder, Layout blockLayout, Layout tilerLayout) {
  auto blockShape = builder.getShape(blockLayout);
  auto tilerShape = builder.getShape(tilerLayout);
  int32_t blockRank = blockShape.isLeaf() ? 1 : blockShape.rank();
  int32_t tilerRank = tilerShape.isLeaf() ? 1 : tilerShape.rank();
  int32_t targetRank = std::max(blockRank, tilerRank);

  Layout paddedBlock = layoutAppendToRank(builder, blockLayout, targetRank);
  Layout paddedTiler = layoutAppendToRank(builder, tilerLayout, targetRank);
  Layout logicalProd = layoutLogicalProduct(builder, paddedBlock, paddedTiler);

  auto outShape = intTupleZip(builder, builder.at(builder.getShape(logicalProd), 1),
                              builder.at(builder.getShape(logicalProd), 0));
  auto outStride = intTupleZip(builder, builder.at(builder.getStride(logicalProd), 1),
                               builder.at(builder.getStride(logicalProd), 0));
  return builder.makeLayout(outShape, outStride);
}

} // namespace mlir::fly

#endif // FLYDSL_DIALECT_UTILS_LAYOUTATTR_H
