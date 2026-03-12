#ifndef FLYDSL_DIALECT_FLY_UTILS_LAYOUTUTILS_H
#define FLYDSL_DIALECT_FLY_UTILS_LAYOUTUTILS_H

#include <algorithm>
#include <numeric>
#include <optional>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/IntTupleUtils.h"

namespace mlir::fly {

namespace detail {

template <class IntTuple>
std::pair<IntTuple, IntTuple> canonicalizeStridePair(const IntTupleBuilder<IntTuple> &builder,
                                                     IntTuple shape, IntTuple stride) {
  if (shape.isLeaf()) {
    if (shape.isLeafStaticValue(1)) {
      return {shape, builder.materializeConstantLeaf(0)};
    }
    return {shape, stride};
  }
  if (shape.rank() == 1) {
    return canonicalizeStridePair(builder, builder.at(shape, 0), builder.at(stride, 0));
  }
  typename IntTupleBuilder<IntTuple>::ElemCollector shapeElems;
  typename IntTupleBuilder<IntTuple>::ElemCollector strideElems;
  for (int i = 0; i < shape.rank(); ++i) {
    auto [cs, cd] = canonicalizeStridePair(builder, builder.at(shape, i), builder.at(stride, i));
    shapeElems.push_back(cs);
    strideElems.push_back(cd);
  }
  return {builder.makeTuple(shapeElems), builder.makeTuple(strideElems)};
}

template <class IntTuple>
IntTuple layoutCrd2IdxTTT(IntTupleBuilder<IntTuple> &builder, IntTuple coord, IntTuple shape,
                          IntTuple stride);

template <class IntTuple>
IntTuple layoutCrd2IdxITT(IntTupleBuilder<IntTuple> &builder, IntTuple coord, IntTuple shape,
                          IntTuple stride) {
  int32_t rank = shape.rank();
  if (rank == 1) {
    return layoutCrd2IdxTTT(builder, coord, builder.at(shape, 0), builder.at(stride, 0));
  }
  IntTuple si = builder.at(shape, 0);
  IntTuple di = builder.at(stride, 0);

  IntTuple siProduct = intTupleProduct(builder, si);
  IntTuple ci = builder.mod(coord, siProduct);
  IntTuple remaining = builder.div(coord, siProduct);

  IntTuple result;
  if (si.isLeaf()) {
    result = builder.mul(ci, di);
  } else {
    result = layoutCrd2IdxITT(builder, ci, si, di);
  }

  for (int i = 1; i < rank; ++i) {
    si = builder.at(shape, i);
    di = builder.at(stride, i);

    if (i == rank - 1) {
      ci = remaining;
    } else {
      siProduct = intTupleProduct(builder, si);
      ci = builder.mod(remaining, siProduct);
      remaining = builder.div(remaining, siProduct);
    }
    if (si.isLeaf()) {
      result = builder.add(result, builder.mul(ci, di));
    } else {
      result = builder.add(result, layoutCrd2IdxITT(builder, ci, si, di));
    }
  }
  return result;
}

template <class IntTuple>
IntTuple layoutCrd2IdxTTT(IntTupleBuilder<IntTuple> &builder, IntTuple coord, IntTuple shape,
                          IntTuple stride) {
  if (coord.isLeaf()) {
    if (shape.isLeaf()) {
      return builder.mul(coord, stride);
    } else {
      return layoutCrd2IdxITT(builder, coord, shape, stride);
    }
  } else {
    assert(coord.rank() == shape.rank() && "Mismatched ranks");
    IntTuple result = layoutCrd2IdxTTT(builder, builder.at(coord, 0), builder.at(shape, 0),
                                       builder.at(stride, 0));
    for (int i = 1; i < coord.rank(); ++i) {
      result = builder.add(result, layoutCrd2IdxTTT(builder, builder.at(coord, i),
                                                    builder.at(shape, i), builder.at(stride, i)));
    }
    return result;
  }
}

template <class IntTuple>
IntTuple layoutIdx2CrdTTT(IntTupleBuilder<IntTuple> &builder, IntTuple index, IntTuple shape,
                          IntTuple stride);

template <class IntTuple>
IntTuple layoutIdx2CrdITT(IntTupleBuilder<IntTuple> &builder, IntTuple index, IntTuple shape,
                          IntTuple stride) {
  typename IntTupleBuilder<IntTuple>::ElemCollector collector;
  for (int i = 0; i < shape.rank(); ++i) {
    collector.push_back(
        layoutIdx2CrdTTT(builder, index, builder.at(shape, i), builder.at(stride, i)));
  }
  return builder.makeTuple(collector);
}

template <class IntTuple>
IntTuple layoutIdx2CrdTTT(IntTupleBuilder<IntTuple> &builder, IntTuple index, IntTuple shape,
                          IntTuple stride) {
  if (index.isLeaf()) {
    if (shape.isLeaf()) {
      if (shape.isLeafStaticValue(1)) {
        return builder.materializeConstantLeaf(0);
      }
      return builder.mod(builder.div(index, stride), shape);
    } else {
      return layoutIdx2CrdITT(builder, index, shape, stride);
    }
  } else {
    assert(index.rank() == shape.rank() && "Mismatched ranks");
    typename IntTupleBuilder<IntTuple>::ElemCollector collector;
    for (int i = 0; i < index.rank(); ++i) {
      collector.push_back(layoutIdx2CrdTTT(builder, builder.at(index, i), builder.at(shape, i),
                                           builder.at(stride, i)));
    }
    return builder.makeTuple(collector);
  }
}

template <class IntTuple>
IntTuple layoutCrd2IdxColMajor(IntTupleBuilder<IntTuple> &builder, IntTuple coord, IntTuple shape) {
  if (coord.isLeaf()) {
    return coord;
  }
  assert(coord.rank() == shape.rank() && "Mismatched ranks");
  IntTuple result = layoutCrd2IdxColMajor(builder, builder.at(coord, coord.rank() - 1),
                                          builder.at(shape, shape.rank() - 1));
  for (int i = coord.rank() - 2; i >= 0; --i) {
    IntTuple si = intTupleProduct(builder, builder.at(shape, i));
    result = builder.add(layoutCrd2IdxColMajor(builder, builder.at(coord, i), builder.at(shape, i)),
                         builder.mul(si, result));
  }
  return result;
}

template <class IntTuple>
IntTuple layoutIdx2CrdColMajor(IntTupleBuilder<IntTuple> &builder, IntTuple index, IntTuple shape) {
  if (shape.isLeaf()) {
    return index;
  }
  typename IntTupleBuilder<IntTuple>::ElemCollector collector;
  IntTuple remaining = index;
  for (int i = 0; i < shape.rank(); ++i) {
    IntTuple si = builder.at(shape, i);
    IntTuple siProduct = intTupleProduct(builder, si);
    if (i == shape.rank() - 1) {
      collector.push_back(layoutIdx2CrdColMajor(builder, remaining, si));
    } else {
      IntTuple ci = builder.mod(remaining, siProduct);
      remaining = builder.div(remaining, siProduct);
      collector.push_back(layoutIdx2CrdColMajor(builder, ci, si));
    }
  }
  return builder.makeTuple(collector);
}

template <class IntTuple>
IntTuple layoutCrd2CrdImpl(IntTupleBuilder<IntTuple> &builder, IntTuple coord, IntTuple srcShape,
                           IntTuple dstShape) {
  if (!coord.isLeaf() && !srcShape.isLeaf() && !dstShape.isLeaf()) {
    assert(coord.rank() == srcShape.rank() && "Mismatched ranks");
    assert(coord.rank() == dstShape.rank() && "Mismatched ranks");
    typename IntTupleBuilder<IntTuple>::ElemCollector collector;
    for (int i = 0; i < coord.rank(); ++i) {
      collector.push_back(layoutCrd2CrdImpl(builder, builder.at(coord, i), builder.at(srcShape, i),
                                            builder.at(dstShape, i)));
    }
    return builder.makeTuple(collector);
  } else {
    auto idx = layoutCrd2IdxColMajor(builder, coord, srcShape);
    return layoutIdx2CrdColMajor(builder, idx, dstShape);
  }
}

} // namespace detail

template <class IntTuple>
IntTuple layoutCrd2Idx(IntTupleBuilder<IntTuple> &builder, IntTuple coord, IntTuple shape,
                       IntTuple stride) {
  return detail::layoutCrd2IdxTTT(builder, coord, shape, stride);
}

template <class IntTuple>
IntTuple layoutIdx2Crd(IntTupleBuilder<IntTuple> &builder, IntTuple index, IntTuple shape,
                       IntTuple stride) {
  return detail::layoutIdx2CrdTTT(builder, index, shape, stride);
}

template <class IntTuple>
IntTuple layoutCrd2Crd(IntTupleBuilder<IntTuple> &builder, IntTuple coord, IntTuple srcShape,
                       IntTuple dstShape) {
  return detail::layoutCrd2CrdImpl(builder, coord, srcShape, dstShape);
}

template <class Layout> class LayoutBuilder;

// NarrowLayout = Attribute for type inference (LayoutAttr | ComposedLayoutAttr)
//              = LayoutValueAdaptor for lowering phase
template <class Layout, class NarrowLayout>
auto layoutCrd2Idx(LayoutBuilder<Layout> &builder, typename LayoutBuilder<Layout>::IntTuple coord,
                   NarrowLayout layout) {
  if (!builder.isComposedLayout(layout)) {
    return layoutCrd2Idx(builder, coord, builder.getShape(layout), builder.getStride(layout));
  } else {
    auto outerResult = layoutCrd2Idx(builder, coord, builder.getOuter(layout));
    auto intermediate = builder.add(builder.getOffset(layout), outerResult);

    auto inner = builder.getInner(layout);
    if (builder.isSwizzle(inner)) {
      return builder.applySwizzle(intermediate, builder.getSwizzleAttr(inner));
    } else {
      return layoutCrd2Idx(builder, intermediate, inner);
    }
  }
}

class LayoutValueAdaptor {
private:
  Value value;
  Attribute attr;

public:
  LayoutValueAdaptor(Value value, Attribute attr) : value(value), attr(attr) {}

  Value getValue() const { return value; }

  friend class LayoutBuilder<LayoutValueAdaptor>;
};

template <> class LayoutBuilder<LayoutAttr> : public IntTupleBuilder<IntTupleAttr> {
public:
  using IntTupleBuilder<IntTupleAttr>::IntTupleBuilder;
  using IntTuple = IntTupleAttr;

  bool isComposedLayout(Attribute attr) const { return isa<ComposedLayoutAttr>(attr); }
  bool isSwizzle(Attribute attr) const { return isa<SwizzleAttr>(attr); }
  SwizzleAttr getSwizzleAttr(Attribute attr) const { return cast<SwizzleAttr>(attr); }

  LayoutAttr getOuter(Attribute attr) const { return cast<ComposedLayoutAttr>(attr).getOuter(); }
  IntTuple getOffset(Attribute attr) const { return cast<ComposedLayoutAttr>(attr).getOffset(); }
  Attribute getInner(Attribute attr) const { return cast<ComposedLayoutAttr>(attr).getInner(); }

  LayoutAttr getLayoutAttr(LayoutAttr attr) const { return attr; }
  IntTuple getShape(LayoutAttr attr) const { return attr.getShape(); }
  IntTuple getShape(Attribute attr) const { return cast<LayoutAttr>(attr).getShape(); }
  IntTuple getStride(LayoutAttr attr) const { return attr.getStride(); }
  IntTuple getStride(Attribute attr) const { return cast<LayoutAttr>(attr).getStride(); }

  LayoutAttr materializeConstantLayout(IntTupleAttr shape, IntTupleAttr stride) const {
    return LayoutAttr::get(materializeConstantTuple(shape), materializeConstantTuple(stride));
  }
  LayoutAttr materializeConstantLayout(LayoutAttr attr) const {
    assert(attr.isStatic() && "Layout must be static");
    return attr;
  }
  Attribute materializeSwizzle(SwizzleAttr swizzle) const { return swizzle; }

  LayoutAttr makeLayout(IntTupleAttr shape, IntTupleAttr stride) const {
    return LayoutAttr::get(shape, stride);
  }
  Attribute makeComposedLayout(Attribute inner, IntTupleAttr offset, LayoutAttr outer) const {
    return ComposedLayoutAttr::get(inner, offset, outer);
  }
};

template <> class LayoutBuilder<LayoutValueAdaptor> : public IntTupleBuilder<IntTupleValueAdaptor> {
public:
  using IntTupleBuilder<IntTupleValueAdaptor>::IntTupleBuilder;
  using IntTuple = IntTupleValueAdaptor;

  bool isComposedLayout(LayoutValueAdaptor adaptor) const {
    return isa<ComposedLayoutAttr>(adaptor.attr);
  }
  bool isSwizzle(LayoutValueAdaptor adaptor) const { return isa<SwizzleAttr>(adaptor.attr); }

  SwizzleAttr getSwizzleAttr(LayoutValueAdaptor adaptor) const {
    return cast<SwizzleAttr>(adaptor.attr);
  }

  ComposedLayoutAttr getComposedLayoutAttr(LayoutValueAdaptor adaptor) const {
    assert(isComposedLayout(adaptor) && "adaptor must be a layout");
    return cast<ComposedLayoutAttr>(adaptor.attr);
  }

  LayoutValueAdaptor getOuter(LayoutValueAdaptor adaptor) const {
    assert(isComposedLayout(adaptor) && "adaptor must be a layout");
    return LayoutValueAdaptor(adaptor.value.getDefiningOp()->getOperand(2),
                              getComposedLayoutAttr(adaptor).getOuter());
  }
  IntTuple getOffset(LayoutValueAdaptor adaptor) const {
    assert(isComposedLayout(adaptor) && "adaptor must be a layout");
    return IntTupleValueAdaptor::create(*this, adaptor.value.getDefiningOp()->getOperand(1),
                                        getComposedLayoutAttr(adaptor).getOffset());
  }
  LayoutValueAdaptor getInner(LayoutValueAdaptor adaptor) const {
    assert(isComposedLayout(adaptor) && "adaptor must be a layout");
    return LayoutValueAdaptor(adaptor.value.getDefiningOp()->getOperand(0),
                              getComposedLayoutAttr(adaptor).getInner());
  }

  LayoutAttr getLayoutAttr(LayoutValueAdaptor adaptor) const {
    assert(!isComposedLayout(adaptor) && "adaptor must be a layout");
    return cast<LayoutAttr>(adaptor.attr);
  }
  IntTuple getShape(LayoutValueAdaptor adaptor) const {
    return IntTupleValueAdaptor::create(*this, adaptor.value.getDefiningOp()->getOperand(0),
                                        getLayoutAttr(adaptor).getShape());
  }
  IntTuple getStride(LayoutValueAdaptor adaptor) const {
    return IntTupleValueAdaptor::create(*this, adaptor.value.getDefiningOp()->getOperand(1),
                                        getLayoutAttr(adaptor).getStride());
  }

  LayoutValueAdaptor materializeConstantLayout(IntTupleAttr shape, IntTupleAttr stride) const {
    return makeLayout(materializeConstantTuple(shape), materializeConstantTuple(stride));
  }
  LayoutValueAdaptor materializeConstantLayout(LayoutAttr attr) const {
    return materializeConstantLayout(attr.getShape(), attr.getStride());
  }
  LayoutValueAdaptor materializeSwizzle(SwizzleAttr swizzle) const {
    auto value = StaticOp::create(this->builder, this->loc, SwizzleType::get(swizzle)).getResult();
    return LayoutValueAdaptor(value, swizzle);
  }

  LayoutValueAdaptor makeLayout(IntTuple shape, IntTuple stride) const {
    auto value = MakeLayoutOp::create(this->builder, this->loc, this->finalize(shape),
                                      this->finalize(stride))
                     .getResult();
    return LayoutValueAdaptor(value, LayoutAttr::get(this->getAttr(shape), this->getAttr(stride)));
  }
  LayoutValueAdaptor makeComposedLayout(LayoutValueAdaptor inner, IntTupleValueAdaptor offset,
                                        LayoutValueAdaptor outer) const {
    auto value = MakeComposedLayoutOp::create(this->builder, this->loc, inner.getValue(),
                                              this->finalize(offset), outer.getValue())
                     .getResult();
    return LayoutValueAdaptor(
        value, ComposedLayoutAttr::get(inner.attr, offset.getAttr(), cast<LayoutAttr>(outer.attr)));
  }
};

//===----------------------------------------------------------------------===//
// MakeLayout operations
//===----------------------------------------------------------------------===//

namespace detail {

inline bool flatOrderLessThan(IntTupleAttr flatOrder, int32_t lhsIdx, int32_t rhsIdx) {
  IntAttr lhs = flatOrder.at(lhsIdx).getLeafAsInt();
  IntAttr rhs = flatOrder.at(rhsIdx).getLeafAsInt();
  bool lhsStatic = lhs.isStatic();
  bool rhsStatic = rhs.isStatic();
  if (lhsStatic && rhsStatic)
    return lhs.getValue() < rhs.getValue();
  if (lhsStatic && !rhsStatic)
    return true;
  if (!lhsStatic && rhsStatic)
    return false;
  return lhsIdx < rhsIdx;
}

template <class Layout>
typename LayoutBuilder<Layout>::IntTuple
compactOrderImpl(LayoutBuilder<Layout> &builder, typename LayoutBuilder<Layout>::IntTuple shape,
                 IntTupleAttr order,
                 SmallVectorImpl<typename LayoutBuilder<Layout>::IntTuple> &refShapeProducts,
                 IntTupleAttr flatOrder, int32_t &flatIdx) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;

  if (!order.isLeaf()) {
    assert(shape.rank() == order.rank() && "Need equal rank of shape and order");
    typename LayoutBuilder<Layout>::ElemCollector collector;
    for (int i = 0; i < order.rank(); ++i) {
      collector.push_back(compactOrderImpl<Layout>(builder, builder.at(shape, i), order.at(i),
                                                   refShapeProducts, flatOrder, flatIdx));
    }
    return builder.makeTuple(collector);
  }

  int32_t curIdx = flatIdx++;
  IntTuple strideStart = builder.materializeConstantLeaf(1);
  for (int i = 0; i < flatOrder.rank(); ++i) {
    if (flatOrderLessThan(flatOrder, i, curIdx)) {
      strideStart = builder.mul(strideStart, refShapeProducts[i]);
    }
  }

  return intTupleCompactColMajor(builder, shape, strideStart);
}

template <class Layout>
void buildRefShapeProducts(
    LayoutBuilder<Layout> &builder, typename LayoutBuilder<Layout>::IntTuple shape,
    IntTupleAttr order,
    SmallVectorImpl<typename LayoutBuilder<Layout>::IntTuple> &refShapeProducts) {
  if (order.isLeaf()) {
    refShapeProducts.push_back(intTupleProduct(builder, shape));
    return;
  }
  assert(shape.rank() == order.rank() && "Need equal rank of shape and order");
  for (int i = 0; i < order.rank(); ++i) {
    buildRefShapeProducts<Layout>(builder, builder.at(shape, i), order.at(i), refShapeProducts);
  }
}

} // namespace detail

template <class Layout>
Layout layoutMakeOrderedLayout(LayoutBuilder<Layout> &builder,
                               typename LayoutBuilder<Layout>::IntTuple shape, IntTupleAttr order) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;

  if (order.isLeaf()) {
    IntTuple compactStride = intTupleCompactColMajor(builder, shape);
    return builder.makeLayout(shape, compactStride);
  }

  IntTupleBuilder<IntTupleAttr> attrBuilder(order.getContext());
  IntTupleAttr flatOrder = intTupleFlatten(attrBuilder, order);

  if (flatOrder.isLeaf()) {
    IntTuple compactStride = intTupleCompactColMajor(builder, shape);
    return builder.makeLayout(shape, compactStride);
  }

  SmallVector<IntTuple> refShapeProducts;
  detail::buildRefShapeProducts<Layout>(builder, shape, order, refShapeProducts);
  assert(refShapeProducts.size() == (size_t)flatOrder.rank() &&
         "refShapeProducts and flatOrder must have the same rank");

  int32_t flatIdx = 0;
  IntTuple resultStride =
      detail::compactOrderImpl<Layout>(builder, shape, order, refShapeProducts, flatOrder, flatIdx);
  return builder.makeLayout(shape, resultStride);
}

template <class Layout>
Layout layoutMakeFragmentLayout(LayoutBuilder<Layout> &builder, Layout layout) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;

  auto shape = builder.getShape(layout);
  auto stride = builder.getStride(layout);
  LayoutAttr layoutAttr = builder.getLayoutAttr(layout);
  int32_t R = layoutAttr.getShape().isLeaf() ? 1 : layoutAttr.getShape().rank();

  if (R > 1) {
    IntTuple mode0Shape = builder.at(shape, 0);
    IntTuple filteredMode0Shape =
        intTupleFilterZero(builder, layoutAttr.getStride().at(0), mode0Shape);
    IntTuple compactMode0Stride = intTupleCompactColMajor(builder, filteredMode0Shape);
    Layout mode0Layout = builder.makeLayout(mode0Shape, compactMode0Stride);

    IntTuple restShape;
    IntTuple restStride;
    if (R == 2) {
      restShape = builder.at(shape, 1);
      restStride = builder.at(stride, 1);
    } else {
      typename LayoutBuilder<Layout>::ElemCollector restShapeElems;
      typename LayoutBuilder<Layout>::ElemCollector restStrideElems;
      for (int i = 1; i < R; ++i) {
        restShapeElems.push_back(builder.at(shape, i));
        restStrideElems.push_back(builder.at(stride, i));
      }
      restShape = builder.makeTuple(restShapeElems);
      restStride = builder.makeTuple(restStrideElems);
    }

    Layout restLayout = builder.makeLayout(restShape, restStride);
    IntTupleAttr restOrderAttr = builder.getLayoutAttr(restLayout).getStride();
    Layout orderedRest = layoutMakeOrderedLayout(builder, restShape, restOrderAttr);

    return layoutTiledProduct(builder, mode0Layout, orderedRest);
  }

  IntTuple compactStride = intTupleCompactColMajor(builder, shape);
  return builder.makeLayout(shape, compactStride);
}

template <class Layout> Layout layoutMakeLayoutLike(LayoutBuilder<Layout> &builder, Layout layout) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;

  auto shape = builder.getShape(layout);
  LayoutAttr layoutAttr = builder.getLayoutAttr(layout);
  IntTupleAttr strideAttr = layoutAttr.getStride();

  IntTuple filteredShape = intTupleFilterZero(builder, strideAttr, shape);
  Layout orderedLayout = layoutMakeOrderedLayout(builder, filteredShape, strideAttr);
  return builder.makeLayout(shape, builder.getStride(orderedLayout));
}

//===----------------------------------------------------------------------===//
// Layout operations
//===----------------------------------------------------------------------===//

template <class Layout>
typename LayoutBuilder<Layout>::IntTuple layoutCoprofile(LayoutBuilder<Layout> &builder,
                                                         Layout layout) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;
  IntTuple stride = builder.getStride(layout);
  IntTuple strideSum = intTupleSum(builder, stride);
  return intTupleTransformLeaf(
      builder, [&](IntTuple) { return builder.materializeConstantLeaf(0); }, strideSum);
}

template <class Layout>
typename LayoutBuilder<Layout>::IntTuple layoutCoshape(LayoutBuilder<Layout> &builder,
                                                       Layout layout) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;
  IntTuple shape = builder.getShape(layout);
  IntTuple stride = builder.getStride(layout);
  IntTuple one = builder.materializeConstantLeaf(1);

  IntTuple m1Shapes = intTupleTransformLeaf(
      builder, [&](IntTuple s) { return builder.sub(s, one); }, shape);
  IntTuple coCoord = intTupleInnerProduct(builder, m1Shapes, stride);
  return intTupleTransformLeaf(
      builder, [&](IntTuple c) { return builder.add(c, one); }, coCoord);
}

template <class Layout>
typename LayoutBuilder<Layout>::IntTuple layoutSize(LayoutBuilder<Layout> &builder, Layout layout) {
  return intTupleProduct(builder, builder.getShape(layout));
}

template <class Layout>
typename LayoutBuilder<Layout>::IntTuple layoutCosize(LayoutBuilder<Layout> &builder,
                                                      Layout layout) {
  return intTupleProduct(builder, layoutCoshape(builder, layout));
}

namespace detail {

template <class IntTuple>
std::pair<IntTuple, IntTuple> coalesceImpl(const IntTupleBuilder<IntTuple> &builder, IntTuple shape,
                                           IntTuple stride) {
  SmallVector<IntTuple> flatShapeLeaves;
  SmallVector<IntTuple> flatStrideLeaves;
  intTupleFlattenToVector(builder, shape, flatShapeLeaves);
  intTupleFlattenToVector(builder, stride, flatStrideLeaves);

  const int flatRank = flatShapeLeaves.size();
  IntTuple currShapeLeaf = flatShapeLeaves[flatRank - 1];
  IntTuple currStrideLeaf = flatStrideLeaves[flatRank - 1];

  if (flatRank == 1) {
    if (currShapeLeaf.isLeafStaticValue(1)) {
      return {builder.materializeConstantLeaf(1), builder.materializeConstantLeaf(0)};
    } else {
      return {shape, stride};
    }
  }

  typename IntTupleBuilder<IntTuple>::ElemCollector resultShape;
  typename IntTupleBuilder<IntTuple>::ElemCollector resultStride;
  for (int i = flatRank - 2; i >= 0; --i) {
    IntTuple nextShapeLeaf = flatShapeLeaves[i];
    IntTuple nextStrideLeaf = flatStrideLeaves[i];

    if (nextShapeLeaf.isLeafStaticValue(1)) {
      continue;
    }
    if (currShapeLeaf.isLeafStaticValue(1)) {
      currShapeLeaf = nextShapeLeaf;
      currStrideLeaf = nextStrideLeaf;
      continue;
    }

    bool merged = false;
    if (nextShapeLeaf.isStatic() && nextStrideLeaf.isStatic() && currShapeLeaf.isStatic() &&
        currStrideLeaf.isStatic()) {
      if (builder.getStaticValue(nextShapeLeaf) * builder.getStaticValue(nextStrideLeaf) ==
          builder.getStaticValue(currStrideLeaf)) {
        currShapeLeaf = builder.mul(flatShapeLeaves[i], currShapeLeaf);
        currStrideLeaf = flatStrideLeaves[i];
        merged = true;
      }
    }
    if (!merged) {
      resultShape.push_back(currShapeLeaf);
      resultStride.push_back(currStrideLeaf);
      currShapeLeaf = nextShapeLeaf;
      currStrideLeaf = nextStrideLeaf;
    }
  }

  if (resultShape.empty()) {
    if (currShapeLeaf.isLeafStaticValue(1)) {
      return {builder.materializeConstantLeaf(1), builder.materializeConstantLeaf(0)};
    }
    return {currShapeLeaf, currStrideLeaf};
  }
  resultShape.push_back(currShapeLeaf);
  resultStride.push_back(currStrideLeaf);
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

  if (rhsStride.isLeafStaticValue(0)) {
    return {rhsShape, rhsStride};
  }
  if (lhsShape.isLeaf()) {
    IntTuple newStride = builder.mul(lhsStride, rhsStride);
    return canonicalizeStridePair(builder, rhsShape, newStride);
  }

  IntTuple restShape = rhsShape;
  IntTuple restStride = rhsStride;

  typename IntTupleBuilder<IntTuple>::ElemCollector resultShape;
  typename IntTupleBuilder<IntTuple>::ElemCollector resultStride;
  int32_t resultCount = 0;
  IntTuple lastShapeElem = rhsShape;
  IntTuple lastStrideElem = rhsStride;

  int R = lhsShape.rank();
  for (int i = 0; i < R - 1; ++i) {
    IntTuple currShape = builder.at(lhsShape, i);
    IntTuple currStride = builder.at(lhsStride, i);

    if (currShape.isStatic() && restStride.isStatic()) {
      int64_t restStrideVal = builder.getStaticValue(restStride);
      int64_t currShapeVal = builder.getStaticValue(currShape);
      assert(restStrideVal % currShapeVal == 0 || restStrideVal < currShapeVal);
    }

    IntTuple nextShape = builder.ceilDiv(currShape, restStride);
    IntTuple nextStride = builder.ceilDiv(restStride, currShape);

    if (nextShape.isLeafStaticValue(1) || restShape.isLeafStaticValue(1)) {
      restStride = nextStride;
      continue;
    }

    IntTuple newShape = builder.min(nextShape, restShape);
    IntTuple newStride = builder.mul(restStride, currStride);

    if (newShape.isStatic() && restShape.isStatic()) {
      int64_t restShapeVal = builder.getStaticValue(restShape);
      int64_t newShapeVal = builder.getStaticValue(newShape);
      assert(restShapeVal % newShapeVal == 0);
    }

    lastShapeElem = newShape;
    lastStrideElem = newStride;
    resultShape.push_back(lastShapeElem);
    resultStride.push_back(lastStrideElem);
    restShape = builder.div(restShape, newShape);
    restStride = nextStride;

    ++resultCount;
  }

  IntTuple lhsLastStride = builder.at(lhsStride, R - 1);
  if (resultCount == 0) {
    IntTuple retStride = builder.mul(restStride, lhsLastStride);
    return canonicalizeStridePair(builder, restShape, retStride);
  }
  if (restShape.isLeafStaticValue(1)) {
    if (resultCount == 1) {
      return canonicalizeStridePair(builder, lastShapeElem, lastStrideElem);
    }
    return canonicalizeStridePair(builder, builder.makeTuple(resultShape),
                                  builder.makeTuple(resultStride));
  }

  resultShape.push_back(restShape);
  resultStride.push_back(builder.mul(restStride, lhsLastStride));
  return canonicalizeStridePair(builder, builder.makeTuple(resultShape),
                                builder.makeTuple(resultStride));
}

template <class IntTuple>
std::pair<IntTuple, IntTuple> complementImpl(const IntTupleBuilder<IntTuple> &builder,
                                             IntTuple filteredShape, IntTuple filteredStride,
                                             IntTuple codomainSize) {
  if (!codomainSize.isLeaf()) {
    assert(false && "this is for basis-strided layout, maybe support this later");
    return {filteredShape, filteredStride};
  }

  auto flatShape = intTupleFlatten(builder, filteredShape);
  auto flatStride = intTupleFlatten(builder, filteredStride);

  if (flatStride.isLeaf()) {
    if (flatStride.isLeafStaticValue(0)) {
      return {codomainSize, builder.materializeConstantLeaf(1)};
    }
  }

  const int R = flatStride.rank();
  assert((R == 1 || filteredStride.isStatic()) && "stride must be static for complement");

  struct ShapeStridePair {
    IntTuple shapeVal;
    IntTuple strideVal;
    int64_t strideStatic;
  };
  SmallVector<ShapeStridePair> modes;
  modes.reserve(R);

  if (!flatStride.isLeaf()) {
    for (int i = 0; i < R; ++i) {
      IntTuple s = builder.at(flatShape, i);
      IntTuple d = builder.at(flatStride, i);
      modes.push_back({s, d, builder.getStaticValue(d)});
    }
    std::sort(modes.begin(), modes.end(), [](const ShapeStridePair &a, const ShapeStridePair &b) {
      return a.strideStatic < b.strideStatic;
    });
  } else {
    modes.push_back({flatShape, flatStride, 0});
  }

  IntTuple lastStride = builder.materializeConstantLeaf(1);
  typename IntTupleBuilder<IntTuple>::ElemCollector resultShapeVals;
  typename IntTupleBuilder<IntTuple>::ElemCollector resultStrideVals;

  resultStrideVals.push_back(lastStride);
  for (int64_t i = 0; i < R - 1; ++i) {
    IntTuple minStride = modes[i].strideVal;
    IntTuple newShape = builder.div(minStride, lastStride);
    IntTuple newStride = builder.mul(minStride, modes[i].shapeVal);

    resultShapeVals.push_back(newShape);
    resultStrideVals.push_back(newStride);
    lastStride = newStride;
  }

  auto lastMode = modes.back();
  IntTuple newShape = builder.div(lastMode.strideVal, lastStride);
  resultShapeVals.push_back(newShape);

  IntTuple newStrideForRest = builder.mul(lastMode.strideVal, lastMode.shapeVal);
  IntTuple restShape = builder.ceilDiv(codomainSize, newStrideForRest);
  IntTuple restStride = newStrideForRest;

  resultShapeVals.push_back(restShape);
  resultStrideVals.push_back(restStride);

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
  auto [canonicalShape, canonicalStride] =
      detail::canonicalizeStridePair(builder, retShape, retStride);
  return builder.makeLayout(canonicalShape, canonicalStride);
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

      auto tileElem = innerTileAttr.at(i);
      if (auto nestedTile = dyn_cast<TileAttr>(tileElem)) {
        Layout subLayout = builder.makeLayout(coalShape, coalStride);
        Layout composed = layoutComposition(builder, subLayout, nestedTile);
        retShape.push_back(builder.getShape(composed));
        retStride.push_back(builder.getStride(composed));
      } else {
        auto makeRhsPair = [&]() -> std::pair<IntTuple, IntTuple> {
          if (auto attr = dyn_cast<LayoutAttr>(tileElem)) {
            return {builder.materializeConstantTuple(attr.getShape()),
                    builder.materializeConstantTuple(attr.getStride())};
          }
          return {builder.materializeConstantLeaf(cast<IntAttr>(tileElem)),
                  builder.materializeConstantLeaf(1)};
        };
        auto [rhsShape, rhsStride] = makeRhsPair();
        auto [elemShape, elemStride] =
            detail::compositionImpl(builder, coalShape, coalStride, rhsShape, rhsStride);
        retShape.push_back(elemShape);
        retStride.push_back(elemStride);
      }
    } else {
      retShape.push_back(builder.at(lhsShape, i));
      retStride.push_back(builder.at(lhsStride, i));
    }
  }
  auto [canonicalShape, canonicalStride] = detail::canonicalizeStridePair(
      builder, builder.makeTuple(retShape), builder.makeTuple(retStride));
  return builder.makeLayout(canonicalShape, canonicalStride);
}

template <class Layout>
Layout layoutComplement(
    LayoutBuilder<Layout> &builder, Layout layout,
    std::optional<typename LayoutBuilder<Layout>::IntTuple> codomainSize = std::nullopt) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;

  auto filteredShape = intTupleFilterZero(builder, builder.getLayoutAttr(layout).getStride(),
                                          builder.getShape(layout));
  auto filteredStride = builder.getStride(layout);

  auto [coalShape, coalStride] = detail::coalesceImpl(builder, filteredShape, filteredStride);

  IntTuple codomain = codomainSize
                          ? *codomainSize
                          : layoutCosize(builder, builder.makeLayout(coalShape, coalStride));
  auto [retShape, retStride] = detail::complementImpl(builder, coalShape, coalStride, codomain);
  return builder.makeLayout(retShape, retStride);
}

template <class Layout> Layout layoutRightInverse(LayoutBuilder<Layout> &builder, Layout layout) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;

  auto coalesced = layoutCoalesce(builder, layout);
  auto shape = builder.getShape(coalesced);
  auto stride = builder.getStride(coalesced);

  SmallVector<IntTuple> flatShapeLeaves;
  SmallVector<IntTuple> flatStrideLeaves;
  intTupleFlattenToVector(builder, shape, flatShapeLeaves);
  intTupleFlattenToVector(builder, stride, flatStrideLeaves);

  SmallVector<IntTuple> prefixProducts;
  prefixProducts.reserve(flatShapeLeaves.size() + 1);
  IntTuple one = builder.materializeConstantLeaf(1);
  prefixProducts.push_back(one);
  for (size_t i = 0; i < flatShapeLeaves.size(); ++i) {
    IntTuple next = builder.mul(prefixProducts.back(), flatShapeLeaves[i]);
    prefixProducts.push_back(next);
  }

  SmallVector<int32_t> sortedIdx;
  sortedIdx.reserve(flatStrideLeaves.size());
  for (size_t i = 0; i < flatStrideLeaves.size(); ++i) {
    if (!flatStrideLeaves[i].isStatic() || flatStrideLeaves[i].isLeafNone())
      continue;
    sortedIdx.push_back(static_cast<int32_t>(i));
  }
  std::sort(sortedIdx.begin(), sortedIdx.end(), [&](int32_t a, int32_t b) {
    return builder.getStaticValue(flatStrideLeaves[a]) <
           builder.getStaticValue(flatStrideLeaves[b]);
  });

  typename LayoutBuilder<Layout>::ElemCollector resultShape;
  typename LayoutBuilder<Layout>::ElemCollector resultStride;
  resultShape.push_back(one);
  resultStride.push_back(builder.materializeConstantLeaf(0));

  IntTuple currStride = one;
  for (int32_t idx : sortedIdx) {
    if (!flatShapeLeaves[idx].isStatic() || !flatStrideLeaves[idx].isStatic())
      continue;
    if (builder.getStaticValue(flatStrideLeaves[idx]) != builder.getStaticValue(currStride))
      continue;

    resultShape.push_back(flatShapeLeaves[idx]);
    resultStride.push_back(prefixProducts[idx]);
    currStride = builder.mul(flatShapeLeaves[idx], flatStrideLeaves[idx]);
  }

  Layout resultLayout =
      builder.makeLayout(builder.makeTuple(resultShape), builder.makeTuple(resultStride));
  return layoutCoalesce(builder, resultLayout);
}

template <class Layout> Layout layoutLeftInverse(LayoutBuilder<Layout> &builder, Layout layout) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;

  auto coalesced = layoutCoalesce(builder, layout);
  auto shape = builder.getShape(coalesced);
  auto stride = builder.getStride(coalesced);

  SmallVector<IntTuple> flatShapeLeaves;
  SmallVector<IntTuple> flatStrideLeaves;
  intTupleFlattenToVector(builder, shape, flatShapeLeaves);
  intTupleFlattenToVector(builder, stride, flatStrideLeaves);

  SmallVector<IntTuple> prefixProducts;
  prefixProducts.reserve(flatShapeLeaves.size() + 1);
  IntTuple one = builder.materializeConstantLeaf(1);
  prefixProducts.push_back(one);
  for (size_t i = 0; i < flatShapeLeaves.size(); ++i) {
    IntTuple next = builder.mul(prefixProducts.back(), flatShapeLeaves[i]);
    prefixProducts.push_back(next);
  }

  SmallVector<int32_t> sortedIdx;
  sortedIdx.reserve(flatStrideLeaves.size());
  for (size_t i = 0; i < flatStrideLeaves.size(); ++i) {
    if (!flatStrideLeaves[i].isStatic())
      continue;
    sortedIdx.push_back(static_cast<int32_t>(i));
  }
  std::sort(sortedIdx.begin(), sortedIdx.end(), [&](int32_t a, int32_t b) {
    return builder.getStaticValue(flatStrideLeaves[a]) <
           builder.getStaticValue(flatStrideLeaves[b]);
  });

  typename LayoutBuilder<Layout>::ElemCollector resultShape;
  typename LayoutBuilder<Layout>::ElemCollector resultStride;
  resultStride.push_back(builder.materializeConstantLeaf(0));

  IntTuple resultShapeProduct = one;
  int32_t lastSortedIdx = -1;
  for (int32_t idx : sortedIdx) {
    if (flatStrideLeaves[idx].isLeafStaticValue(0))
      continue;

    IntTuple gapShape = builder.div(flatStrideLeaves[idx], resultShapeProduct);
    resultShape.push_back(gapShape);
    resultStride.push_back(prefixProducts[idx]);
    resultShapeProduct = builder.mul(resultShapeProduct, gapShape);
    lastSortedIdx = idx;
  }

  if (lastSortedIdx >= 0) {
    resultShape.push_back(flatShapeLeaves[lastSortedIdx]);
  } else {
    resultShape.push_back(one);
  }
  Layout resultLayout =
      builder.makeLayout(builder.makeTuple(resultShape), builder.makeTuple(resultStride));
  return layoutCoalesce(builder, resultLayout);
}

namespace detail {

template <class Layout>
std::pair<typename LayoutBuilder<Layout>::IntTuple, typename LayoutBuilder<Layout>::IntTuple>
layoutUpcastImpl(LayoutBuilder<Layout> &builder, typename LayoutBuilder<Layout>::IntTuple shape,
                 typename LayoutBuilder<Layout>::IntTuple stride, int32_t factor) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;

  if (shape.isLeaf()) {
    if (stride.isLeafStaticValue(0)) {
      return {shape, stride};
    }

    IntTuple factorTuple = builder.materializeConstantLeaf(factor);
    if (!stride.isStatic()) {
      return {shape, builder.safeDiv(stride, factorTuple)};
    }

    int32_t staticStride = builder.getStaticValue(stride);
    int32_t absStride = std::abs(staticStride);
    assert((absStride % factor == 0 || factor % absStride == 0) &&
           "layoutUpcast: divisibility condition failed between factor and stride");
    int32_t sign = staticStride < 0 ? -1 : 1;

    IntTuple absStrideTuple = builder.materializeConstantLeaf(absStride);
    IntTuple strideShapeScale = builder.ceilDiv(factorTuple, absStrideTuple);
    IntTuple newShapeVal = builder.ceilDiv(shape, strideShapeScale);
    IntTuple newStrideAbs = builder.ceilDiv(absStrideTuple, factorTuple);
    IntTuple newStrideVal =
        sign > 0 ? newStrideAbs : builder.sub(builder.materializeConstantLeaf(0), newStrideAbs);
    return {newShapeVal, newStrideVal};
  }

  typename LayoutBuilder<Layout>::ElemCollector outShape;
  typename LayoutBuilder<Layout>::ElemCollector outStride;
  for (int i = 0; i < shape.rank(); ++i) {
    auto [childShape, childStride] =
        layoutUpcastImpl(builder, builder.at(shape, i), builder.at(stride, i), factor);
    outShape.push_back(childShape);
    outStride.push_back(childStride);
  }
  return {builder.makeTuple(outShape), builder.makeTuple(outStride)};
}

template <class Layout>
auto layoutUpcastImpl(LayoutBuilder<Layout> &builder,
                      typename LayoutBuilder<Layout>::IntTuple offset, int32_t factor) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;
  assert(offset.isLeaf());
  IntTuple factorTuple = builder.materializeConstantLeaf(factor);
  return builder.safeDiv(offset, factorTuple);
}

template <class Layout>
auto layoutUpcastImpl(LayoutBuilder<Layout> &builder, SwizzleAttr swizzle, int32_t factor) {
  assert(utils::isPowerOf2(factor) && "layoutUpcast: factor must be a power of 2");
  int32_t log_factor = std::log2(factor);
  int32_t base = swizzle.getBase();
  assert(base >= log_factor);
  return builder.materializeSwizzle(SwizzleAttr::get(swizzle.getContext(), base - log_factor,
                                                     swizzle.getMask(), swizzle.getShift()));
}

template <class Layout>
std::pair<typename LayoutBuilder<Layout>::IntTuple, typename LayoutBuilder<Layout>::IntTuple>
layoutDowncastImpl(LayoutBuilder<Layout> &builder, typename LayoutBuilder<Layout>::IntTuple shape,
                   typename LayoutBuilder<Layout>::IntTuple stride, int32_t factor) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;

  if (shape.isLeaf()) {
    IntTuple factorTuple = builder.materializeConstantLeaf(factor);
    if (stride.isLeafStaticValue(1) || stride.isLeafStaticValue(-1)) {
      return {builder.mul(shape, factorTuple), stride};
    }
    return {shape, builder.mul(stride, factorTuple)};
  }

  typename LayoutBuilder<Layout>::ElemCollector outShape;
  typename LayoutBuilder<Layout>::ElemCollector outStride;
  for (int i = 0; i < shape.rank(); ++i) {
    auto [childShape, childStride] =
        layoutDowncastImpl(builder, builder.at(shape, i), builder.at(stride, i), factor);
    outShape.push_back(childShape);
    outStride.push_back(childStride);
  }
  return {builder.makeTuple(outShape), builder.makeTuple(outStride)};
}

template <class Layout>
auto layoutDowncastImpl(LayoutBuilder<Layout> &builder,
                        typename LayoutBuilder<Layout>::IntTuple offset, int32_t factor) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;
  assert(offset.isLeaf());
  IntTuple factorTuple = builder.materializeConstantLeaf(factor);
  return builder.mul(offset, factorTuple);
}

template <class Layout>
auto layoutDowncastImpl(LayoutBuilder<Layout> &builder, SwizzleAttr swizzle, int32_t factor) {
  assert(utils::isPowerOf2(factor) && "layoutDowncast: factor must be a power of 2");
  int32_t log_factor = std::log2(factor);
  return builder.materializeSwizzle(SwizzleAttr::get(
      swizzle.getContext(), swizzle.getBase() + log_factor, swizzle.getMask(), swizzle.getShift()));
}

} // namespace detail

template <class Layout, class NarrowLayout>
auto layoutUpcast(LayoutBuilder<Layout> &builder, NarrowLayout layout, int32_t factor) {
  if constexpr (std::is_same_v<LayoutAttr, NarrowLayout>) {
    if (factor == 1) {
      return layout;
    }
    auto [newShape, newStride] = detail::layoutUpcastImpl(builder, builder.getShape(layout),
                                                          builder.getStride(layout), factor);
    return builder.makeLayout(newShape, newStride);
  } else {
    if (factor == 1) {
      return layout;
    }
    if (!builder.isComposedLayout(layout)) {
      auto [newShape, newStride] = detail::layoutUpcastImpl(builder, builder.getShape(layout),
                                                            builder.getStride(layout), factor);
      return static_cast<NarrowLayout>(builder.makeLayout(newShape, newStride));
    } else {
      auto outer = builder.getOuter(layout);
      auto inner = builder.getInner(layout);
      auto offset = builder.getOffset(layout);
      auto newOuter = layoutUpcast(builder, outer, factor);
      auto newOffset = detail::layoutUpcastImpl(builder, offset, factor);

      if (builder.isSwizzle(inner)) {
        inner = detail::layoutUpcastImpl(builder, builder.getSwizzleAttr(inner), factor);
      } else {
        inner = layoutUpcast(builder, inner, factor);
      }
      return builder.makeComposedLayout(inner, newOffset, newOuter);
    }
  }
}

template <class Layout, class NarrowLayout>
auto layoutDowncast(LayoutBuilder<Layout> &builder, NarrowLayout layout, int32_t factor) {
  if constexpr (std::is_same_v<LayoutAttr, NarrowLayout>) {
    if (factor == 1) {
      return layout;
    }
    auto [newShape, newStride] = detail::layoutDowncastImpl(builder, builder.getShape(layout),
                                                            builder.getStride(layout), factor);
    return builder.makeLayout(newShape, newStride);
  } else {
    if (factor == 1) {
      return layout;
    }
    if (!builder.isComposedLayout(layout)) {
      auto [newShape, newStride] = detail::layoutDowncastImpl(builder, builder.getShape(layout),
                                                              builder.getStride(layout), factor);
      return static_cast<NarrowLayout>(builder.makeLayout(newShape, newStride));
    } else {
      auto outer = builder.getOuter(layout);
      auto inner = builder.getInner(layout);
      auto offset = builder.getOffset(layout);
      auto newOuter = layoutDowncast(builder, outer, factor);
      auto newOffset = detail::layoutDowncastImpl(builder, offset, factor);

      if (builder.isSwizzle(inner)) {
        inner = detail::layoutDowncastImpl(builder, builder.getSwizzleAttr(inner), factor);
      } else {
        inner = layoutDowncast(builder, inner, factor);
      }
      return builder.makeComposedLayout(inner, newOffset, newOuter);
    }
  }
}

template <class Layout, class NarrowLayout>
auto layoutRecast(LayoutBuilder<Layout> &builder, NarrowLayout layout, int32_t oldTypeBits,
                  int32_t newTypeBits) {
  if (oldTypeBits <= 0 || newTypeBits <= 0) {
    return layout;
  }
  int32_t g = std::gcd(oldTypeBits, newTypeBits);
  int32_t num = newTypeBits / g;
  int32_t den = oldTypeBits / g;

  if (num == 1 && den == 1) {
    return layout;
  }
  if (num == 1) {
    return layoutDowncast(builder, layout, den);
  }
  if (den == 1) {
    return layoutUpcast(builder, layout, num);
  }
  return layoutDowncast(builder, layoutUpcast(builder, layout, num), den);
}

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
    if (auto nestedTile = dyn_cast<TileAttr>(divisor)) {
      return layoutLogicalDivide(builder, currentLayout, nestedTile);
    } else if (auto attr = dyn_cast<LayoutAttr>(divisor)) {
      return layoutLogicalDivide(builder, currentLayout, builder.materializeConstantLayout(attr));
    } else if (auto intDivisor = dyn_cast<IntAttr>(divisor)) {
      IntTuple divisorShape = builder.materializeConstantTuple(IntTupleAttr::get(intDivisor));
      IntTuple divisorStride = builder.materializeConstantLeaf(1);
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
    shapeElems.push_back(builder.materializeConstantLeaf(1));
    strideElems.push_back(builder.materializeConstantLeaf(0));
  }
  return builder.makeLayout(builder.makeTuple(shapeElems), builder.makeTuple(strideElems));
}

template <class Layout>
Layout layoutLogicalProduct(LayoutBuilder<Layout> &builder, Layout blockLayout,
                            Layout tilerLayout) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;

  IntTuple blockSize = layoutSize(builder, blockLayout);
  IntTuple tilerCosize = layoutCosize(builder, tilerLayout);

  IntTuple codomainSize = builder.mul(blockSize, tilerCosize);
  Layout complement = layoutComplement(builder, blockLayout, codomainSize);
  Layout composed = layoutComposition(builder, complement, tilerLayout);

  typename LayoutBuilder<Layout>::ElemCollector retShapeElems;
  typename LayoutBuilder<Layout>::ElemCollector retStrideElems;
  retShapeElems.push_back(builder.getShape(blockLayout));
  retShapeElems.push_back(builder.getShape(composed));
  retStrideElems.push_back(builder.getStride(blockLayout));
  retStrideElems.push_back(builder.getStride(composed));

  auto [canonicalShape, canonicalStride] = detail::canonicalizeStridePair(
      builder, builder.makeTuple(retShapeElems), builder.makeTuple(retStrideElems));
  return builder.makeLayout(canonicalShape, canonicalStride);
}

template <class Layout>
Layout layoutZippedProduct(LayoutBuilder<Layout> &builder, Layout blockLayout, Layout tilerLayout) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;

  Layout logicalProd = layoutLogicalProduct(builder, blockLayout, tilerLayout);

  auto *ctx = builder.getLayoutAttr(blockLayout).getContext();
  IntTupleAttr guide = IntTupleAttr::getLeafStatic(ctx, 1);
  IntTuple retShape = intTupleZip2By(builder, builder.getShape(logicalProd), guide);
  IntTuple retStride = intTupleZip2By(builder, builder.getStride(logicalProd), guide);
  return builder.makeLayout(retShape, retStride);
}

template <class Layout>
Layout layoutTiledProduct(LayoutBuilder<Layout> &builder, Layout blockLayout, Layout tilerLayout) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;
  Layout zipped = layoutZippedProduct(builder, blockLayout, tilerLayout);
  IntTuple retShape = intTupleExpand(builder, builder.getShape(zipped), {1});
  IntTuple retStride = intTupleExpand(builder, builder.getStride(zipped), {1});
  return builder.makeLayout(retShape, retStride);
}

template <class Layout>
Layout layoutFlatProduct(LayoutBuilder<Layout> &builder, Layout blockLayout, Layout tilerLayout) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;
  Layout zipped = layoutZippedProduct(builder, blockLayout, tilerLayout);
  IntTuple retShape = intTupleExpand(builder, builder.getShape(zipped), {0, 1});
  IntTuple retStride = intTupleExpand(builder, builder.getStride(zipped), {0, 1});
  return builder.makeLayout(retShape, retStride);
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

template <class Layout>
Layout layoutTileToShape(LayoutBuilder<Layout> &builder, Layout blockLayout,
                         typename LayoutBuilder<Layout>::IntTuple targetShape, IntTupleAttr order) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;

  IntTupleAttr targetShapeAttr = builder.getAttr(targetShape);
  int32_t targetRank = targetShapeAttr.isLeaf() ? 1 : targetShapeAttr.rank();

  Layout paddedBlock = layoutAppendToRank(builder, blockLayout, targetRank);

  IntTuple blockShape = intTupleProductEach(builder, builder.getShape(paddedBlock));
  IntTuple targetShapeFlat = intTupleProductEach(builder, targetShape);
  IntTuple productShape = intTupleCeilDiv(builder, targetShapeFlat, blockShape);

  Layout orderedTiler = layoutMakeOrderedLayout(builder, productShape, order);
  return layoutBlockedProduct(builder, paddedBlock, orderedTiler);
}

} // namespace mlir::fly

#endif // FLYDSL_DIALECT_FLY_UTILS_LAYOUTUTILS_H
