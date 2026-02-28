#ifndef FLYDSL_DIALECT_UTILS_TILEDOPUTILS_H
#define FLYDSL_DIALECT_UTILS_TILEDOPUTILS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/LayoutUtils.h"

namespace mlir::fly {

namespace detail {

template <class Layout>
Layout layoutTiledCopyThrValView(LayoutBuilder<Layout> &builder, CopyAtomType copyAtom,
                                 Layout trgLayout, LayoutAttr tiledLayoutThrVal,
                                 LayoutAttr ref2trg) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;

  auto *ctx = tiledLayoutThrVal.getContext();
  LayoutBuilder<LayoutAttr> attrBuilder(ctx);

  auto atomLayoutRef = cast<LayoutAttr>(copyAtom.getThrValLayoutRef());
  auto atomNumThr = intTupleProductImpl(attrBuilder, atomLayoutRef.getShape().at(0));
  auto atomNumVal = intTupleProductImpl(attrBuilder, atomLayoutRef.getShape().at(1));

  TileAttr atomTile = TileAttr::get(
      ArrayAttr::get(ctx, {IntAttr::getStatic(ctx, attrBuilder.getStaticValue(atomNumThr)),
                           IntAttr::getStatic(ctx, attrBuilder.getStaticValue(atomNumVal))}));
  LayoutAttr atomLayoutTV = layoutZippedDivide(attrBuilder, tiledLayoutThrVal, atomTile);

  LayoutAttr firstMode = atomLayoutTV.at(0);
  LayoutAttr composedFirstMode = layoutComposition(attrBuilder, firstMode, ref2trg);
  LayoutAttr trgLayoutTV = LayoutAttr::get(
      IntTupleAttr::get(
          ArrayAttr::get(ctx, {composedFirstMode.getShape(), atomLayoutTV.getShape().at(1)})),
      IntTupleAttr::get(
          ArrayAttr::get(ctx, {composedFirstMode.getStride(), atomLayoutTV.getStride().at(1)})));

  IntTupleAttr zippedShape = intTupleZip(attrBuilder, trgLayoutTV.getShape());
  IntTupleAttr zippedStride = intTupleZip(attrBuilder, trgLayoutTV.getStride());
  auto one = IntTupleAttr::getLeafStatic(ctx, 1);
  IntTupleAttr profile = IntTupleAttr::get(
      ArrayAttr::get(ctx, {one, IntTupleAttr::get(ArrayAttr::get(ctx, {one, one}))}));
  LayoutAttr thrval2mn =
      layoutCoalesce(attrBuilder, LayoutAttr::get(zippedShape, zippedStride), profile);

  IntTuple firstModeShape = builder.at(builder.getShape(trgLayout), 0);
  IntTuple firstModeStride = builder.at(builder.getStride(trgLayout), 0);
  Layout firstModeLayout = builder.makeLayout(firstModeShape, firstModeStride);
  Layout thrval2mnLayout = builder.materializeConstantLayout(thrval2mn);
  Layout composedTV = layoutComposition(builder, firstModeLayout, thrval2mnLayout);

  typename LayoutBuilder<Layout>::ElemCollector retShapeElems;
  typename LayoutBuilder<Layout>::ElemCollector retStrideElems;
  retShapeElems.push_back(builder.getShape(composedTV));
  retStrideElems.push_back(builder.getStride(composedTV));
  retShapeElems.push_back(builder.at(builder.getShape(trgLayout), 1));
  retStrideElems.push_back(builder.at(builder.getStride(trgLayout), 1));
  Layout resultLayout =
      builder.makeLayout(builder.makeTuple(retShapeElems), builder.makeTuple(retStrideElems));

  IntTuple retShape = intTupleExpand(builder, builder.getShape(resultLayout), {0});
  IntTuple retStride = intTupleExpand(builder, builder.getStride(resultLayout), {0});
  return builder.makeLayout(retShape, retStride);
}

template <class Layout>
Layout layoutTiledMmaThrValView(LayoutBuilder<Layout> &builder, MmaAtomTypeInterface mmaAtom,
                                LayoutAttr tiledShape2D, IntTupleAttr atomShape2D,
                                LayoutAttr atomLayoutThrVal, TileAttr permutation2D,
                                Layout trgLayout) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;

  auto *ctx = tiledShape2D.getContext();
  LayoutBuilder<LayoutAttr> attrBuilder(ctx);

  Layout permuted = layoutLogicalDivide(builder, trgLayout, permutation2D);

  SmallVector<Attribute> atomTileElems;
  for (int i = 0; i < atomShape2D.rank(); ++i) {
    auto size = intTupleProductImpl(attrBuilder, atomShape2D.at(i));
    atomTileElems.push_back(IntAttr::getStatic(ctx, attrBuilder.getStaticValue(size)));
  }
  TileAttr atomTile = TileAttr::get(ArrayAttr::get(ctx, atomTileElems));
  Layout atomDiv = layoutZippedDivide(builder, permuted, atomTile);

  IntTuple firstModeShape = builder.at(builder.getShape(atomDiv), 0);
  IntTuple firstModeStride = builder.at(builder.getStride(atomDiv), 0);
  Layout firstModeLayout = builder.makeLayout(firstModeShape, firstModeStride);
  Layout atomTVLayout = builder.materializeConstantLayout(atomLayoutThrVal);
  Layout composedTV = layoutComposition(builder, firstModeLayout, atomTVLayout);

  IntTuple restShape = builder.at(builder.getShape(atomDiv), 1);
  IntTuple restStride = builder.at(builder.getStride(atomDiv), 1);
  Layout restLayout = builder.makeLayout(restShape, restStride);

  SmallVector<Attribute> thrTileElems;
  for (int i = 0; i < tiledShape2D.getShape().rank(); ++i) {
    auto size = intTupleProductImpl(attrBuilder, tiledShape2D.getShape().at(i));
    thrTileElems.push_back(IntAttr::getStatic(ctx, attrBuilder.getStaticValue(size)));
  }
  TileAttr thrTile = TileAttr::get(ArrayAttr::get(ctx, thrTileElems));
  Layout thrDiv = layoutZippedDivide(builder, restLayout, thrTile);

  IntTuple thrVShape = builder.at(builder.getShape(composedTV), 0);
  IntTuple thrVStride = builder.at(builder.getStride(composedTV), 0);
  IntTuple frgVShape = builder.at(builder.getShape(composedTV), 1);
  IntTuple frgVStride = builder.at(builder.getStride(composedTV), 1);

  IntTuple thrDimsShape = builder.at(builder.getShape(thrDiv), 0);
  IntTuple thrDimsStride = builder.at(builder.getStride(thrDiv), 0);
  IntTuple restDimsShape = builder.at(builder.getShape(thrDiv), 1);
  IntTuple restDimsStride = builder.at(builder.getStride(thrDiv), 1);

  typename LayoutBuilder<Layout>::ElemCollector thr0Shape, thr0Stride;
  thr0Shape.push_back(thrVShape);
  thr0Shape.push_back(thrDimsShape);
  thr0Stride.push_back(thrVStride);
  thr0Stride.push_back(thrDimsStride);

  typename LayoutBuilder<Layout>::ElemCollector val0Shape, val0Stride;
  val0Shape.push_back(frgVShape);
  val0Shape.push_back(restDimsShape);
  val0Stride.push_back(frgVStride);
  val0Stride.push_back(restDimsStride);

  typename LayoutBuilder<Layout>::ElemCollector retShapeElems, retStrideElems;
  retShapeElems.push_back(builder.makeTuple(thr0Shape));
  retShapeElems.push_back(builder.makeTuple(val0Shape));
  retStrideElems.push_back(builder.makeTuple(thr0Stride));
  retStrideElems.push_back(builder.makeTuple(val0Stride));

  return builder.makeLayout(builder.makeTuple(retShapeElems), builder.makeTuple(retStrideElems));
}

} // namespace detail

template <class Layout>
Layout layoutTiledCopyThrValViewSrc(LayoutBuilder<Layout> &builder, CopyAtomType copyAtom,
                                    LayoutAttr tiledLayoutThrVal, TileAttr tileMN,
                                    Layout srcLayout) {
  auto *ctx = tiledLayoutThrVal.getContext();
  LayoutBuilder<LayoutAttr> attrBuilder(ctx);

  auto atomLayoutRef = cast<LayoutAttr>(copyAtom.getThrValLayoutRef());
  auto atomLayoutSrc = cast<LayoutAttr>(copyAtom.getThrValLayoutSrc());
  LayoutAttr refInv = layoutRightInverse(attrBuilder, atomLayoutRef);
  LayoutAttr ref2src = layoutComposition(attrBuilder, refInv, atomLayoutSrc);

  Layout zippedDiv = layoutZippedDivide(builder, srcLayout, tileMN);
  return detail::layoutTiledCopyThrValView(builder, copyAtom, zippedDiv, tiledLayoutThrVal,
                                           ref2src);
}

template <class Layout>
Layout layoutTiledCopyThrValViewDst(LayoutBuilder<Layout> &builder, CopyAtomType copyAtom,
                                    LayoutAttr tiledLayoutThrVal, TileAttr tileMN,
                                    Layout dstLayout) {
  auto *ctx = tiledLayoutThrVal.getContext();
  LayoutBuilder<LayoutAttr> attrBuilder(ctx);

  auto atomLayoutRef = cast<LayoutAttr>(copyAtom.getThrValLayoutRef());
  auto atomLayoutDst = cast<LayoutAttr>(copyAtom.getThrValLayoutDst());
  LayoutAttr refInv = layoutRightInverse(attrBuilder, atomLayoutRef);
  LayoutAttr ref2dst = layoutComposition(attrBuilder, refInv, atomLayoutDst);

  Layout zippedDiv = layoutZippedDivide(builder, dstLayout, tileMN);
  return detail::layoutTiledCopyThrValView(builder, copyAtom, zippedDiv, tiledLayoutThrVal,
                                           ref2dst);
}

template <class Layout>
Layout layoutTiledCopyRetile(LayoutBuilder<Layout> &builder, CopyAtomType copyAtom,
                             LayoutAttr tiledLayoutThrVal, TileAttr tilerMN, Layout inputLayout) {
  using IntTuple = typename LayoutBuilder<Layout>::IntTuple;

  auto *ctx = tiledLayoutThrVal.getContext();
  LayoutBuilder<LayoutAttr> attrBuilder(ctx);

  auto atomLayoutRef = cast<LayoutAttr>(copyAtom.getThrValLayoutRef());
  auto atomNumVal = intTupleProductImpl(attrBuilder, atomLayoutRef.getShape().at(1));
  auto tiledNumThr = intTupleProductImpl(attrBuilder, tiledLayoutThrVal.getShape().at(0));

  IntTuple inputShape = builder.getShape(inputLayout);
  IntTuple V = builder.at(inputShape, 0);

  auto vVal = intTupleProductImpl(builder, V);
  int32_t tiledNumThrStatic = attrBuilder.getStaticValue(tiledNumThr);
  int32_t atomNumValStatic = attrBuilder.getStaticValue(atomNumVal);
  int32_t vStatic = builder.getStaticValue(vVal);

  int32_t upcastFactor = tiledNumThrStatic * vStatic;

  LayoutAttr tiledLayoutTVInv = layoutRightInverse(attrBuilder, tiledLayoutThrVal);

  SmallVector<Attribute> tilerShapeElems;
  for (int i = 0; i < tilerMN.rank(); ++i) {
    auto elem = tilerMN.at(i);
    if (auto layoutElem = dyn_cast<LayoutAttr>(elem)) {
      auto sz = intTupleProductImpl(attrBuilder, layoutElem.getShape());
      tilerShapeElems.push_back(IntTupleAttr::getLeafStatic(ctx, attrBuilder.getStaticValue(sz)));
    } else if (auto intElem = dyn_cast<IntAttr>(elem)) {
      tilerShapeElems.push_back(IntTupleAttr::getLeafStatic(ctx, intElem.getValue()));
    }
  }
  IntTupleAttr tilerShape = IntTupleAttr::get(ArrayAttr::get(ctx, tilerShapeElems));
  IntTupleAttr tilerCompactStride = intTupleCompactColMajor(attrBuilder, tilerShape);
  LayoutAttr tilerShapeLayout = LayoutAttr::get(tilerShape, tilerCompactStride);

  LayoutAttr tiledLayoutTVInvWithShape =
      layoutComposition(attrBuilder, tiledLayoutTVInv, tilerShapeLayout);
  LayoutAttr frgLayoutMN = layoutUpcast(attrBuilder, tiledLayoutTVInvWithShape, upcastFactor);

  LayoutAttr vLayout = LayoutAttr::get(IntTupleAttr::getLeafStatic(ctx, vStatic),
                                       IntTupleAttr::getLeafStatic(ctx, 1));

  LayoutAttr frgLayoutMNInv = layoutRightInverse(attrBuilder, frgLayoutMN);
  LayoutAttr vProduct = layoutLogicalProduct(attrBuilder, vLayout, frgLayoutMNInv);

  LayoutAttr atomNumValLayout = LayoutAttr::get(IntTupleAttr::getLeafStatic(ctx, atomNumValStatic),
                                                IntTupleAttr::getLeafStatic(ctx, 1));
  LayoutAttr frgLayoutV = layoutZippedDivide(attrBuilder, vProduct, atomNumValLayout);

  IntTupleAttr frgMNShapeProductEach = intTupleProductEach(attrBuilder, frgLayoutMN.getShape());
  IntTupleAttr divisorShapeForTensor = intTuplePrepend(attrBuilder, frgMNShapeProductEach,
                                                       IntTupleAttr::getLeafStatic(ctx, vStatic));
  SmallVector<Attribute> divisorTileElems;
  if (divisorShapeForTensor.isLeaf()) {
    divisorTileElems.push_back(IntAttr::getStatic(
        ctx, attrBuilder.getStaticValue(attrBuilder.getArithValue(divisorShapeForTensor))));
  } else {
    for (int i = 0; i < divisorShapeForTensor.rank(); ++i) {
      auto elemVal = intTupleProductImpl(attrBuilder, attrBuilder.at(divisorShapeForTensor, i));
      divisorTileElems.push_back(IntAttr::getStatic(ctx, attrBuilder.getStaticValue(elemVal)));
    }
  }
  TileAttr divisorTile = TileAttr::get(ArrayAttr::get(ctx, divisorTileElems));

  Layout tTensor = layoutZippedDivide(builder, inputLayout, divisorTile);

  Layout frgLayoutVMat = builder.materializeConstantLayout(frgLayoutV);
  IntTuple tTensorFirstModeShape = builder.at(builder.getShape(tTensor), 0);
  IntTuple tTensorFirstModeStride = builder.at(builder.getStride(tTensor), 0);
  Layout tTensorFirstMode = builder.makeLayout(tTensorFirstModeShape, tTensorFirstModeStride);
  Layout composedFirstMode = layoutComposition(builder, tTensorFirstMode, frgLayoutVMat);

  IntTuple restModeShape = builder.at(builder.getShape(tTensor), 1);
  IntTuple restModeStride = builder.at(builder.getStride(tTensor), 1);

  typename LayoutBuilder<Layout>::ElemCollector retShapeElems;
  typename LayoutBuilder<Layout>::ElemCollector retStrideElems;
  retShapeElems.push_back(builder.getShape(composedFirstMode));
  retStrideElems.push_back(builder.getStride(composedFirstMode));

  for (int i = 1; i < restModeShape.rank(); ++i) {
    retShapeElems.push_back(builder.at(restModeShape, i));
    retStrideElems.push_back(builder.at(restModeStride, i));
  }
  Layout resultLayout =
      builder.makeLayout(builder.makeTuple(retShapeElems), builder.makeTuple(retStrideElems));
  return resultLayout;
}

template <class Layout>
Layout layoutTiledMmaThrValOperandView(LayoutBuilder<Layout> &builder, MmaAtomTypeInterface mmaAtom,
                                       LayoutAttr atomLayoutMNK, TileAttr permutationMNK,
                                       MmaOperand operandId, Layout trgLayout) {
  auto *ctx = atomLayoutMNK.getContext();
  LayoutBuilder<LayoutAttr> attrBuilder(ctx);

  IntTupleAttr shapeMNK = cast<IntTupleAttr>(mmaAtom.getShapeMNK());

  int idx0, idx1;
  LayoutAttr atomLayoutThrVal;
  switch (operandId) {
  case MmaOperand::C:
    [[fallthrough]];
  case MmaOperand::D:
    idx0 = 0;
    idx1 = 1;
    atomLayoutThrVal = cast<LayoutAttr>(mmaAtom.getThrValLayoutC());
    break;
  case MmaOperand::A:
    idx0 = 0;
    idx1 = 2;
    atomLayoutThrVal = cast<LayoutAttr>(mmaAtom.getThrValLayoutA());
    break;
  case MmaOperand::B:
    idx0 = 1;
    idx1 = 2;
    atomLayoutThrVal = cast<LayoutAttr>(mmaAtom.getThrValLayoutB());
    break;
  }

  IntTupleAttr atomShape2D =
      IntTupleAttr::get(ArrayAttr::get(ctx, {shapeMNK.at(idx0), shapeMNK.at(idx1)}));

  LayoutAttr tiledShape2D =
      LayoutAttr::get(IntTupleAttr::get(ArrayAttr::get(ctx, {atomLayoutMNK.getShape().at(idx0),
                                                             atomLayoutMNK.getShape().at(idx1)})),
                      IntTupleAttr::get(ArrayAttr::get(ctx, {atomLayoutMNK.getStride().at(idx0),
                                                             atomLayoutMNK.getStride().at(idx1)})));

  SmallVector<Attribute> permElems;
  for (int i : {idx0, idx1}) {
    if (i >= permutationMNK.rank() || permutationMNK.isNoneMode(i)) {
      auto atomShapeI =
          attrBuilder.getStaticValue(intTupleProductImpl(attrBuilder, shapeMNK.at(i)));
      auto thrSizeI = attrBuilder.getStaticValue(
          intTupleProductImpl(attrBuilder, atomLayoutMNK.getShape().at(i)));
      permElems.push_back(IntAttr::getStatic(ctx, atomShapeI * thrSizeI));
    } else {
      permElems.push_back(permutationMNK.at(i));
    }
  }
  TileAttr permutation2D = TileAttr::get(ArrayAttr::get(ctx, permElems));

  return detail::layoutTiledMmaThrValView(builder, mmaAtom, tiledShape2D, atomShape2D,
                                          atomLayoutThrVal, permutation2D, trgLayout);
}

} // namespace mlir::fly

#endif // FLYDSL_DIALECT_UTILS_TILEDOPUTILS_H
