// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "flydsl/Dialect/Fly/Utils/IntUtils.h"

#include "TiledOpTraits.h"

namespace mlir::fly {

static LayoutAttr tiledCopyGetTiledThrValLayoutImpl(CopyAtomType copyAtom,
                                                    LayoutAttr tiledLayoutThrVal, TileAttr tileMN,
                                                    LayoutAttr atomLayoutTrg) {
  auto *ctx = tiledLayoutThrVal.getContext();
  LayoutBuilder<LayoutAttr> attrBuilder(ctx);

  auto atomLayoutRef = cast<LayoutAttr>(copyAtom.getThrValLayoutRef());
  LayoutAttr refInv = layoutRightInverse(attrBuilder, atomLayoutRef);
  LayoutAttr ref2trg = layoutComposition(attrBuilder, refInv, atomLayoutTrg);

  SmallVector<Attribute> tilerShapeElems;
  SmallVector<Attribute> tilerStrideElems;
  IntAttr runningStride = IntAttr::getStatic(ctx, 1);
  for (int i = 0; i < tileMN.rank(); ++i) {
    auto tileElem = tileMN.at(i);
    IntAttr tileSize;
    if (auto intVal = dyn_cast<IntAttr>(tileElem))
      tileSize = intVal;
    else if (auto layoutVal = dyn_cast<LayoutAttr>(tileElem))
      tileSize = intTupleProduct(attrBuilder, layoutVal.getShape()).getLeafAsInt();
    else
      llvm_unreachable("unsupported tile element type");
    tilerShapeElems.push_back(IntTupleAttr::get(tileSize));
    tilerStrideElems.push_back(IntTupleAttr::get(runningStride));
    runningStride = runningStride * tileSize;
  }
  IntTupleAttr tilerShape = IntTupleAttr::get(ArrayAttr::get(ctx, tilerShapeElems));
  IntTupleAttr tilerStride = IntTupleAttr::get(ArrayAttr::get(ctx, tilerStrideElems));
  LayoutAttr refLayout = LayoutAttr::get(
      IntTupleAttr::get(ArrayAttr::get(ctx, {tilerShape, IntTupleAttr::getLeafStatic(ctx, 1)})),
      IntTupleAttr::get(ArrayAttr::get(ctx, {tilerStride, IntTupleAttr::getLeafStatic(ctx, 0)})));

  LayoutAttr thrValView = detail::layoutTiledCopyThrValView(attrBuilder, copyAtom, refLayout,
                                                            tiledLayoutThrVal, ref2trg);

  SmallVector<Attribute> sliceElems;
  sliceElems.push_back(IntTupleAttr::getLeafNone(ctx));
  sliceElems.push_back(IntTupleAttr::getLeafNone(ctx));
  sliceElems.push_back(IntTupleAttr::getLeafStatic(ctx, 0));
  IntTupleAttr sliceCoord = IntTupleAttr::get(ArrayAttr::get(ctx, sliceElems));

  IntTupleAttr resultShape = intTupleSlice(attrBuilder, thrValView.getShape(), sliceCoord);
  IntTupleAttr resultStride = intTupleSlice(attrBuilder, thrValView.getStride(), sliceCoord);
  return LayoutAttr::get(resultShape, resultStride);
}

LayoutAttr tiledCopyGetTiledThrValLayoutSrc(CopyAtomType copyAtom, LayoutAttr tiledLayoutThrVal,
                                            TileAttr tileMN) {
  return tiledCopyGetTiledThrValLayoutImpl(copyAtom, tiledLayoutThrVal, tileMN,
                                           cast<LayoutAttr>(copyAtom.getThrValLayoutSrc()));
}

LayoutAttr tiledCopyGetTiledThrValLayoutDst(CopyAtomType copyAtom, LayoutAttr tiledLayoutThrVal,
                                            TileAttr tileMN) {
  return tiledCopyGetTiledThrValLayoutImpl(copyAtom, tiledLayoutThrVal, tileMN,
                                           cast<LayoutAttr>(copyAtom.getThrValLayoutDst()));
}

LayoutAttr tiledMmaGetTiledThrValLayout(MmaAtomTypeInterface mmaAtom, LayoutAttr atomLayoutMNK,
                                        TileAttr permutationMNK, MmaOperand operandId) {
  auto *ctx = atomLayoutMNK.getContext();
  LayoutBuilder<LayoutAttr> attrBuilder(ctx);

  IntTupleAttr shapeMNK = cast<IntTupleAttr>(mmaAtom.getShapeMNK());

  int idx0, idx1;
  switch (operandId) {
  case MmaOperand::C:
  case MmaOperand::D:
    idx0 = 0;
    idx1 = 1;
    break;
  case MmaOperand::A:
    idx0 = 0;
    idx1 = 2;
    break;
  case MmaOperand::B:
    idx0 = 1;
    idx1 = 2;
    break;
  }

  SmallVector<Attribute> tileSizeElems;
  for (int i : {idx0, idx1}) {
    if (i >= permutationMNK.rank() || permutationMNK.isNoneMode(i)) {
      auto atomShapeI = intTupleProduct(attrBuilder, shapeMNK.at(i)).getLeafAsInt();
      auto thrSizeI = intTupleProduct(attrBuilder, atomLayoutMNK.getShape().at(i)).getLeafAsInt();
      tileSizeElems.push_back(IntTupleAttr::get(atomShapeI * thrSizeI));
    } else {
      auto permLayout = cast<LayoutAttr>(permutationMNK.at(i));
      auto sizeI = intTupleProduct(attrBuilder, permLayout.getShape()).getLeafAsInt();
      tileSizeElems.push_back(IntTupleAttr::get(sizeI));
    }
  }
  IntTupleAttr refShape = IntTupleAttr::get(ArrayAttr::get(ctx, tileSizeElems));
  IntTupleAttr refStride =
      IntTupleAttr::get(ArrayAttr::get(ctx, {IntTupleAttr::getLeafStatic(ctx, 1), refShape.at(0)}));
  LayoutAttr refLayout = LayoutAttr::get(refShape, refStride);

  LayoutAttr thrfrgResult = layoutTiledMmaThrValOperandView(attrBuilder, mmaAtom, atomLayoutMNK,
                                                            permutationMNK, operandId, refLayout);

  IntTupleAttr thrModeShape = thrfrgResult.getShape().at(0);
  IntTupleAttr thrModeStride = thrfrgResult.getStride().at(0);
  LayoutAttr thrModeLayout = LayoutAttr::get(thrModeShape, thrModeStride);

  LayoutAttr atomThrLayout = cast<LayoutAttr>(mmaAtom.getThrLayout());
  LayoutAttr thrLayoutVMNK = layoutTiledProduct(
      attrBuilder, atomThrLayout, attrBuilder.materializeConstantLayout(atomLayoutMNK));

  if (operandId == MmaOperand::A || operandId == MmaOperand::B) {
    IntTupleAttr thrMSize = thrLayoutVMNK.getShape().at(1);
    IntTupleAttr thrNSize = thrLayoutVMNK.getShape().at(2);
    IntTupleAttr stride0, stride1;
    if (operandId == MmaOperand::A) {
      stride0 = IntTupleAttr::getLeafStatic(ctx, 1);
      stride1 = IntTupleAttr::getLeafStatic(ctx, 0);
    } else {
      stride0 = IntTupleAttr::getLeafStatic(ctx, 0);
      stride1 = IntTupleAttr::getLeafStatic(ctx, 1);
    }
    IntTupleAttr mnShape = IntTupleAttr::get(ArrayAttr::get(ctx, {thrMSize, thrNSize}));
    IntTupleAttr mnStride = IntTupleAttr::get(ArrayAttr::get(ctx, {stride0, stride1}));
    LayoutAttr mnLayout = LayoutAttr::get(mnShape, mnStride);

    TileAttr innerTile = TileAttr::get(ArrayAttr::get(ctx, {mnLayout, IntAttr::getNone(ctx)}));
    TileAttr outerTile = TileAttr::get(ArrayAttr::get(ctx, {IntAttr::getNone(ctx), innerTile}));
    thrModeLayout = layoutComposition(attrBuilder, thrModeLayout, outerTile);
  }

  IntTupleAttr valModeShape = thrfrgResult.getShape().at(1);
  IntTupleAttr valModeStride = thrfrgResult.getStride().at(1);

  LayoutAttr complementThrVMNK = layoutComplement(attrBuilder, thrLayoutVMNK);

  LayoutAttr thrCrd2idxLayout = LayoutAttr::get(
      IntTupleAttr::get(
          ArrayAttr::get(ctx, {thrLayoutVMNK.getShape(), complementThrVMNK.getShape()})),
      IntTupleAttr::get(
          ArrayAttr::get(ctx, {thrLayoutVMNK.getStride(), complementThrVMNK.getStride()})));
  LayoutAttr thrIdx2CrdLayout = layoutRightInverse(attrBuilder, thrCrd2idxLayout);

  IntTupleAttr vmnkSize = intTupleProduct(attrBuilder, thrLayoutVMNK.getShape());
  LayoutAttr vmnkSizeLayout = LayoutAttr::get(
      IntTupleAttr::get(ArrayAttr::get(ctx, {vmnkSize, IntTupleAttr::getLeafStatic(ctx, 1)})),
      IntTupleAttr::get(ArrayAttr::get(
          ctx, {IntTupleAttr::getLeafStatic(ctx, 1), IntTupleAttr::getLeafStatic(ctx, 0)})));

  LayoutAttr thrIdx2Offset = layoutComposition(attrBuilder, vmnkSizeLayout, thrIdx2CrdLayout);
  LayoutAttr resThrLayout = layoutComposition(attrBuilder, thrModeLayout, thrIdx2Offset);

  IntTupleAttr finalShape =
      IntTupleAttr::get(ArrayAttr::get(ctx, {resThrLayout.getShape(), valModeShape}));
  IntTupleAttr finalStride =
      IntTupleAttr::get(ArrayAttr::get(ctx, {resThrLayout.getStride(), valModeStride}));
  return LayoutAttr::get(finalShape, finalStride);
}

IntTupleAttr tiledMmaGetTileSizeMNK(MmaAtomTypeInterface mmaAtom, LayoutAttr atomLayoutMNK,
                                    TileAttr permutationMNK) {
  auto *ctx = atomLayoutMNK.getContext();
  LayoutBuilder<LayoutAttr> attrBuilder(ctx);

  IntTupleAttr shapeMNK = cast<IntTupleAttr>(mmaAtom.getShapeMNK());

  SmallVector<Attribute> tileSizeElems;
  for (int i = 0; i < 3; ++i) {
    if (i >= permutationMNK.rank() || permutationMNK.isNoneMode(i)) {
      auto atomShapeI = intTupleProduct(attrBuilder, shapeMNK.at(i)).getLeafAsInt();
      auto thrSizeI = intTupleProduct(attrBuilder, atomLayoutMNK.getShape().at(i)).getLeafAsInt();
      tileSizeElems.push_back(IntTupleAttr::get(atomShapeI * thrSizeI));
    } else {
      auto permLayout = cast<LayoutAttr>(permutationMNK.at(i));
      auto sizeI = intTupleProduct(attrBuilder, permLayout.getShape()).getLeafAsInt();
      tileSizeElems.push_back(IntTupleAttr::get(sizeI));
    }
  }
  return IntTupleAttr::get(ArrayAttr::get(ctx, tileSizeElems));
}

LayoutAttr tiledMmaGetThrLayoutVMNK(MmaAtomTypeInterface mmaAtom, LayoutAttr atomLayoutMNK) {
  auto *ctx = atomLayoutMNK.getContext();
  LayoutBuilder<LayoutAttr> attrBuilder(ctx);

  LayoutAttr atomThrLayout = cast<LayoutAttr>(mmaAtom.getThrLayout());
  return layoutTiledProduct(attrBuilder, atomThrLayout,
                            attrBuilder.materializeConstantLayout(atomLayoutMNK));
}

} // namespace mlir::fly
