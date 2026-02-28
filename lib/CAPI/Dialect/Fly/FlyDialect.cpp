#include "flydsl-c/FlyDialect.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Transforms/Passes.h"
#include "flydsl/Dialect/Fly/Utils/LayoutUtils.h"
#include "flydsl/Dialect/Fly/Utils/TiledOpUtils.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

using namespace mlir;
using namespace mlir::fly;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Fly, fly, mlir::fly::FlyDialect)

//===----------------------------------------------------------------------===//
// IntTupleType
//===----------------------------------------------------------------------===//

bool mlirTypeIsAFlyIntTupleType(MlirType type) { return isa<IntTupleType>(unwrap(type)); }

MlirTypeID mlirFlyIntTupleTypeGetTypeID(void) { return wrap(IntTupleType::getTypeID()); }

bool mlirFlyIntTupleTypeIsLeaf(MlirType type) { return cast<IntTupleType>(unwrap(type)).isLeaf(); }

int32_t mlirFlyIntTupleTypeGetRank(MlirType type) {
  return cast<IntTupleType>(unwrap(type)).rank();
}

int32_t mlirFlyIntTupleTypeGetDepth(MlirType type) {
  return cast<IntTupleType>(unwrap(type)).depth();
}

bool mlirFlyIntTupleTypeIsStatic(MlirType type) {
  return cast<IntTupleType>(unwrap(type)).isStatic();
}

int32_t mlirFlyIntTupleTypeGetStaticValue(MlirType type) {
  auto intTupleTy = cast<IntTupleType>(unwrap(type));
  assert(intTupleTy.isLeaf() && intTupleTy.isStatic());
  return intTupleTy.getAttr().getLeafAsInt().getValue();
}

//===----------------------------------------------------------------------===//
// LayoutType
//===----------------------------------------------------------------------===//

bool mlirTypeIsAFlyLayoutType(MlirType type) { return isa<LayoutType>(unwrap(type)); }

MlirTypeID mlirFlyLayoutTypeGetTypeID(void) { return wrap(LayoutType::getTypeID()); }

MlirType mlirFlyLayoutTypeGet(MlirType shape, MlirType stride) {
  auto shapeType = cast<IntTupleType>(unwrap(shape));
  auto strideType = cast<IntTupleType>(unwrap(stride));
  LayoutAttr attr = LayoutAttr::get(shapeType.getAttr(), strideType.getAttr());
  return wrap(LayoutType::get(attr));
}

MlirType mlirFlyLayoutTypeGetShape(MlirType type) {
  auto layoutType = cast<LayoutType>(unwrap(type));
  IntTupleAttr shapeAttr = layoutType.getAttr().getShape();
  return wrap(IntTupleType::get(shapeAttr));
}

MlirType mlirFlyLayoutTypeGetStride(MlirType type) {
  auto layoutType = cast<LayoutType>(unwrap(type));
  IntTupleAttr strideAttr = layoutType.getAttr().getStride();
  return wrap(IntTupleType::get(strideAttr));
}

bool mlirFlyLayoutTypeIsLeaf(MlirType type) { return cast<LayoutType>(unwrap(type)).isLeaf(); }

int32_t mlirFlyLayoutTypeGetRank(MlirType type) { return cast<LayoutType>(unwrap(type)).rank(); }

int32_t mlirFlyLayoutTypeGetDepth(MlirType type) { return cast<LayoutType>(unwrap(type)).depth(); }

bool mlirFlyLayoutTypeIsStatic(MlirType type) { return cast<LayoutType>(unwrap(type)).isStatic(); }

bool mlirFlyLayoutTypeIsStaticShape(MlirType type) {
  return cast<LayoutType>(unwrap(type)).isStaticShape();
}

bool mlirFlyLayoutTypeIsStaticStride(MlirType type) {
  return cast<LayoutType>(unwrap(type)).isStaticStride();
}

//===----------------------------------------------------------------------===//
// SwizzleType
//===----------------------------------------------------------------------===//

bool mlirTypeIsAFlySwizzleType(MlirType type) { return isa<SwizzleType>(unwrap(type)); }

MlirTypeID mlirFlySwizzleTypeGetTypeID(void) { return wrap(SwizzleType::getTypeID()); }

MlirType mlirFlySwizzleTypeGet(MlirContext ctx, int32_t mask, int32_t base, int32_t shift) {
  MLIRContext *context = unwrap(ctx);
  SwizzleAttr attr = SwizzleAttr::get(context, mask, base, shift);
  return wrap(SwizzleType::get(attr));
}

int32_t mlirFlySwizzleTypeGetMask(MlirType type) {
  return cast<SwizzleType>(unwrap(type)).getAttr().getMask();
}

int32_t mlirFlySwizzleTypeGetBase(MlirType type) {
  return cast<SwizzleType>(unwrap(type)).getAttr().getBase();
}

int32_t mlirFlySwizzleTypeGetShift(MlirType type) {
  return cast<SwizzleType>(unwrap(type)).getAttr().getShift();
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

bool mlirTypeIsAFlyPointerType(MlirType type) { return isa<fly::PointerType>(unwrap(type)); }

MlirTypeID mlirFlyPointerTypeGetTypeID(void) { return wrap(fly::PointerType::getTypeID()); }

MlirType mlirFlyPointerTypeGet(MlirType elemType, int32_t addressSpace, int32_t alignment) {
  Type elemTy = unwrap(elemType);
  MLIRContext *ctx = elemTy.getContext();
  AddressSpaceAttr addrSpaceAttr =
      AddressSpaceAttr::get(ctx, static_cast<AddressSpace>(addressSpace));
  AlignAttr alignAttr = AlignAttr::get(ctx, alignment);
  return wrap(fly::PointerType::get(elemTy, addrSpaceAttr, alignAttr));
}

MlirType mlirFlyPointerTypeGetElementType(MlirType type) {
  return wrap(cast<fly::PointerType>(unwrap(type)).getElemTy());
}

int32_t mlirFlyPointerTypeGetAddressSpace(MlirType type) {
  return static_cast<int32_t>(cast<fly::PointerType>(unwrap(type)).getAddressSpace().getValue());
}

int32_t mlirFlyPointerTypeGetAlignment(MlirType type) {
  return cast<fly::PointerType>(unwrap(type)).getAlignment().getAlignment();
}

MlirType mlirFlyPointerTypeGetSwizzle(MlirType type) {
  return wrap(SwizzleType::get(cast<fly::PointerType>(unwrap(type)).getSwizzle()));
}

//===----------------------------------------------------------------------===//
// MemRefType
//===----------------------------------------------------------------------===//

bool mlirTypeIsAFlyMemRefType(MlirType type) { return isa<fly::MemRefType>(unwrap(type)); }

MlirTypeID mlirFlyMemRefTypeGetTypeID(void) { return wrap(fly::MemRefType::getTypeID()); }

MlirType mlirFlyMemRefTypeGet(MlirType elemType, MlirType layout, int32_t addressSpace,
                              int32_t alignment) {
  Type elemTy = unwrap(elemType);
  auto layoutType = cast<LayoutType>(unwrap(layout));
  MLIRContext *ctx = elemTy.getContext();
  AddressSpaceAttr addrSpaceAttr =
      AddressSpaceAttr::get(ctx, static_cast<AddressSpace>(addressSpace));
  AlignAttr alignAttr = AlignAttr::get(ctx, alignment);
  return wrap(fly::MemRefType::get(elemTy, addrSpaceAttr, layoutType.getAttr(), alignAttr));
}

MlirType mlirFlyMemRefTypeGetElementType(MlirType type) {
  return wrap(cast<fly::MemRefType>(unwrap(type)).getElemTy());
}

MlirType mlirFlyMemRefTypeGetLayout(MlirType type) {
  auto memrefType = cast<fly::MemRefType>(unwrap(type));
  return wrap(LayoutType::get(memrefType.getLayout()));
}

int32_t mlirFlyMemRefTypeGetAddressSpace(MlirType type) {
  return static_cast<int32_t>(cast<fly::MemRefType>(unwrap(type)).getAddressSpace().getValue());
}

int32_t mlirFlyMemRefTypeGetAlignment(MlirType type) {
  return cast<fly::MemRefType>(unwrap(type)).getAlignment().getAlignment();
}

MlirType mlirFlyMemRefTypeGetSwizzle(MlirType type) {
  return wrap(SwizzleType::get(cast<fly::MemRefType>(unwrap(type)).getSwizzle()));
}

//===----------------------------------------------------------------------===//
// CopyOpUniversalCopyType
//===----------------------------------------------------------------------===//

bool mlirTypeIsAFlyCopyOpUniversalCopyType(MlirType type) {
  return isa<CopyOpUniversalCopyType>(unwrap(type));
}

MlirTypeID mlirFlyCopyOpUniversalCopyTypeGetTypeID(void) {
  return wrap(CopyOpUniversalCopyType::getTypeID());
}

MlirType mlirFlyCopyOpUniversalCopyTypeGet(MlirContext ctx, int32_t bitSize) {
  return wrap(CopyOpUniversalCopyType::get(unwrap(ctx), bitSize));
}

int32_t mlirFlyCopyOpUniversalCopyTypeGetBitSize(MlirType type) {
  return cast<CopyOpUniversalCopyType>(unwrap(type)).getBitSize();
}

//===----------------------------------------------------------------------===//
// CopyAtomType
//===----------------------------------------------------------------------===//

bool mlirTypeIsAFlyCopyAtomType(MlirType type) {
  return isa<CopyAtomType>(unwrap(type));
}

MlirTypeID mlirFlyCopyAtomTypeGetTypeID(void) {
  return wrap(CopyAtomType::getTypeID());
}

MlirType mlirFlyCopyAtomTypeGet(MlirType copyOp, int32_t valBits) {
  return wrap(CopyAtomType::get(unwrap(copyOp), valBits));
}

MlirType mlirFlyCopyAtomTypeGetCopyOp(MlirType type) {
  return wrap(cast<CopyAtomType>(unwrap(type)).getCopyOp());
}

int32_t mlirFlyCopyAtomTypeGetValBits(MlirType type) {
  return cast<CopyAtomType>(unwrap(type)).getValBits();
}

MlirType mlirFlyCopyAtomTypeGetThrLayout(MlirType type) {
  auto copyAtomTy = cast<CopyAtomType>(unwrap(type));
  auto attr = cast<LayoutAttr>(copyAtomTy.getThrLayout());
  return wrap(LayoutType::get(attr));
}

MlirType mlirFlyCopyAtomTypeGetThrValLayoutSrc(MlirType type) {
  auto copyAtomTy = cast<CopyAtomType>(unwrap(type));
  auto attr = cast<LayoutAttr>(copyAtomTy.getThrValLayoutSrc());
  return wrap(LayoutType::get(attr));
}

MlirType mlirFlyCopyAtomTypeGetThrValLayoutDst(MlirType type) {
  auto copyAtomTy = cast<CopyAtomType>(unwrap(type));
  auto attr = cast<LayoutAttr>(copyAtomTy.getThrValLayoutDst());
  return wrap(LayoutType::get(attr));
}

MlirType mlirFlyCopyAtomTypeGetThrValLayoutRef(MlirType type) {
  auto copyAtomTy = cast<CopyAtomType>(unwrap(type));
  auto attr = cast<LayoutAttr>(copyAtomTy.getThrValLayoutRef());
  return wrap(LayoutType::get(attr));
}

//===----------------------------------------------------------------------===//
// MmaAtomUniversalFMAType
//===----------------------------------------------------------------------===//

bool mlirTypeIsAFlyMmaAtomUniversalFMAType(MlirType type) {
  return isa<MmaAtomUniversalFMAType>(unwrap(type));
}

MlirTypeID mlirFlyMmaAtomUniversalFMATypeGetTypeID(void) {
  return wrap(MmaAtomUniversalFMAType::getTypeID());
}

MlirType mlirFlyMmaAtomUniversalFMATypeGet(MlirContext ctx, MlirType elemTy) {
  return wrap(MmaAtomUniversalFMAType::get(unwrap(ctx), unwrap(elemTy)));
}

MlirType mlirFlyMmaAtomUniversalFMATypeGetElemTy(MlirType type) {
  return wrap(cast<MmaAtomUniversalFMAType>(unwrap(type)).getElemTy());
}

//===----------------------------------------------------------------------===//
// TiledCopyType
//===----------------------------------------------------------------------===//

bool mlirTypeIsAFlyTiledCopyType(MlirType type) { return isa<TiledCopyType>(unwrap(type)); }

MlirTypeID mlirFlyTiledCopyTypeGetTypeID(void) { return wrap(TiledCopyType::getTypeID()); }

MlirType mlirFlyTiledCopyTypeGetCopyAtom(MlirType type) {
  return wrap(cast<TiledCopyType>(unwrap(type)).getCopyAtom());
}

MlirType mlirFlyTiledCopyTypeGetLayoutThrVal(MlirType type) {
  return wrap(static_cast<Type>(cast<TiledCopyType>(unwrap(type)).getLayoutThrVal()));
}

MlirType mlirFlyTiledCopyTypeGetTileMN(MlirType type) {
  return wrap(static_cast<Type>(cast<TiledCopyType>(unwrap(type)).getTileMN()));
}

static MlirType tiledCopyGetTiledTVLayout(MlirType type, bool isSrc) {
  auto tiledCopyTy = cast<TiledCopyType>(unwrap(type));
  auto copyAtom = cast<CopyAtomType>(tiledCopyTy.getCopyAtom());
  LayoutAttr tiledLayoutThrVal = tiledCopyTy.getLayoutThrVal().getAttr();
  TileAttr tileMN = tiledCopyTy.getTileMN().getAttr();
  auto *ctx = tiledLayoutThrVal.getContext();
  LayoutBuilder<LayoutAttr> attrBuilder(ctx);

  auto atomLayoutRef = cast<LayoutAttr>(copyAtom.getThrValLayoutRef());
  LayoutAttr refInv = layoutRightInverse(attrBuilder, atomLayoutRef);
  LayoutAttr atomLayoutTrg =
      cast<LayoutAttr>(isSrc ? copyAtom.getThrValLayoutSrc() : copyAtom.getThrValLayoutDst());
  LayoutAttr ref2trg = layoutComposition(attrBuilder, refInv, atomLayoutTrg);

  SmallVector<Attribute> tilerShapeElems;
  SmallVector<Attribute> tilerStrideElems;
  int64_t runningStride = 1;
  for (int i = 0; i < tileMN.rank(); ++i) {
    auto tileElem = tileMN.at(i);
    int64_t tileSize;
    if (auto intVal = dyn_cast<IntAttr>(tileElem))
      tileSize = intVal.getValue();
    else if (auto layoutVal = dyn_cast<LayoutAttr>(tileElem))
      tileSize = attrBuilder.getStaticValue(intTupleProductImpl(attrBuilder, layoutVal.getShape()));
    else
      llvm_unreachable("unsupported tile element type");
    tilerShapeElems.push_back(IntTupleAttr::getLeafStatic(ctx, tileSize));
    tilerStrideElems.push_back(IntTupleAttr::getLeafStatic(ctx, runningStride));
    runningStride *= tileSize;
  }
  IntTupleAttr tilerShape = IntTupleAttr::get(ArrayAttr::get(ctx, tilerShapeElems));
  IntTupleAttr tilerStride = IntTupleAttr::get(ArrayAttr::get(ctx, tilerStrideElems));
  LayoutAttr refLayout = LayoutAttr::get(
      IntTupleAttr::get(ArrayAttr::get(ctx, {tilerShape, IntTupleAttr::getLeafStatic(ctx, 1)})),
      IntTupleAttr::get(ArrayAttr::get(ctx, {tilerStride, IntTupleAttr::getLeafStatic(ctx, 0)})));

  LayoutAttr thrValView = fly::detail::layoutTiledCopyThrValView(attrBuilder, copyAtom, refLayout,
                                                                 tiledLayoutThrVal, ref2trg);

  SmallVector<Attribute> sliceElems;
  sliceElems.push_back(IntTupleAttr::getLeafNone(ctx));
  sliceElems.push_back(IntTupleAttr::getLeafNone(ctx));
  sliceElems.push_back(IntTupleAttr::getLeafStatic(ctx, 0));
  IntTupleAttr sliceCoord = IntTupleAttr::get(ArrayAttr::get(ctx, sliceElems));

  IntTupleAttr resultShape = intTupleSlice(attrBuilder, thrValView.getShape(), sliceCoord);
  IntTupleAttr resultStride = intTupleSlice(attrBuilder, thrValView.getStride(), sliceCoord);
  LayoutAttr result = LayoutAttr::get(resultShape, resultStride);
  return wrap(LayoutType::get(result));
}

MlirType mlirFlyTiledCopyTypeGetTiledTVLayoutSrc(MlirType type) {
  return tiledCopyGetTiledTVLayout(type, true);
}

MlirType mlirFlyTiledCopyTypeGetTiledTVLayoutDst(MlirType type) {
  return tiledCopyGetTiledTVLayout(type, false);
}

//===----------------------------------------------------------------------===//
// TiledMmaType
//===----------------------------------------------------------------------===//

bool mlirTypeIsAFlyTiledMmaType(MlirType type) { return isa<TiledMmaType>(unwrap(type)); }

MlirTypeID mlirFlyTiledMmaTypeGetTypeID(void) { return wrap(TiledMmaType::getTypeID()); }

MlirType mlirFlyTiledMmaTypeGetMmaAtom(MlirType type) {
  return wrap(cast<TiledMmaType>(unwrap(type)).getMmaAtom());
}

MlirType mlirFlyTiledMmaTypeGetAtomLayout(MlirType type) {
  return wrap(static_cast<Type>(cast<TiledMmaType>(unwrap(type)).getAtomLayout()));
}

MlirType mlirFlyTiledMmaTypeGetPermutation(MlirType type) {
  return wrap(static_cast<Type>(cast<TiledMmaType>(unwrap(type)).getPermutation()));
}

MlirType mlirFlyTiledMmaTypeGetTileSizeMNK(MlirType type) {
  auto tiledMmaTy = cast<TiledMmaType>(unwrap(type));
  auto mmaAtom = cast<MmaAtomTypeInterface>(tiledMmaTy.getMmaAtom());
  auto atomLayoutMNK = tiledMmaTy.getAtomLayout().getAttr();
  auto permutationMNK = tiledMmaTy.getPermutation().getAttr();
  auto *ctx = atomLayoutMNK.getContext();
  LayoutBuilder<LayoutAttr> attrBuilder(ctx);

  IntTupleAttr shapeMNK = cast<IntTupleAttr>(mmaAtom.getShapeMNK());

  SmallVector<Attribute> tileSizeElems;
  for (int i = 0; i < 3; ++i) {
    if (i >= permutationMNK.rank() || permutationMNK.isNoneMode(i)) {
      auto atomShapeI =
          attrBuilder.getStaticValue(intTupleProductImpl(attrBuilder, shapeMNK.at(i)));
      auto thrSizeI = attrBuilder.getStaticValue(
          intTupleProductImpl(attrBuilder, atomLayoutMNK.getShape().at(i)));
      tileSizeElems.push_back(IntTupleAttr::getLeafStatic(ctx, atomShapeI * thrSizeI));
    } else {
      auto permLayout = cast<LayoutAttr>(permutationMNK.at(i));
      auto sizeI =
          attrBuilder.getStaticValue(intTupleProductImpl(attrBuilder, permLayout.getShape()));
      tileSizeElems.push_back(IntTupleAttr::getLeafStatic(ctx, sizeI));
    }
  }
  auto result = IntTupleAttr::get(ArrayAttr::get(ctx, tileSizeElems));
  return wrap(IntTupleType::get(result));
}

MlirType mlirFlyTiledMmaTypeGetThrLayoutVMNK(MlirType type) {
  auto tiledMmaTy = cast<TiledMmaType>(unwrap(type));
  auto mmaAtom = cast<MmaAtomTypeInterface>(tiledMmaTy.getMmaAtom());
  auto atomLayoutMNK = tiledMmaTy.getAtomLayout().getAttr();
  auto *ctx = atomLayoutMNK.getContext();
  LayoutBuilder<LayoutAttr> attrBuilder(ctx);

  LayoutAttr atomThrLayout = cast<LayoutAttr>(mmaAtom.getThrLayout());
  LayoutAttr thrLayoutVMNK = layoutTiledProduct(
      attrBuilder, atomThrLayout, attrBuilder.materializeConstantLayout(atomLayoutMNK));
  return wrap(LayoutType::get(thrLayoutVMNK));
}

static MlirType tiledMmaGetTiledTVLayout(MlirType type, MmaOperand operandId) {
  auto tiledMmaTy = cast<TiledMmaType>(unwrap(type));
  auto mmaAtom = cast<MmaAtomTypeInterface>(tiledMmaTy.getMmaAtom());
  auto atomLayoutMNK = tiledMmaTy.getAtomLayout().getAttr();
  auto permutationMNK = tiledMmaTy.getPermutation().getAttr();
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
      auto atomShapeI =
          attrBuilder.getStaticValue(intTupleProductImpl(attrBuilder, shapeMNK.at(i)));
      auto thrSizeI = attrBuilder.getStaticValue(
          intTupleProductImpl(attrBuilder, atomLayoutMNK.getShape().at(i)));
      tileSizeElems.push_back(IntTupleAttr::getLeafStatic(ctx, atomShapeI * thrSizeI));
    } else {
      auto permLayout = cast<LayoutAttr>(permutationMNK.at(i));
      auto sizeI =
          attrBuilder.getStaticValue(intTupleProductImpl(attrBuilder, permLayout.getShape()));
      tileSizeElems.push_back(IntTupleAttr::getLeafStatic(ctx, sizeI));
    }
  }
  IntTupleAttr refShape = IntTupleAttr::get(ArrayAttr::get(ctx, tileSizeElems));
  IntTupleAttr refStride = IntTupleAttr::get(
      ArrayAttr::get(ctx, {IntTupleAttr::getLeafStatic(ctx, 1), refShape.at(0)})); // TODO
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

  LayoutAttr thrCrd2idx_layout = LayoutAttr::get(
      IntTupleAttr::get(
          ArrayAttr::get(ctx, {thrLayoutVMNK.getShape(), complementThrVMNK.getShape()})),
      IntTupleAttr::get(
          ArrayAttr::get(ctx, {thrLayoutVMNK.getStride(), complementThrVMNK.getStride()})));
  LayoutAttr thrIdx2Crd_layout = layoutRightInverse(attrBuilder, thrCrd2idx_layout);

  IntTupleAttr vmnkSize = intTupleProduct(attrBuilder, thrLayoutVMNK.getShape());
  LayoutAttr vmnkSizeLayout = LayoutAttr::get(
      IntTupleAttr::get(ArrayAttr::get(ctx, {vmnkSize, IntTupleAttr::getLeafStatic(ctx, 1)})),
      IntTupleAttr::get(ArrayAttr::get(
          ctx, {IntTupleAttr::getLeafStatic(ctx, 1), IntTupleAttr::getLeafStatic(ctx, 0)})));

  LayoutAttr thrIdx2Offset = layoutComposition(attrBuilder, vmnkSizeLayout, thrIdx2Crd_layout);
  LayoutAttr resThrLayout = layoutComposition(attrBuilder, thrModeLayout, thrIdx2Offset);

  IntTupleAttr finalShape =
      IntTupleAttr::get(ArrayAttr::get(ctx, {resThrLayout.getShape(), valModeShape}));
  IntTupleAttr finalStride =
      IntTupleAttr::get(ArrayAttr::get(ctx, {resThrLayout.getStride(), valModeStride}));
  LayoutAttr result = LayoutAttr::get(finalShape, finalStride);
  return wrap(LayoutType::get(result));
}

MlirType mlirFlyTiledMmaTypeGetTiledTVLayoutA(MlirType type) {
  return tiledMmaGetTiledTVLayout(type, MmaOperand::A);
}

MlirType mlirFlyTiledMmaTypeGetTiledTVLayoutB(MlirType type) {
  return tiledMmaGetTiledTVLayout(type, MmaOperand::B);
}

MlirType mlirFlyTiledMmaTypeGetTiledTVLayoutC(MlirType type) {
  return tiledMmaGetTiledTVLayout(type, MmaOperand::C);
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

void mlirRegisterFlyPasses(void) { mlir::fly::registerFlyPasses(); }
