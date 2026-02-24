#include "flydsl-c/FlyDialect.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Transforms/Passes.h"
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
// CopyAtomUniversalCopyType
//===----------------------------------------------------------------------===//

bool mlirTypeIsAFlyCopyAtomUniversalCopyType(MlirType type) {
  return isa<CopyAtomUniversalCopyType>(unwrap(type));
}

MlirTypeID mlirFlyCopyAtomUniversalCopyTypeGetTypeID(void) {
  return wrap(CopyAtomUniversalCopyType::getTypeID());
}

MlirType mlirFlyCopyAtomUniversalCopyTypeGet(MlirContext ctx, int32_t bitSize) {
  return wrap(CopyAtomUniversalCopyType::get(unwrap(ctx), bitSize));
}

int32_t mlirFlyCopyAtomUniversalCopyTypeGetBitSize(MlirType type) {
  return cast<CopyAtomUniversalCopyType>(unwrap(type)).getBitSize();
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
// Pass Registration
//===----------------------------------------------------------------------===//

void mlirRegisterFlyPasses(void) { mlir::fly::registerFlyPasses(); }
