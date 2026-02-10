#ifndef FLYDSL_C_FLYDIALECT_H
#define FLYDSL_C_FLYDIALECT_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Fly, fly);

//===----------------------------------------------------------------------===//
// IntTupleType
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyIntTupleType(MlirType type);
MLIR_CAPI_EXPORTED MlirTypeID mlirFlyIntTupleTypeGetTypeID(void);

// Accessors
MLIR_CAPI_EXPORTED bool mlirFlyIntTupleTypeIsLeaf(MlirType type);
MLIR_CAPI_EXPORTED int32_t mlirFlyIntTupleTypeGetRank(MlirType type);
MLIR_CAPI_EXPORTED int32_t mlirFlyIntTupleTypeGetDepth(MlirType type);
MLIR_CAPI_EXPORTED bool mlirFlyIntTupleTypeIsStatic(MlirType type);

//===----------------------------------------------------------------------===//
// LayoutType
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyLayoutType(MlirType type);
MLIR_CAPI_EXPORTED MlirTypeID mlirFlyLayoutTypeGetTypeID(void);

// Constructor
MLIR_CAPI_EXPORTED MlirType mlirFlyLayoutTypeGet(MlirType shape, MlirType stride);

// Accessors
MLIR_CAPI_EXPORTED MlirType mlirFlyLayoutTypeGetShape(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyLayoutTypeGetStride(MlirType type);
MLIR_CAPI_EXPORTED bool mlirFlyLayoutTypeIsLeaf(MlirType type);
MLIR_CAPI_EXPORTED int32_t mlirFlyLayoutTypeGetRank(MlirType type);
MLIR_CAPI_EXPORTED int32_t mlirFlyLayoutTypeGetDepth(MlirType type);
MLIR_CAPI_EXPORTED bool mlirFlyLayoutTypeIsStatic(MlirType type);
MLIR_CAPI_EXPORTED bool mlirFlyLayoutTypeIsStaticShape(MlirType type);
MLIR_CAPI_EXPORTED bool mlirFlyLayoutTypeIsStaticStride(MlirType type);

//===----------------------------------------------------------------------===//
// SwizzleType
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlySwizzleType(MlirType type);
MLIR_CAPI_EXPORTED MlirTypeID mlirFlySwizzleTypeGetTypeID(void);

// Constructor
MLIR_CAPI_EXPORTED MlirType mlirFlySwizzleTypeGet(MlirContext ctx, int32_t mask, int32_t base,
                                                  int32_t shift);

// Accessors
MLIR_CAPI_EXPORTED int32_t mlirFlySwizzleTypeGetMask(MlirType type);
MLIR_CAPI_EXPORTED int32_t mlirFlySwizzleTypeGetBase(MlirType type);
MLIR_CAPI_EXPORTED int32_t mlirFlySwizzleTypeGetShift(MlirType type);

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyPointerType(MlirType type);
MLIR_CAPI_EXPORTED MlirTypeID mlirFlyPointerTypeGetTypeID(void);

// Constructor
MLIR_CAPI_EXPORTED MlirType mlirFlyPointerTypeGet(MlirType elemType, int32_t addressSpace,
                                                  int32_t alignment);

// Accessors
MLIR_CAPI_EXPORTED MlirType mlirFlyPointerTypeGetElementType(MlirType type);
MLIR_CAPI_EXPORTED int32_t mlirFlyPointerTypeGetAddressSpace(MlirType type);
MLIR_CAPI_EXPORTED int32_t mlirFlyPointerTypeGetAlignment(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyPointerTypeGetSwizzle(MlirType type);

//===----------------------------------------------------------------------===//
// MemRefType
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyMemRefType(MlirType type);
MLIR_CAPI_EXPORTED MlirTypeID mlirFlyMemRefTypeGetTypeID(void);

// Constructor - layout must be LayoutType
MLIR_CAPI_EXPORTED MlirType mlirFlyMemRefTypeGet(MlirType elemType, MlirType layout,
                                                 int32_t addressSpace, int32_t alignment);

// Accessors
MLIR_CAPI_EXPORTED MlirType mlirFlyMemRefTypeGetElementType(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyMemRefTypeGetLayout(MlirType type);
MLIR_CAPI_EXPORTED int32_t mlirFlyMemRefTypeGetAddressSpace(MlirType type);
MLIR_CAPI_EXPORTED int32_t mlirFlyMemRefTypeGetAlignment(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyMemRefTypeGetSwizzle(MlirType type);

//===----------------------------------------------------------------------===//
// CopyAtomUniversalCopyType
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyCopyAtomUniversalCopyType(MlirType type);
MLIR_CAPI_EXPORTED MlirTypeID mlirFlyCopyAtomUniversalCopyTypeGetTypeID(void);

// Constructor
MLIR_CAPI_EXPORTED MlirType mlirFlyCopyAtomUniversalCopyTypeGet(MlirContext ctx, int32_t bitSize);

// Accessors
MLIR_CAPI_EXPORTED int32_t mlirFlyCopyAtomUniversalCopyTypeGetBitSize(MlirType type);

//===----------------------------------------------------------------------===//
// MmaAtomUniversalFMAType
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyMmaAtomUniversalFMAType(MlirType type);
MLIR_CAPI_EXPORTED MlirTypeID mlirFlyMmaAtomUniversalFMATypeGetTypeID(void);

// Constructor
MLIR_CAPI_EXPORTED MlirType mlirFlyMmaAtomUniversalFMATypeGet(MlirContext ctx, MlirType elemTy);

// Accessors
MLIR_CAPI_EXPORTED MlirType mlirFlyMmaAtomUniversalFMATypeGetElemTy(MlirType type);

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

/// Register all Fly dialect passes (fly-canonicalize, fly-layout-lowering).
MLIR_CAPI_EXPORTED void mlirRegisterFlyPasses(void);

#ifdef __cplusplus
}
#endif

#endif // FLYDSL_C_FLYDIALECT_H
