// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLY_TRANSFORMS_MEMREF_LOWERING
#define FLY_TRANSFORMS_MEMREF_LOWERING

include "mlir/IR/PatternBase.td"
include "flydsl/Dialect/Fly/IR/FlyOps.td"

def : Pat<(Fly_PtrLoadOp Fly_IntTuple:$int_tuple),
          (replaceWithValue $int_tuple)>;

def : Pat<(Fly_AddOffsetOp Fly_IntTuple:$int_tuple, Fly_IntTuple:$offset),
          (Fly_IntTupleAddOp $int_tuple, $offset)>;
// Nested add_offset is canonicalized by AddOffsetCanonicalization in
// LayoutLowering.cpp.  Fusing unconditionally here merged a runtime offset with
// a compile-time constant, which gave every LDS read its own base register
// instead of sharing one (issue #898).

def : Pat<(Fly_DecompositionOp Fly_MemRef:$memref),
          (replaceWithValue $memref),
          [(Fly_SimpleLayoutMemRef $memref)]>;
def : Pat<(Fly_DecompositionOp Fly_CoordTensor:$tensor),
          (replaceWithValue $tensor),
          [(Fly_SimpleLayoutCoordTensor $tensor)]>;
def : Pat<(Fly_MemRefLoadOp Fly_MemRef:$memref, $indices),
          (Fly_PtrLoadOp (Fly_AddOffsetOp (Fly_GetIterOp $memref),
                                          (Fly_Crd2IdxOp $indices, (Fly_GetLayoutOp $memref))))>;
def : Pat<(Fly_MemRefStoreOp $value, Fly_MemRef:$memref, $indices),
          (Fly_PtrStoreOp $value, (Fly_AddOffsetOp (Fly_GetIterOp $memref),
                                                   (Fly_Crd2IdxOp $indices, (Fly_GetLayoutOp $memref))))>;

def : Pat<(Fly_MemRefLoadOp Fly_CoordTensor:$tensor, $indices),
          (Fly_AddOffsetOp (Fly_GetIterOp $tensor),
                           (Fly_Crd2IdxOp $indices, (Fly_GetLayoutOp $tensor)))>;

#endif // FLY_TRANSFORMS_MEMREF_LOWERING
