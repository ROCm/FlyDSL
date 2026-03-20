// Copyright (c) 2025 FlyDSL Project Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef FLY_TRANSFORMS_MEMREF_LOWERING
#define FLY_TRANSFORMS_MEMREF_LOWERING

include "mlir/IR/PatternBase.td"
include "flydsl/Dialect/Fly/IR/FlyOps.td"

def : Pat<(Fly_PtrLoadOp Fly_IntTuple:$int_tuple),
          (replaceWithValue $int_tuple)>;
def : Pat<(Fly_AddOffsetOp Fly_IntTuple:$int_tuple, Fly_IntTuple:$offset),
          (Fly_IntTupleAddOp $int_tuple, $offset)>;

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
