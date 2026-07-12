// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYROCDL_ATOM
#define FLYROCDL_ATOM

include "flydsl/Dialect/FlyROCDL/IR/Dialect.td"

def FlyROCDL_AtomStateField : I32EnumAttr<"AtomStateField", "", [
  I32EnumAttrCase<"Soffset",            0, "soffset">,
  I32EnumAttrCase<"ImmOffset",          1, "imm_offset">,
  I32EnumAttrCase<"ScaleA",             2, "scale_a">,
  I32EnumAttrCase<"ScaleB",             3, "scale_b">,
  I32EnumAttrCase<"WorkgroupMask",      4, "workgroup_mask">,
  // Runtime outer-dim tensor bound (rows from the tile start) for OOB / ragged
  // tiles: tensor_dim1 = max(0, oob_outer - warp_off_outer), tile_dim1 stays
  // static so the HW fault-guards (load) / drops (store) the partial-tile
  // overhang. Default INT32_MAX = no clamp (full tile).
  I32EnumAttrCase<"OobOuter",           5, "oob_outer">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = FlyROCDL_Dialect.cppNamespace;
}

include "flydsl/Dialect/FlyROCDL/IR/CopyAtom.td"
include "flydsl/Dialect/FlyROCDL/IR/MmaAtom.td"

#endif // FLYROCDL_ATOM
