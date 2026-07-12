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
  // TDM 2D descriptor carried on the copy atom: global base pointer plus the
  // tensor's runtime 2D extent (for OOB handling) and outer stride (elements).
  I32EnumAttrCase<"Base",               5, "base">,
  I32EnumAttrCase<"OuterExtent",        6, "outer_extent">,
  I32EnumAttrCase<"InnerExtent",        7, "inner_extent">,
  I32EnumAttrCase<"OuterStride",        8, "outer_stride">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = FlyROCDL_Dialect.cppNamespace;
}

include "flydsl/Dialect/FlyROCDL/IR/CopyAtom.td"
include "flydsl/Dialect/FlyROCDL/IR/MmaAtom.td"

#endif // FLYROCDL_ATOM
