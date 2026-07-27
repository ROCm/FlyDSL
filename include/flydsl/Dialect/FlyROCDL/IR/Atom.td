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
  I32EnumAttrCase<"WorkgroupMask",      4, "workgroup_mask">
  // NOTE: the gfx1250 TDM N-D descriptor's per-dim tensor extent (OOB) and per-dim
  // tensor stride are NOT shared enum fields. They are resolved privately by
  // CopyOpGFX1250TDMType from the field names "extent_0".."extent_4" /
  // "stride_0".."stride_3" (set via fly.atom.set_value), keeping this cross-atom
  // vocabulary free of TDM's per-dim sprawl. See GFX1250/CopyAtom.cpp (tdmGeomSlot).
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = FlyROCDL_Dialect.cppNamespace;
}

include "flydsl/Dialect/FlyROCDL/IR/CopyAtom.td"
include "flydsl/Dialect/FlyROCDL/IR/MmaAtom.td"

#endif // FLYROCDL_ATOM
