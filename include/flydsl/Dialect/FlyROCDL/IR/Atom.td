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
  // TDM N-D descriptor carried on the copy atom: global base pointer, per-dim
  // tensor extent (extent_i, i32, tensor dim order 0=outermost) for OOB handling,
  // and per-dim tensor stride in elements (stride_i, i64; the innermost stride is
  // assumed 1 and not stored, so stride_i covers dims 0..rank-2).
  I32EnumAttrCase<"Base",               5, "base">,
  I32EnumAttrCase<"Extent0",            6, "extent_0">,
  I32EnumAttrCase<"Extent1",            7, "extent_1">,
  I32EnumAttrCase<"Extent2",            8, "extent_2">,
  I32EnumAttrCase<"Extent3",            9, "extent_3">,
  I32EnumAttrCase<"Extent4",            10, "extent_4">,
  I32EnumAttrCase<"Stride0",            11, "stride_0">,
  I32EnumAttrCase<"Stride1",            12, "stride_1">,
  I32EnumAttrCase<"Stride2",            13, "stride_2">,
  I32EnumAttrCase<"Stride3",            14, "stride_3">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = FlyROCDL_Dialect.cppNamespace;
}

include "flydsl/Dialect/FlyROCDL/IR/CopyAtom.td"
include "flydsl/Dialect/FlyROCDL/IR/MmaAtom.td"

#endif // FLYROCDL_ATOM
