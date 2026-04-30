// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYROCDL_ATTR_DEFS
#define FLYROCDL_ATTR_DEFS

include "mlir/IR/AttrTypeBase.td"
include "flydsl/Dialect/FlyROCDL/IR/Dialect.td"

def FlyROCDL_TargetAddressSpace : FlyROCDL_I32EnumAttr<"TargetAddressSpace", "", [
  I32EnumAttrCase<"BufferDesc", 8, "buffer_desc">
]>;

def FlyROCDL_TargetAddressSpaceAttr
    : FlyROCDL_EnumAttr<FlyROCDL_TargetAddressSpace, "address_space"> {}

#endif // FLYROCDL_ATTR_DEFS
