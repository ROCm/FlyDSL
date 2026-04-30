// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYROCDL_ATTR_DEFS
#define FLYROCDL_ATTR_DEFS

include "mlir/IR/AttrTypeBase.td"
include "flydsl/Dialect/FlyROCDL/IR/Dialect.td"

def FlyROCDL_BufferDescAddressAttr : FlyROCDL_Attr<"BufferDescAddress", "buffer_desc"> {}

#endif // FLYROCDL_ATTR_DEFS
