// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYROCDL_ATTR_DEFS
#define FLYROCDL_ATTR_DEFS

include "mlir/IR/AttrTypeBase.td"
include "flydsl/Dialect/FlyROCDL/IR/Dialect.td"

def FlyROCDL_BufferDescAddressAttr : FlyROCDL_Attr<"BufferDescAddress", "buffer_desc"> {}

// A global tensor operand that carries its 2D runtime extent (globalDim) with the
// base pointer, for descriptor-driven DMA (TDM) out-of-bounds handling. Unlike
// buffer_desc it is not a fat hardware resource: the base stays a plain global
// pointer (PtrToInt-able) and the extents are separate i32 fields.
def FlyROCDL_GlobalTensorDescAddressAttr : FlyROCDL_Attr<"GlobalTensorDescAddress", "global_tensor_desc"> {}

#endif // FLYROCDL_ATTR_DEFS
