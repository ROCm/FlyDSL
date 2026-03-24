// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYROCDL_COPYATOM
#define FLYROCDL_COPYATOM

include "flydsl/Dialect/FlyROCDL/IR/Dialect.td"

def FlyROCDL_CopyOpBufferCopy : FlyxROCL_CopyOp<"CopyOpCDNA3BufferCopy", "cdna3.buffer_copy",[]> {
  let parameters = (ins
    "int32_t":$bitSize
  );
  let assemblyFormat = "`<` $bitSize `>`";
}

#endif // FLYROCDL_COPYATOM
