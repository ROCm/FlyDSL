// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYROCDL_OPS
#define FLYROCDL_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

include "flydsl/Dialect/FlyROCDL/IR/Dialect.td"

class FlyROCDL_Op<string mnemonic, list<Trait> traits = []>
    : Op<FlyROCDL_Dialect, mnemonic, traits>;

def FlyROCDL_GetBufferRsrcOp : FlyROCDL_Op<"get_buffer_rsrc",
    [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Extract the raw ROCDL buffer resource from a buffer-descriptor pointer.";
  let description = [{
    A `fly.ptr` with the `#fly_rocdl.buffer_desc` address space converts to a
    `BufferFatPtr` (an `!llvm.struct<(ptr<8>, i32)>` of buffer resource + offset),
    not to a bare buffer resource. This op extracts element 0, the ROCDL buffer
    resource (`!llvm.ptr<8>`), so it can be fed to raw ROCDL buffer intrinsics.

    The operand must be a `fly.ptr` in the `buffer_desc` address space.
  }];
  let arguments = (ins AnyType:$ptr);
  let results = (outs AnyType:$result);
  let assemblyFormat = "`(` $ptr `)` attr-dict `:` functional-type($ptr, $result)";
}

#endif // FLYROCDL_OPS
