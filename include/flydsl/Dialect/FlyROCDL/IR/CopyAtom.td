// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYROCDL_COPYATOM
#define FLYROCDL_COPYATOM

include "flydsl/Dialect/FlyROCDL/IR/Dialect.td"

def FlyROCDL_CopyOpBufferCopy : FlyROCDL_StatefulCopyOp<"CopyOpCDNA3BufferCopy", "cdna3.buffer_copy",[]> {
  let parameters = (ins
    "int32_t":$bitSize,
    // raw buffer load/store cachepolicy: 0 = cached, 2 = non-temporal (nt)
    DefaultValuedParameter<"int32_t", "0">:$cacheModifier
  );
  let assemblyFormat = "`<` $bitSize (`,` `cache` `=` $cacheModifier^)? `>`";
}

def FlyROCDL_CopyOpBufferCopyLDS : FlyROCDL_StatefulCopyOp<"CopyOpCDNA3BufferCopyLDS", "cdna3.buffer_copy_lds",[]> {
  let parameters = (ins
    "int32_t":$bitSize
  );
  let assemblyFormat = "`<` $bitSize `>`";
  let genVerifyDecl = 1;
}

def FlyROCDL_CopyOpBufferAtomic : FlyROCDL_StatefulCopyOp<"CopyOpCDNA3BufferAtomic", "cdna3.buffer_atomic",[]> {
  let parameters = (ins
    "::mlir::fly::AtomicOpAttr":$atomicOp,
    "Type":$valType
  );
  let assemblyFormat = "`<` $atomicOp `,` $valType `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::fly::AtomicOpAttr":$atomicOp, "Type":$valType), [{
      return $_get(atomicOp.getContext(), atomicOp, valType);
    }]>
  ];
}

def FlyROCDL_CopyOpLdsReadTranspose : FlyROCDL_CopyOp<"CopyOpCDNA4LdsReadTranspose", "cdna4.lds_read_trans",[]> {
  let parameters = (ins
    "int32_t":$transGranularity, "int32_t":$bitSize
  );
  let assemblyFormat = "`<` `trans` `=` $transGranularity `` `b` `,` $bitSize  `>`";
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// CopyOp GFX1250 — 2D TDM (Tensor Data Mover) async Global<->LDS copy.
//
// Descriptor-driven whole-tile DMA. The tile geometry (outer/inner extent,
// outer stride) is taken from the Global-side memref layout; padding / warp
// distribution / cache policy are compile-time atom parameters. Direction
// (load vs store) is inferred from which side is Global vs Shared.
//
// Stateful: carries a runtime `workgroup_mask` (default 0) for TDM multicast
// (MCAST) loads, ORed into the descriptor's GROUP1 config word [15:0]. Set via
// fly.atom.set_value(atom, "workgroup_mask", %mask).
//===----------------------------------------------------------------------===//

def FlyROCDL_CopyOpGFX1250TDM2D : FlyROCDL_StatefulCopyOp<"CopyOpGFX1250TDM2D", "gfx1250.tdm_2d",[]> {
  let parameters = (ins
    "int32_t":$numWarps,
    // Padding in elements (both zero disables LDS padding).
    "int32_t":$padInterval,
    "int32_t":$padAmount,
    "int32_t":$cacheModifier
  );
  let assemblyFormat = [{
    `<` `warps` `=` $numWarps `,` `pad` `=` $padInterval `,` $padAmount
    `,` `cache` `=` $cacheModifier `>`
  }];
  let genVerifyDecl = 1;
}

#endif // FLYROCDL_COPYATOM
