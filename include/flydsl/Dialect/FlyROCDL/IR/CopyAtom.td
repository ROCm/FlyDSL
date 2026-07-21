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
// CopyOp GFX1250 — N-D TDM (Tensor Data Mover) async Global<->LDS copy (1-5D).
//
// Whole-tile DMA. The global operand of the copy_atom_call supplies the base
// pointer (its pointer, NOT atom state) and the compile-time tile shape (its
// layout); its address space picks the direction (load vs store). `rank` (1-5) is
// the tensor/tile rank. Stateful fields (via fly.atom.set_value):
//   `workgroup_mask` (default 0): MCAST mask, ORed into GROUP1 config [15:0].
//   `extent_i` (default INT32_MAX = no clamp): per-dim tensor extent (tensor dim
//     order, 0=outermost) for OOB handling; tensor_dim = max(0, extent - warp
//     start) while the tile dims stay static, so a ragged tile is HW-guarded.
//   `stride_i` (default = unset sentinel): per-dim tensor stride in elements for
//     dims 0..rank-2 (innermost stride is assumed 1). Unset falls back to the tile
//     memref's static layout stride; set it for a dynamic/true global stride that
//     differs from the packed tile-internal stride.
//   `imm_offset` (default 0): i64 byte offset added to base (K-loop tile bump).
//===----------------------------------------------------------------------===//

// TDM moves the whole N-D tile in one DMA, so it carries the WholeTileCopy trait
// to keep the expand-copy lowering from decomposing the tiled memref per element.
def FlyROCDL_CopyOpGFX1250TDM : FlyROCDL_StatefulCopyOp<"CopyOpGFX1250TDM", "gfx1250.tdm",
    /*traits=*/[Fly_WholeTileCopyTrait]> {
  let parameters = (ins
    // Tensor/tile rank (1-5).
    "int32_t":$rank,
    "int32_t":$numWarps,
    // Padding in elements (both zero disables LDS padding).
    "int32_t":$padInterval,
    "int32_t":$padAmount,
    "int32_t":$cacheModifier,
    // Descriptor config bits (GROUP1 sgpr0): atomic_barrier_enable [18] sets the
    // HW auto-barrier bit; early_timeout [21] is a multicast-load GL1 knob.
    "bool":$atomicBarrier,
    "bool":$earlyTimeout
  );
  let assemblyFormat = [{
    `<` `rank` `=` $rank `,` `warps` `=` $numWarps `,` `pad` `=` $padInterval `,` $padAmount
    `,` `cache` `=` $cacheModifier `,` `barrier` `=` $atomicBarrier
    `,` `timeout` `=` $earlyTimeout `>`
  }];
  let genVerifyDecl = 1;
}

#endif // FLYROCDL_COPYATOM
