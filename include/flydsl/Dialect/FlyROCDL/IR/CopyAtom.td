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
// Whole-tile DMA. The atom carries the tile descriptor as runtime state. The
// global operand of the copy_atom_call is a *shape/direction token only*: only
// its layout (compile-time tile shape) and address space (load vs store) are
// read — its pointer is unused; the base address comes from the `base` state.
// Stateful fields (via fly.atom.set_value):
//   `workgroup_mask` (default 0): MCAST mask, ORed into GROUP1 config [15:0].
//   `base`: global tile base pointer (advance it to move to the next tile).
//   `outer_extent` / `inner_extent` (default INT32_MAX = no clamp): the tensor's
//     2D extent for OOB handling; tensor_dim0/1 = max(0, extent - warp start)
//     while the tile dims stay static, so a ragged tile is HW-guarded.
//   `outer_stride` (default = unset sentinel): outer stride in elements. Unset
//     falls back to the tile memref's static layout stride; a tile with a
//     dynamic outer stride must set this (else the descriptor stride faults
//     rather than silently reading stride 0).
//===----------------------------------------------------------------------===//

// TDM moves the whole 2D tile in one DMA, so it overrides `getCopyRank` (== 2)
// to keep the expand-copy lowering from decomposing the tiled memref per element.
def FlyROCDL_CopyOpGFX1250TDM2D : FlyROCDL_StatefulCopyOp<"CopyOpGFX1250TDM2D", "gfx1250.tdm_2d",
    /*traits=*/[], /*copyOverrides=*/["getCopyRank"]> {
  let parameters = (ins
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
    `<` `warps` `=` $numWarps `,` `pad` `=` $padInterval `,` $padAmount
    `,` `cache` `=` $cacheModifier `,` `barrier` `=` $atomicBarrier
    `,` `timeout` `=` $earlyTimeout `>`
  }];
  let genVerifyDecl = 1;
}

#endif // FLYROCDL_COPYATOM
