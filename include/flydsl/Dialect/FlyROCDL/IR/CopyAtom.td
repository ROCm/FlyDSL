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

//===----------------------------------------------------------------------===//
// CopyOp GFX1250 — TDM gather Global<->LDS copy.
//
// Gather port of tdm_ops.py::make_tensor_gather_descriptor. Loads/stores a set of
// rows selected by explicit row indices carried as atom state (a packed
// vector<8xi32>) rather than iterating a contiguous outer dim. The global operand
// supplies only the tile shape (outer = rows, inner = per-row width) and the copy
// direction. Stateful fields (via fly.atom.set_value):
//   `workgroup_mask` (default 0): MCAST mask, ORed into GROUP1 config [15:0].
//   `base`: global matrix base pointer.
//   `outer_extent` (default INT32_MAX): tensor_dim1, the row count for HW OOB on
//     the gather indices (index >= extent => OOB row, zero-filled on load).
//   `inner_extent` (default INT32_MAX): tensor_dim0, the full matrix row width for
//     column OOB.
//   `outer_stride` (default = unset sentinel): row stride in elements; falls back
//     to the static layout stride when unset.
//   `row_indices`: vector<8xi32> of packed row indices (32-bit: 8 indices; 16-bit:
//     16 indices, 2 per lane).
//   `gather_count` (default = unset sentinel): number of valid indices; falls back
//     to the operand's outer tile extent when unset.
//   `imm_offset` (default 0, i64): byte offset added to `base` (K-column advance).
//===----------------------------------------------------------------------===//

// Like TDM2D, the gather DMA moves the whole tile in one shot; override
// `getCopyRank` (== 2) so the expand-copy lowering emits a single rank-2 call.
def FlyROCDL_CopyOpGFX1250TDMGather : FlyROCDL_StatefulCopyOp<"CopyOpGFX1250TDMGather",
    "gfx1250.tdm_gather", /*traits=*/[], /*copyOverrides=*/["getCopyRank"]> {
  let parameters = (ins
    // Row-index width in bits: 32 (up to 8 indices) or 16 (up to 16 indices).
    "int32_t":$indexSize,
    // Padding in elements (both zero disables LDS padding).
    "int32_t":$padInterval,
    "int32_t":$padAmount,
    "int32_t":$cacheModifier
  );
  let assemblyFormat = [{
    `<` `index` `=` $indexSize `,` `pad` `=` $padInterval `,` $padAmount
    `,` `cache` `=` $cacheModifier `>`
  }];
  let genVerifyDecl = 1;
}

#endif // FLYROCDL_COPYATOM
