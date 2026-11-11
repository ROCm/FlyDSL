// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLYROCDL_UTILS_TDMATOMBUILDER_H
#define FLYDSL_DIALECT_FLYROCDL_UTILS_TDMATOMBUILDER_H

#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir {
class IRRewriter;
}

namespace mlir::fly_rocdl::tdm {

// TDM descriptors hold at most five axes. Excess modes share the last axis
// for positioning, with their transfers left in software ITER.
constexpr int32_t kMaxRank = 5;

// Tile dimensions are direct 16-bit lengths; zero disables a dimension (dim 0: NOP).
constexpr int32_t kMaxTileDim = 0xFFFF;

// Keep merged axes within half the encodable range. This is a coalescing policy,
// not a restriction on an individual descriptor dimension.
constexpr int32_t kMaxCoalescedTileDim = 32768;

// `tensor_dimN_stride` is a 48-bit field, in elements of data_size.
constexpr uint64_t kMaxTensorStride = (uint64_t{1} << 48) - 1;

// `iterate_count` is a 16-bit field encoded as "value minus one".
constexpr int32_t kMaxIterateCount = 256;

// Iteration is paid for out of the descriptor's own slots: `tensor_dim2_stride` becomes
// `global_addr_increment`, `tensor_dim3` becomes `lds_addr_increment`, and `tile_dim3`
// becomes `iterate_count`. Losing dim 2's *stride* is what sets the bound -- a dim the
// descriptor cannot step is not a dim -- so an iterating descriptor holds two. The third
// axis is the one the iteration itself walks, which is why iteration covers a 3D tensor
// with a 2D descriptor.
constexpr int32_t kMaxIterateRank = 2;

/// `tensor2tdm` read the other way round: per global mode, the descriptor axis index
/// its flag lands on, flattened into mode order; `-1` for a mode with no bound to give.
///
/// The basis keeps the tensor's nesting so an `boundary_check` tuple can be profile-checked against
/// it, but every consumer works mode by mode. A leaf's *scale* says how the mode's
/// coordinate maps onto its axis, which the extent — measured on that axis — has already
/// accounted for, so only the axis index survives here.
void boundaryCheckAxes(fly::IntTupleAttr tensor2tdm, SmallVectorImpl<int32_t> &axes);

} // namespace mlir::fly_rocdl::tdm

namespace mlir::fly_rocdl {

/// Materialize a tiled TDM builder using the same construction as type inference.
LogicalResult expandTdmAtom(MakeTiledTdmLoadAtomOp op, IRRewriter &rewriter);
LogicalResult expandTdmAtom(MakeTiledTdmStoreAtomOp op, IRRewriter &rewriter);

FailureOr<fly::LayoutType> tdmPartitionLayout(Type atomType, Type stensorType, Type gtensorType,
                                              int32_t numWarps,
                                              function_ref<InFlightDiagnostic()> emitError);

} // namespace mlir::fly_rocdl

#endif // FLYDSL_DIALECT_FLYROCDL_UTILS_TDMATOMBUILDER_H
