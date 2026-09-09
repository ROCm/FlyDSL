// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLYROCDL_UTILS_TDMGEOMETRY_H
#define FLYDSL_DIALECT_FLYROCDL_UTILS_TDMGEOMETRY_H

#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <tuple>

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

namespace mlir::fly_rocdl::tdm {

// TDM descriptors hold at most five dims. Beyond that the trailing modes are packed into
// the last one.
constexpr int32_t kMaxRank = 5;

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

/// One descriptor scalar — an extent or a stride.
///
/// Either a compile-time constant, or a reference to a mode of the global tensor whose
/// value only exists at run time. The reference is what lets type inference and the
/// expansion share one derivation: inference needs the geometry's *shape* and reads it
/// off the static half, while the expansion needs the SSA value and looks it up by
/// `mode` in the layout operand. `divisor` is the ratio applied to the looked-up value
/// when the descriptor counts in a wider unit than the tensor's element, 1 otherwise.
struct Scalar {
  int32_t value = 0; ///< valid when `isStatic`
  bool isStatic = true;
  int32_t mode = -1;      ///< global mode (depth-first leaf index) when dynamic
  bool fromShape = false; ///< dynamic: read the mode's extent rather than its stride
  int32_t divisor = 1;

  static Scalar getStatic(int32_t v) { return Scalar{v, true, -1, false, 1}; }
  static Scalar getDynamic(int32_t mode, bool fromShape) {
    return Scalar{0, false, mode, fromShape, 1};
  }
};

/// One descriptor dim.
struct Dim {
  int32_t box = 1;            ///< tile extent along this dim
  SmallVector<int32_t> modes; ///< global modes feeding it, depth-first leaf indices
  Scalar stride;              ///< global stride, in descriptor elements
  Scalar tensorDim;           ///< global extent, for the hardware boundary check
  /// Whether this dim's bound is a faithful per-mode rectangular bound. A dim fed by one
  /// mode always is. The rank-5 packing's is not — its extent comes from a gcd recurrence
  /// over modes that are unrelated in memory.
  bool clampable = true;
};

/// The derived descriptor geometry, in descriptor dim order (innermost first).
struct Geometry {
  SmallVector<Dim> dims;
  int32_t padInterval = 0;
  int32_t padAmount = 0;
  int32_t ratio = 1;     ///< tensor elements per descriptor element
  int32_t iterCount = 1; ///< descriptor replays, 1 when it does not iterate
  Scalar iterStride;     ///< global step between replays, valid when iterCount > 1
  int32_t iterMode = -1; ///< the global mode hardware iteration absorbed

  /// Per global mode, its geometry *after* the recast — what the descriptor counts in.
  /// Kept because the coordinate tensor's basis scales are read back off it.
  SmallVector<Scalar> modeExtent;
  SmallVector<Scalar> modeStride;
  /// The single stride-1 global mode, or -1. Under a recast this is the mode whose
  /// coordinate has to be divided by `ratio`.
  int32_t contiguousMode = -1;

  int32_t rank() const { return static_cast<int32_t>(dims.size()); }
};

struct Request {
  fly::LayoutAttr gLayout;    ///< the global tensor's layout
  fly::LayoutAttr smemLayout; ///< the LDS tile layout, fully static
  /// The value map, from tile values to global modes: `makeValueMap` of the tiler.
  fly::LayoutAttr valueMap;
  int32_t elemBits = 0;
  int32_t internalBits = 0;
  int32_t numWarps = 1;
  /// Whether the atom starts out bounding what it can. One flag for the whole tensor:
  /// which *individual* modes clamp is per-call state, so the builder only says whether
  /// to switch the checkable ones on. A mode with no bound to give is left off either
  /// way — this is "clamp what can be clamped", never a request that can be refused.
  bool initBoundaryCheck = true;
};

/// Derive the descriptor geometry: invert the LDS tile layout to find the largest
/// contiguous run, trace that run back through the global tensor's modes, and read the
/// per-dim extents and strides off the result.
FailureOr<Geometry> derive(const Request &request, function_ref<InFlightDiagnostic()> emitError);

/// Split an LDS tile layout into `(padInterval, padAmount, compactStride)`: the descriptor
/// carries the skip as a pad field and the geometry runs on the tile with the skip taken
/// back out. Exposed because `tdm_partition` cuts the tile by its *compact* order too.
FailureOr<std::tuple<int32_t, int32_t, fly::IntTupleAttr>>
analyzeLdsTile(fly::LayoutAttr smemLayout, function_ref<InFlightDiagnostic()> emitError);

/// The value map: `identity(gshape) . tiler`, with the tiler right-padded with 1s to the
/// tensor's profile.
FailureOr<fly::LayoutAttr> makeValueMap(fly::IntTupleAttr gshape, fly::TileAttr tiler,
                                        function_ref<InFlightDiagnostic()> emitError);

/// Per global mode, the descriptor axis its boundary-check flag lands on — the atom type's
/// `tensor2tdm`. Shaped like `gshape`, so an `boundary_check` tuple can be checked against it
/// for profile. A leaf is `scale E axis`, or `0` for a mode with no bound to give.
///
/// Axis indices are in *tensor dim order* (the reverse of descriptor order), matching the
/// atom's `tileShape` and its state slots.
FailureOr<fly::IntTupleAttr> makeTensor2Tdm(const Geometry &geometry, fly::IntTupleAttr gshape,
                                            function_ref<InFlightDiagnostic()> emitError);

/// `tensor2tdm` read the other way round: per global mode, the descriptor axis index
/// its flag lands on, flattened into mode order; `-1` for a mode with no bound to give.
///
/// The basis keeps the tensor's nesting so an `boundary_check` tuple can be profile-checked against
/// it, but every consumer works mode by mode. A leaf's *scale* says how the mode's
/// coordinate maps onto its axis, which the extent — measured on that axis — has already
/// accounted for, so only the axis index survives here.
void boundaryCheckAxes(fly::IntTupleAttr tensor2tdm, SmallVectorImpl<int32_t> &axes);

/// The coordinate tensor's layout: per global mode, the descriptor axis it moves along, as
/// a basis stride. Slicing it in logical order yields an origin already expressed in
/// descriptor axes.
FailureOr<fly::LayoutAttr> coordLayout(const Geometry &geometry, fly::IntTupleAttr gshape,
                                       function_ref<InFlightDiagnostic()> emitError);

/// The atom's initial `boundary_check` state, shaped like the tensor: `enable` on every
/// mode that has a bound to give, and off on every mode that has none.
///
/// A mode with nothing to clamp — size-1, stride-0, not spanned by the box, or sharing the
/// rank-5 packing's dim — simply comes out off. The builder's flag is "clamp what can be
/// clamped" and so has no way to fail; naming such a mode is only an error when a call
/// site names it, which is `fly.atom.set_value "boundary_check"`.
FailureOr<fly::IntTupleAttr> initialBoundaryCheck(const Geometry &geometry,
                                                  fly::IntTupleAttr gshape, bool enable,
                                                  function_ref<InFlightDiagnostic()> emitError);

/// The tile shape the atom type carries, in tensor dim order.
SmallVector<int32_t> tileShape(const Geometry &geometry);

/// The layout `tdm_partition` cuts both the LDS tile and the coordinate tile by, so the
/// two keep describing the same elements.
///
/// The result is `((ATOM), (ITER))`, or `((ATOM), (WARP), (ITER))` when the workgroup's
/// warps split the tile: mode 0 is one atom call's worth of values, the middle mode (when
/// present) is the warp the caller slices out, and the last counts the calls. Composing a
/// tensor with it is what the caller still does in IR; everything up to it is static and
/// is folded here, which is why this hands back a layout rather than a tensor.
///
/// Only mode-0 -- the TDM box one call fills -- is related here: `smemLayout` and
/// `coordShape` are cut by their mode-0, whose sizes must agree, while any remaining "rest"
/// modes (e.g. PIPE on the LDS side, k-tiles on the coordinate side) pass through untouched
/// and may differ. The caller applies the returned mode-0 tiler by-mode (padding the rest
/// with pass-through), so mode 0 is reshaped and the rest is left alone.
///
/// `smemLayout`'s mode-0 decides the split. Inverting it gives the order the box is laid out
/// in LDS; that order is cut into `numWarps` equal chunks and the first ATOM values of a
/// chunk are what one descriptor fills. The padding is divided out first, because a pad is
/// a hole in the addresses and not in the values.
///
/// `atomValBits` / `atomValShape` come from the copy atom: how many values one *call*
/// moves, counted in the atom's own unit, which a recast makes wider than the LDS tile's
/// element. `ldsElemBits` is that element, and the two are reconciled in bits.
FailureOr<fly::LayoutAttr> partitionLayout(fly::IntTupleAttr atomValShape, int32_t atomValBits,
                                           fly::LayoutAttr smemLayout, int32_t ldsElemBits,
                                           fly::IntTupleAttr coordShape, int32_t numWarps,
                                           function_ref<InFlightDiagnostic()> emitError);

} // namespace mlir::fly_rocdl::tdm

#endif // FLYDSL_DIALECT_FLYROCDL_UTILS_TDMGEOMETRY_H
