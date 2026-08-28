// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Descriptor construction for the CDNA5 TDM copy atom.
//
// The question this answers: *given a global tensor and an LDS tile layout, which global
// modes does the DMA box span, in what order, and what are the per-dim extents and
// strides?*
//
// A global mode is identified by its depth-first **leaf index** into the (possibly
// hierarchical) tensor shape. A Fly basis stride carries the *path* instead (`1E1E0`), so
// the two are translated by `flatIndexOfPath`; a hierarchical global tensor therefore
// needs no flattening of its own.
//
// What the LDS tile's padding does to all of this: TDM has no smem swizzle, it has an LDS
// padding field that the DMA engine walks on its own. So the tile is split as
// `padded tile -> (pad field, compact tile)` and the algebra runs on the compact tile.
// Without that split the smem vector would stop at the first padded row and the box would
// collapse to a single row.

#include "flydsl/Dialect/FlyROCDL/Utils/TdmGeometry.h"

#include "flydsl/Dialect/Fly/Utils/IntTupleUtils.h"
#include "flydsl/Dialect/Fly/Utils/LayoutUtils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <numeric>

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_rocdl::tdm {

namespace {

using TupleBuilder = IntTupleBuilder<IntTupleAttr>;
using AttrLayoutBuilder = LayoutBuilder<LayoutAttr>;

//===----------------------------------------------------------------------===//
// Nested int tuples, addressed by depth-first leaf index
//===----------------------------------------------------------------------===//

void flattenLeaves(IntTupleAttr t, SmallVectorImpl<IntTupleAttr> &out) {
  intTupleFlattenToVector(TupleBuilder(t.getContext()), t, out);
}

int32_t leafCount(IntTupleAttr t) {
  SmallVector<IntTupleAttr> leaves;
  flattenLeaves(t, leaves);
  return static_cast<int32_t>(leaves.size());
}

/// The depth-first leaf index a basis stride's mode path names, or -1 when the path does
/// not address a leaf of `profile`.
int32_t flatIndexOfPath(IntTupleAttr profile, ArrayRef<int32_t> path) {
  int32_t base = 0;
  IntTupleAttr cur = profile;
  for (int32_t step : path) {
    if (cur.isLeaf() || step < 0 || step >= cur.rank())
      return -1;
    for (int32_t i = 0; i < step; ++i)
      base += leafCount(cur.at(i));
    cur = cur.at(step);
  }
  return cur.isLeaf() ? base : -1;
}

/// An int tuple's elements are int tuples, never the bare `IntAttr` / `BasisAttr`
/// underneath them; wrap one that is bare, and pass a tuple through.
IntTupleAttr asLeaf(Attribute a) {
  if (auto tuple = dyn_cast<IntTupleAttr>(a))
    return tuple;
  return IntTupleAttr::get(a);
}

/// Rebuild a tuple shaped like `profile`, taking leaves from `flat` in depth-first order.
/// The leaf at index `splitLeaf` is replaced by `splitValue` instead, which makes the
/// result one level deeper there — that is how a recast's integer division is written
/// into the coordinate tensor's shape. `intTupleUnflatten` takes whatever attribute sits
/// at a profile leaf, so a two-element `splitValue` needs no special case.
IntTupleAttr unflattenLike(MLIRContext *ctx, IntTupleAttr profile, ArrayRef<Attribute> flat,
                           int32_t splitLeaf = -1, Attribute splitValue = nullptr) {
  SmallVector<Attribute> leaves;
  for (auto [i, elem] : llvm::enumerate(flat))
    leaves.push_back(static_cast<int32_t>(i) == splitLeaf ? splitValue : Attribute(asLeaf(elem)));
  return intTupleUnflatten(TupleBuilder(ctx), IntTupleAttr::get(ArrayAttr::get(ctx, leaves)),
                           profile);
}

/// A two-element int tuple, as an attribute usable as a leaf replacement.
Attribute makePair(MLIRContext *ctx, Attribute a, Attribute b) {
  return IntTupleAttr::get(ArrayAttr::get(ctx, {asLeaf(a), asLeaf(b)}));
}

//===----------------------------------------------------------------------===//
// LDS padding: split the tile into a pad field and the compact tile
//===----------------------------------------------------------------------===//

struct LdsEntry {
  int32_t leaf;
  int32_t extent;
  int32_t step;
};

} // namespace

FailureOr<std::tuple<int32_t, int32_t, IntTupleAttr>>
analyzeLdsTile(LayoutAttr smemLayout, function_ref<InFlightDiagnostic()> emitError) {
  MLIRContext *ctx = smemLayout.getContext();
  IntTupleAttr shape = smemLayout.getShape();
  IntTupleAttr stride = smemLayout.getStride();

  SmallVector<IntTupleAttr> shapeLeaves, strideLeaves;
  flattenLeaves(shape, shapeLeaves);
  flattenLeaves(stride, strideLeaves);
  if (shapeLeaves.size() != strideLeaves.size())
    return emitError() << "the LDS tile's shape and stride do not have the same profile";

  SmallVector<LdsEntry> entries;
  for (auto [i, extentLeaf, stepLeaf] : llvm::enumerate(shapeLeaves, strideLeaves)) {
    if (!extentLeaf.isLeafInt() || !extentLeaf.isStatic() || !stepLeaf.isLeafInt() ||
        !stepLeaf.isStatic())
      return emitError() << "the LDS tile must be fully static, mode " << i << " is not";
    int32_t extent = extentLeaf.getLeafAsInt().getValue();
    int32_t step = stepLeaf.getLeafAsInt().getValue();
    if (extent < 1)
      return emitError() << "LDS mode " << i << " has extent " << extent;
    if (extent > 1 && step == 0)
      return emitError() << "LDS mode " << i << " has extent " << extent
                         << " and stride 0; TDM walks LDS linearly and cannot broadcast one "
                            "location over several tile elements";
    entries.push_back({static_cast<int32_t>(i), extent, step});
  }

  // The modes are ordered by stride rather than assumed to run outermost-first, so a
  // column-major, permuted or hierarchical tile is as ordinary as a row-major one, and the
  // pad may sit at any level rather than only on the innermost row.
  SmallVector<LdsEntry> active;
  for (const LdsEntry &e : entries)
    if (e.extent > 1)
      active.push_back(e);
  llvm::stable_sort(active, [](const LdsEntry &a, const LdsEntry &b) { return a.step < b.step; });

  SmallVector<int32_t> compact(active.size(), 1);
  for (size_t k = 1; k < active.size(); ++k)
    compact[k] = compact[k - 1] * active[k - 1].extent;
  if (!active.empty() && active[0].step != 1)
    return emitError() << "the LDS tile must be contiguous along its fastest axis, but its "
                          "smallest stride is "
                       << active[0].step << " (mode " << active[0].leaf << ")";

  int32_t padInterval = 0, padAmount = 0;
  size_t split = active.size();
  for (size_t k = 0; k < active.size(); ++k)
    if (active[k].step != compact[k]) {
      split = k;
      break;
    }
  if (split < active.size()) {
    padAmount = active[split].step - compact[split];
    if (padAmount < 0)
      return emitError() << "LDS mode " << active[split].leaf << " has stride "
                         << active[split].step << ", smaller than the " << compact[split]
                         << " elements inside it; the tile overlaps itself";
    padInterval = compact[split];
    SmallVector<int32_t> padded(compact);
    padded[split] = active[split].step;
    for (size_t k = split + 1; k < active.size(); ++k)
      padded[k] = padded[k - 1] * active[k - 1].extent;
    for (size_t k = 0; k < active.size(); ++k)
      if (active[k].step != padded[k])
        return emitError() << "the LDS tile's stride (modes sorted by stride) is not a "
                              "single-pad form; TDM can express one constant skip every "
                              "pad_interval elements, so mode "
                           << active[k].leaf << " must have stride " << padded[k] << ", got "
                           << active[k].step;
  }

  // `compactStride` is the same layout with the skip taken back out, shaped like `shape`.
  // That is what the geometry algebra runs on.
  Attribute one = IntAttr::getStatic(ctx, 1);
  SmallVector<Attribute> flatStride(shapeLeaves.size(), one);
  for (auto [k, e] : llvm::enumerate(active))
    flatStride[e.leaf] = IntAttr::getStatic(ctx, compact[k]);
  return std::make_tuple(padInterval, padAmount, unflattenLike(ctx, shape, flatStride));
}

namespace {

//===----------------------------------------------------------------------===//
// The value map: the tensor's identity layout composed with the tiler
//===----------------------------------------------------------------------===//

/// `profile` with every extent replaced by 1.
Attribute onesLike(MLIRContext *ctx, IntTupleAttr profile) {
  if (profile.isLeaf())
    return IntAttr::getStatic(ctx, 1);
  SmallVector<Attribute> elems;
  for (int32_t i = 0; i < profile.rank(); ++i)
    elems.push_back(onesLike(ctx, profile.at(i)));
  return TileAttr::get(ArrayAttr::get(ctx, elems));
}

/// Right-pad a tiler with 1s until it has the tensor's profile.
///
/// Fly's composition keeps every mode a shorter tiler does not reach — at every nesting
/// level, not just the top — so the 1s are written out here: taking one element from a mode
/// is the same thing as not tiling it, and it is what keeps the value map's size equal to
/// the LDS tile's.
///
/// A scalar against a hierarchical mode is left alone: the tiler splits itself across the
/// sub-modes there, and composition already gets it right.
///
/// A `Tile`'s modes are bare attributes rather than a uniform tuple — an `IntAttr` for a
/// plain extent, a nested `TileAttr` for a hierarchical one, a `LayoutAttr` for a
/// strided tiler — so this walks `Attribute` and names what it cannot take.
FailureOr<Attribute> padTiler(MLIRContext *ctx, Attribute tiler, IntTupleAttr profile,
                              int32_t depth, function_ref<InFlightDiagnostic()> emitError) {
  if (auto extent = dyn_cast<IntAttr>(tiler)) {
    if (!extent.isStatic())
      return emitError() << "the tiler must be static, mode at nesting depth " << depth
                         << " is not";
    return Attribute(IntAttr::getStatic(ctx, extent.getValue()));
  }
  auto tile = dyn_cast<TileAttr>(tiler);
  if (!tile)
    return emitError() << "the tiler's mode at nesting depth " << depth
                       << " must be an extent or a nested tile, got " << tiler
                       << "; a strided (layout) tiler mode is not modelled";
  // A one-element `Tile` wraps its mode rather than nesting it.
  if (tile.isLeaf())
    return padTiler(ctx, tile.getValue(), profile, depth, emitError);
  if (profile.isLeaf())
    return emitError() << "the tiler is deeper than the tensor at nesting depth " << depth
                       << "; the tensor has a single extent there but the tiler has " << tile.rank()
                       << " sub-modes";
  if (tile.rank() > profile.rank())
    return emitError() << "the tiler has " << tile.rank() << " modes at nesting depth " << depth
                       << " but the tensor has " << profile.rank()
                       << "; a tiler selects from a mode's leading sub-modes";
  SmallVector<Attribute> elems;
  for (int32_t i = 0; i < profile.rank(); ++i) {
    if (i < tile.rank()) {
      FailureOr<Attribute> sub = padTiler(ctx, tile.at(i), profile.at(i), depth + 1, emitError);
      if (failed(sub))
        return failure();
      elems.push_back(*sub);
    } else {
      elems.push_back(onesLike(ctx, profile.at(i)));
    }
  }
  return Attribute(TileAttr::get(ArrayAttr::get(ctx, elems)));
}

} // namespace

FailureOr<LayoutAttr> makeValueMap(IntTupleAttr gshape, TileAttr tiler,
                                   function_ref<InFlightDiagnostic()> emitError) {
  MLIRContext *ctx = gshape.getContext();
  FailureOr<Attribute> padded = padTiler(ctx, tiler, gshape, 0, emitError);
  if (failed(padded))
    return failure();
  auto tileAttr = dyn_cast<TileAttr>(*padded);
  if (!tileAttr)
    tileAttr = TileAttr::get(*padded);

  AttrLayoutBuilder layoutBuilder(ctx);
  LayoutAttr identity = LayoutAttr::get(gshape, intTupleMakeBasisTupleLike(gshape));
  return layoutComposition(layoutBuilder, identity, tileAttr);
}

namespace {

//===----------------------------------------------------------------------===//
// The transfer box: which modes it spans, and how far
//===----------------------------------------------------------------------===//

/// One entry of the smem run: how many elements it takes, and from which global mode.
struct VectorEntry {
  int32_t extent;
  int32_t mode;
};

/// `coalesce(composition(valueMap, right_inverse(slayout)))`.
///
/// The vector is the smem run innermost first, truncated at the first mode whose basis
/// coefficient is not 1 (no starting in the middle of a global
/// mode). `nextAfterCut` is the mode the cut stopped on, or -1 when the vector runs to the
/// end — it matters because it is the mode the box is *adjacent to in LDS*, and hardware
/// iteration walks LDS with a constant increment of one box, so that mode and only that
/// mode can be folded into the instruction.
FailureOr<SmallVector<VectorEntry>> ldsRun(MLIRContext *ctx, LayoutAttr valueMap,
                                           IntTupleAttr gshape, IntTupleAttr smemShape,
                                           IntTupleAttr compactStride, int32_t &nextAfterCut,
                                           function_ref<InFlightDiagnostic()> emitError) {
  AttrLayoutBuilder layoutBuilder(ctx);
  LayoutAttr compactSmem = LayoutAttr::get(smemShape, compactStride);
  LayoutAttr invSmem = layoutRightInverse(layoutBuilder, compactSmem);
  LayoutAttr sidx2gmode =
      layoutCoalesce(layoutBuilder, layoutComposition(layoutBuilder, valueMap, invSmem));

  SmallVector<IntTupleAttr> extents, strides;
  flattenLeaves(sidx2gmode.getShape(), extents);
  flattenLeaves(sidx2gmode.getStride(), strides);

  SmallVector<VectorEntry> vector;
  nextAfterCut = -1;
  for (auto [extentLeaf, strideLeaf] : llvm::zip(extents, strides)) {
    if (!extentLeaf.isLeafInt() || !extentLeaf.isStatic())
      return emitError() << "the tile/global vectorization must be static";
    int32_t extent = extentLeaf.getLeafAsInt().getValue();
    int32_t coeff = 0, mode = -1;
    if (strideLeaf.isLeafBasis()) {
      BasisAttr basis = strideLeaf.getLeafAsBasis();
      if (!basis.getValue().isStatic())
        return emitError() << "a dynamic basis coefficient is not supported";
      coeff = basis.getValue().getValue();
      mode = flatIndexOfPath(gshape, basis.getModes());
    } else if (strideLeaf.isLeafInt() && strideLeaf.isStatic()) {
      // A stride-0 / stride-1 constant leaf carries no global mode.
      coeff = strideLeaf.getLeafAsInt().getValue();
    } else {
      return emitError() << "unsupported stride leaf in the tile/global vectorization";
    }
    if (coeff != 1 || mode < 0) {
      nextAfterCut = mode; // -1 when the leaf carried no mode at all
      break;               // stop at the first non-unit basis
    }
    vector.push_back({extent, mode});
  }
  if (vector.empty())
    return emitError() << "no common tile/global vectorization — the LDS tile and the global "
                          "tile do not share a contiguous innermost run. Does the tiler select "
                          "out the major global mode?";
  return vector;
}

/// One warp's share of the smem run: its leading `1 / numWarps`.
///
/// The whole run belongs to the workgroup; the box only has to be one participant's share
/// of it, and the participants take equal contiguous chunks. Every chunk is then the same
/// box translated along a single global mode, which is what lets one descriptor serve all
/// of them: the caller moves the origin, not the geometry.
FailureOr<SmallVector<VectorEntry>> splitAcrossWarps(ArrayRef<VectorEntry> vector, int32_t numWarps,
                                                     function_ref<InFlightDiagnostic()> emitError) {
  if (numWarps == 1)
    return SmallVector<VectorEntry>(vector);
  int32_t total = 1;
  for (const VectorEntry &e : vector)
    total *= e.extent;
  if (total % numWarps)
    return emitError() << numWarps << " warps cannot split this transfer's " << total
                       << "-element contiguous run evenly; num_warps has to divide it";
  int32_t want = total / numWarps;

  SmallVector<VectorEntry> out;
  int32_t acc = 1;
  for (const VectorEntry &e : vector) {
    int32_t room = want / acc;
    if (room == 1)
      break;
    if (e.extent <= room) {
      out.push_back(e);
      acc *= e.extent;
      continue;
    }
    if (e.extent % room)
      return emitError() << "splitting the transfer run " << numWarps << " ways cuts global mode "
                         << e.mode << " at " << room << " of the " << e.extent
                         << " elements the tile takes from it, which does not divide it";
    out.push_back({room, e.mode});
    break;
  }
  if (out.empty())
    out.push_back({1, vector.front().mode});
  return out;
}

//===----------------------------------------------------------------------===//
// The multi-mode gcd recurrence
//===----------------------------------------------------------------------===//

struct RawDim {
  int32_t box;
  SmallVector<int32_t> modes;
  Scalar stride;
};

/// The extent/stride recurrence for a dim fed by several global modes.
///
/// A descriptor dim fed by several global modes takes `gcd` of their strides and grows its
/// extent to span them: `g_shape = (s_i - 1) * (d_i / gcd) + ... + 1`. Only defined over
/// static values — a dynamic extent or stride has no build-time gcd — and the rank-5
/// packing is the only thing that builds such a dim, so a tensor whose tail it has to lump
/// together is refused here unless that tail's geometry is static.
FailureOr<std::pair<Scalar, Scalar>> foldModes(ArrayRef<int32_t> modes, ArrayRef<Scalar> modeExtent,
                                               ArrayRef<Scalar> modeStride,
                                               function_ref<InFlightDiagnostic()> emitError) {
  if (modes.size() == 1)
    return std::make_pair(modeExtent[modes[0]], modeStride[modes[0]]);

  // The recurrence below is compile-time `int32_t` arithmetic, so the packing is static
  // or it does not happen. That is the companion of the restriction in
  // `axisBasisPerMode`: see the TODO there for what lifting both would take.
  for (int32_t m : modes)
    if (!modeExtent[m].isStatic || !modeStride[m].isStatic)
      return emitError() << "a descriptor dim spans several global modes but mode " << m
                         << " has a dynamic extent or stride; a multi-mode dim needs static "
                            "geometry";
  int32_t stride = 0;
  for (int32_t m : modes)
    stride = std::gcd(stride, modeStride[m].value);
  if (stride == 0)
    return std::make_pair(modeExtent[modes[0]], Scalar::getStatic(0));
  int32_t extent = 1;
  for (int32_t m : modes)
    extent += (modeExtent[m].value - 1) * (modeStride[m].value / stride);
  return std::make_pair(Scalar::getStatic(extent), Scalar::getStatic(stride));
}

/// Divide a scalar by the recast ratio. A static value is checked; a dynamic one records
/// the divisor for the expansion, because the caller is asserting that its tensor is laid
/// out in whole internal units and a run-time value cannot be checked against that any more
/// than a static check can verify a dynamic value.
LogicalResult recastDivide(Scalar &s, int32_t ratio, const Twine &what,
                           function_ref<InFlightDiagnostic()> emitError) {
  if (!s.isStatic) {
    s.divisor *= ratio;
    return success();
  }
  if (s.value % ratio)
    return emitError() << what << " is " << s.value
                       << ", which is not divisible by the recast ratio " << ratio;
  s.value /= ratio;
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// derive
//===----------------------------------------------------------------------===//

FailureOr<Geometry> derive(const Request &request, function_ref<InFlightDiagnostic()> emitError) {
  LayoutAttr gLayout = request.gLayout;
  MLIRContext *ctx = gLayout.getContext();
  TupleBuilder tupleBuilder(ctx);

  IntTupleAttr gshape = gLayout.getShape();
  SmallVector<IntTupleAttr> gshapeLeaves, gstrideLeaves;
  flattenLeaves(gshape, gshapeLeaves);
  flattenLeaves(gLayout.getStride(), gstrideLeaves);
  if (gshapeLeaves.size() != gstrideLeaves.size())
    return emitError() << "the global tensor's shape and stride do not have the same profile";
  int32_t numModes = static_cast<int32_t>(gshapeLeaves.size());

  // Per global mode, its extent and stride — static leaves as values, dynamic ones as
  // references the expansion resolves against the layout operand.
  SmallVector<Scalar> modeExtent(numModes), modeStride(numModes);
  for (int32_t i = 0; i < numModes; ++i) {
    if (!gshapeLeaves[i].isLeafInt() || !gstrideLeaves[i].isLeafInt())
      return emitError() << "global mode " << i << " must have an integer extent and stride";
    modeExtent[i] = gshapeLeaves[i].isStatic()
                        ? Scalar::getStatic(gshapeLeaves[i].getLeafAsInt().getValue())
                        : Scalar::getDynamic(i, /*fromShape=*/true);
    modeStride[i] = gstrideLeaves[i].isStatic()
                        ? Scalar::getStatic(gstrideLeaves[i].getLeafAsInt().getValue())
                        : Scalar::getDynamic(i, /*fromShape=*/false);
  }

  // The single contiguous mode, recorded before the recast rewrites the strides.
  int32_t contiguousMode = -1;
  {
    int32_t found = 0;
    for (int32_t i = 0; i < numModes; ++i)
      if (modeStride[i].isStatic && modeStride[i].value == 1) {
        contiguousMode = i;
        ++found;
      }
    if (found != 1)
      contiguousMode = -1;
  }

  FailureOr<std::tuple<int32_t, int32_t, IntTupleAttr>> lds =
      analyzeLdsTile(request.smemLayout, emitError);
  if (failed(lds))
    return failure();

  int32_t padInterval = std::get<0>(*lds);
  int32_t padAmount = std::get<1>(*lds);
  IntTupleAttr compactStride = std::get<2>(*lds);

  LayoutAttr valueMap = request.valueMap;
  if (!valueMap)
    return emitError() << "the request carries no value map; build one from the tiler with "
                          "`makeValueMap`";

  // The tiler cuts the global tile that the LDS tile receives, so the two hold the same
  // elements. Only the
  // sizes are compared, so the tiler may be hierarchical where the LDS tile is flat.
  IntTupleAttr smemSizeAttr = intTupleProduct(tupleBuilder, request.smemLayout.getShape());
  IntTupleAttr vSizeAttr = intTupleProduct(tupleBuilder, valueMap.getShape());
  if (!smemSizeAttr.isStatic() || !vSizeAttr.isStatic())
    return emitError() << "the tiler and the LDS tile must both be static";

  int32_t smemSize = smemSizeAttr.getLeafAsInt().getValue();
  int32_t vSize = vSizeAttr.getLeafAsInt().getValue();
  if (smemSize != vSize)
    return emitError() << "the tiler holds " << vSize << " elements but the LDS tile holds "
                       << smemSize << "; a tiler reshapes the tile, it does not shrink it";

  // Per global mode, how many elements the tile takes and the step it takes them by. The
  // step is the basis coefficient: 1 for the ordinary case where the tile walks a mode
  // element by element, and more when it strides — `8:2E0` takes 8 rows two apart.
  SmallVector<int32_t> tileExtent(numModes, 1), tileStep(numModes, 1);
  {
    llvm::DenseSet<int32_t> tiled;
    SmallVector<IntTupleAttr> vExtents, vStrides;
    flattenLeaves(valueMap.getShape(), vExtents);
    flattenLeaves(valueMap.getStride(), vStrides);
    for (auto [extentLeaf, strideLeaf] : llvm::zip(vExtents, vStrides)) {
      if (!strideLeaf.isLeafBasis())
        continue;
      BasisAttr basis = strideLeaf.getLeafAsBasis();
      if (!basis.getValue().isStatic())
        return emitError() << "a dynamic basis coefficient is not supported in the tiler";
      int32_t mode = flatIndexOfPath(gshape, basis.getModes());
      if (mode < 0)
        continue;
      if (!extentLeaf.isLeafInt() || !extentLeaf.isStatic())
        return emitError() << "the tiler must be static";
      int32_t extent = extentLeaf.getLeafAsInt().getValue();
      int32_t coeff = basis.getValue().getValue();
      if (tiled.insert(mode).second) {
        tileExtent[mode] = extent;
        tileStep[mode] = coeff;
      } else {
        tileExtent[mode] *= extent;
        tileStep[mode] = std::min(tileStep[mode], coeff);
      }
    }
  }

  int32_t nextAfterCut = -1;
  FailureOr<SmallVector<VectorEntry>> vectorOr = ldsRun(
      ctx, valueMap, gshape, request.smemLayout.getShape(), compactStride, nextAfterCut, emitError);
  if (failed(vectorOr))
    return failure();
  SmallVector<VectorEntry> vector = *vectorOr;

  // How much of each mode the box covers, counted before the recast rewrites the run in
  // wider units — `tileExtent` is in the tensor's own elements and the two are compared
  // below.
  SmallVector<int32_t> coveredExtent(numModes, 1);
  for (const VectorEntry &e : vector)
    coveredExtent[e.mode] *= e.extent;

  // Cut before the recast, so the other warps' share turns into
  // neither a wider unit nor an iteration — it is simply not this box.
  FailureOr<SmallVector<VectorEntry>> splitOr =
      splitAcrossWarps(vector, request.numWarps, emitError);
  if (failed(splitOr))
    return failure();
  vector = *splitOr;

  // The recast: view the transfer in a wider unit. The innermost run,
  // the contiguous global mode's extent, and every global stride are divided by the ratio.
  // This is what lets a sub-byte or awkward element type ride on a TDM data size the
  // hardware can encode.
  if (request.elemBits <= 0 || request.internalBits % request.elemBits != 0)
    return emitError() << "internal width " << request.internalBits
                       << " is not a multiple of element width " << request.elemBits;
  int32_t ratio = request.internalBits / request.elemBits;
  if (ratio != 1) {
    if (vector.front().extent % ratio)
      return emitError() << "the innermost run is " << vector.front().extent
                         << ", which is not divisible by the recast ratio " << ratio;
    vector.front().extent /= ratio;
    for (int32_t i = 0; i < numModes; ++i) {
      if (modeStride[i].isStatic && modeStride[i].value == 0)
        continue;
      if (modeStride[i].isStatic && modeStride[i].value == 1) {
        // The contiguous mode is the one measured in elements; recast shrinks it.
        if (failed(recastDivide(modeExtent[i], ratio,
                                "the extent of the contiguous global mode " + Twine(i), emitError)))
          return failure();
      } else if (failed(recastDivide(modeStride[i], ratio, "the stride of global mode " + Twine(i),
                                     emitError))) {
        return failure();
      }
    }
    // The pad fields ride in the same units as the box (the lowering rebuilds the LDS pitch
    // as `tileShape[-1] + padAmount`), so a recast rescales them too.
    if (padAmount) {
      if (padInterval % ratio || padAmount % ratio)
        return emitError() << "LDS padding (" << padInterval << ", " << padAmount
                           << ") is not divisible by the recast ratio " << ratio
                           << "; the pad must be a whole number of internal units";
      padInterval /= ratio;
      padAmount /= ratio;
    }
  }

  // The box need not span the whole tile. What the cut above leaves behind is not an
  // error: a TiledCopy says how many values one *tiled*
  // operation covers, the atom says how many one *call* covers, and the V mode carries the
  // difference as more calls.
  //
  // What TDM adds is one optimization on top: `iterate_enable` replays a single descriptor
  // `iterate_count` times, advancing global and LDS by a constant increment each time. That
  // folds *one* of those axes back into the instruction. Which one is not a free choice —
  // the LDS increment is one whole box, so it has to be the mode the box is adjacent to in
  // LDS, which is exactly the mode the cut stopped on. Hardware iteration also
  // steps LDS by one box, which is the other warps' space once the box is only a share of
  // the tile, so a split gives the axis up.
  llvm::DenseSet<int32_t> coveredModes;
  for (const VectorEntry &e : vector)
    coveredModes.insert(e.mode);

  int32_t iterCount = 1, iterMode = -1;
  Scalar iterStride;
  if (request.numWarps == 1 && nextAfterCut >= 0 &&
      tileExtent[nextAfterCut] > coveredExtent[nextAfterCut]) {
    int32_t count = tileExtent[nextAfterCut] / coveredExtent[nextAfterCut];
    if (count <= kMaxIterateCount && !padAmount) {
      Scalar step = modeStride[nextAfterCut];
      if (!step.isStatic && tileStep[nextAfterCut] != 1) {
        // The step would need a run-time multiply the expansion does not model; keep the
        // axis in the V mode instead, which is always correct and only costs instructions.
      } else {
        iterMode = nextAfterCut;
        iterCount = count;
        iterStride = step.isStatic ? Scalar::getStatic(step.value * tileStep[nextAfterCut]) : step;
      }
    }
  }

  SmallVector<RawDim> dims;
  for (const VectorEntry &e : vector)
    dims.push_back({e.extent, {e.mode}, modeStride[e.mode]});
  // A tile extent of one still needs a descriptor coordinate when the global tensor can
  // move on that mode. Without this filler, changing a batch index leaves global_addr at
  // batch zero.
  for (int32_t i = 0; i < numModes; ++i) {
    if (coveredModes.contains(i))
      continue;
    if (modeExtent[i].isStatic && modeExtent[i].value == 1)
      continue;
    if (modeStride[i].isStatic && modeStride[i].value == 0)
      continue;
    dims.push_back({1, {i}, modeStride[i]});
  }

  // The lowering hardwires descriptor dim 0's stride to 1, so dim 0 must be the contiguous
  // global mode. It can legitimately be missing: `coalesce` drops a size-1 mode from the
  // smem vector, so a tile that is one element wide along the contiguous mode loses it. Put
  // it back as an extent-1 dim instead of rejecting the copy.
  if (!dims.front().stride.isStatic || dims.front().stride.value != 1) {
    llvm::DenseSet<int32_t> seen;
    for (const RawDim &d : dims)
      for (int32_t m : d.modes)
        seen.insert(m);
    if (contiguousMode >= 0 && !seen.contains(contiguousMode) && tileExtent[contiguousMode] == 1)
      dims.insert(dims.begin(), {1, {contiguousMode}, Scalar::getStatic(1)});
    else
      return emitError() << "the innermost descriptor dim must be contiguous in global memory "
                            "(stride 1), got global mode "
                         << dims.front().modes.front()
                         << " — the LDS tile's majorness does not match the global tensor's";
  }

  // Every mode the box spans keeps its own descriptor dim: two dims that happen to be
  // adjacent in global memory are not folded into one, so a dim's bound is always a
  // per-mode rectangular bound and the descriptor says exactly what the tile said.
  SmallVector<bool> clampable(dims.size(), true);

  // Past five dims the trailing modes are packed into the
  // last descriptor dim, whose extent and stride come from a gcd recurrence over modes that
  // are unrelated in memory. That is the one dim whose single bound cannot speak for the
  // modes under it, so it is marked unclampable and the boundary-check state refuses to switch it
  // on.
  if (static_cast<int32_t>(dims.size()) > kMaxRank) {
    SmallVector<RawDim> head(dims.begin(), dims.begin() + (kMaxRank - 1));
    SmallVector<int32_t> packedModes;
    int32_t packedBox = 1;
    for (const RawDim &d : llvm::drop_begin(dims, kMaxRank - 1)) {
      packedModes.append(d.modes.begin(), d.modes.end());
      packedBox *= d.box;
    }
    FailureOr<std::pair<Scalar, Scalar>> packed =
        foldModes(packedModes, modeExtent, modeStride, emitError);
    if (failed(packed))
      return failure();
    head.push_back({packedBox, packedModes, packed->second});
    dims = head;
    clampable.truncate(kMaxRank - 1);
    clampable.push_back(false);
  }

  // TDM writes LDS strictly linearly -- the address only ever advances, across all the
  // dim loops -- and the pad counter runs with it rather than restarting per row. So a
  // skip every N elements is expressible for any N the tile is a whole number of, and
  // the interval is free to span several rows. It has to divide the *whole* box because
  // each instruction re-seeds that counter: a tile issued as several calls would
  // otherwise put the later calls' holes in the wrong place.
  if (padAmount) {
    int64_t boxElems = 1;
    for (const RawDim &d : dims)
      boxElems *= d.box;
    if (boxElems % padInterval != 0)
      return emitError() << "the padded LDS tile skips every " << padInterval
                         << " elements, which does not divide the descriptor's " << boxElems
                         << "-element box; one call's holes would fall in the wrong place";
  }

  Geometry geometry;
  geometry.padInterval = padInterval;
  geometry.padAmount = padAmount;
  geometry.ratio = ratio;
  geometry.modeExtent = modeExtent;
  geometry.modeStride = modeStride;
  geometry.contiguousMode = contiguousMode;
  for (auto [d, raw] : llvm::enumerate(dims)) {
    FailureOr<std::pair<Scalar, Scalar>> folded =
        foldModes(raw.modes, modeExtent, modeStride, emitError);
    if (failed(folded))
      return failure();
    if (folded->second.isStatic && (folded->second.value < 0 ||
                                    static_cast<uint64_t>(folded->second.value) > kMaxTensorStride))
      return emitError() << "descriptor dim " << d << " stride " << folded->second.value
                         << " is outside the unsigned 48-bit tensor_dim_stride range";
    Dim dim;
    dim.box = raw.box;
    dim.modes = raw.modes;
    dim.tensorDim = folded->first;
    dim.stride = folded->second;
    dim.clampable = clampable[d];
    geometry.dims.push_back(dim);
  }

  // Iteration is paid for out of the descriptor's own slots (`tdm::kMaxIterateRank`), so a
  // descriptor that needs them keeps its residual axis in the V mode instead. Declining is
  // not a failure: the copy still moves the whole tile, it just spends one more instruction
  // per step.
  if (iterCount > 1 && geometry.rank() > kMaxIterateRank)
    iterCount = 1;
  if (iterCount > 1) {
    geometry.iterCount = iterCount;
    geometry.iterStride = iterStride;
    geometry.iterMode = iterMode;
    // The residual axis is stepped by the hardware, not by a `tensor_dim`, so whichever dim
    // carries its tile origin has no bound to give.
    for (Dim &dim : geometry.dims)
      if (llvm::is_contained(dim.modes, iterMode))
        dim.clampable = false;
  }

  return geometry;
}

FailureOr<IntTupleAttr> initialBoundaryCheck(const Geometry &geometry, IntTupleAttr gshape,
                                             bool enable,
                                             function_ref<InFlightDiagnostic()> emitError) {
  MLIRContext *ctx = gshape.getContext();
  FailureOr<IntTupleAttr> basis = makeTensor2Tdm(geometry, gshape, emitError);
  if (failed(basis))
    return failure();

  // `makeTensor2Tdm` already answers "does this mode have a bound to give": a basis
  // leaf if it does, a `0` if it does not. So the initial state is that map with the
  // caller's one flag written onto the modes that can carry it.
  SmallVector<IntTupleAttr> basisLeaves;
  flattenLeaves(*basis, basisLeaves);
  SmallVector<Attribute> flat;
  for (IntTupleAttr basisLeaf : basisLeaves)
    flat.push_back(IntAttr::getStatic(ctx, enable && basisLeaf.isLeafBasis() ? 1 : 0));
  return unflattenLike(ctx, gshape, flat);
}

//===----------------------------------------------------------------------===//
// The inverse of the mode map: per global mode, the descriptor axis it moves along
//===----------------------------------------------------------------------===//

namespace {

/// Per global mode, the descriptor axis it moves along, as a basis leaf (`0` = none).
///
/// Shared by the coordinate tensor's strides and by the atom's boundary-check mode map,
/// which differ only in whether a dim the descriptor cannot put a
/// single bound on still contributes: a coordinate on such a dim is still a coordinate, a
/// bound on it would not be one.
FailureOr<SmallVector<Attribute>> axisBasisPerMode(MLIRContext *ctx, const Geometry &geometry,
                                                   IntTupleAttr gshape, bool clampableOnly,
                                                   function_ref<InFlightDiagnostic()> emitError) {
  SmallVector<IntTupleAttr> gshapeLeaves;
  flattenLeaves(gshape, gshapeLeaves);
  int32_t numModes = static_cast<int32_t>(gshapeLeaves.size());
  if (numModes != static_cast<int32_t>(geometry.modeStride.size()))
    return emitError() << "internal: the coordinate tensor's shape has " << numModes
                       << " modes but the derivation recorded " << geometry.modeStride.size();

  Attribute zero = IntAttr::getStatic(ctx, 0);
  SmallVector<Attribute> byMode(numModes, zero);

  int32_t descRank = geometry.rank();
  for (auto [descDim, dim] : llvm::enumerate(geometry.dims)) {
    if (clampableOnly && !dim.clampable)
      continue;
    // Axis indices are in tensor dim order, the reverse of descriptor order, matching the
    // atom's `tileShape` and its state slots.
    int32_t axis = descRank - 1 - static_cast<int32_t>(descDim);
    for (int32_t mode : dim.modes) {
      // A size-1 axis has only coordinate 0 and a stride-0 one is a
      // broadcast; neither can move the tile, so neither gets a basis.
      const Scalar &extent = geometry.modeExtent[mode];
      const Scalar &stride = geometry.modeStride[mode];
      if (extent.isStatic && extent.value == 1)
        continue;
      if (stride.isStatic && stride.value == 0)
        continue;
      if (dim.modes.size() == 1) {
        byMode[mode] = BasisAttr::get(IntAttr::getStatic(ctx, 1), axis);
        continue;
      }
      // A dim covering several modes only ever comes out of the rank-5 packing. Each of
      // those modes rides the shared axis at its own scale, `mode_stride / dim_stride`,
      // and the map has nowhere to put a scale that is not a compile-time integer: a
      // `tensor2tdm` leaf is a `BasisAttr`, so the whole 1-to-many mapping has to be
      // static or it is not expressible.
      //
      // TODO(rank>5, dynamic strides): a tensor with more than five modes and dynamic
      // strides is refused right here, need to support tensor2tdm with dynamic basis strides.
      if (!stride.isStatic || !dim.stride.isStatic || dim.stride.value == 0)
        return emitError() << "descriptor dim " << descDim << " covers global modes but mode "
                           << mode
                           << " has a dynamic stride; a shared axis needs static strides to "
                              "scale its coordinate";
      if (stride.value % dim.stride.value)
        return emitError() << "global mode " << mode << " has stride " << stride.value
                           << ", which is not a multiple of the descriptor dim " << descDim
                           << " stride " << dim.stride.value
                           << "; its coordinate cannot be expressed on that axis";
      byMode[mode] = BasisAttr::get(IntAttr::getStatic(ctx, stride.value / dim.stride.value), axis);
    }
  }
  return byMode;
}

} // namespace

FailureOr<IntTupleAttr> makeTensor2Tdm(const Geometry &geometry, IntTupleAttr gshape,
                                       function_ref<InFlightDiagnostic()> emitError) {
  MLIRContext *ctx = gshape.getContext();
  FailureOr<SmallVector<Attribute>> byMode =
      axisBasisPerMode(ctx, geometry, gshape, /*clampableOnly=*/true, emitError);
  if (failed(byMode))
    return failure();
  return unflattenLike(ctx, gshape, *byMode);
}

void boundaryCheckAxes(IntTupleAttr tensor2tdm, SmallVectorImpl<int32_t> &axes) {
  SmallVector<IntTupleAttr> leaves;
  flattenLeaves(tensor2tdm, leaves);
  for (IntTupleAttr leaf : leaves)
    axes.push_back(leaf.isLeafBasis() ? leaf.getLeafAsBasis().getModes().front() : -1);
}

FailureOr<LayoutAttr> coordLayout(const Geometry &geometry, IntTupleAttr gshape,
                                  function_ref<InFlightDiagnostic()> emitError) {
  MLIRContext *ctx = gshape.getContext();
  FailureOr<SmallVector<Attribute>> byMode =
      axisBasisPerMode(ctx, geometry, gshape, /*clampableOnly=*/false, emitError);
  if (failed(byMode))
    return failure();

  if (geometry.ratio == 1)
    return LayoutAttr::get(gshape, unflattenLike(ctx, gshape, *byMode));

  // Under a recast the descriptor counts in wider units than the tensor does, so the
  // contiguous mode's coordinate has to be divided by the ratio — and a rational basis
  // scale is not something a Fly basis can hold (its coefficient is an
  // integer). The shape carries it instead: that mode becomes `(ratio, N / ratio)` with
  // strides `(0, 1E<axis>)`, so a logical coordinate `n` decomposes into
  // `(n % ratio, n / ratio)` and only the second half reaches the axis. Integer division,
  // done by the layout algebra, invisible to the kernel — which goes on tiling in tensor
  // elements.
  if (geometry.contiguousMode < 0)
    return emitError() << "a recast needs exactly one contiguous global mode to divide the "
                          "coordinate on, found none";
  int32_t split = geometry.contiguousMode;
  const Scalar &recastExtent = geometry.modeExtent[split];
  Attribute extentAttr = recastExtent.isStatic
                             ? Attribute(IntAttr::getStatic(ctx, recastExtent.value))
                             : Attribute(IntAttr::getDynamic(ctx));
  SmallVector<IntTupleAttr> gshapeLeaves;
  flattenLeaves(gshape, gshapeLeaves);
  SmallVector<Attribute> flatShape;
  for (IntTupleAttr leaf : gshapeLeaves)
    flatShape.push_back(leaf.getValue());

  IntTupleAttr shape =
      unflattenLike(ctx, gshape, flatShape, split,
                    makePair(ctx, IntAttr::getStatic(ctx, geometry.ratio), extentAttr));
  IntTupleAttr stride = unflattenLike(ctx, gshape, *byMode, split,
                                      makePair(ctx, IntAttr::getStatic(ctx, 0), (*byMode)[split]));
  return LayoutAttr::get(shape, stride);
}

SmallVector<int32_t> tileShape(const Geometry &geometry) {
  SmallVector<int32_t> shape;
  for (const Dim &dim : llvm::reverse(geometry.dims))
    shape.push_back(dim.box);
  return shape;
}

//===----------------------------------------------------------------------===//
// tdm_partition
//===----------------------------------------------------------------------===//

namespace {

/// The product of a fully static int tuple's leaves, or failure if any is not.
FailureOr<int64_t> staticSize(IntTupleAttr t, const Twine &what,
                              function_ref<InFlightDiagnostic()> emitError) {
  SmallVector<IntTupleAttr> leaves;
  flattenLeaves(t, leaves);
  int64_t n = 1;
  for (IntTupleAttr leaf : leaves) {
    if (!leaf.isLeafInt() || !leaf.isStatic())
      return emitError() << what << " must be a static int tuple, got a dynamic or basis leaf";
    n *= leaf.getLeafAsInt().getValue();
  }
  return n;
}

} // namespace

FailureOr<LayoutAttr> partitionLayout(IntTupleAttr atomValShape, int32_t atomValBits,
                                      LayoutAttr smemLayout, int32_t ldsElemBits,
                                      IntTupleAttr coordShape, int32_t numWarps,
                                      function_ref<InFlightDiagnostic()> emitError) {
  MLIRContext *ctx = smemLayout.getContext();
  AttrLayoutBuilder layoutBuilder(ctx);

  // Only mode-0 -- the TDM box one call fills -- is related. The remaining "rest" modes
  // (e.g. PIPE on the LDS side, k-tiles on the coordinate side) pass through untouched and
  // need not agree in size, so cut both operands by their mode-0 only.
  LayoutAttr smemBox = smemLayout.rank() > 1 ? smemLayout.at(0) : smemLayout;
  IntTupleAttr coordBox = coordShape.rank() > 1 ? coordShape.at(0) : coordShape;

  FailureOr<int64_t> vSize =
      staticSize(smemBox.getShape(), "the LDS tile's mode-0 shape", emitError);
  if (failed(vSize))
    return failure();
  FailureOr<int64_t> gSize = staticSize(coordBox, "the coordinate tile's mode-0 shape", emitError);
  if (failed(gSize))
    return failure();
  if (*vSize != *gSize)
    return emitError() << "the LDS tile's mode-0 holds " << *vSize << " values but the "
                       << "coordinate tile's mode-0 holds " << *gSize
                       << "; the two are cut by the same mode-0 layout, so they must agree";
  if (numWarps < 1)
    return emitError() << "the warp layout must have a positive size, got " << numWarps;
  if (*vSize % numWarps)
    return emitError() << numWarps << " warps do not divide the tile's " << *vSize
                       << " values evenly";
  int64_t want = *vSize / numWarps;

  // Tensor elements one atom *call* moves. The atom counts in its own `val_bits`, which a
  // recast makes wider than the tile's element, while the tile
  // counts in its own. TDM's two sides carry the same value layout -- one instruction
  // moves one whole box either way -- so which of src / dst is read does not matter.
  FailureOr<int64_t> vals = staticSize(atomValShape, "the atom's value mode", emitError);
  if (failed(vals))
    return failure();
  int64_t bits = *vals * atomValBits;
  if (ldsElemBits <= 0 || bits % ldsElemBits)
    return emitError() << "one call moves " << bits
                       << " bits, which is not a whole number of the LDS tile's " << ldsElemBits
                       << "-bit elements";
  int64_t numElems = bits / ldsElemBits;
  if (numElems <= 0 || want % numElems)
    return emitError() << "each of the " << numWarps << " warps takes " << want
                       << " values but one call moves " << numElems << ", which does not divide it";

  // The inverse of the compact tile, composed directly: the pad has already been taken
  // back out, so it covers the tile exactly and needs no tiling up to the tile's size.
  FailureOr<std::tuple<int32_t, int32_t, IntTupleAttr>> lds = analyzeLdsTile(smemBox, emitError);
  if (failed(lds))
    return failure();
  LayoutAttr invSmem =
      layoutRightInverse(layoutBuilder, LayoutAttr::get(smemBox.getShape(), std::get<2>(*lds)));

  auto flat = [&](int64_t extent, int64_t stride) {
    return LayoutAttr::get(IntTupleAttr::get(IntAttr::getStatic(ctx, extent)),
                           IntTupleAttr::get(IntAttr::getStatic(ctx, stride)));
  };
  LayoutAttr layoutV = layoutComposition(layoutBuilder, invSmem, flat(numElems, 1));
  LayoutAttr layoutIter =
      layoutComposition(layoutBuilder, invSmem, flat(want / numElems, numElems));

  SmallVector<Attribute> shapes{layoutV.getShape()};
  SmallVector<Attribute> strides{layoutV.getStride()};
  if (numWarps > 1) {
    // `((ATOM), (WARP), (ITER))` before the warp coordinate is sliced out: the
    // warps take equal contiguous chunks of the LDS order, so warp `w` starts at
    // `w * want` -- the stride of the WARP mode.
    LayoutAttr layoutWarp = layoutComposition(layoutBuilder, invSmem, flat(numWarps, want));
    shapes.push_back(layoutWarp.getShape());
    strides.push_back(layoutWarp.getStride());
  }
  shapes.push_back(layoutIter.getShape());
  strides.push_back(layoutIter.getStride());
  return LayoutAttr::get(IntTupleAttr::get(ArrayAttr::get(ctx, shapes)),
                         IntTupleAttr::get(ArrayAttr::get(ctx, strides)));
}

} // namespace mlir::fly_rocdl::tdm
