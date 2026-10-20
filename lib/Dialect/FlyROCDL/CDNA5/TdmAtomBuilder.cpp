// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#include "flydsl/Dialect/FlyROCDL/Utils/TdmAtomBuilder.h"
#include "flydsl/Core/Trait.hpp"
#include "flydsl/Dialect/Fly/Utils/CoreBridge.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <numeric>

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_rocdl {
namespace tdm {
namespace {

using EmitErrorFn = function_ref<InFlightDiagnostic()>;
using Tuple = core::IntTuple<core::Trait>;
using TupleRef = core::IntTupleRef<core::Trait>;
using Layout = core::Layout<core::Trait>;
using Leaf = core::Leaf<core::Trait>;
using Tile = core::Tile<core::Trait>;

struct TdmTypeParams {
  int32_t elemBits = 8;     ///< the tensor's element
  int32_t internalBits = 8; ///< the descriptor's element, `internal_type`
  int32_t numWarps = 1;
};

struct TdmAuxParams {
  Layout gbasis;           ///< descriptor box -> original global modes (TmaGmemBasis)
  bool packed = false;     ///< the trailing global modes share the last axis
  int32_t padInterval = 0; ///< LDS padding, in internal descriptor elements
  int32_t padAmount = 0;
  SmallVector<int32_t, 5> boxShape; ///< per-warp box, outermost first
  Tuple tensor2tdm;                 ///< tensor modes -> reversed state axes; 0 means no bound
  int32_t rank() const { return gbasis.rank(); }
};

template <class Cat> struct TdmCopyDesc {
  core::Pointer<Cat> gmemAddress; ///< the tensor origin
  core::Layout<Cat> gmemProb;     ///< descriptor extents:strides, innermost axis first
  core::IntTuple<Cat> gStride;    ///< original global profile -> descriptor coordinates
};

template <class Cat> struct TdmCopyAtom {
  TdmAuxParams aux;
  TdmCopyDesc<Cat> desc;
  core::CoordTensor<Cat> tdmTensor; ///< get_tma_tensor
};

/// `profile` with every extent replaced by 1.
Tile ones_like(TupleRef profile) {
  if (profile.isLeaf())
    return Tile(Leaf::getSInt(1));
  SmallVector<Tile> elems;
  for (auto child : profile)
    elems.push_back(ones_like(child));
  return Tile::getTuple(elems);
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
/// A tile mode is a plain extent, a nested tile for a hierarchical mode, a layout for
/// a strided tiler, or None; this names the ones it cannot take.
FailureOr<Tile> pad_tiler(MLIRContext *ctx, const Tile &tile, TupleRef profile, int32_t depth,
                          EmitErrorFn emitError) {
  if (tile.isScalar() || tile.isNone()) {
    auto extent = tile.isNone() ? Leaf::getNone() : tile.getScalar();
    if (extent.isNone() || extent.isBasis() || extent.isRatio())
      return emitError() << "the tiler must have integer extents";
    if (!extent.isStatic())
      return emitError() << "the tiler must be static, mode at nesting depth " << depth
                         << " is not";
    return Tile(Leaf::getSInt(extent.staticValue()));
  }
  if (tile.isLayout())
    return emitError() << "the tiler's mode at nesting depth " << depth
                       << " must be an extent or a nested tile, got " << TileAttr::get(ctx, tile)
                       << "; a strided (layout) tiler mode is not modelled";
  if (profile.isLeaf())
    return emitError() << "the tiler is deeper than the tensor at nesting depth " << depth
                       << "; the tensor has a single extent there but the tiler has " << tile.rank()
                       << " sub-modes";
  if (tile.rank() > profile.rank())
    return emitError() << "the tiler has " << tile.rank() << " modes at nesting depth " << depth
                       << " but the tensor has " << profile.rank()
                       << "; a tiler selects from a mode's leading sub-modes";
  SmallVector<Tile> elems;
  for (int32_t i = 0; i < profile.rank(); ++i) {
    if (i < tile.rank()) {
      auto sub = pad_tiler(ctx, tile.at(i), profile.at(i), depth + 1, emitError);
      if (failed(sub))
        return failure();
      elems.push_back(std::move(*sub));
    } else {
      elems.push_back(ones_like(profile.at(i)));
    }
  }
  return Tile::getTuple(elems);
}

/// `make_identity_layout(shape(gtensor)).compose(cta_tiler)`: tile values -> global
/// modes, with the tiler right-padded to the tensor's profile.
FailureOr<Layout> make_tdm_value_map(MLIRContext *ctx, TupleRef gshape, const Tile &ctaTiler,
                                     EmitErrorFn emitError) {
  auto padded = pad_tiler(ctx, ctaTiler, gshape, 0, emitError);
  if (failed(padded))
    return failure();
  auto identity = core::make_identity_layout(gshape);
  // A single-mode tiler canonicalizes to a scalar. Tile composition indexes
  // tuple modes, so use ordinary layout composition for this scalar case.
  if (padded->isScalar())
    return core::composition(identity, core::make_layout(Tuple::fromLeaf(padded->getScalar())));
  return core::composition(identity, *padded);
}

LogicalResult check_tdm_gtensor(const Layout &gtensor, const Layout &slayout, const Layout &ctaVMap,
                                const TdmTypeParams &params, EmitErrorFn emitError) {
  if (params.elemBits <= 0 || params.internalBits <= 0)
    return emitError() << "TDM element widths must be positive";
  int32_t factor = params.internalBits / std::gcd(params.elemBits, params.internalBits);
  for (auto [i, s, d] : llvm::enumerate(gtensor.shape().leaves(), gtensor.stride().leaves())) {
    if (!s.isInt() || !d.isInt())
      return emitError() << "global mode " << i << " must have an integer extent and stride";
    if (s.isSInt() && s.staticValue() < 1)
      return emitError() << "global mode " << i << " has extent " << s.staticValue()
                         << "; TDM cannot describe an empty tensor";
    // A mode of extent 1 only has coordinate 0, so its stride never reaches an address.
    if (s.isOne())
      continue;
    if (d.isSInt() && d.staticValue() < 0)
      return emitError() << "global mode " << i << " has stride " << d.staticValue()
                         << "; TDM descriptor strides are unsigned";
    // Recasting may not round a row's extent or stride and access extra data.
    auto aligned = d.isOne() ? s : d;
    StringRef what = d.isOne() ? "extent" : "stride";
    if (aligned.isSInt() && aligned.staticValue() % factor)
      return emitError() << "internal_type packs " << factor
                         << " tensor elements into one, so global mode " << i << "'s " << what
                         << " must be a multiple of " << factor << ", got "
                         << aligned.staticValue();
    if (!aligned.isSInt() && aligned.divisibility() % factor)
      return emitError() << "internal_type packs " << factor
                         << " tensor elements into one, so global mode " << i << "'s dynamic "
                         << what << " must be known divisible by " << factor << ", but it is only "
                         << "known divisible by " << aligned.divisibility()
                         << "; annotate it in the layout, e.g. ?{div=" << factor << "}";
  }
  auto smemSize = slayout.size();
  auto valueSize = ctaVMap.size();
  if (!smemSize.isSInt() || !valueSize.isSInt())
    return emitError() << "the tiler and the LDS tile must both be static";
  if (smemSize != valueSize)
    return emitError() << "the tiler holds " << valueSize.staticValue()
                       << " elements but the LDS tile holds " << smemSize.staticValue()
                       << "; a tiler reshapes the tile, it does not shrink it";
  return success();
}

struct LdsEntry {
  int32_t leaf;
  int32_t extent;
  int32_t step;
};

/// The LDS tile with its padding taken out, and the padding as a descriptor field.
struct LdsTile {
  Layout compact; ///< same shape, the skip removed from the stride
  int32_t padInterval = 0;
  int32_t padAmount = 0;
};

/// TDM expresses a skip of `padAmount` elements every `padInterval` elements, and the algebra runs
/// on the tile with that skip taken back out.
FailureOr<LdsTile> get_nonpad_portion(const Layout &slayout, EmitErrorFn emitError) {
  auto shapeLeaves = slayout.shape().leaves();
  auto strideLeaves = slayout.stride().leaves();
  if (shapeLeaves.size() != strideLeaves.size())
    return emitError() << "the LDS tile's shape and stride do not have the same profile";

  SmallVector<LdsEntry> entries;
  for (auto [i, extentLeaf, stepLeaf] : llvm::enumerate(shapeLeaves, strideLeaves)) {
    if (!extentLeaf.isSInt() || !stepLeaf.isSInt())
      return emitError() << "the LDS tile must be fully static, mode " << i << " is not";
    int32_t extent = extentLeaf.staticValue();
    int32_t step = stepLeaf.staticValue();
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

  SmallVector<Leaf> compactStride(shapeLeaves.size(), Leaf::getOne());
  for (auto [k, e] : llvm::enumerate(active))
    compactStride[e.leaf] = Leaf::getSInt(compact[k]);
  return LdsTile{
      core::make_layout(slayout.shape(), Tuple::fromParts(slayout.shape().nodes(), compactStride)),
      padInterval, padAmount};
}

/// greedily merge each mode into the back of the result while the merged extent stays within
/// kMaxCoalescedTileDim. The merge itself is ordinary layout coalescing, and walking innermost
/// first makes the limit deterministic.
Layout coalesce_tdm_box(const Layout &layout) {
  auto result = core::make_layout(Tuple::getEmpty(), Tuple::getEmpty());
  auto flat = core::flatten(layout);
  for (int32_t i = 0; i < flat.rank(); ++i) {
    auto mode = core::get(flat, i);
    // shape<I>(layout) == _1, skip it and continue
    if (mode.size().isOne())
      continue;
    if (result.rank() > 0) {
      // Merge modes because the shapes and strides match and the merge is within the limit
      auto last = core::get(result, result.rank() - 1);
      if (last.size().staticValue() * mode.size().staticValue() <= kMaxCoalescedTileDim) {
        auto merged = core::coalesce(core::make_layout(last, mode));
        if (merged.rank() == 1) {
          result = core::replace(result, result.rank() - 1, merged);
          continue;
        }
      }
    }
    // Can't replace or merge, so append a new mode
    result = core::append(result, mode);
  }
  return result;
}

struct TdmGbasis {
  Layout gbasis;
  bool packed;
};

/// CuTe's construct_tma_gbasis, step for step: invert the LDS tile, trace its contiguous
/// run back through the global modes, coalesce it to the box limit, and add every global
/// mode the box does not span with extent 1. TDM additionally needs a contiguous axis 0,
/// and folds a rank above five into an axis 4 of extent 1 whose traversal stays in ITER.
FailureOr<TdmGbasis> construct_tdm_gbasis(const Layout &gtensor, const Layout &slayout,
                                          const Layout &ctaVMap, const TdmTypeParams &params,
                                          EmitErrorFn emitError) {
  auto gbasis = core::make_identity_layout(gtensor.shape());
  auto globalBases = gbasis.stride().leaves();
  // Also validate residual modes, which do not participate in this descriptor.
  for (auto stride : ctaVMap.stride().leaves())
    if (stride.isBasis() && !stride.scale().isStatic())
      return emitError() << "a dynamic basis coefficient is not supported in the tiler";

  //
  // TDM slayout manipulation
  //

  // Invert the smem to get the largest contiguous vector in the smem layout
  // smem idx -> smem coord
  auto invSmemLayout = core::right_inverse(slayout);
  // Compose with the V-Map to convert smem coord (CTA val idx) to gmem mode
  // smem idx -> gmem mode
  auto sidx2gmodeFull = core::coalesce(core::composition(ctaVMap, invSmemLayout));

  //
  // TDM gtensor truncation
  //

  // Truncate any incompatibilities -- no starting in the middle of gmodes
  auto isUnitGlobalBasis = [&](TupleRef e) {
    return e.isBasis() && e.getLeaf().scale().isOne() &&
           llvm::is_contained(globalBases, e.getLeaf());
  };
  int32_t smemRank =
      core::find_if(sidx2gmodeFull.stride(), [&](TupleRef e) { return !isUnitGlobalBasis(e); });
  for (auto [i, extent, stride] :
       llvm::enumerate(sidx2gmodeFull.shape().leaves(), sidx2gmodeFull.stride().leaves())) {
    if (static_cast<int32_t>(i) > smemRank)
      break;
    if (!extent.isSInt())
      return emitError() << "the tile/global vectorization must be static";
    if (!stride.isBasis() && !stride.isSInt())
      return emitError() << "unsupported stride leaf in the tile/global vectorization";
  }
  if (!smemRank)
    return emitError() << "no common tile/global vectorization — the LDS tile and the global "
                          "tile do not share a contiguous innermost run. Does the tiler select "
                          "out the major global mode?";

  // Keep only the static-1 basis modes into gmem
  auto sidx2gmode = core::take(sidx2gmodeFull, 0, smemRank);

  //
  // TDM gtensor manipulation
  //

  // The smem vector is the same units as gtensor, so compose first and then recast
  // tdm_val_idx:gmem_strides
  auto tileGstride = core::recast_layout(params.internalBits, params.elemBits,
                                         core::composition(gtensor, sidx2gmode));
  if (tileGstride.isError() || tileGstride.size().staticValue() * params.internalBits !=
                                   sidx2gmode.size().staticValue() * params.elemBits)
    return emitError() << "the contiguous TDM tile must contain whole internal_type elements";
  // Coalesce modes up to the maximum box extent, in units of internal_type
  // tdm_box_shape:gmem_strides
  auto tdmGstride = coalesce_tdm_box(tileGstride);

  // Perform the tiling, recast, and coalesce to the gmem vector again, but with indirections
  // to the gtensor modes. Instead of the recast (gbasis doesn't have type info), replace the
  // shape with the already-recasted shape. tdm_box_shape:gmem_mode
  auto tileGbasis =
      core::make_layout(tileGstride.shape(), core::composition(gbasis, sidx2gmode).stride());

  // "Coalesce" the tile basis into a compatible shape with the tdm_gstride
  auto tdmGbasisTile =
      core::composition(tileGbasis, core::make_layout(core::wrap(tdmGstride.shape())));
  if (tdmGbasisTile.isError())
    return emitError() << "cannot express the coalesced TDM box in global modes";

  // Find missing bases that don't appear in tile_gbasis
  auto tileGbasisFlatStride = core::flatten(tdmGbasisTile.stride());
  auto tileGbasisRemainingStride = core::filter_tuple(
      core::flatten(gtensor.shape()), core::flatten(gtensor.stride()),
      core::flatten(gbasis.stride()), [&](TupleRef s, TupleRef d, TupleRef e) -> Tuple {
        // If size-1 or stride-0, then don't append
        if (s.isOne() || d.isZero())
          return Tuple::getEmpty();
        // If e was found, then don't append
        if (core::any_of(tileGbasisFlatStride,
                         [&](TupleRef tb) { return tb.getLeaf() == e.getLeaf(); }))
          return Tuple::getEmpty();
        // Else, this is missing so append
        return core::make_tuple(e);
      });

  // Append the remaining basis modes that contribute to the TDM with size-1
  auto tileGbasisRemainingShape =
      core::transform_leaf(tileGbasisRemainingStride, [](TupleRef) { return Tuple::getOne(); });
  auto tdmGbasis = core::make_layout(
      core::concat(core::wrap(tdmGbasisTile.shape()), core::wrap(tileGbasisRemainingShape)),
      core::concat(core::wrap(tdmGbasisTile.stride()), core::wrap(tileGbasisRemainingStride)));

  // TDM only: descriptor axis 0 has an implicit stride of 1. A dropped size-1 contiguous
  // mode can still be needed there; require a unique contiguous global mode and an
  // extent-1 selection of that mode.
  auto firstBasis = tdmGbasis.stride().leaves();
  if (firstBasis.empty() ||
      !core::basis_get(Tuple::fromLeaf(firstBasis.front()), gtensor.stride()).isOne()) {
    int32_t contiguousMode = -1, found = 0;
    for (auto [i, stride] : llvm::enumerate(gtensor.stride().leaves()))
      if (stride.isOne()) {
        contiguousMode = i;
        ++found;
      }
    int64_t tileExtent = 1;
    if (found == 1)
      for (auto [extent, stride] : llvm::zip(ctaVMap.shape().leaves(), ctaVMap.stride().leaves()))
        if (stride.isBasis() && stride.modes() == globalBases[contiguousMode].modes())
          tileExtent *= extent.staticValue();
    if (found == 1 && tileExtent == 1 &&
        !llvm::is_contained(tdmGbasis.stride().leaves(), globalBases[contiguousMode]))
      tdmGbasis =
          core::prepend(tdmGbasis, core::make_layout(Tuple::getOne(),
                                                     Tuple::fromLeaf(globalBases[contiguousMode])));
    else
      return emitError() << "the innermost descriptor dim must be contiguous in global memory "
                            "(stride 1), got global mode "
                         << (firstBasis.empty()
                                 ? -1
                                 : std::distance(globalBases.begin(),
                                                 llvm::find(globalBases, firstBasis.front())))
                         << " — the LDS tile's majorness does not match the global tensor's";
  }

  // Group the trailing modes to make this max rank-5 -- TDM rank limitation.
  // tdm_box_shape:gmem_mode
  bool packed = tdmGbasis.rank() > kMaxRank;
  if (packed) {
    tdmGbasis = core::group(tdmGbasis, kMaxRank - 1, tdmGbasis.rank());
    // Unrelated modes do not form a dense interval at their gcd stride. Move
    // their traversal to ITER; the fifth axis only carries the tile origin.
    auto tail = core::get(tdmGbasis, kMaxRank - 1);
    auto singleton = core::transform_leaf(tail.shape(), [](TupleRef) { return Tuple::getOne(); });
    tdmGbasis = core::replace(tdmGbasis, kMaxRank - 1, core::make_layout(singleton, tail.stride()));
  }
  if (tdmGbasis.isError())
    return emitError() << "cannot express the coalesced TDM box in global modes";
  return TdmGbasis{tdmGbasis, packed};
}

template <class Cat>
core::Layout<Cat> fill_tdm_gmem_shape_stride(const core::Layout<Cat> &gtensorT,
                                             TupleRef tdmGbasisStride, bool packed) {
  using CatTuple = core::IntTuple<Cat>;
  SmallVector<CatTuple, kMaxRank> gmemProbShape, gmemProbStride;
  int32_t tdmRank = tdmGbasisStride.rank();
  for (int32_t i = 0; i < tdmRank; ++i) {
    auto tdmModes = tdmGbasisStride.at(i);
    if (tdmModes.leafCount() == 1) {
      // Trivial contribution of this gmem mode to this tdm mode
      auto ej = core::unwrap(tdmModes);
      gmemProbShape.push_back(core::basis_get(ej, gtensorT.shape()));
      gmemProbStride.push_back(core::basis_get(ej, gtensorT.stride()));
      continue;
    }
    auto stride = core::leaf_abs(
        core::basis_get(Tuple::fromLeaf(tdmModes.leaves().front()), gtensorT.stride()));
    for (auto ej : tdmModes.leaves().drop_front())
      stride = core::leaf_gcd(stride, core::basis_get(Tuple::fromLeaf(ej), gtensorT.stride()));
    // Its boundary is disabled and global_addr already includes the origin, so no folded
    // bound is needed (and a large dynamic tail could overflow the hardware extent).
    if (packed && i == tdmRank - 1) {
      gmemProbShape.push_back(CatTuple::getOne());
      gmemProbStride.push_back(stride);
      continue;
    }
    // A runtime broadcast tail has gcd == 0; it contributes no address offset.
    auto divisor = core::leaf_max(stride, CatTuple::getOne());
    // Division is exact by construction: divisor is the gcd (or 1 for zero).
    // Trait cannot express this relation between dynamic leaves.
    auto shape = CatTuple::getOne();
    for (auto ej : tdmModes.leaves()) {
      auto e = Tuple::fromLeaf(ej);
      auto s = core::basis_get(e, gtensorT.shape());
      auto d = core::basis_get(e, gtensorT.stride());
      shape = shape + (s - CatTuple::getOne()) * core::leaf_div(d, divisor);
    }
    gmemProbShape.push_back(shape);
    gmemProbStride.push_back(stride);
  }
  return core::make_layout(core::make_tuple(gmemProbShape), core::make_tuple(gmemProbStride));
}

/// each global mode's coordinate stride in descriptor coordinates. Axis 0 carries the
/// recast ratio and a merged axis the mode's share of the gcd stride.
template <class Cat>
core::IntTuple<Cat>
make_tdm_gmem_basis_stride(const core::Layout<Cat> &gtensor, const core::Layout<Cat> &gtensorT,
                           TupleRef tdmGbasisStride, const core::Layout<Cat> &gmemProb,
                           const TdmTypeParams &params) {
  using CatTuple = core::IntTuple<Cat>;
  auto gbasis = core::make_basis_like(gtensor.shape());
  return core::transform_leaf(gbasis, [&](core::IntTupleRef<Cat> ei) {
    auto si = core::basis_get(ei, gtensor.shape());
    auto di = core::basis_get(ei, gtensor.stride());
    // If size-1 or stride-0, return arithmetic identity -- no contribution to the TDM
    if (si.isOne() || di.isZero())
      return CatTuple::getZero();
    // Find j such that E<i> is in stride<j>(tdm_gbasis)
    int32_t j = core::find_if(tdmGbasisStride, [&](TupleRef tdmStrideJ) {
      return core::any_of(core::flatten(tdmStrideJ),
                          [&](TupleRef dj) { return dj.getLeaf() == Leaf(ei.getLeaf()); });
    });
    // If not-found, return arithmetic identity -- no contribution to the TDM
    if (j == tdmGbasisStride.rank())
      return CatTuple::getZero();
    auto scale = CatTuple::getOne();
    if (j == 0)
      // Return TDM Coord basis -- with a recast scale factor
      scale = core::leaf_exact_div(di * CatTuple::getSInt(params.elemBits),
                                   CatTuple::getSInt(params.internalBits));
    else if (tdmGbasisStride.at(j).leafCount() > 1)
      // Return TDM Coord basis -- with a dynamic scale factor
      scale = core::leaf_div(core::basis_get(ei, gtensorT.stride()),
                             core::leaf_max(gmemProb.stride().at(j), CatTuple::getOne()));
    return scale * CatTuple::getBasis({j});
  });
}

/// recast the tensor to the descriptor's element, read the descriptor through the basis, and invert
/// the basis for the TDM tensor's stride.
template <class Cat>
TdmCopyDesc<Cat> make_tdm_copy_desc(const core::MemRef<Cat> &gtensor, const Layout &tdmGbasis,
                                    bool packed, const TdmTypeParams &params) {
  using CatTuple = core::IntTuple<Cat>;
  const auto &gmemLayout = std::get<core::Layout<Cat>>(gtensor.layout());
  // A mode of extent 1 never reaches an address, so drop its stride before the recast,
  // which could not divide an odd one or would carry a negative one into the descriptor.
  auto gmemStride = core::transform_leaf(gmemLayout.shape(), gmemLayout.stride(),
                                         [](core::IntTupleRef<Cat> s, core::IntTupleRef<Cat> d) {
                                           return s.isOne() ? CatTuple::getZero() : CatTuple(d);
                                         });
  // Recast the original tensor for shape/stride inspections
  auto gtensorT = core::recast_layout(params.internalBits, params.elemBits,
                                      core::make_layout(gmemLayout.shape(), gmemStride));
  auto gmemProb = fill_tdm_gmem_shape_stride(gtensorT, tdmGbasis.stride(), packed);
  auto gStride =
      make_tdm_gmem_basis_stride(gmemLayout, gtensorT, tdmGbasis.stride(), gmemProb, params);
  return {gtensor.data(), gmemProb, gStride};
}

/// The box before warp splitting, in tensor dim order (descriptor axes reversed).
SmallVector<int32_t, 5> tdm_box_shape(const Layout &tdmGbasis) {
  auto shape = core::product_each(tdmGbasis.shape());
  SmallVector<int32_t, 5> result;
  for (auto leaf : llvm::reverse(shape.leaves()))
    result.push_back(leaf.staticValue());
  return result;
}

/// cooperating warps split one box into contiguous shares of the compact LDS traversal.
/// `box` is outermost first, so consuming outermost axes here matches the
/// (ATOM, WARP, ITER) partition map.
FailureOr<SmallVector<int32_t, 5>> truncate_box_by_warps(ArrayRef<int32_t> box, int32_t numWarps,
                                                         EmitErrorFn emitError) {
  if (numWarps < 1)
    return emitError() << "num_warps must be positive";
  SmallVector<int32_t, 5> shape(box);
  int32_t remaining = numWarps;
  for (int32_t &extent : shape) {
    if (extent < 1)
      return emitError() << "TDM tile shape dims must be positive";
    if (remaining == 1)
      continue;
    if (extent % remaining == 0) {
      extent /= remaining;
      remaining = 1;
    } else if (remaining % extent == 0) {
      remaining /= extent;
      extent = 1;
    } else {
      return emitError() << "num_warps cuts a descriptor axis unevenly";
    }
  }
  if (remaining != 1)
    return emitError() << "num_warps exceeds the logical descriptor box";
  return shape;
}

/// `tensor2tdm`: the TDM tensor's basis stride read the other way round, per global mode
/// the reversed state axis its boundary flag lands on. The packed axis has no bound.
Tuple make_tdm_boundary_map(const Tuple &gStride, int32_t rank, bool packed) {
  return core::transform_leaf(gStride, [&](TupleRef leaf) {
    if (!leaf.isBasis() || (packed && leaf.mode(0) == kMaxRank - 1))
      return Tuple::getZero();
    return Tuple::getBasis(leaf.scale(), {rank - 1 - leaf.mode(0)});
  });
}

/// The atom's initial `boundary_check` state, shaped like the tensor: `enable` on every
/// mode that has a bound to give, and off on every mode that has none.
///
/// A mode with nothing to clamp — size-1, stride-0, not spanned by the box, or sharing the
/// rank-5 packing's dim — simply comes out off. The builder's flag is "clamp what can be
/// clamped" and so has no way to fail; naming such a mode is only an error when a call
/// site names it, which is `fly.atom.set_value "boundary_check"`.
Tuple initial_boundary_check(const TdmAuxParams &aux, bool enable) {
  return core::transform_leaf(
      aux.tensor2tdm, [&](TupleRef leaf) { return Tuple::getSInt(enable && leaf.isBasis()); });
}

template <class Cat>
FailureOr<TdmCopyAtom<Cat>> make_tdm_atom(MLIRContext *ctx, const core::MemRef<Cat> &gtensor,
                                          const Layout &slayout, const Tile &ctaTiler,
                                          const TdmTypeParams &params, EmitErrorFn emitError) {
  Layout glayout = core::to_trait(std::get<core::Layout<Cat>>(gtensor.layout()));
  auto ctaVMap = make_tdm_value_map(ctx, glayout.shape(), ctaTiler, emitError);
  if (failed(ctaVMap))
    return failure();
  if (params.numWarps < 1)
    return emitError() << "num_warps must be positive, got " << params.numWarps;
  if (failed(check_tdm_gtensor(glayout, slayout, *ctaVMap, params, emitError)))
    return failure();

  // TDM truncated layout
  auto smemLayout = get_nonpad_portion(slayout, emitError);
  if (failed(smemLayout))
    return failure();
  auto tdmGbasis = construct_tdm_gbasis(glayout, smemLayout->compact, *ctaVMap, params, emitError);
  if (failed(tdmGbasis))
    return failure();

  // Construct the TDM descriptor and the strides of the TDM tensor
  auto desc = make_tdm_copy_desc(gtensor, tdmGbasis->gbasis, tdmGbasis->packed, params);
  if (desc.gmemProb.isError() || desc.gStride.isError())
    return emitError() << "cannot represent TDM descriptor geometry or coordinate scales";
  int32_t rank = tdmGbasis->gbasis.rank();
  for (auto stride : core::to_trait(desc.gmemProb).stride().leaves())
    if (stride.isSInt() && (stride.staticValue() < 0 ||
                            static_cast<uint64_t>(stride.staticValue()) > kMaxTensorStride))
      return emitError() << "descriptor stride is outside the unsigned 48-bit range";

  // The padding, in TDM internal elements.
  int64_t intervalBits = int64_t(smemLayout->padInterval) * params.elemBits;
  int64_t amountBits = int64_t(smemLayout->padAmount) * params.elemBits;
  if (intervalBits % params.internalBits || amountBits % params.internalBits)
    return emitError() << "LDS padding must contain whole internal_type elements";
  int32_t padInterval = intervalBits / params.internalBits;
  int32_t padAmount = amountBits / params.internalBits;

  // Finally, truncate the TDM box by the cooperating warps.
  auto boxShape =
      truncate_box_by_warps(tdm_box_shape(tdmGbasis->gbasis), params.numWarps, emitError);
  if (failed(boxShape))
    return failure();
  int64_t boxElems = 1;
  for (int32_t dim : *boxShape) {
    if (dim > kMaxTileDim)
      return emitError() << "physical TDM tile dimension exceeds " << kMaxTileDim;
    boxElems *= dim;
  }
  if (padAmount && boxElems % padInterval)
    return emitError() << "padInterval must divide each warp's physical TDM box";
  // Padding is inserted per dword, so a padded box's per-issue dimension (after
  // coalescing and warp splitting) must span whole dwords.
  if (padAmount) {
    int64_t innerBits = static_cast<int64_t>(boxShape->back()) * params.internalBits;
    if (innerBits % (4 * 8))
      return emitError() << "a padded TDM tile_dim0 must span a whole multiple of 4 bytes, got "
                         << boxShape->back() << " elements at " << params.internalBits << " bits";
  }

  TdmAuxParams aux{tdmGbasis->gbasis,
                   tdmGbasis->packed,
                   padInterval,
                   padAmount,
                   std::move(*boxShape),
                   make_tdm_boundary_map(core::to_trait(desc.gStride), rank, tdmGbasis->packed)};
  // get_tma_tensor: the coordinates start at the tensor origin, one per descriptor axis.
  core::CoordTensor<Cat> tdmTensor(
      core::tuple_repeat(core::IntTuple<Cat>::getZero(), rank),
      core::make_layout(std::get<core::Layout<Cat>>(gtensor.layout()).shape(), desc.gStride));
  return TdmCopyAtom<Cat>{std::move(aux), std::move(desc), std::move(tdmTensor)};
}

//===----------------------------------------------------------------------===//
// tma_partition
//===----------------------------------------------------------------------===//

FailureOr<Layout> tdm_partition_layout(const core::CopyAtom<core::Trait> &copyAtom,
                                       const Layout &slayout, int32_t ldsElemBits,
                                       TupleRef targetShape, int32_t numWarps,
                                       EmitErrorFn emitError) {
  if (slayout.shape().isLeaf() || targetShape.isLeaf())
    return emitError() << "partition inputs must have shape (box, rest...)";
  if (numWarps < 1)
    return emitError() << "num_warps must be positive, got " << numWarps;

  // NumValSrc counted in LDS elements: one warp's issue
  auto atomBits = core::size<1>(copyAtom.thrBitLayoutSrc());
  auto smemSize = core::size<0>(slayout);
  auto targetSize = core::size<0>(targetShape);
  if (!atomBits.isSInt() || !smemSize.isSInt() || !targetSize.isSInt())
    return emitError() << "the atom and both mode-0 boxes must be static";
  if (ldsElemBits <= 0 || atomBits.staticValue() % ldsElemBits)
    return emitError() << "the atom must move a whole number of LDS elements";
  int64_t numVal = atomBits.staticValue() / ldsElemBits;
  if (smemSize.staticValue() % (numVal * numWarps))
    return emitError() << "the LDS box holds " << smemSize.staticValue()
                       << " elements, not a multiple of the cooperative tile size "
                       << numVal * numWarps << " (atom size * num_warps)";
  if (targetSize.staticValue() % smemSize.staticValue())
    return emitError() << "the target's mode 0 holds " << targetSize.staticValue()
                       << " elements, not a multiple of the LDS box's " << smemSize.staticValue();

  auto smemLayout = get_nonpad_portion(core::layout<0>(slayout), emitError);
  if (failed(smemLayout))
    return failure();

  // Invert the smem to get the largest contiguous vector in the smem layout
  auto invSmemLayout = core::right_inverse(smemLayout->compact);
  // Scale that up to cover all of the target's mode-0 coords
  auto layoutV = core::tile_to_shape(core::make_layout(invSmemLayout), targetSize);

  // Factor out the single-instruction portion, once per cooperating warp
  auto tdmLayoutV =
      core::make_layout(core::make_tuple(Tuple::getSInt(numVal), Tuple::getSInt(numWarps)));
  auto layout = core::logical_divide(layoutV, tdmLayoutV);
  if (layout.isError())
    return emitError() << "cannot split the LDS traversal into " << numVal
                       << "-element issues: " << core::toString(layout.errorInfo());
  return layout;
}

} // namespace

void boundaryCheckAxes(IntTupleAttr tensor2tdm, SmallVectorImpl<int32_t> &axes) {
  for (auto leaf : tensor2tdm.getRef().leaves())
    axes.push_back(leaf.isBasis() ? leaf.mode(0) : -1);
}

} // namespace tdm

//===----------------------------------------------------------------------===//
// Op glue: operands in, Core in the middle, atom and coordinate tensor out
//===----------------------------------------------------------------------===//

namespace {

/// A tiled TDM builder's operands, checked. Nothing is emitted for them: the LDS layout
/// and the tiler are static, and the tensor is imported by the expansion.
struct TdmOperands {
  fly::MemRefType tensorType;
  tdm::Layout slayout;
  tdm::Tile ctaTiler;
  Type dataType;
  tdm::TdmTypeParams params;
  bool initBoundaryCheck;
};

template <typename AdaptorT>
FailureOr<TdmOperands> convertToTdmOperands(AdaptorT adaptor, tdm::EmitErrorFn emitError) {
  auto memRefTy = dyn_cast<fly::MemRefType>(adaptor.getTensor().getType());
  if (!memRefTy)
    return emitError() << "expected a !fly.memref tensor, got " << adaptor.getTensor().getType();
  if (!isa<LayoutAttr>(memRefTy.getLayout()))
    return emitError() << "expected a plain #fly.layout on the tensor, got "
                       << memRefTy.getLayout();

  auto smemLayoutTy = dyn_cast<LayoutType>(adaptor.getSmemLayout().getType());
  if (!smemLayoutTy)
    return emitError() << "expected !fly.layout for the LDS layout";

  Type elemTy = memRefTy.getElemTy();
  if (!elemTy.isIntOrFloat())
    return emitError() << "the tensor's element type must be an integer or float, got " << elemTy;

  std::optional<Type> internalType = adaptor.getInternalType();
  Type dataType = internalType.value_or(elemTy);
  if (!dataType.isIntOrFloat())
    return emitError() << "internal_type must be an integer or float type, got " << dataType;

  tdm::TdmTypeParams params;
  params.elemBits = elemTy.getIntOrFloatBitWidth();
  params.internalBits = dataType.getIntOrFloatBitWidth();
  if (params.internalBits != 8 && params.internalBits != 16 && params.internalBits != 32 &&
      params.internalBits != 64) {
    if (internalType)
      return emitError() << "internal_type " << dataType << " is a " << params.internalBits
                         << "-bit type, but TDM moves 8, 16, 32 or 64-bit elements";
    return emitError() << "the tensor's element type " << elemTy << " is a " << params.elemBits
                       << "-bit type, but TDM moves 8, 16, 32 or 64-bit elements; pass an "
                          "internal_type such as i8 to move it as wider elements";
  }
  params.numWarps = adaptor.getNumWarps();

  auto tilerTy = dyn_cast<TileType>(adaptor.getTiler().getType());
  if (!tilerTy)
    return emitError() << "expected !fly.tile for the tiler, got " << adaptor.getTiler().getType();
  return TdmOperands{memRefTy, smemLayoutTy.getRef(),         tilerTy.getRef(), dataType,
                     params,   adaptor.getInitBoundaryCheck()};
}

FailureOr<Type> tdmLoadOpType(MLIRContext *ctx, const tdm::TdmAuxParams &aux, Type dataType,
                              bool atomicBarrier, int32_t cacheModifier,
                              tdm::EmitErrorFn emitError) {
  if (aux.padAmount &&
      static_cast<int64_t>(aux.boxShape.back()) * dataType.getIntOrFloatBitWidth() % 32)
    return emitError() << "padded physical TDM tile_dim0 must span whole dwords";
  Type ty = CopyOpCDNA5TensorLoadType::getChecked(emitError, ctx, ArrayRef<int32_t>(aux.boxShape),
                                                  dataType, IntTupleAttr::get(ctx, aux.tensor2tdm),
                                                  atomicBarrier, cacheModifier, /*iterCount=*/1,
                                                  aux.padInterval, aux.padAmount);
  if (!ty)
    return failure();
  return ty;
}

FailureOr<Type> tdmStoreOpType(MLIRContext *ctx, const tdm::TdmAuxParams &aux, Type dataType,
                               bool atomicBarrier, int32_t cacheModifier,
                               tdm::EmitErrorFn emitError) {
  if (aux.padAmount)
    return emitError() << "a TDM store cannot drain a padded LDS tile (pad interval "
                       << aux.padInterval << ", amount " << aux.padAmount
                       << "); TENSOR_STORE_FROM_LDS has no de-padding, it walks LDS with the "
                          "packed tile stride";
  Type ty = CopyOpCDNA5TensorStoreType::getChecked(emitError, ctx, ArrayRef<int32_t>(aux.boxShape),
                                                   dataType, IntTupleAttr::get(ctx, aux.tensor2tdm),
                                                   atomicBarrier, cacheModifier, /*iterCount=*/1);
  if (!ty)
    return failure();
  return ty;
}

template <typename OpT>
LogicalResult inferTdmAtomTypes(MLIRContext *context, std::optional<Location> location,
                                ValueRange operands, DictionaryAttr attributes,
                                PropertyRef properties, RegionRange regions,
                                SmallVectorImpl<Type> &inferredReturnTypes) {
  using Adaptor = typename OpT::Adaptor;
  constexpr bool isLoad = std::is_same_v<OpT, MakeTiledTdmLoadAtomOp>;

  Location loc = location.value_or(UnknownLoc::get(context));
  auto emitError = [&]() -> InFlightDiagnostic {
    return mlir::emitError(loc) << OpT::getOperationName() << ": ";
  };

  typename OpT::Properties parsed;
  PropertyRef effective = properties;
  if (attributes && !attributes.empty()) {
    if (failed(OpT::setPropertiesFromAttr(parsed, attributes, emitError)))
      return failure();
    effective = PropertyRef(TypeID::get<typename OpT::Properties>(), &parsed);
  }
  Adaptor adaptor(operands, attributes, effective, regions);

  auto in = convertToTdmOperands(adaptor, emitError);
  if (failed(in))
    return failure();

  auto atom = tdm::make_tdm_atom(context, in->tensorType.getRef(), in->slayout, in->ctaTiler,
                                 in->params, emitError);
  if (failed(atom))
    return failure();

  FailureOr<Type> copyOp =
      isLoad ? tdmLoadOpType(context, atom->aux, in->dataType, adaptor.getAtomicBarrier(),
                             adaptor.getCacheModifier(), emitError)
             : tdmStoreOpType(context, atom->aux, in->dataType, adaptor.getAtomicBarrier(),
                              adaptor.getCacheModifier(), emitError);
  if (failed(copyOp))
    return failure();

  auto coordTy = CoordTensorType::get(context, atom->tdmTensor);
  if (!coordTy)
    return emitError() << "cannot represent the TDM coordinate tensor";

  inferredReturnTypes.assign({CopyAtomType::get(*copyOp, in->params.elemBits), coordTy});
  return success();
}

/// An all-static int tuple, as a value.
Value staticTuple(OpBuilder &builder, Location loc, IntTupleAttr attr) {
  return MakeIntTupleOp::create(builder, loc, IntTupleType::get(attr), ValueRange{});
}

/// Build the runtime descriptor and its coordinate tensor: import the tensor, run the same
/// construction as type inference in EmitOp, and materialize the results.
template <typename OpT> LogicalResult expandTdmAtomImpl(OpT op, IRRewriter &rewriter) {
  Location loc = op.getLoc();
  MLIRContext *context = rewriter.getContext();
  rewriter.setInsertionPoint(op);
  auto emitError = [&]() -> InFlightDiagnostic { return op.emitOpError(); };

  auto in = convertToTdmOperands(typename OpT::Adaptor(op), emitError);
  if (failed(in))
    return failure();

  auto atomTy = dyn_cast<CopyAtomType>(op.getAtom().getType());
  auto coordTy = dyn_cast<CoordTensorType>(op.getCoordTensor().getType());
  if (!atomTy || !coordTy)
    return failure();

  // Operands in: the LDS layout and the tiler are static, the tensor is imported once.
  auto ctx = makeDefaultEmitContext(rewriter, loc);
  auto gtensor = importValue<fly::MemRefType>(op.getTensor(), ctx);
  if (failed(gtensor))
    return op.emitOpError() << "could not bind the TDM tensor";

  auto atom =
      tdm::make_tdm_atom(context, *gtensor, in->slayout, in->ctaTiler, in->params, emitError);

  if (failed(atom))
    return failure();

  // Results out. Construction arguments, in tensor dim order (the reverse of descriptor
  // order, which is what the atom's `tileShape` and its state slots use): base pointer,
  // then the strides of dims 0..rank-2 (the innermost is 1 by construction and is not
  // passed), then every dim's extent. Every dim passes its extent whether or not it
  // currently clamps: `boundary_check` is per-call state, so a later call site may switch
  // its dim on and needs the bound already in the atom, and an extent no call reads dies
  // with the rest of the scalarized state struct.
  auto scalar = [&](core::IntTupleRef<core::EmitOp> value, unsigned width) -> Value {
    auto result = value.getLeaf().materialize(ctx);
    return failed(result)
               ? Value{}
               : convertScalarToDtype(rewriter, loc, *result, rewriter.getIntegerType(width),
                                      /*isUnsignedCast=*/true);
  };

  int32_t rank = atom->aux.rank();
  const auto &gmemProb = atom->desc.gmemProb;
  auto base = materializeValue(atom->desc.gmemAddress, ctx);
  if (failed(base))
    return op.emitOpError() << "could not materialize the TDM tensor's base pointer";

  SmallVector<Value> args{*base};
  for (int32_t i = 0; i < rank - 1; ++i) {
    Value v = scalar(gmemProb.stride().at(rank - 1 - i), 64);
    if (!v)
      return op.emitOpError() << "could not materialize the stride of descriptor dim "
                              << (rank - 1 - i);
    args.push_back(v);
  }
  for (int32_t i = 0; i < rank; ++i) {
    Value v = scalar(gmemProb.shape().at(rank - 1 - i), 32);
    if (!v)
      return op.emitOpError() << "could not materialize the extent of descriptor dim "
                              << (rank - 1 - i);
    args.push_back(v);
  }

  Value atomValue = MakeCopyAtomOp::create(rewriter, loc, atomTy, args, in->params.elemBits);

  // The initial `boundary_check` state; explicit requests to clamp a mode with no bound are
  // checked by the pass's boundary normalization.
  auto boundaryCheck =
      IntTupleAttr::get(context, tdm::initial_boundary_check(atom->aux, in->initBoundaryCheck));
  atomValue = AtomSetValueOp::create(rewriter, loc, atomValue.getType(), atomValue,
                                     rewriter.getStringAttr("boundary_check"),
                                     staticTuple(rewriter, loc, boundaryCheck));

  // The coordinate tensor starts at the tensor origin. Static coordinates fold
  // into its type; dynamic basis coefficients remain SSA values in its layout.
  assert(CoordTensorType::get(context, core::to_trait(atom->tdmTensor)) == coordTy &&
         "the expansion and type inference disagree on the TDM tensor");

  const auto &coordLayout = std::get<core::Layout<core::EmitOp>>(atom->tdmTensor.layout());
  Value origin = staticTuple(rewriter, loc, coordTy.getBase());
  auto layout = materializeValue(coordLayout, ctx);
  if (failed(layout))
    return op.emitOpError() << "could not materialize the TDM coordinate layout";
  Value coord = MakeViewOp::create(rewriter, loc, coordTy, origin, *layout);

  rewriter.replaceOp(op, {atomValue, coord});
  return success();
}

} // namespace

LogicalResult MakeTiledTdmLoadAtomOp::inferReturnTypes(MLIRContext *context,
                                                       std::optional<Location> location,
                                                       ValueRange operands,
                                                       DictionaryAttr attributes,
                                                       PropertyRef properties, RegionRange regions,
                                                       SmallVectorImpl<Type> &inferredReturnTypes) {
  return inferTdmAtomTypes<MakeTiledTdmLoadAtomOp>(context, location, operands, attributes,
                                                   properties, regions, inferredReturnTypes);
}

LogicalResult
MakeTiledTdmStoreAtomOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                                          ValueRange operands, DictionaryAttr attributes,
                                          PropertyRef properties, RegionRange regions,
                                          SmallVectorImpl<Type> &inferredReturnTypes) {
  return inferTdmAtomTypes<MakeTiledTdmStoreAtomOp>(context, location, operands, attributes,
                                                    properties, regions, inferredReturnTypes);
}

LogicalResult expandTdmAtom(MakeTiledTdmLoadAtomOp op, IRRewriter &rewriter) {
  return expandTdmAtomImpl(op, rewriter);
}

LogicalResult expandTdmAtom(MakeTiledTdmStoreAtomOp op, IRRewriter &rewriter) {
  return expandTdmAtomImpl(op, rewriter);
}

FailureOr<fly::LayoutType> tdmPartitionLayout(Type atomType, Type stensorType, Type gtensorType,
                                              int32_t numWarps,
                                              function_ref<InFlightDiagnostic()> emitError) {
  auto atomTy = dyn_cast<CopyAtomType>(atomType);
  if (!atomTy)
    return emitError() << "expected a !fly.copy_atom, got " << atomType;

  auto smemTy = dyn_cast<fly::MemRefType>(stensorType);
  if (!smemTy)
    return emitError() << "expected a !fly.memref LDS tile, got " << stensorType;
  auto smemLayout = dyn_cast<LayoutAttr>(smemTy.getLayout());
  if (!smemLayout)
    return emitError() << "the LDS tile needs a plain layout, got " << smemTy.getLayout();
  if (!smemTy.getElemTy().isIntOrFloat())
    return emitError() << "the LDS tile's element needs a width, got " << smemTy.getElemTy();

  // The coordinate tile is a `!fly.coord_tensor` in a kernel; a memref stands in for it
  // where a test cuts the tile out of an ordinary tensor. Only its shape is read either way.
  Attribute coordLayoutAttr;
  if (auto coordTy = dyn_cast<CoordTensorType>(gtensorType))
    coordLayoutAttr = coordTy.getLayout();
  else if (auto memTy = dyn_cast<fly::MemRefType>(gtensorType))
    coordLayoutAttr = memTy.getLayout();
  else
    return emitError() << "expected a !fly.coord_tensor or !fly.memref coordinate tile, got "
                       << gtensorType;
  auto coordLayout = dyn_cast<LayoutAttr>(coordLayoutAttr);
  if (!coordLayout)
    return emitError() << "the coordinate tile needs a plain layout, got " << coordLayoutAttr;

  auto layout =
      tdm::tdm_partition_layout(atomTy.getRef(), smemLayout.getRef(),
                                static_cast<int32_t>(smemTy.getElemTy().getIntOrFloatBitWidth()),
                                coordLayout.getRef().shape(), numWarps, emitError);
  if (failed(layout))
    return failure();
  return LayoutType::get(LayoutAttr::get(atomType.getContext(), *layout));
}

} // namespace mlir::fly_rocdl
