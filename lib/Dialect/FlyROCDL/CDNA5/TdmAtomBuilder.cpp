// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#include "flydsl/Dialect/FlyROCDL/Utils/TdmAtomBuilder.h"
#include "flydsl/Dialect/Fly/Utils/IntTupleUtils.h"

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_rocdl {

template <typename AdaptorT>
FailureOr<tdm::Geometry> deriveTdmAtom(AdaptorT adaptor, tdm::Request &request, Type &dataType,
                                       function_ref<InFlightDiagnostic()> emitError) {
  auto memRefTy = dyn_cast<fly::MemRefType>(adaptor.getTensor().getType());
  if (!memRefTy)
    return emitError() << "expected a !fly.memref tensor, got " << adaptor.getTensor().getType();

  auto gLayout = dyn_cast<LayoutAttr>(memRefTy.getLayout());
  if (!gLayout)
    return emitError() << "expected a plain #fly.layout on the tensor, got "
                       << memRefTy.getLayout();

  // Stage one limits the original profile, before coalescing or descriptor
  // packing. Nested and extent-one modes count toward this flat rank too.
  SmallVector<IntTupleAttr> globalModes;
  intTupleFlattenToVector(IntTupleBuilder<IntTupleAttr>(gLayout.getContext()), gLayout.getShape(),
                          globalModes);
  if (globalModes.size() > tdm::kMaxRank)
    return emitError() << "tiled TDM currently supports global tensor flat-rank <= "
                       << tdm::kMaxRank << ", got " << globalModes.size();

  auto smemLayoutTy = dyn_cast<LayoutType>(adaptor.getSmemLayout().getType());
  if (!smemLayoutTy)
    return emitError() << "expected !fly.layout for the LDS layout";

  Type elemTy = memRefTy.getElemTy();
  if (!elemTy.isIntOrFloat())
    return emitError() << "the tensor's element type must be an integer or float, got " << elemTy;
  // Reserve internal_type until rational basis coefficients can preserve gshape.
  std::optional<Type> internalType = adaptor.getInternalType();
  if (internalType)
    return emitError()
           << "internal_type must be omitted until rational basis strides are supported";
  dataType = elemTy;

  request.gLayout = gLayout;
  request.smemLayout = smemLayoutTy.getAttr();
  // The tiler is entirely static, so like `smemLayout` it is read off its operand's type
  // and nothing is emitted for the operand itself.
  auto tilerTy = dyn_cast<TileType>(adaptor.getTiler().getType());
  if (!tilerTy)
    return emitError() << "expected !fly.tile for the tiler, got " << adaptor.getTiler().getType();

  FailureOr<LayoutAttr> valueMap =
      tdm::makeValueMap(gLayout.getShape(), tilerTy.getAttr(), emitError);
  if (failed(valueMap))
    return failure();

  request.valueMap = *valueMap;
  request.numWarps = adaptor.getNumWarps();
  request.initBoundaryCheck = adaptor.getInitBoundaryCheck();

  if (request.numWarps < 1)
    return emitError() << "num_warps must be positive, got " << request.numWarps;

  auto geometry = tdm::derive(request, emitError);
  if (failed(geometry))
    return failure();
  // Check the actual per-issue dimension after coalescing and warp splitting:
  // a sufficiently wide collective tile can still leave each warp below 16 B.
  int64_t innerBits =
      static_cast<int64_t>(geometry->descriptorShape.back()) * dataType.getIntOrFloatBitWidth();
  if (innerBits < 16 * 8)
    return emitError() << "physical TDM tile_dim0 must span at least 16 bytes, got "
                       << geometry->descriptorShape.back() << " elements at "
                       << dataType.getIntOrFloatBitWidth() << " bits";
  return geometry;
}

template FailureOr<tdm::Geometry>
deriveTdmAtom<MakeTiledTdmLoadAtomOpAdaptor>(MakeTiledTdmLoadAtomOpAdaptor, tdm::Request &, Type &,
                                             function_ref<InFlightDiagnostic()>);
template FailureOr<tdm::Geometry>
deriveTdmAtom<MakeTiledTdmStoreAtomOpAdaptor>(MakeTiledTdmStoreAtomOpAdaptor, tdm::Request &,
                                              Type &, function_ref<InFlightDiagnostic()>);

FailureOr<fly::LayoutType> tdmPartitionLayout(Type atomType, Type stensorType, Type gtensorType,
                                              int32_t numWarps,
                                              function_ref<InFlightDiagnostic()> emitError) {
  auto atomTy = dyn_cast<CopyAtomType>(atomType);
  if (!atomTy)
    return emitError() << "expected a !fly.copy_atom, got " << atomType;
  auto atomVal = dyn_cast<LayoutAttr>(atomTy.getThrValLayoutSrc());
  if (!atomVal || atomVal.getShape().isLeaf())
    return emitError() << "the atom has no (thread, value) layout";

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

  FailureOr<LayoutAttr> layout =
      tdm::partitionLayout(atomVal.getShape().at(1), atomTy.getValBits(), smemLayout,
                           static_cast<int32_t>(smemTy.getElemTy().getIntOrFloatBitWidth()),
                           coordLayout.getShape(), numWarps, emitError);
  if (failed(layout))
    return failure();
  return LayoutType::get(*layout);
}

FailureOr<Type> tdmLoadOpType(MLIRContext *ctx, const tdm::Geometry &geometry, Type dataType,
                              IntTupleAttr tensor2tdm, bool atomicBarrier, int32_t cacheModifier,
                              function_ref<InFlightDiagnostic()> emitError) {
  if (geometry.padAmount &&
      static_cast<int64_t>(geometry.descriptorShape.back()) * dataType.getIntOrFloatBitWidth() % 32)
    return emitError() << "padded physical TDM tile_dim0 must span whole dwords";
  ArrayRef<int32_t> shape(geometry.descriptorShape);

  Type ty = CopyOpCDNA5TensorLoadType::getChecked(emitError, ctx, shape, dataType, tensor2tdm,
                                                  atomicBarrier, cacheModifier, /*iterCount=*/1,
                                                  geometry.padInterval, geometry.padAmount);
  if (!ty)
    return failure();
  return ty;
}

FailureOr<Type> tdmStoreOpType(MLIRContext *ctx, const tdm::Geometry &geometry, Type dataType,
                               IntTupleAttr tensor2tdm, bool atomicBarrier, int32_t cacheModifier,
                               function_ref<InFlightDiagnostic()> emitError) {
  if (geometry.padAmount)
    return emitError() << "a TDM store cannot drain a padded LDS tile (pad interval "
                       << geometry.padInterval << ", amount " << geometry.padAmount
                       << "); TENSOR_STORE_FROM_LDS has no de-padding, it walks LDS with the "
                          "packed tile stride";
  ArrayRef<int32_t> shape(geometry.descriptorShape);
  Type ty = CopyOpCDNA5TensorStoreType::getChecked(emitError, ctx, shape, dataType, tensor2tdm,
                                                   atomicBarrier, cacheModifier, /*iterCount=*/1);
  if (!ty)
    return failure();
  return ty;
}

} // namespace mlir::fly_rocdl
