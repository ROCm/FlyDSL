// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "flydsl/Dialect/FlyROCDL/Utils/BufferFatPtr.h"
#include "flydsl/Dialect/FlyROCDL/Utils/TdmAtomBuilder.h"

using namespace mlir;
using namespace mlir::fly;
using namespace mlir::fly_rocdl;

LogicalResult GetBufferRsrcOp::inferReturnTypes(MLIRContext *context,
                                                std::optional<Location> location,
                                                ValueRange operands, DictionaryAttr attributes,
                                                PropertyRef properties, RegionRange regions,
                                                SmallVectorImpl<Type> &inferredReturnTypes) {
  auto ptrTy = dyn_cast<PointerType>(operands[0].getType());
  if (!ptrTy)
    return emitOptionalError(location, "GetBufferRsrcOp: expected a fly.ptr, got ",
                             operands[0].getType());
  if (!isTargetAddressSpace<BufferDescAddressAttr>(ptrTy.getAddressSpace()))
    return emitOptionalError(location,
                             "GetBufferRsrcOp: expected a buffer_desc address space pointer, got ",
                             ptrTy.getAddressSpace());
  inferredReturnTypes.assign({BufferFatPtr::getRsrcPtrType(context)});
  return success();
}

//===----------------------------------------------------------------------===//
// MakeTiledTdmLoadAtomOp / MakeTiledTdmStoreAtomOp
//===----------------------------------------------------------------------===//

namespace {

/// Both builders infer their results the same way: one derivation, and only the atom type
/// it lands on is the direction's.
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

  tdm::Request request;
  Type dataType;
  FailureOr<tdm::Geometry> geometry = deriveTdmAtom(adaptor, request, dataType, emitError);
  if (failed(geometry))
    return failure();

  FailureOr<IntTupleAttr> tensor2tdm =
      tdm::makeTensor2Tdm(*geometry, request.gLayout.getShape(), emitError);
  if (failed(tensor2tdm))
    return failure();

  FailureOr<Type> copyOp =
      isLoad ? tdmLoadOpType(context, *geometry, dataType, *tensor2tdm, adaptor.getAtomicBarrier(),
                             adaptor.getCacheModifier(), emitError)
             : tdmStoreOpType(context, *geometry, dataType, *tensor2tdm, adaptor.getAtomicBarrier(),
                              adaptor.getCacheModifier(), emitError);
  if (failed(copyOp))
    return failure();

  FailureOr<LayoutAttr> coord = tdm::coordLayout(*geometry, request.gLayout.getShape(), emitError);
  if (failed(coord))
    return failure();

  SmallVector<Attribute> zeros(geometry->rank(), IntTupleAttr::get(IntAttr::getStatic(context, 0)));
  IntTupleAttr base = IntTupleAttr::get(ArrayAttr::get(context, zeros));

  inferredReturnTypes.assign(
      {CopyAtomType::get(*copyOp, static_cast<int32_t>(dataType.getIntOrFloatBitWidth())),
       CoordTensorType::get(base, *coord)});
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
