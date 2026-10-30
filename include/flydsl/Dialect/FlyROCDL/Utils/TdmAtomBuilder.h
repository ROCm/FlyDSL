// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLYROCDL_UTILS_TDMATOMBUILDER_H
#define FLYDSL_DIALECT_FLYROCDL_UTILS_TDMATOMBUILDER_H

#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "flydsl/Dialect/FlyROCDL/Utils/TdmGeometry.h"

namespace mlir::fly_rocdl {

/// Read a builder's operands and attributes into a `tdm::Request`, run the derivation, and
/// report the descriptor's data type.
template <typename AdaptorT>
FailureOr<tdm::Geometry> deriveTdmAtom(AdaptorT adaptor, tdm::Request &request, Type &dataType,
                                       function_ref<InFlightDiagnostic()> emitError);

extern template FailureOr<tdm::Geometry>
deriveTdmAtom<MakeTiledTdmLoadAtomOpAdaptor>(MakeTiledTdmLoadAtomOpAdaptor, tdm::Request &, Type &,
                                             function_ref<InFlightDiagnostic()>);
extern template FailureOr<tdm::Geometry>
deriveTdmAtom<MakeTiledTdmStoreAtomOpAdaptor>(MakeTiledTdmStoreAtomOpAdaptor, tdm::Request &,
                                              Type &, function_ref<InFlightDiagnostic()>);

FailureOr<Type> tdmLoadOpType(MLIRContext *ctx, const tdm::Geometry &geometry, Type dataType,
                              fly::IntTupleAttr tensor2tdm, bool atomicBarrier,
                              int32_t cacheModifier, int32_t indexBits,
                              function_ref<InFlightDiagnostic()> emitError);
FailureOr<Type> tdmStoreOpType(MLIRContext *ctx, const tdm::Geometry &geometry, Type dataType,
                               fly::IntTupleAttr tensor2tdm, bool atomicBarrier,
                               int32_t cacheModifier, int32_t indexBits,
                               function_ref<InFlightDiagnostic()> emitError);

/// Emit one CDNA5 TDM call whose outer dim is addressed by the row list in `index`.
/// Kept off `CopyOpTypeInterface` because the index operand is specific to this target:
/// every other atom would carry a parameter it can never use.
LogicalResult emitTdmIndexedCall(OpBuilder &builder, Location loc, Type copyAtomTy, Type srcTy,
                                 Type dstTy, Type indexTy, Value atomVal, Value src, Value dst,
                                 Value index);

FailureOr<fly::LayoutType> tdmPartitionLayout(Type atomType, Type stensorType, Type gtensorType,
                                              int32_t numWarps,
                                              function_ref<InFlightDiagnostic()> emitError);

} // namespace mlir::fly_rocdl

#endif // FLYDSL_DIALECT_FLYROCDL_UTILS_TDMATOMBUILDER_H
