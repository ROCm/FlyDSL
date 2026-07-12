// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/MathExtras.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_rocdl {

//===----------------------------------------------------------------------===//
// CopyOpGFX1250TDM2DType — 2D TDM async Global<->LDS copy
//
// Ports the 2D (non-gather) descriptor build of
// python/flydsl/expr/rocdl/tdm_ops.py::make_tensor_descriptor_2d (including the
// runtime oob_outer_bound / ragged-tile clamp via the oob_outer atom state) plus
// tensor_load_2d / tensor_store_2d into a copy atom. The Global-side memref
// layout supplies the tile geometry (outer/inner extent + outer stride); the
// atom parameters supply padding, warp count, and cache policy.
//===----------------------------------------------------------------------===//

namespace {

// (encoded_interval, encoded_amount) for the TDM padding bitfield, following the
// Triton TDMUtility convention used by compute_padding_encoding().
struct PadEncoding {
  int32_t interval = 0;
  int32_t amount = 0;
  bool enable = false;
};

PadEncoding computePadEncoding(int32_t padIntervalElems, int32_t padAmountElems, int32_t elemBits) {
  PadEncoding e;
  if (padIntervalElems <= 0 || padAmountElems <= 0)
    return e;
  int32_t intervalDw = padIntervalElems * elemBits / 32;
  int32_t amountDw = padAmountElems * elemBits / 32;
  if (intervalDw <= 0 || amountDw <= 0)
    return e;
  e.interval = llvm::Log2_32(static_cast<uint32_t>(intervalDw)) - 1;
  e.amount = amountDw - 1;
  e.enable = true;
  return e;
}

// Mirror of tdm_ops.compute_warp_distribution for a 2D [outer, inner] block.
void computeWarpDistribution(int32_t outer, int32_t inner, int32_t numWarps, int32_t &warpsOuter,
                             int32_t &warpsInner, int32_t &bpwOuter, int32_t &bpwInner) {
  int32_t dims[2] = {outer, inner};
  int32_t warps[2] = {1, 1};
  int32_t remaining = numWarps;
  for (int i = 0; i < 2; ++i) {
    while (remaining > 1 && warps[i] * 2 <= dims[i]) {
      warps[i] *= 2;
      remaining /= 2;
    }
  }
  if (remaining > 1)
    warps[1] *= remaining;
  warpsOuter = warps[0];
  warpsInner = warps[1];
  bpwOuter = (outer + warps[0] - 1) / warps[0];
  bpwInner = (inner + warps[1] - 1) / warps[1];
}

Value i32Const(OpBuilder &b, Location loc, int32_t v) {
  return arith::ConstantIntOp::create(b, loc, v, 32);
}

} // namespace

// Stateful: two i32 fields — workgroup_mask (MCAST), and oob_outer (runtime
// outer-dim tensor bound for ragged tiles; default INT32_MAX = no clamp).
std::optional<unsigned> CopyOpGFX1250TDM2DType::getFieldIndex(AtomStateField field) {
  switch (field) {
  case AtomStateField::WorkgroupMask:
    return 0;
  case AtomStateField::OobOuter:
    return 1;
  default:
    return std::nullopt;
  }
}

Type CopyOpGFX1250TDM2DType::getConvertedType(MLIRContext *ctx) const {
  auto i32 = IntegerType::get(ctx, 32);
  return LLVM::LLVMStructType::getLiteral(ctx, {i32, i32});
}

Value CopyOpGFX1250TDM2DType::getDefaultState(OpBuilder &builder, Location loc) const {
  auto structTy = cast<LLVM::LLVMStructType>(getConvertedType(builder.getContext()));
  Value state = LLVM::UndefOp::create(builder, loc, structTy);
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  state = LLVM::InsertValueOp::create(
      builder, loc, state, zero, ArrayRef<int64_t>{*getFieldIndex(AtomStateField::WorkgroupMask)});
  // oob_outer default INT32_MAX: tensor_dim1 >= tile_dim1 => whole tile in-bounds
  // (byte-identical behavior to the pre-OOB atom for callers that never set it).
  Value noClamp = arith::ConstantIntOp::create(builder, loc, 0x7FFFFFFF, 32);
  state = LLVM::InsertValueOp::create(builder, loc, state, noClamp,
                                      ArrayRef<int64_t>{*getFieldIndex(AtomStateField::OobOuter)});
  return state;
}

Value CopyOpGFX1250TDM2DType::setAtomState(OpBuilder &builder, Location loc, Value atomStruct,
                                           Attribute fieldAttr, Value fieldValue) const {
  auto fieldStr = dyn_cast<StringAttr>(fieldAttr);
  if (!fieldStr)
    return nullptr;
  auto field = symbolizeAtomStateField(fieldStr.getValue());
  if (!field)
    return nullptr;
  auto idx = getFieldIndex(*field);
  if (!idx)
    return nullptr;
  return LLVM::InsertValueOp::create(builder, loc, atomStruct, fieldValue, ArrayRef<int64_t>{*idx});
}

// TDM is a whole-tile DMA: one copy_atom_call transfers the entire 2D tile, so
// the expand-copy lowering must keep the full tiled memref (not decompose it).
bool CopyOpGFX1250TDM2DType::isWholeTileCopy() const { return true; }

Attribute CopyOpGFX1250TDM2DType::getThrLayout() const { return FxLayout(FxC(1), FxC(1)); }

Attribute CopyOpGFX1250TDM2DType::getThrBitLayoutSrc() const {
  return FxLayout(FxShape(FxC(1), FxC(1)), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpGFX1250TDM2DType::getThrBitLayoutDst() const {
  return FxLayout(FxShape(FxC(1), FxC(1)), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpGFX1250TDM2DType::getThrBitLayoutRef() const {
  return FxLayout(FxShape(FxC(1), FxC(1)), FxStride(FxC(1), FxC(1)));
}

LogicalResult CopyOpGFX1250TDM2DType::verify(function_ref<InFlightDiagnostic()> emitError,
                                             int32_t numWarps, int32_t padInterval,
                                             int32_t padAmount, int32_t cacheModifier) {
  if (numWarps < 1 || (numWarps & (numWarps - 1)) != 0)
    return emitError() << "numWarps must be a positive power of two, got " << numWarps;
  if ((padInterval == 0) != (padAmount == 0))
    return emitError() << "padInterval and padAmount must both be zero or both non-zero";
  return success();
}

FailureOr<Value> CopyOpGFX1250TDM2DType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                         Type resultTy, Type copyAtomTyArg,
                                                         Type srcTyArg, Type dstTyArg,
                                                         Value atomVal, Value src,
                                                         Value dst) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, atomVal, src, dst)))
    return failure();
  return Value{};
}

FailureOr<Value> CopyOpGFX1250TDM2DType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                         Type resultTy, Type copyAtomTyArg,
                                                         Type srcTyArg, Type dstTyArg,
                                                         Type predTyArg, Value atomVal, Value src,
                                                         Value dst, Value pred) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, predTyArg, atomVal, src,
                          dst, pred)))
    return failure();
  return Value{};
}

LogicalResult CopyOpGFX1250TDM2DType::emitAtomCall(OpBuilder &builder, Location loc,
                                                   Type copyAtomTyArg, Type srcMemTyArg,
                                                   Type dstMemTyArg, Value atomVal, Value src,
                                                   Value dst) const {
  auto srcMemTy = dyn_cast<fly::MemRefType>(srcMemTyArg);
  auto dstMemTy = dyn_cast<fly::MemRefType>(dstMemTyArg);
  if (!srcMemTy || !dstMemTy)
    return failure();

  bool srcGlobal = isGenericAddressSpace<fly::AddressSpace::Global>(srcMemTy.getAddressSpace());
  bool srcShared = isGenericAddressSpace<fly::AddressSpace::Shared>(srcMemTy.getAddressSpace());
  bool dstGlobal = isGenericAddressSpace<fly::AddressSpace::Global>(dstMemTy.getAddressSpace());
  bool dstShared = isGenericAddressSpace<fly::AddressSpace::Shared>(dstMemTy.getAddressSpace());

  bool isLoad = srcGlobal && dstShared;
  bool isStore = srcShared && dstGlobal;
  if (!isLoad && !isStore)
    return failure();

  // Global side carries the tile geometry; LDS side is the scratch tile.
  fly::MemRefType glbMemTy = isLoad ? srcMemTy : dstMemTy;
  Value glbPtr = isLoad ? src : dst;
  Value ldsPtr = isLoad ? dst : src;

  auto layout = dyn_cast<fly::LayoutAttr>(glbMemTy.getLayout());
  if (!layout || !layout.isStatic() || layout.rank() != 2)
    return failure();

  int32_t outer = layout.getShape().at(0).getLeafAsInt().getValue();
  int32_t inner = layout.getShape().at(1).getLeafAsInt().getValue();
  int32_t outerStride = layout.getStride().at(0).getLeafAsInt().getValue();
  int32_t innerStride = layout.getStride().at(1).getLeafAsInt().getValue();

  int32_t elemBits = glbMemTy.getElemTy().getIntOrFloatBitWidth();
  if (elemBits % 8 != 0)
    return failure(); // TDM operates on byte-granular elements
  int32_t elemBytes = elemBits / 8;
  if ((elemBytes & (elemBytes - 1)) != 0)
    return failure(); // data_size = log2(elem_bytes)
  int32_t dataSizeCode = llvm::Log2_32(static_cast<uint32_t>(elemBytes));

  int32_t numWarps = getNumWarps();
  int32_t warpsOuter, warpsInner, bpwOuter, bpwInner;
  computeWarpDistribution(outer, inner, numWarps, warpsOuter, warpsInner, bpwOuter, bpwInner);

  bool padActive = getPadInterval() > 0 && getPadAmount() > 0;
  // Stores fold LDS padding into the tile extent (no de-pad on the store path).
  int32_t ldsInnerStride = (padActive && !isStore) ? (inner + getPadAmount()) : inner;

  Type i64Ty = builder.getI64Type();
  Value glbBase = LLVM::PtrToIntOp::create(builder, loc, i64Ty, glbPtr);
  Value ldsBase = LLVM::PtrToIntOp::create(builder, loc, builder.getI32Type(), ldsPtr);

  // Per-wave outer offset (in outer-dim elements), needed both for the descriptor
  // address and the OOB tensor-bound clamp below.
  Value startOuter;
  Value glbAddr, ldsAddr;
  if (numWarps > 1) {
    // Per-wave outer/inner offsets from the hardware wave id.
    Value waveId = ROCDL::WaveId::create(builder, loc, builder.getI32Type());
    Value wOuterN = i32Const(builder, loc, warpsOuter);
    Value coordOuter = arith::RemUIOp::create(builder, loc, waveId, wOuterN);
    Value coordInner = arith::DivUIOp::create(builder, loc, waveId, wOuterN);
    Value warpOffOuter =
        arith::MulIOp::create(builder, loc, coordOuter, i32Const(builder, loc, bpwOuter));
    Value warpOffInner =
        arith::MulIOp::create(builder, loc, coordInner, i32Const(builder, loc, bpwInner));
    startOuter = warpOffOuter;

    auto toI64 = [&](Value v) { return arith::ExtUIOp::create(builder, loc, i64Ty, v); };
    Value glbElemOff = arith::AddIOp::create(
        builder, loc,
        arith::MulIOp::create(builder, loc, toI64(warpOffOuter),
                              arith::ConstantIntOp::create(builder, loc, outerStride, 64)),
        arith::MulIOp::create(builder, loc, toI64(warpOffInner),
                              arith::ConstantIntOp::create(builder, loc, innerStride, 64)));
    Value glbByteOff = arith::MulIOp::create(
        builder, loc, glbElemOff, arith::ConstantIntOp::create(builder, loc, elemBytes, 64));
    glbAddr = arith::AddIOp::create(builder, loc, glbBase, glbByteOff);

    Value ldsElemOff = arith::AddIOp::create(
        builder, loc,
        arith::MulIOp::create(builder, loc, warpOffOuter, i32Const(builder, loc, ldsInnerStride)),
        warpOffInner);
    Value ldsByteOff =
        arith::MulIOp::create(builder, loc, ldsElemOff, i32Const(builder, loc, elemBytes));
    ldsAddr = arith::AddIOp::create(builder, loc, ldsBase, ldsByteOff);
  } else {
    // Single wave covers the whole tile; the memref base pointers are the
    // descriptor addresses directly.
    glbAddr = glbBase;
    ldsAddr = ldsBase;
    startOuter = i32Const(builder, loc, 0);
  }

  // ================================================================
  // GROUP0 (vector<4xi32>): pred, lds_addr, glb_lo, glb_hi | type
  //
  // The global address is computed and split in full 64 bits: glb_lo is the low
  // word and glb_hi is (glbAddr >> 32). In a K-reduction loop the kernel passes
  // a per-iteration sliced global memref (base advanced by k*delta); because the
  // split happens on the full i64 address, any carry out of the low 32 bits
  // lands in glb_hi automatically. This makes the atom carry-safe by
  // construction over >4 GiB buffers — no separate addr64 helper is required
  // (unlike the raw tdm_ops addr-lo shortcut).
  // ================================================================
  Value g0s0 = i32Const(builder, loc, /*pred=*/1);
  Value g0s1 = ldsAddr;
  Value g0s2 = LLVM::TruncOp::create(builder, loc, builder.getI32Type(), glbAddr);
  Value glbHiRaw = LLVM::LShrOp::create(builder, loc, glbAddr,
                                        arith::ConstantIntOp::create(builder, loc, 32, 64));
  Value glbHi = LLVM::TruncOp::create(builder, loc, builder.getI32Type(), glbHiRaw);
  Value g0s3 = arith::OrIOp::create(builder, loc, glbHi,
                                    i32Const(builder, loc, /*type field [31:30]=2*/ 1 << 31));
  Value dgroup0 = vector::FromElementsOp::create(
      builder, loc, VectorType::get({4}, builder.getI32Type()), ValueRange{g0s0, g0s1, g0s2, g0s3});

  // ================================================================
  // GROUP1 (vector<8xi32>): all compile-time — config + dims + tile + stride
  // ================================================================
  int32_t tdim0 = bpwInner;  // innermost extent per warp (compile-time)
  int32_t tileD0 = bpwInner; // block dim0 per warp
  int32_t tileD1 = bpwOuter; // block dim1 per warp (full per-warp tile extent)

  int32_t padInterval = getPadInterval();
  int32_t padAmount = getPadAmount();
  if (isStore && padActive) {
    tileD0 += padAmount;
    padInterval = 0;
    padAmount = 0;
  }
  PadEncoding pad = computePadEncoding(padInterval, padAmount, elemBits);

  int32_t g1s0Upper = (dataSizeCode << 16) | ((pad.enable ? 1 : 0) << 20) | (pad.interval << 22) |
                      (pad.amount << 25);
  int32_t g1s1 = (tdim0 & 0xFFFF) << 16;
  int32_t g1s4 = tileD1 & 0xFFFF;
  int32_t g1s5 = outerStride;

  // GROUP1 config word [15:0] carries the MCAST workgroup mask (runtime atom
  // state); upper bits are the compile-time descriptor config.
  Value maskRaw = LLVM::ExtractValueOp::create(
      builder, loc, atomVal, ArrayRef<int64_t>{*getFieldIndex(AtomStateField::WorkgroupMask)});
  Value maskLow = arith::AndIOp::create(builder, loc, maskRaw, i32Const(builder, loc, 0xFFFF));
  Value g1s0 = arith::OrIOp::create(builder, loc, i32Const(builder, loc, g1s0Upper), maskLow);

  // tensor_dim1 is runtime: max(0, oob_outer - startOuter). Default oob_outer =
  // INT32_MAX makes this >= tileD1 (whole tile in-bounds). When the caller sets
  // oob_outer = rows-from-tile-start, the partial last tile exceeds the tensor
  // bound and the HW OOB-handles the overhang (load: zero-fill; store: drop).
  // tile_dim1 (g1s4) stays the full static per-warp extent.
  Value oobOuter = LLVM::ExtractValueOp::create(
      builder, loc, atomVal, ArrayRef<int64_t>{*getFieldIndex(AtomStateField::OobOuter)});
  Value tdim1 = arith::MaxSIOp::create(builder, loc,
                                       arith::SubIOp::create(builder, loc, oobOuter, startOuter),
                                       i32Const(builder, loc, 0));
  Value c16 = i32Const(builder, loc, 16);
  Value mask16 = i32Const(builder, loc, 0xFFFF);
  Value td1Lo = arith::AndIOp::create(builder, loc, tdim1, mask16);
  Value g1s2 = arith::OrIOp::create(
      builder, loc, i32Const(builder, loc, (tdim0 >> 16) & 0xFFFF),
      arith::ShLIOp::create(builder, loc, td1Lo, c16));
  Value td1Hi = arith::AndIOp::create(builder, loc,
                                      arith::ShRUIOp::create(builder, loc, tdim1, c16), mask16);
  Value g1s3 = arith::OrIOp::create(builder, loc, td1Hi, i32Const(builder, loc, tileD0 << 16));

  Value dgroup1 = vector::FromElementsOp::create(
      builder, loc, VectorType::get({8}, builder.getI32Type()),
      ValueRange{g1s0, i32Const(builder, loc, g1s1), g1s2, g1s3, i32Const(builder, loc, g1s4),
                 i32Const(builder, loc, g1s5), i32Const(builder, loc, 0),
                 i32Const(builder, loc, 0)});

  // Unused descriptor groups.
  Value zero = i32Const(builder, loc, 0);
  Value dg2 = vector::FromElementsOp::create(
      builder, loc, VectorType::get({4}, builder.getI32Type()), ValueRange{zero, zero, zero, zero});
  Value dg3 = vector::FromElementsOp::create(
      builder, loc, VectorType::get({4}, builder.getI32Type()), ValueRange{zero, zero, zero, zero});
  Value dg4 =
      vector::FromElementsOp::create(builder, loc, VectorType::get({8}, builder.getI32Type()),
                                     ValueRange{zero, zero, zero, zero, zero, zero, zero, zero});

  uint32_t cachePolicy = static_cast<uint32_t>(getCacheModifier());
  ArrayAttr noAliasScopes;
  if (isLoad)
    ROCDL::TensorLoadToLDSOp::create(builder, loc, dgroup0, dgroup1, dg2, dg3, dg4, cachePolicy,
                                     noAliasScopes, noAliasScopes, noAliasScopes);
  else
    ROCDL::TensorStoreFromLDSOp::create(builder, loc, dgroup0, dgroup1, dg2, dg3, dg4, cachePolicy,
                                        noAliasScopes, noAliasScopes, noAliasScopes);

  return success();
}

LogicalResult CopyOpGFX1250TDM2DType::emitAtomCall(OpBuilder &builder, Location loc,
                                                   Type copyAtomTyArg, Type srcMemTyArg,
                                                   Type dstMemTyArg, Type predMemTyArg,
                                                   Value atomVal, Value src, Value dst,
                                                   Value pred) const {
  OpBuilder::InsertionGuard guard(builder);
  auto predMemTy = cast<fly::MemRefType>(predMemTyArg);
  Value predVal = LLVM::LoadOp::create(builder, loc, predMemTy.getElemTy(), pred);
  auto ifOp = scf::IfOp::create(builder, loc, TypeRange{}, predVal, /*withElse=*/false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  return emitAtomCall(builder, loc, copyAtomTyArg, srcMemTyArg, dstMemTyArg, atomVal, src, dst);
}

} // namespace mlir::fly_rocdl
