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
#include "flydsl/Dialect/FlyROCDL/Utils/GlobalTensorDesc.h"

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_rocdl {

//===----------------------------------------------------------------------===//
// CopyOpGFX1250TDM2DType — 2D TDM async Global<->LDS copy
//
// 2D (non-gather) port of tdm_ops.py::make_tensor_descriptor_2d. The Global-side
// memref layout supplies the tile geometry; the atom parameters supply padding,
// warp count, and cache policy. Direction is inferred from the address spaces.
//===----------------------------------------------------------------------===//

namespace {

// TDM padding descriptor bitfield:
//   encoded_interval = log2(interval_dw) - 1  -> bits [24:22] (3 bits)
//   encoded_amount   = amount_dw - 1          -> bits [31:25] (7 bits)
//   where *_dw = pad_*_elems * elem_bits / 32
struct PadEncoding {
  int32_t interval = 0;
  int32_t amount = 0;
  bool enable = false;
};

// Encoding for active padding, a disabled encoding when none is requested, or
// failure() when it cannot be represented (not dword-aligned, dword interval not
// a power of two, or a field out of range) — failing here avoids silently
// emitting a wrong descriptor.
FailureOr<PadEncoding> computePadEncoding(int32_t padIntervalElems, int32_t padAmountElems,
                                          int32_t elemBits) {
  PadEncoding e;
  if (padIntervalElems <= 0 || padAmountElems <= 0)
    return e; // disabled
  if ((padIntervalElems * elemBits) % 32 != 0 || (padAmountElems * elemBits) % 32 != 0)
    return failure();
  int32_t intervalDw = padIntervalElems * elemBits / 32;
  int32_t amountDw = padAmountElems * elemBits / 32;
  if (intervalDw <= 0 || amountDw <= 0 || (intervalDw & (intervalDw - 1)) != 0)
    return failure();
  int32_t encInterval = llvm::Log2_32(static_cast<uint32_t>(intervalDw)) - 1;
  int32_t encAmount = amountDw - 1;
  if (encInterval < 0 || encInterval > 0x7 || encAmount < 0 || encAmount > 0x7F)
    return failure();
  e.interval = encInterval;
  e.amount = encAmount;
  e.enable = true;
  return e;
}

// 2D [outer, inner] warp distribution (mirrors tdm_ops.compute_warp_distribution).
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

// Stateful: one i32 field — workgroup_mask (MCAST). The OOB tensor bound is
// carried on the global operand (global_tensor_desc), not as atom state.
std::optional<unsigned> CopyOpGFX1250TDM2DType::getFieldIndex(AtomStateField field) {
  switch (field) {
  case AtomStateField::WorkgroupMask:
    return 0;
  default:
    return std::nullopt;
  }
}

Type CopyOpGFX1250TDM2DType::getConvertedType(MLIRContext *ctx) const {
  return LLVM::LLVMStructType::getLiteral(ctx, {IntegerType::get(ctx, 32)});
}

Value CopyOpGFX1250TDM2DType::getDefaultState(OpBuilder &builder, Location loc) const {
  auto structTy = cast<LLVM::LLVMStructType>(getConvertedType(builder.getContext()));
  Value state = LLVM::UndefOp::create(builder, loc, structTy);
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  state = LLVM::InsertValueOp::create(
      builder, loc, state, zero, ArrayRef<int64_t>{*getFieldIndex(AtomStateField::WorkgroupMask)});
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

// TDM moves the whole 2D tile in one DMA; its geometry lives in the rank-2
// memref layout, so the expand-copy lowering must emit a single rank-2 call.
unsigned CopyOpGFX1250TDM2DType::getCopyRank() const { return 2; }

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
                                             int32_t padAmount, int32_t cacheModifier,
                                             bool atomicBarrier, bool earlyTimeout) {
  if (numWarps < 1 || (numWarps & (numWarps - 1)) != 0)
    return emitError() << "numWarps must be a positive power of two, got " << numWarps;
  if ((padInterval == 0) != (padAmount == 0))
    return emitError() << "padInterval and padAmount must both be zero or both non-zero";
  if (padInterval != 0) {
    if (padInterval < 0 || padAmount < 0)
      return emitError() << "padInterval and padAmount must be non-negative, got " << padInterval
                         << ", " << padAmount;
    // interval_in_dwords is a power of two iff padInterval (in elements) is, since
    // element bits are a power of two; the exact dword/bitfield check needs the
    // element type and runs at lowering.
    if ((padInterval & (padInterval - 1)) != 0)
      return emitError() << "padInterval must be a power of two (in elements), got " << padInterval;
  }
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

  // A global_tensor_desc operand (base pointer + runtime 2D extent) counts as the
  // global side; a plain global memref carries no bound (whole tile in-bounds).
  auto isGlobalLike = [](fly::MemRefType t) {
    return isGenericAddressSpace<fly::AddressSpace::Global>(t.getAddressSpace()) ||
           isTargetAddressSpace<GlobalTensorDescAddressAttr>(t.getAddressSpace());
  };
  bool srcGlobal = isGlobalLike(srcMemTy);
  bool srcShared = isGenericAddressSpace<fly::AddressSpace::Shared>(srcMemTy.getAddressSpace());
  bool dstGlobal = isGlobalLike(dstMemTy);
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

  // Global operand: a global_tensor_desc carries {base ptr, outer extent, inner
  // extent}; a plain global memref is a bare base ptr with no bound (extents
  // default to INT32_MAX => whole tile in-bounds). tensorDim0/tensorDim1 follow
  // the descriptor's innermost-first convention (dim0 = inner, dim1 = outer).
  bool glbIsDesc = isTargetAddressSpace<GlobalTensorDescAddressAttr>(glbMemTy.getAddressSpace());
  Value glbBasePtr = glbIsDesc ? GlobalTensorDesc::base(builder, loc, glbPtr) : glbPtr;
  Value tensorDim0 = glbIsDesc ? GlobalTensorDesc::innerExtent(builder, loc, glbPtr)
                               : i32Const(builder, loc, 0x7FFFFFFF);
  Value tensorDim1 = glbIsDesc ? GlobalTensorDesc::outerExtent(builder, loc, glbPtr)
                               : i32Const(builder, loc, 0x7FFFFFFF);

  Type i64Ty = builder.getI64Type();
  Value glbBase = LLVM::PtrToIntOp::create(builder, loc, i64Ty, glbBasePtr);
  Value ldsBase = LLVM::PtrToIntOp::create(builder, loc, builder.getI32Type(), ldsPtr);

  // Per-wave outer/inner offsets (in elements), needed both for the descriptor
  // address and the per-wave OOB tensor-bound clamp below.
  Value startOuter, startInner;
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
    startInner = warpOffInner;

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
    startInner = i32Const(builder, loc, 0);
  }

  // GROUP0 (vector<4xi32>): pred, lds_addr, glb_lo, glb_hi | type. The global
  // address is split from the full i64, so a K-loop advancing the base carries
  // into glb_hi automatically (carry-safe over >4 GiB buffers).
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

  // GROUP1 (vector<8xi32>): config + dims + tile + stride.
  int32_t tileD0 = bpwInner; // block dim0 per warp
  int32_t tileD1 = bpwOuter; // block dim1 per warp (full per-warp tile extent)

  int32_t padInterval = getPadInterval();
  int32_t padAmount = getPadAmount();
  if (isStore && padActive) {
    tileD0 += padAmount;
    padInterval = 0;
    padAmount = 0;
  }
  FailureOr<PadEncoding> padOr = computePadEncoding(padInterval, padAmount, elemBits);
  if (failed(padOr))
    return mlir::emitError(loc)
           << "gfx1250 TDM: padding (interval=" << padInterval << ", amount=" << padAmount
           << " elements at " << elemBits
           << "-bit) is not encodable — the dword interval must be a power of two and the encoded "
              "fields must fit the descriptor bitfield";
  PadEncoding pad = *padOr;

  int32_t g1s0Upper = (dataSizeCode << 16) | ((getAtomicBarrier() ? 1 : 0) << 18) |
                      ((pad.enable ? 1 : 0) << 20) | ((getEarlyTimeout() ? 1 : 0) << 21) |
                      (pad.interval << 22) | (pad.amount << 25);
  int32_t g1s4 = tileD1 & 0xFFFF;
  int32_t g1s5 = outerStride;

  // Config word [15:0] carries the runtime MCAST workgroup mask; upper bits are
  // the compile-time config.
  Value maskRaw = LLVM::ExtractValueOp::create(
      builder, loc, atomVal, ArrayRef<int64_t>{*getFieldIndex(AtomStateField::WorkgroupMask)});
  Value maskLow = arith::AndIOp::create(builder, loc, maskRaw, i32Const(builder, loc, 0xFFFF));
  Value g1s0 = arith::OrIOp::create(builder, loc, i32Const(builder, loc, g1s0Upper), maskLow);

  // tensor_dim0/tensor_dim1 are per-wave runtime bounds = max(0, globalDim - warp
  // start); the tile dims (g1s4 / tileD0) stay static, so a ragged tile is HW
  // OOB-handled (load: zero-fill; store: drop). A plain global operand supplies
  // globalDim = INT32_MAX => whole tile in-bounds.
  Value c16 = i32Const(builder, loc, 16);
  Value mask16 = i32Const(builder, loc, 0xFFFF);
  Value zeroC = i32Const(builder, loc, 0);
  Value tdim0 = arith::MaxSIOp::create(
      builder, loc, arith::SubIOp::create(builder, loc, tensorDim0, startInner), zeroC);
  Value tdim1 = arith::MaxSIOp::create(
      builder, loc, arith::SubIOp::create(builder, loc, tensorDim1, startOuter), zeroC);
  Value td0Lo = arith::AndIOp::create(builder, loc, tdim0, mask16);
  Value td0Hi =
      arith::AndIOp::create(builder, loc, arith::ShRUIOp::create(builder, loc, tdim0, c16), mask16);
  Value td1Lo = arith::AndIOp::create(builder, loc, tdim1, mask16);
  Value td1Hi =
      arith::AndIOp::create(builder, loc, arith::ShRUIOp::create(builder, loc, tdim1, c16), mask16);
  // s1: tensor_dim0_lo [31:16]
  Value g1s1 = arith::ShLIOp::create(builder, loc, td0Lo, c16);
  // s2: tensor_dim0_hi [15:0] | tensor_dim1_lo [31:16]
  Value g1s2 =
      arith::OrIOp::create(builder, loc, td0Hi, arith::ShLIOp::create(builder, loc, td1Lo, c16));
  // s3: tensor_dim1_hi [15:0] | tile_dim0 [31:16]
  Value g1s3 = arith::OrIOp::create(builder, loc, td1Hi, i32Const(builder, loc, tileD0 << 16));

  Value dgroup1 = vector::FromElementsOp::create(
      builder, loc, VectorType::get({8}, builder.getI32Type()),
      ValueRange{g1s0, g1s1, g1s2, g1s3, i32Const(builder, loc, g1s4), i32Const(builder, loc, g1s5),
                 i32Const(builder, loc, 0), i32Const(builder, loc, 0)});

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
