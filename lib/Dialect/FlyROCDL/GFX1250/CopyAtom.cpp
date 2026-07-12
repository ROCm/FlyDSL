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
// CopyOpGFX1250TDMType — 2D TDM async Global<->LDS copy
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

// N-D warp distribution (mirrors Triton tdmGetWarpDistribution /
// tdmGetAdjustedBlockShape, TDMCommon.h). `tileShape` is tensor dim order
// (0 = outermost). Assigns warps greedily from the outer dims; `bpw[i]` is the
// per-warp block size of dim i.
void computeWarpDistribution(ArrayRef<int32_t> tileShape, int32_t numWarps,
                             SmallVectorImpl<int32_t> &warps, SmallVectorImpl<int32_t> &bpw) {
  unsigned n = tileShape.size();
  warps.assign(n, 1);
  int32_t remaining = numWarps;
  for (unsigned i = 0; i < n && remaining > 1; ++i) {
    while (remaining > 1 && warps[i] * 2 <= tileShape[i]) {
      warps[i] *= 2;
      remaining /= 2;
    }
  }
  if (remaining > 1)
    warps[n - 1] *= remaining;
  bpw.assign(n, 0);
  for (unsigned i = 0; i < n; ++i)
    bpw[i] = (tileShape[i] + warps[i] - 1) / warps[i];
}

Value i32Const(OpBuilder &b, Location loc, int32_t v) {
  return arith::ConstantIntOp::create(b, loc, v, 32);
}

// Sentinel for an unset `outer_stride` state field: fall back to the tile
// memref's static layout stride. Distinct from any real stride (including a
// legitimate 0 broadcast stride), so "unset" and "explicitly 0" don't alias.
constexpr int32_t kOuterStrideUnset = static_cast<int32_t>(0x80000000);

} // namespace

// Stateful: the atom carries the whole TDM N-D descriptor — workgroup_mask
// (MCAST) plus the global tile descriptor. Struct slots: {mask, base,
// extent_0..4 (i32, per-dim tensor extent for OOB), stride_0..3 (i64, per-dim
// tensor stride in elements; innermost stride is assumed 1), imm_offset (i64)}.
// See CopyAtom.td for field semantics.
static constexpr unsigned kMaxTdmRank = 5;

std::optional<unsigned> CopyOpGFX1250TDMType::getFieldIndex(AtomStateField field) {
  switch (field) {
  case AtomStateField::WorkgroupMask:
    return 0;
  case AtomStateField::Base:
    return 1;
  case AtomStateField::Extent0:
    return 2;
  case AtomStateField::Extent1:
    return 3;
  case AtomStateField::Extent2:
    return 4;
  case AtomStateField::Extent3:
    return 5;
  case AtomStateField::Extent4:
    return 6;
  case AtomStateField::Stride0:
    return 7;
  case AtomStateField::Stride1:
    return 8;
  case AtomStateField::Stride2:
    return 9;
  case AtomStateField::Stride3:
    return 10;
  // Byte offset added to the global base at lowering (i64, carry-safe). Lets a
  // K-loop advance the tile by bumping one scalar (imm_offset) instead of
  // re-deriving the base pointer. Default 0 folds away.
  case AtomStateField::ImmOffset:
    return 11;
  default:
    return std::nullopt;
  }
}

// The i'th extent / stride state field (Extent0.., Stride0..).
static AtomStateField extentField(unsigned i) {
  return static_cast<AtomStateField>(static_cast<unsigned>(AtomStateField::Extent0) + i);
}
static AtomStateField strideField(unsigned i) {
  return static_cast<AtomStateField>(static_cast<unsigned>(AtomStateField::Stride0) + i);
}

Type CopyOpGFX1250TDMType::getConvertedType(MLIRContext *ctx) const {
  auto i32 = IntegerType::get(ctx, 32);
  auto i64 = IntegerType::get(ctx, 64);
  auto glbPtr = LLVM::LLVMPointerType::get(ctx, /*addrSpace=*/1);
  SmallVector<Type> fields = {i32, glbPtr};
  fields.append(kMaxTdmRank, i32);     // extent_0..4
  fields.append(kMaxTdmRank - 1, i64); // stride_0..3
  fields.push_back(i64);               // imm_offset
  return LLVM::LLVMStructType::getLiteral(ctx, fields);
}

Value CopyOpGFX1250TDMType::getDefaultState(OpBuilder &builder, Location loc) const {
  auto *ctx = builder.getContext();
  auto structTy = cast<LLVM::LLVMStructType>(getConvertedType(ctx));
  Value state = LLVM::UndefOp::create(builder, loc, structTy);
  auto insert = [&](AtomStateField f, Value v) {
    state =
        LLVM::InsertValueOp::create(builder, loc, state, v, ArrayRef<int64_t>{*getFieldIndex(f)});
  };
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  Value noClamp = arith::ConstantIntOp::create(builder, loc, 0x7FFFFFFF, 32);
  Value strideUnset = arith::ConstantIntOp::create(builder, loc, kOuterStrideUnset, 64);
  Value nullBase =
      LLVM::ZeroOp::create(builder, loc, LLVM::LLVMPointerType::get(ctx, /*addrSpace=*/1));
  Value zero64 = arith::ConstantIntOp::create(builder, loc, 0, 64);
  insert(AtomStateField::WorkgroupMask, zero);
  insert(AtomStateField::Base, nullBase);
  for (unsigned i = 0; i < kMaxTdmRank; ++i)
    insert(extentField(i), noClamp);
  for (unsigned i = 0; i < kMaxTdmRank - 1; ++i)
    insert(strideField(i), strideUnset);
  insert(AtomStateField::ImmOffset, zero64);
  return state;
}

Value CopyOpGFX1250TDMType::setAtomState(OpBuilder &builder, Location loc, Value atomStruct,
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

// TDM moves the whole N-D tile in one DMA; its geometry lives in the rank-N
// memref layout, so the expand-copy lowering must emit a single rank-N call.
unsigned CopyOpGFX1250TDMType::getCopyRank() const { return static_cast<unsigned>(getRank()); }

Attribute CopyOpGFX1250TDMType::getThrLayout() const { return FxLayout(FxC(1), FxC(1)); }

Attribute CopyOpGFX1250TDMType::getThrBitLayoutSrc() const {
  return FxLayout(FxShape(FxC(1), FxC(1)), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpGFX1250TDMType::getThrBitLayoutDst() const {
  return FxLayout(FxShape(FxC(1), FxC(1)), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpGFX1250TDMType::getThrBitLayoutRef() const {
  return FxLayout(FxShape(FxC(1), FxC(1)), FxStride(FxC(1), FxC(1)));
}

LogicalResult CopyOpGFX1250TDMType::verify(function_ref<InFlightDiagnostic()> emitError,
                                           int32_t rank, int32_t numWarps, int32_t padInterval,
                                           int32_t padAmount, int32_t cacheModifier,
                                           bool atomicBarrier, bool earlyTimeout) {
  if (rank < 1 || rank > static_cast<int32_t>(kMaxTdmRank))
    return emitError() << "TDM rank must be in [1, " << kMaxTdmRank << "], got " << rank;
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

FailureOr<Value> CopyOpGFX1250TDMType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                       Type resultTy, Type copyAtomTyArg,
                                                       Type srcTyArg, Type dstTyArg, Value atomVal,
                                                       Value src, Value dst) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, atomVal, src, dst)))
    return failure();
  return Value{};
}

FailureOr<Value> CopyOpGFX1250TDMType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                       Type resultTy, Type copyAtomTyArg,
                                                       Type srcTyArg, Type dstTyArg, Type predTyArg,
                                                       Value atomVal, Value src, Value dst,
                                                       Value pred) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, predTyArg, atomVal, src,
                          dst, pred)))
    return failure();
  return Value{};
}

LogicalResult CopyOpGFX1250TDMType::emitAtomCall(OpBuilder &builder, Location loc,
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

  // The global operand only marks the direction; its base pointer, extents, and
  // strides come from the atom descriptor state. The tile shape is compile-time on
  // the operand layout (tensor dim order: index 0 = outermost, rank-1 = innermost).
  fly::MemRefType glbMemTy = isLoad ? srcMemTy : dstMemTy;
  Value ldsPtr = isLoad ? dst : src;
  int32_t rank = getRank();

  auto layout = dyn_cast<fly::LayoutAttr>(glbMemTy.getLayout());
  if (!layout || !layout.isStaticShape() || layout.rank() != rank)
    return failure();

  SmallVector<int32_t> tileShape(rank);
  for (int32_t i = 0; i < rank; ++i)
    tileShape[i] = layout.getShape().at(i).getLeafAsInt().getValue();
  bool hasStaticStride = layout.isStaticStride();

  int32_t elemBits = glbMemTy.getElemTy().getIntOrFloatBitWidth();
  if (elemBits % 8 != 0)
    return failure(); // TDM operates on byte-granular elements
  int32_t elemBytes = elemBits / 8;
  if ((elemBytes & (elemBytes - 1)) != 0)
    return failure(); // data_size = log2(elem_bytes)
  int32_t dataSizeCode = llvm::Log2_32(static_cast<uint32_t>(elemBytes));

  int32_t numWarps = getNumWarps();
  SmallVector<int32_t> warps, bpw;
  computeWarpDistribution(tileShape, numWarps, warps, bpw);

  bool padActive = getPadInterval() > 0 && getPadAmount() > 0;

  Type i64Ty = builder.getI64Type();
  Value zeroC = i32Const(builder, loc, 0);
  Value c16 = i32Const(builder, loc, 16);
  Value mask16 = i32Const(builder, loc, 0xFFFF);
  auto stateField = [&](AtomStateField f) {
    return LLVM::ExtractValueOp::create(builder, loc, atomVal,
                                        ArrayRef<int64_t>{*getFieldIndex(f)});
  };

  // Per-dim (tensor order) tensor extent (i32) and stride in elements (i64). The
  // innermost stride (dim rank-1) is assumed 1; stride_i covers dims 0..rank-2,
  // falling back to the static layout stride when the state slot is left unset.
  SmallVector<Value> extent(rank), strideElems(rank);
  for (int32_t i = 0; i < rank; ++i)
    extent[i] = stateField(extentField(i));
  for (int32_t i = 0; i < rank - 1; ++i) {
    Value st = stateField(strideField(i));
    if (hasStaticStride) {
      int64_t s = layout.getStride().at(i).getLeafAsInt().getValue();
      Value unset =
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, st,
                                arith::ConstantIntOp::create(builder, loc, kOuterStrideUnset, 64));
      strideElems[i] = arith::SelectOp::create(
          builder, loc, unset, arith::ConstantIntOp::create(builder, loc, s, 64), st);
    } else {
      strideElems[i] = st; // no static fallback; sentinel faults loudly
    }
  }
  strideElems[rank - 1] = arith::ConstantIntOp::create(builder, loc, 1, 64); // innermost contiguous

  Value glbBasePtr = stateField(AtomStateField::Base);
  Value glbBase = LLVM::PtrToIntOp::create(builder, loc, i64Ty, glbBasePtr);
  // i64 byte offset (default 0) added to the base so a K-loop can advance the tile
  // by bumping one scalar; the carry into glb_hi is handled by the i64 split below.
  glbBase = arith::AddIOp::create(builder, loc, glbBase, stateField(AtomStateField::ImmOffset));
  Value ldsBase = LLVM::PtrToIntOp::create(builder, loc, builder.getI32Type(), ldsPtr);

  // Compile-time LDS per-warp strides (elements): innermost contiguous, with row
  // padding folded into the next-outer stride on the load path.
  SmallVector<int32_t> ldsStride(rank);
  ldsStride[rank - 1] = 1;
  if (rank >= 2) {
    int32_t innerRow = tileShape[rank - 1];
    if (padActive && !isStore)
      innerRow += getPadAmount();
    ldsStride[rank - 2] = innerRow;
    for (int32_t i = rank - 3; i >= 0; --i)
      ldsStride[i] = tileShape[i + 1] * ldsStride[i + 1];
  }

  // Per-wave start offsets (elements, tensor order); wave id decomposed with dim 0
  // fastest-varying, matching the warp distribution order.
  SmallVector<Value> warpOff(rank, zeroC);
  Value glbAddr = glbBase, ldsAddr = ldsBase;
  if (numWarps > 1) {
    Value waveId = ROCDL::WaveId::create(builder, loc, builder.getI32Type());
    Value rem = waveId;
    for (int32_t i = 0; i < rank; ++i) {
      Value wN = i32Const(builder, loc, warps[i]);
      Value coord = arith::RemUIOp::create(builder, loc, rem, wN);
      rem = arith::DivUIOp::create(builder, loc, rem, wN);
      warpOff[i] = arith::MulIOp::create(builder, loc, coord, i32Const(builder, loc, bpw[i]));
    }
    auto toI64 = [&](Value v) { return arith::ExtUIOp::create(builder, loc, i64Ty, v); };
    Value glbElemOff = arith::ConstantIntOp::create(builder, loc, 0, 64);
    Value ldsElemOff = zeroC;
    for (int32_t i = 0; i < rank; ++i) {
      glbElemOff = arith::AddIOp::create(
          builder, loc, glbElemOff,
          arith::MulIOp::create(builder, loc, toI64(warpOff[i]), strideElems[i]));
      ldsElemOff = arith::AddIOp::create(
          builder, loc, ldsElemOff,
          arith::MulIOp::create(builder, loc, warpOff[i], i32Const(builder, loc, ldsStride[i])));
    }
    Value elemBytes64 = arith::ConstantIntOp::create(builder, loc, elemBytes, 64);
    glbAddr = arith::AddIOp::create(builder, loc, glbBase,
                                    arith::MulIOp::create(builder, loc, glbElemOff, elemBytes64));
    Value ldsByteOff =
        arith::MulIOp::create(builder, loc, ldsElemOff, i32Const(builder, loc, elemBytes));
    ldsAddr = arith::AddIOp::create(builder, loc, ldsBase, ldsByteOff);
  }

  // GROUP0 (vector<4xi32>): pred, lds_addr, glb_lo, glb_hi | type. The global
  // address is split from the full i64, so a K-loop advancing the base carries
  // into glb_hi automatically (carry-safe over >4 GiB buffers).
  Value g0s2 = LLVM::TruncOp::create(builder, loc, builder.getI32Type(), glbAddr);
  Value glbHiRaw = LLVM::LShrOp::create(builder, loc, glbAddr,
                                        arith::ConstantIntOp::create(builder, loc, 32, 64));
  Value g0s3 = arith::OrIOp::create(
      builder, loc, LLVM::TruncOp::create(builder, loc, builder.getI32Type(), glbHiRaw),
      i32Const(builder, loc, /*type field [31:30]=2*/ 1 << 31));
  Value dgroup0 = vector::FromElementsOp::create(
      builder, loc, VectorType::get({4}, builder.getI32Type()),
      ValueRange{i32Const(builder, loc, /*pred=*/1), ldsAddr, g0s2, g0s3});

  // Padding config (folded into the tile extent on the store path).
  int32_t padInterval = getPadInterval();
  int32_t padAmount = getPadAmount();
  if (isStore && padActive) {
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

  // Descriptor dims are innermost-first: descriptor dim j maps to tensor dim
  // (rank-1-j). tensor_dim_j is the per-wave OOB clamp max(0, extent - warp start);
  // tile_dim_j is the static per-warp block size. Store folds row padding into the
  // innermost tile dim (descriptor dim 0).
  auto descTensorDim = [&](int32_t j) -> Value { // runtime i32, clamped
    int32_t d = rank - 1 - j;
    return arith::MaxSIOp::create(
        builder, loc, arith::SubIOp::create(builder, loc, extent[d], warpOff[d]), zeroC);
  };
  auto descTileDim = [&](int32_t j) -> int32_t {
    int32_t d = rank - 1 - j;
    int32_t t = bpw[d];
    if (j == 0 && isStore && padActive)
      t += getPadAmount();
    return t;
  };
  // 48-bit stride slots: descriptor stride k = stride of tensor dim (rank-2-k).
  auto descStrideLo32 = [&](int32_t k) -> Value {
    return LLVM::TruncOp::create(builder, loc, builder.getI32Type(), strideElems[rank - 2 - k]);
  };
  auto descStrideHi16 = [&](int32_t k) -> Value {
    Value hi = LLVM::LShrOp::create(builder, loc, strideElems[rank - 2 - k],
                                    arith::ConstantIntOp::create(builder, loc, 32, 64));
    return arith::AndIOp::create(
        builder, loc, LLVM::TruncOp::create(builder, loc, builder.getI32Type(), hi), mask16);
  };
  auto lo16 = [&](Value v) { return arith::AndIOp::create(builder, loc, v, mask16); };
  auto hi16 = [&](Value v) {
    return arith::AndIOp::create(builder, loc, arith::ShRUIOp::create(builder, loc, v, c16),
                                 mask16);
  };
  auto shl16 = [&](Value v) { return arith::ShLIOp::create(builder, loc, v, c16); };
  auto orr = [&](Value a, Value b) { return arith::OrIOp::create(builder, loc, a, b); };

  // GROUP1: config | mask, tensor_dim0/1, tile_dim0/1/2, stride0/1.
  int32_t g1s0Upper = (dataSizeCode << 16) | ((getAtomicBarrier() ? 1 : 0) << 18) |
                      ((pad.enable ? 1 : 0) << 20) | ((getEarlyTimeout() ? 1 : 0) << 21) |
                      (pad.interval << 22) | (pad.amount << 25);
  Value maskLow = lo16(stateField(AtomStateField::WorkgroupMask));
  Value g1s0 = orr(i32Const(builder, loc, g1s0Upper), maskLow);

  Value td0 = descTensorDim(0);
  Value td1 = rank >= 2 ? descTensorDim(1) : zeroC;
  int32_t tile0 = descTileDim(0);
  int32_t tile1 = rank >= 2 ? descTileDim(1) : 0;
  int32_t tile2 = rank >= 3 ? descTileDim(2) : 0;
  Value g1s1 = shl16(lo16(td0));                                    // tensor_dim0 lo16 in [31:16]
  Value g1s2 = orr(hi16(td0), shl16(lo16(td1)));                    // dim0 hi16 | dim1 lo16
  Value g1s3 = orr(hi16(td1), i32Const(builder, loc, tile0 << 16)); // dim1 hi16 | tile0
  int32_t g1s4c = (tile1 & 0xFFFF) | (tile2 << 16);
  Value g1s5 = zeroC, g1s6 = zeroC, g1s7 = zeroC;
  if (rank >= 2) {
    g1s5 = descStrideLo32(0);
    g1s6 = descStrideHi16(0);
    if (rank >= 3) {
      g1s6 = orr(g1s6, shl16(lo16(descStrideLo32(1))));
      g1s7 = orr(hi16(descStrideLo32(1)), shl16(descStrideHi16(1)));
    }
  }
  Value dgroup1 = vector::FromElementsOp::create(
      builder, loc, VectorType::get({8}, builder.getI32Type()),
      ValueRange{g1s0, g1s1, g1s2, g1s3, i32Const(builder, loc, g1s4c), g1s5, g1s6, g1s7});

  // GROUP2 (rank>=3): tensor_dim2, tensor_dim3, stride2, tile_dim3.
  Value zero = zeroC;
  Value g2s0 = zero, g2s1 = zero, g2s2 = zero, g2s3 = zero;
  if (rank >= 3) {
    g2s0 = descTensorDim(2);
    if (rank >= 4) {
      g2s1 = descTensorDim(3);
      g2s2 = descStrideLo32(2);
      g2s3 = orr(descStrideHi16(2), i32Const(builder, loc, descTileDim(3) << 16));
    }
  }
  Value dg2 = vector::FromElementsOp::create(
      builder, loc, VectorType::get({4}, builder.getI32Type()), ValueRange{g2s0, g2s1, g2s2, g2s3});

  // GROUP3 (rank==5): stride3, tensor_dim4, tile_dim4.
  Value g3s0 = zero, g3s1 = zero, g3s2 = zero, g3s3 = zero;
  if (rank == 5) {
    Value td4 = descTensorDim(4);
    g3s0 = descStrideLo32(3);
    g3s1 = orr(descStrideHi16(3), shl16(lo16(td4)));
    g3s2 = orr(hi16(td4), i32Const(builder, loc, descTileDim(4) << 16));
  }
  Value dg3 = vector::FromElementsOp::create(
      builder, loc, VectorType::get({4}, builder.getI32Type()), ValueRange{g3s0, g3s1, g3s2, g3s3});

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

LogicalResult CopyOpGFX1250TDMType::emitAtomCall(OpBuilder &builder, Location loc,
                                                 Type copyAtomTyArg, Type srcMemTyArg,
                                                 Type dstMemTyArg, Type predMemTyArg, Value atomVal,
                                                 Value src, Value dst, Value pred) const {
  OpBuilder::InsertionGuard guard(builder);
  auto predMemTy = cast<fly::MemRefType>(predMemTyArg);
  Value predVal = LLVM::LoadOp::create(builder, loc, predMemTy.getElemTy(), pred);
  auto ifOp = scf::IfOp::create(builder, loc, TypeRange{}, predVal, /*withElse=*/false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  return emitAtomCall(builder, loc, copyAtomTyArg, srcMemTyArg, dstMemTyArg, atomVal, src, dst);
}

//===----------------------------------------------------------------------===//
// CopyOpGFX1250TDMGatherType — TDM gather async Global<->LDS copy
//
// Gather port of tdm_ops.py::make_tensor_gather_descriptor. Row indices ride as a
// packed vector<8xi32> atom state field; groups 2/3 of the descriptor carry them.
//===----------------------------------------------------------------------===//

namespace {
// gather_count sentinel: "unset" => fall back to the operand's outer tile extent.
constexpr int32_t kGatherCountUnset = static_cast<int32_t>(0x80000000);
} // namespace

std::optional<unsigned> CopyOpGFX1250TDMGatherType::getFieldIndex(AtomStateField field) {
  switch (field) {
  case AtomStateField::WorkgroupMask:
    return 0;
  case AtomStateField::Base:
    return 1;
  case AtomStateField::Extent0: // tensor_dim1 (row count, OOB on indices)
    return 2;
  case AtomStateField::Extent1: // tensor_dim0 (row width, column OOB)
    return 3;
  case AtomStateField::Stride0: // row stride in elements
    return 4;
  case AtomStateField::RowIndices: // packed vector<8xi32>
    return 5;
  case AtomStateField::GatherCount: // number of valid indices
    return 6;
  case AtomStateField::ImmOffset: // i64 byte offset added to base (K advance)
    return 7;
  default:
    return std::nullopt;
  }
}

Type CopyOpGFX1250TDMGatherType::getConvertedType(MLIRContext *ctx) const {
  auto i32 = IntegerType::get(ctx, 32);
  auto i64 = IntegerType::get(ctx, 64);
  auto glbPtr = LLVM::LLVMPointerType::get(ctx, /*addrSpace=*/1);
  auto idxVec = VectorType::get({8}, i32);
  return LLVM::LLVMStructType::getLiteral(ctx, {i32, glbPtr, i32, i32, i32, idxVec, i32, i64});
}

Value CopyOpGFX1250TDMGatherType::getDefaultState(OpBuilder &builder, Location loc) const {
  auto *ctx = builder.getContext();
  auto structTy = cast<LLVM::LLVMStructType>(getConvertedType(ctx));
  Value state = LLVM::UndefOp::create(builder, loc, structTy);
  auto insert = [&](AtomStateField f, Value v) {
    state =
        LLVM::InsertValueOp::create(builder, loc, state, v, ArrayRef<int64_t>{*getFieldIndex(f)});
  };
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  Value zero64 = arith::ConstantIntOp::create(builder, loc, 0, 64);
  Value noClamp = arith::ConstantIntOp::create(builder, loc, 0x7FFFFFFF, 32);
  Value strideUnset = arith::ConstantIntOp::create(builder, loc, kOuterStrideUnset, 32);
  Value countUnset = arith::ConstantIntOp::create(builder, loc, kGatherCountUnset, 32);
  Value nullBase =
      LLVM::ZeroOp::create(builder, loc, LLVM::LLVMPointerType::get(ctx, /*addrSpace=*/1));
  Value zeroIdx = LLVM::ZeroOp::create(builder, loc, VectorType::get({8}, builder.getI32Type()));
  insert(AtomStateField::WorkgroupMask, zero);
  insert(AtomStateField::Base, nullBase);
  insert(AtomStateField::Extent0, noClamp);
  insert(AtomStateField::Extent1, noClamp);
  insert(AtomStateField::Stride0, strideUnset);
  insert(AtomStateField::RowIndices, zeroIdx);
  insert(AtomStateField::GatherCount, countUnset);
  insert(AtomStateField::ImmOffset, zero64);
  return state;
}

Value CopyOpGFX1250TDMGatherType::setAtomState(OpBuilder &builder, Location loc, Value atomStruct,
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

unsigned CopyOpGFX1250TDMGatherType::getCopyRank() const { return 2; }

Attribute CopyOpGFX1250TDMGatherType::getThrLayout() const { return FxLayout(FxC(1), FxC(1)); }

Attribute CopyOpGFX1250TDMGatherType::getThrBitLayoutSrc() const {
  return FxLayout(FxShape(FxC(1), FxC(1)), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpGFX1250TDMGatherType::getThrBitLayoutDst() const {
  return FxLayout(FxShape(FxC(1), FxC(1)), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpGFX1250TDMGatherType::getThrBitLayoutRef() const {
  return FxLayout(FxShape(FxC(1), FxC(1)), FxStride(FxC(1), FxC(1)));
}

LogicalResult CopyOpGFX1250TDMGatherType::verify(function_ref<InFlightDiagnostic()> emitError,
                                                 int32_t indexSize, int32_t padInterval,
                                                 int32_t padAmount, int32_t cacheModifier) {
  if (indexSize != 16 && indexSize != 32)
    return emitError() << "indexSize must be 16 or 32, got " << indexSize;
  if ((padInterval == 0) != (padAmount == 0))
    return emitError() << "padInterval and padAmount must both be zero or both non-zero";
  if (padInterval != 0) {
    if (padInterval < 0 || padAmount < 0)
      return emitError() << "padInterval and padAmount must be non-negative, got " << padInterval
                         << ", " << padAmount;
    if ((padInterval & (padInterval - 1)) != 0)
      return emitError() << "padInterval must be a power of two (in elements), got " << padInterval;
  }
  return success();
}

FailureOr<Value> CopyOpGFX1250TDMGatherType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                             Type resultTy, Type copyAtomTyArg,
                                                             Type srcTyArg, Type dstTyArg,
                                                             Value atomVal, Value src,
                                                             Value dst) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, atomVal, src, dst)))
    return failure();
  return Value{};
}

FailureOr<Value> CopyOpGFX1250TDMGatherType::emitAtomCallSSA(
    OpBuilder &builder, Location loc, Type resultTy, Type copyAtomTyArg, Type srcTyArg,
    Type dstTyArg, Type predTyArg, Value atomVal, Value src, Value dst, Value pred) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, predTyArg, atomVal, src,
                          dst, pred)))
    return failure();
  return Value{};
}

LogicalResult CopyOpGFX1250TDMGatherType::emitAtomCall(OpBuilder &builder, Location loc,
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

  // The global operand marks the direction and supplies the tile shape (outer =
  // rows to gather, inner = per-row width). Base / extents / stride / indices come
  // from atom state.
  fly::MemRefType glbMemTy = isLoad ? srcMemTy : dstMemTy;
  Value ldsPtr = isLoad ? dst : src;

  auto layout = dyn_cast<fly::LayoutAttr>(glbMemTy.getLayout());
  if (!layout || !layout.isStaticShape() || layout.rank() != 2)
    return failure();

  int32_t outer = layout.getShape().at(0).getLeafAsInt().getValue();    // rows in tile
  int32_t rowWidth = layout.getShape().at(1).getLeafAsInt().getValue(); // tile_dim0
  bool hasStaticOuterStride = layout.isStaticStride();

  int32_t elemBits = glbMemTy.getElemTy().getIntOrFloatBitWidth();
  if (elemBits % 8 != 0)
    return failure();
  int32_t elemBytes = elemBits / 8;
  if ((elemBytes & (elemBytes - 1)) != 0)
    return failure();
  int32_t dataSizeCode = llvm::Log2_32(static_cast<uint32_t>(elemBytes));

  auto stateField = [&](AtomStateField f) {
    return LLVM::ExtractValueOp::create(builder, loc, atomVal,
                                        ArrayRef<int64_t>{*getFieldIndex(f)});
  };
  Value glbBasePtr = stateField(AtomStateField::Base);
  Value tensorDim1 = stateField(AtomStateField::Extent0); // rows (OOB on indices)
  Value tensorDim0 = stateField(AtomStateField::Extent1); // row width (column OOB)
  Value stateStride = stateField(AtomStateField::Stride0);
  Value rowIndices = stateField(AtomStateField::RowIndices);
  Value countState = stateField(AtomStateField::GatherCount);
  Value immOffset = stateField(AtomStateField::ImmOffset);

  // Row stride: state value overrides; unset sentinel falls back to the static
  // layout stride (dynamic layout must set it or the descriptor stride faults).
  Value outerStride;
  if (hasStaticOuterStride) {
    int32_t s = layout.getStride().at(0).getLeafAsInt().getValue();
    Value unset = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, stateStride,
                                        i32Const(builder, loc, kOuterStrideUnset));
    outerStride =
        arith::SelectOp::create(builder, loc, unset, i32Const(builder, loc, s), stateStride);
  } else {
    outerStride = stateStride;
  }

  // gather_count: state value overrides; unset falls back to the tile row count.
  Value countUnset = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, countState,
                                           i32Const(builder, loc, kGatherCountUnset));
  Value gatherCount =
      arith::SelectOp::create(builder, loc, countUnset, i32Const(builder, loc, outer), countState);

  Type i64Ty = builder.getI64Type();
  Value glbBase = LLVM::PtrToIntOp::create(builder, loc, i64Ty, glbBasePtr);
  glbBase = arith::AddIOp::create(builder, loc, glbBase, immOffset);
  Value ldsAddr = LLVM::PtrToIntOp::create(builder, loc, builder.getI32Type(), ldsPtr);

  int32_t padInterval = getPadInterval();
  int32_t padAmount = getPadAmount();
  FailureOr<PadEncoding> padOr = computePadEncoding(padInterval, padAmount, elemBits);
  if (failed(padOr))
    return mlir::emitError(loc)
           << "gfx1250 TDM gather: padding (interval=" << padInterval << ", amount=" << padAmount
           << " elements at " << elemBits
           << "-bit) is not encodable — the dword interval must be a power of two and the encoded "
              "fields must fit the descriptor bitfield";
  PadEncoding pad = *padOr;

  // GROUP0 (vector<4xi32>): pred (gather-index bit [30] set for 32-bit mode,
  // type field [31]), lds_addr, glb_lo, glb_hi | type.
  int32_t gatherIndexBit = (getIndexSize() == 32) ? 1 : 0;
  Value g0s0 = i32Const(builder, loc, 1 | (gatherIndexBit << 30) | (1 << 31));
  Value g0s1 = ldsAddr;
  Value g0s2 = LLVM::TruncOp::create(builder, loc, builder.getI32Type(), glbBase);
  Value glbHiRaw = LLVM::LShrOp::create(builder, loc, glbBase,
                                        arith::ConstantIntOp::create(builder, loc, 32, 64));
  Value glbHi = LLVM::TruncOp::create(builder, loc, builder.getI32Type(), glbHiRaw);
  Value g0s3 = arith::OrIOp::create(builder, loc, glbHi, i32Const(builder, loc, 1 << 31));
  Value dgroup0 = vector::FromElementsOp::create(
      builder, loc, VectorType::get({4}, builder.getI32Type()), ValueRange{g0s0, g0s1, g0s2, g0s3});

  // GROUP1 (vector<8xi32>): config + tensor dims + tile row width + count + stride.
  int32_t g1s0Upper = (dataSizeCode << 16) | ((pad.enable ? 1 : 0) << 20) | (pad.interval << 22) |
                      (pad.amount << 25);
  Value maskRaw = LLVM::ExtractValueOp::create(
      builder, loc, atomVal, ArrayRef<int64_t>{*getFieldIndex(AtomStateField::WorkgroupMask)});
  Value maskLow = arith::AndIOp::create(builder, loc, maskRaw, i32Const(builder, loc, 0xFFFF));
  Value g1s0 = arith::OrIOp::create(builder, loc, i32Const(builder, loc, g1s0Upper), maskLow);

  Value c16 = i32Const(builder, loc, 16);
  Value mask16 = i32Const(builder, loc, 0xFFFF);
  Value td0Lo = arith::AndIOp::create(builder, loc, tensorDim0, mask16);
  Value td0Hi = arith::AndIOp::create(
      builder, loc, arith::ShRUIOp::create(builder, loc, tensorDim0, c16), mask16);
  Value td1Lo = arith::AndIOp::create(builder, loc, tensorDim1, mask16);
  Value td1Hi = arith::AndIOp::create(
      builder, loc, arith::ShRUIOp::create(builder, loc, tensorDim1, c16), mask16);
  // s1: tensor_dim0_lo [31:16]
  Value g1s1 = arith::ShLIOp::create(builder, loc, td0Lo, c16);
  // s2: tensor_dim0_hi [15:0] | tensor_dim1_lo [31:16]
  Value g1s2 =
      arith::OrIOp::create(builder, loc, td0Hi, arith::ShLIOp::create(builder, loc, td1Lo, c16));
  // s3: tensor_dim1_hi [15:0] | row_width (tile_dim0) [31:16]
  Value g1s3 = arith::OrIOp::create(builder, loc, td1Hi, i32Const(builder, loc, rowWidth << 16));
  // s4: gather tile_dim1 = number of valid indices [15:0]
  Value g1s4 = arith::AndIOp::create(builder, loc, gatherCount, mask16);
  Value g1s5 = outerStride; // dim0 (row) stride
  Value dgroup1 = vector::FromElementsOp::create(
      builder, loc, VectorType::get({8}, builder.getI32Type()),
      ValueRange{g1s0, g1s1, g1s2, g1s3, g1s4, g1s5, i32Const(builder, loc, 0),
                 i32Const(builder, loc, 0)});

  // GROUP2 / GROUP3: the packed row-index vector split into two vector<4xi32>.
  Value dg2 = vector::ShuffleOp::create(builder, loc, rowIndices, rowIndices,
                                        ArrayRef<int64_t>{0, 1, 2, 3});
  Value dg3 = vector::ShuffleOp::create(builder, loc, rowIndices, rowIndices,
                                        ArrayRef<int64_t>{4, 5, 6, 7});
  Value zero = i32Const(builder, loc, 0);
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

LogicalResult CopyOpGFX1250TDMGatherType::emitAtomCall(OpBuilder &builder, Location loc,
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
