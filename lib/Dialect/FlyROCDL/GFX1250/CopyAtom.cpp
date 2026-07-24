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
// CopyOpGFX1250TDMType — N-D TDM async Global<->LDS whole-tile copy (rank 1-5).
//
// The Global-side memref layout supplies the compile-time tile geometry; the
// atom parameters supply padding, warp count, and cache policy; the tile
// descriptor (base / per-dim extent / per-dim stride) is runtime atom state.
// Direction (load vs store) is inferred from the operand address spaces.
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

// Stateful: the atom carries the TDM N-D descriptor — workgroup_mask (MCAST) plus
// the tile geometry, EXCEPT the global base, which comes from the copy_atom_call
// operand pointer. Struct slots: {mask, extent_0..4 (i32, per-dim tensor extent
// for OOB), stride_0..3 (i64, per-dim tensor stride in elements; innermost stride
// is assumed 1), imm_offset (i64)}. See CopyAtom.td for field semantics.
static constexpr unsigned kMaxTdmRank = 5;

// Struct slots for the per-dim geometry: extent_i at kExtentSlot0+i (i32, i in
// 0..kMaxTdmRank-1), stride_i at kStrideSlot0+i (i64, i in 0..kMaxTdmRank-2).
// These are TDM-private (not shared AtomStateField cases); see tdmGeomSlot.
static constexpr unsigned kExtentSlot0 = 1;
static constexpr unsigned kStrideSlot0 = kExtentSlot0 + kMaxTdmRank; // 6
// imm_offset occupies kStrideSlot0 + (kMaxTdmRank - 1) = 10 (see getFieldIndex).
// The gather row-index buffer pointer (global) rides in the next slot.
static constexpr unsigned kIndexPtrSlot = kStrideSlot0 + (kMaxTdmRank - 1) + 1; // 11

std::optional<unsigned> CopyOpGFX1250TDMType::getFieldIndex(AtomStateField field) {
  switch (field) {
  case AtomStateField::WorkgroupMask:
    return 0;
  // Byte offset added to the global base at lowering (i64, carry-safe). Lets a
  // K-loop advance the tile by bumping one scalar (imm_offset) instead of
  // re-deriving the base pointer. Default 0 folds away.
  case AtomStateField::ImmOffset:
    return kStrideSlot0 + (kMaxTdmRank - 1); // 10, last slot after the strides
  default:
    return std::nullopt;
  }
}

// TDM-private state fields, resolved by name instead of via the shared
// AtomStateField enum: "extent_i" -> kExtentSlot0+i, "stride_i" -> kStrideSlot0+i,
// "index_ptr" -> kIndexPtrSlot (gather row-index buffer).
static std::optional<unsigned> tdmGeomSlot(StringRef name) {
  unsigned i;
  if (name.consume_front("extent_") && !name.getAsInteger(10, i) && i < kMaxTdmRank)
    return kExtentSlot0 + i;
  if (name.consume_front("stride_") && !name.getAsInteger(10, i) && i < kMaxTdmRank - 1)
    return kStrideSlot0 + i;
  if (name == "index_ptr")
    return kIndexPtrSlot;
  return std::nullopt;
}

Type CopyOpGFX1250TDMType::getConvertedType(MLIRContext *ctx) const {
  auto i32 = IntegerType::get(ctx, 32);
  auto i64 = IntegerType::get(ctx, 64);
  SmallVector<Type> fields = {i32};    // workgroup_mask
  fields.append(kMaxTdmRank, i32);     // extent_0..4
  fields.append(kMaxTdmRank - 1, i64); // stride_0..3
  fields.push_back(i64);               // imm_offset
  // Gather mode carries one extra slot: the row-index buffer pointer. Tiled mode
  // keeps the original layout so its lowering/tests are unaffected.
  if (getIndexWidth() != 0)
    fields.push_back(LLVM::LLVMPointerType::get(ctx, /*addrSpace=*/1)); // index_ptr
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
  auto insertSlot = [&](unsigned slot, Value v) {
    state = LLVM::InsertValueOp::create(builder, loc, state, v, ArrayRef<int64_t>{slot});
  };
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  Value noClamp = arith::ConstantIntOp::create(builder, loc, 0x7FFFFFFF, 32);
  Value strideUnset = arith::ConstantIntOp::create(builder, loc, kOuterStrideUnset, 64);
  Value zero64 = arith::ConstantIntOp::create(builder, loc, 0, 64);
  insert(AtomStateField::WorkgroupMask, zero);
  for (unsigned i = 0; i < kMaxTdmRank; ++i)
    insertSlot(kExtentSlot0 + i, noClamp);
  for (unsigned i = 0; i < kMaxTdmRank - 1; ++i)
    insertSlot(kStrideSlot0 + i, strideUnset);
  insert(AtomStateField::ImmOffset, zero64);
  if (getIndexWidth() != 0) {
    Value nullIdx =
        LLVM::ZeroOp::create(builder, loc, LLVM::LLVMPointerType::get(ctx, /*addrSpace=*/1));
    insertSlot(kIndexPtrSlot, nullIdx);
  }
  return state;
}

Value CopyOpGFX1250TDMType::setAtomState(OpBuilder &builder, Location loc, Value atomStruct,
                                         Attribute fieldAttr, Value fieldValue) const {
  auto fieldStr = dyn_cast<StringAttr>(fieldAttr);
  if (!fieldStr)
    return nullptr;
  // Per-dim extent/stride are TDM-private (resolved by name); everything else goes
  // through the shared AtomStateField enum.
  std::optional<unsigned> idx = tdmGeomSlot(fieldStr.getValue());
  if (!idx) {
    if (auto field = symbolizeAtomStateField(fieldStr.getValue()))
      idx = getFieldIndex(*field);
  }
  if (!idx)
    return nullptr;
  return LLVM::InsertValueOp::create(builder, loc, atomStruct, fieldValue, ArrayRef<int64_t>{*idx});
}

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
                                           bool atomicBarrier, bool earlyTimeout,
                                           int32_t indexWidth) {
  if (rank < 1 || rank > static_cast<int32_t>(kMaxTdmRank))
    return emitError() << "TDM rank must be in [1, " << kMaxTdmRank << "], got " << rank;
  // index_width == 0 is tiled; 16/32 selects rank-2 row gather/scatter.
  if (indexWidth != 0) {
    if (indexWidth != 16 && indexWidth != 32)
      return emitError() << "TDM index_width must be 0 (tiled) or 16/32 (gather), got "
                         << indexWidth;
    if (rank != 2)
      return emitError() << "TDM gather (index_width " << indexWidth << ") requires rank 2, got "
                         << rank;
  }
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

// Gather-mode emit: load the rows selected by the atom's `index_ptr` row-index
// buffer (element width from the atom's index_width: i16 or i32; row count from the
// rank-2 tile layout's outer extent) into the descriptor index groups, and pack the
// rank-2 gather descriptor. The base pointer comes from the copy_atom_call global
// operand (glbPtr); OOB extents / row stride / index_ptr come from atom state.
// The row-index descriptor packing mirrors the Triton AMD backend (TDMUtility.cpp
// createTDMDescriptor gather path), matching the tiled emit's Triton-derived layout.
static LogicalResult emitTdmGather(OpBuilder &builder, Location loc, CopyOpGFX1250TDMType atomTy,
                                   fly::MemRefType glbMemTy, Value glbPtr, Value ldsPtr,
                                   bool isLoad, Value atomVal) {
  auto layout = dyn_cast<fly::LayoutAttr>(glbMemTy.getLayout());
  if (!layout || !layout.isStaticShape() || layout.rank() != 2)
    return failure();
  int32_t outer = layout.getShape().at(0).getLeafAsInt().getValue();    // rows gathered
  int32_t rowWidth = layout.getShape().at(1).getLeafAsInt().getValue(); // tile_dim0
  bool hasStaticStride = layout.isStaticStride();

  int32_t elemBits = glbMemTy.getElemTy().getIntOrFloatBitWidth();
  if (elemBits % 8 != 0)
    return failure();
  int32_t elemBytes = elemBits / 8;
  if ((elemBytes & (elemBytes - 1)) != 0)
    return failure();
  int32_t dataSizeCode = llvm::Log2_32(static_cast<uint32_t>(elemBytes));

  // Row-index width (16/32) from the atom's index_width; row count is the tile's
  // outer extent, bounded by the descriptor's 8x i32 / 16x i16 index slots.
  int32_t indexSize = atomTy.getIndexWidth();
  if (indexSize != 16 && indexSize != 32)
    return mlir::emitError(loc) << "gfx1250 TDM gather: index_width must be 16 or 32";
  int32_t count = outer;
  int32_t maxCount = indexSize == 32 ? 8 : 16;
  if (count < 1 || count > maxCount)
    return mlir::emitError(loc) << "gfx1250 TDM gather: " << indexSize << "-bit row count must be "
                                << "in [1, " << maxCount << "], got " << count;

  Type i32Ty = builder.getI32Type();
  Type i64Ty = builder.getI64Type();
  Type idxElemTy = builder.getIntegerType(indexSize);
  auto slotField = [&](unsigned slot) {
    return LLVM::ExtractValueOp::create(builder, loc, atomVal, ArrayRef<int64_t>{slot});
  };
  auto stateField = [&](AtomStateField f) {
    return LLVM::ExtractValueOp::create(builder, loc, atomVal,
                                        ArrayRef<int64_t>{*CopyOpGFX1250TDMType::getFieldIndex(f)});
  };

  // Row-index buffer pointer from atom state.
  Value indices = slotField(kIndexPtrSlot);
  auto idxPtrTy = cast<LLVM::LLVMPointerType>(indices.getType());
  Value zeroC = i32Const(builder, loc, 0);
  Value c16v = i32Const(builder, loc, 16);
  // Load index j from the buffer and zero-extend to i32.
  auto loadIdx = [&](int32_t j) -> Value {
    Value gep = LLVM::GEPOp::create(builder, loc, idxPtrTy, idxElemTy, indices,
                                    ArrayRef<LLVM::GEPArg>{LLVM::GEPArg(j)});
    Value v = LLVM::LoadOp::create(builder, loc, idxElemTy, gep);
    if (indexSize == 32)
      return v;
    return LLVM::ZExtOp::create(builder, loc, i32Ty, v);
  };

  // Pack the loaded indices into 8x i32 descriptor words (32-bit: one per word;
  // 16-bit: two per word, lo | hi<<16). Unfilled slots are zero.
  SmallVector<Value> words(8, zeroC);
  if (indexSize == 32) {
    for (int32_t j = 0; j < count; ++j)
      words[j] = loadIdx(j);
  } else {
    for (int32_t w = 0; w < 8; ++w) {
      Value lo = (2 * w < count) ? loadIdx(2 * w) : zeroC;
      Value hi = (2 * w + 1 < count) ? loadIdx(2 * w + 1) : zeroC;
      Value hiSh = arith::ShLIOp::create(builder, loc, hi, c16v);
      words[w] = arith::OrIOp::create(builder, loc, lo, hiSh);
    }
  }

  Value tensorDim1 = slotField(kExtentSlot0 + 0); // rows (OOB on indices)
  Value tensorDim0 = slotField(kExtentSlot0 + 1); // row width (column OOB)
  Value stateStride = slotField(kStrideSlot0 + 0);
  Value immOffset = stateField(AtomStateField::ImmOffset);

  // Row stride (i64 state; unset sentinel falls back to the static layout stride),
  // truncated to the descriptor's 32-bit row-stride slot.
  Value outerStride64;
  if (hasStaticStride) {
    int64_t s = layout.getStride().at(0).getLeafAsInt().getValue();
    Value unset =
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, stateStride,
                              arith::ConstantIntOp::create(builder, loc, kOuterStrideUnset, 64));
    outerStride64 = arith::SelectOp::create(
        builder, loc, unset, arith::ConstantIntOp::create(builder, loc, s, 64), stateStride);
  } else {
    outerStride64 = stateStride;
  }
  Value outerStride = LLVM::TruncOp::create(builder, loc, i32Ty, outerStride64);

  // Global base = the copy_atom_call operand pointer (not atom state), plus the
  // i64 imm_offset (carry-safe into glb_hi via the i64 split below).
  Value glbBase = LLVM::PtrToIntOp::create(builder, loc, i64Ty, glbPtr);
  glbBase = arith::AddIOp::create(builder, loc, glbBase, immOffset);
  Value ldsAddr = LLVM::PtrToIntOp::create(builder, loc, i32Ty, ldsPtr);

  int32_t padInterval = atomTy.getPadInterval();
  int32_t padAmount = atomTy.getPadAmount();
  FailureOr<PadEncoding> padOr = computePadEncoding(padInterval, padAmount, elemBits);
  if (failed(padOr))
    return mlir::emitError(loc) << "gfx1250 TDM gather: padding (interval=" << padInterval
                                << ", amount=" << padAmount << " elements at " << elemBits
                                << "-bit) is not encodable";
  PadEncoding pad = *padOr;

  // GROUP0: pred (gather-index bit [30] set for 32-bit mode, type field [31]),
  // lds_addr, glb_lo, glb_hi | type.
  int32_t gatherIndexBit = (indexSize == 32) ? 1 : 0;
  Value g0s0 = i32Const(builder, loc, 1 | (gatherIndexBit << 30) | (1 << 31));
  Value g0s2 = LLVM::TruncOp::create(builder, loc, i32Ty, glbBase);
  Value glbHiRaw = LLVM::LShrOp::create(builder, loc, glbBase,
                                        arith::ConstantIntOp::create(builder, loc, 32, 64));
  Value glbHi = LLVM::TruncOp::create(builder, loc, i32Ty, glbHiRaw);
  Value g0s3 = arith::OrIOp::create(builder, loc, glbHi, i32Const(builder, loc, 1 << 31));
  Value dgroup0 = vector::FromElementsOp::create(builder, loc, VectorType::get({4}, i32Ty),
                                                 ValueRange{g0s0, ldsAddr, g0s2, g0s3});

  // GROUP1: config + tensor dims + tile row width + count + row stride.
  int32_t g1s0Upper = (dataSizeCode << 16) | ((pad.enable ? 1 : 0) << 20) | (pad.interval << 22) |
                      (pad.amount << 25);
  Value maskRaw = stateField(AtomStateField::WorkgroupMask);
  Value maskLow = arith::AndIOp::create(builder, loc, maskRaw, i32Const(builder, loc, 0xFFFF));
  Value g1s0 = arith::OrIOp::create(builder, loc, i32Const(builder, loc, g1s0Upper), maskLow);

  Value mask16 = i32Const(builder, loc, 0xFFFF);
  Value td0Lo = arith::AndIOp::create(builder, loc, tensorDim0, mask16);
  Value td0Hi = arith::AndIOp::create(
      builder, loc, arith::ShRUIOp::create(builder, loc, tensorDim0, c16v), mask16);
  Value td1Lo = arith::AndIOp::create(builder, loc, tensorDim1, mask16);
  Value td1Hi = arith::AndIOp::create(
      builder, loc, arith::ShRUIOp::create(builder, loc, tensorDim1, c16v), mask16);
  Value g1s1 = arith::ShLIOp::create(builder, loc, td0Lo, c16v);
  Value g1s2 =
      arith::OrIOp::create(builder, loc, td0Hi, arith::ShLIOp::create(builder, loc, td1Lo, c16v));
  Value g1s3 = arith::OrIOp::create(builder, loc, td1Hi, i32Const(builder, loc, rowWidth << 16));
  Value g1s4 = i32Const(builder, loc, count & 0xFFFF); // gather tile_dim1 = valid row count
  Value dgroup1 = vector::FromElementsOp::create(
      builder, loc, VectorType::get({8}, i32Ty),
      ValueRange{g1s0, g1s1, g1s2, g1s3, g1s4, outerStride, zeroC, zeroC});

  // GROUP2 / GROUP3: the packed row-index words.
  Value dg2 = vector::FromElementsOp::create(builder, loc, VectorType::get({4}, i32Ty),
                                             ValueRange{words[0], words[1], words[2], words[3]});
  Value dg3 = vector::FromElementsOp::create(builder, loc, VectorType::get({4}, i32Ty),
                                             ValueRange{words[4], words[5], words[6], words[7]});
  Value dg4 = vector::FromElementsOp::create(
      builder, loc, VectorType::get({8}, i32Ty),
      ValueRange{zeroC, zeroC, zeroC, zeroC, zeroC, zeroC, zeroC, zeroC});

  uint32_t cachePolicy = static_cast<uint32_t>(atomTy.getCacheModifier());
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

  // The global operand supplies the base pointer and the compile-time tile shape;
  // its per-dim extents (OOB) and strides come from the atom descriptor state (the
  // tile view's static stride is the packed tile-internal stride, not necessarily
  // the true global stride, which may be dynamic). Tensor dim order on the layout:
  // index 0 = outermost, rank-1 = innermost.
  fly::MemRefType glbMemTy = isLoad ? srcMemTy : dstMemTy;
  Value glbPtr = isLoad ? src : dst;
  Value ldsPtr = isLoad ? dst : src;

  if (getIndexWidth() != 0)
    return emitTdmGather(builder, loc, *this, glbMemTy, glbPtr, ldsPtr, isLoad, atomVal);

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
  auto slotField = [&](unsigned slot) {
    return LLVM::ExtractValueOp::create(builder, loc, atomVal, ArrayRef<int64_t>{slot});
  };

  // Per-dim (tensor order) tensor extent (i32) and stride in elements (i64). The
  // innermost stride (dim rank-1) is assumed 1; stride_i covers dims 0..rank-2,
  // falling back to the static layout stride when the state slot is left unset.
  SmallVector<Value> extent(rank), strideElems(rank);
  for (int32_t i = 0; i < rank; ++i)
    extent[i] = slotField(kExtentSlot0 + i);
  for (int32_t i = 0; i < rank - 1; ++i) {
    Value st = slotField(kStrideSlot0 + i);
    if (hasStaticStride) {
      int64_t s = layout.getStride().at(i).getLeafAsInt().getValue();
      Value unset =
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq, st,
                                arith::ConstantIntOp::create(builder, loc, kOuterStrideUnset, 64));
      strideElems[i] = arith::SelectOp::create(
          builder, loc, unset, arith::ConstantIntOp::create(builder, loc, s, 64), st);
    } else {
      // Dynamic outer stride: there is no static layout stride to fall back to,
      // so the caller must set stride_i explicitly (make_tdm_atom's `strides`).
      // An unset slot carries the sentinel (a large negative stride) straight
      // into the descriptor, which produces a visibly-wrong transfer rather than
      // silently aliasing a legitimate stride 0.
      strideElems[i] = st;
    }
  }
  strideElems[rank - 1] = arith::ConstantIntOp::create(builder, loc, 1, 64); // innermost contiguous

  // Global base = the copy_atom_call operand pointer (not atom state).
  Value glbBase = LLVM::PtrToIntOp::create(builder, loc, i64Ty, glbPtr);
  // i64 byte offset (default 0) added to the base so a K-loop can advance the tile
  // by bumping one scalar; the carry into glb_hi is handled by the i64 split below.
  glbBase = arith::AddIOp::create(builder, loc, glbBase, stateField(AtomStateField::ImmOffset));
  Value ldsBase = LLVM::PtrToIntOp::create(builder, loc, builder.getI32Type(), ldsPtr);

  // Compile-time LDS per-warp strides (elements): innermost contiguous, with row
  // padding folded into the next-outer stride. Padding describes the LDS layout,
  // so it applies to both directions (load fills padded LDS, store reads it) —
  // matching the Triton reference, which encodes padding identically for load and
  // store.
  SmallVector<int32_t> ldsStride(rank);
  ldsStride[rank - 1] = 1;
  if (rank >= 2) {
    int32_t innerRow = tileShape[rank - 1];
    if (padActive)
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

  // Padding descriptor bitfield: set for both directions (it describes the LDS
  // layout the DMA engine walks — pad-per-row on fill and on drain alike).
  int32_t padInterval = getPadInterval();
  int32_t padAmount = getPadAmount();
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
  // tile_dim_j is the static per-warp block size. LDS padding is carried by the
  // pad bitfield + ldsStride, never by widening tile_dim, so the global transfer
  // extent stays the true tile size for both load and store.
  auto descTensorDim = [&](int32_t j) -> Value { // runtime i32, clamped
    int32_t d = rank - 1 - j;
    return arith::MaxSIOp::create(
        builder, loc, arith::SubIOp::create(builder, loc, extent[d], warpOff[d]), zeroC);
  };
  auto descTileDim = [&](int32_t j) -> int32_t {
    int32_t d = rank - 1 - j;
    return bpw[d];
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

} // namespace mlir::fly_rocdl
