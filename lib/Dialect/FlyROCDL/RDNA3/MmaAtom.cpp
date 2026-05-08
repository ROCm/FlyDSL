// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
//
// MmaOpRDNA3_WMMAType — wave32 WMMA atom for RDNA 3 / 3.5 (gfx1100/gfx1150).
//
// RDNA 3 / 3.5 expose six 16x16x16 V_WMMA opcodes (VOP3P 64-69):
//   V_WMMA_F32_16X16X16_F16    F16 -> F32
//   V_WMMA_F32_16X16X16_BF16   BF16 -> F32
//   V_WMMA_F16_16X16X16_F16    F16 -> F16 (op_sel chooses high/low half)
//   V_WMMA_BF16_16X16X16_BF16  BF16 -> BF16 (op_sel chooses high/low half)
//   V_WMMA_I32_16X16X16_IU8    IU8 -> I32  (clamp + signed/unsigned per operand)
//   V_WMMA_I32_16X16X16_IU4    IU4 -> I32  (clamp + signed/unsigned per operand)
//
// The hardware uses the "WMMA256bInsts" convention: lanes 0-15 hold the
// matrix data; lanes 16-31 are duplicates populated by the LLVM AMDGPU
// codegen (round-tripped through the LLVM intrinsic interface). At the
// IR / atom level we use the same lane-group-split logical layout as the
// gfx1250 K=4 path, with each lane holding K/2 = 8 unique elements.
// Concretely, lane group 0 (lane/16=0) covers K = 0..7 and lane group 1
// (lane/16=1) covers K = 8..15. This mirrors what `kernels/rdna_f16_gemm.py`
// (the existing wave32 raw-intrinsic kernel) constructs by hand.
//
// FP8/FP4/scaled/sparse WMMA, K=4 / K=32 / K=64 / K=128 shapes, and the
// transpose-load atoms are gfx12+ only and rejected by `verify` here.

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"

using namespace mlir;
using namespace mlir::fly;

namespace rdna3 {

// A/B operand register layout for RDNA 3 / 3.5 WMMA (wave32, M=N=K=16).
//
// Reference space is column-major (M, K) with stride (1, M=16). Each lane
// holds K/2 = 8 unique elements; the duplication of lanes 0-15 into lanes
// 16-31 required by the hardware is materialised by the LLVM AMDGPU
// backend at codegen time.
//
//   lane_pos = (l%16)*1   // M coord
//            + (l/16)*128 // K-group offset (8 K elements * stride 16)
//            + v*16       // K within group (val v -> K=v in group 0,
//                         //                  val v -> K=v+8 in group 1)
//
// shape   = ((16, 2), 8)  -> outer thr is (16,2), val is 8 (flat)
// stride  = ((1, 128), 16)
LayoutAttr getThrValLayoutAB(MLIRContext *ctx) {
  auto getContext = [&]() { return ctx; };
  return FxLayout(FxShape(FxThr(16, 2), FxVal(8)),
                  FxStride(FxThr(1, 128), FxVal(16)));
}

// C/D operand register layout for RDNA 3 / 3.5 WMMA (wave32, M=N=16).
//
// The accumulator is always 16x16. Lane l covers N = l%16 (one column).
// The two lane groups split M into two halves of 8 rows each:
//   group 0 -> M = 0..7
//   group 1 -> M = 8..15
//
// 32-bit accumulator (f32, i32): 8 VGPRs, one element per VGPR.
//   M = (l/16)*8 + v
//
// 16-bit accumulator (f16, bf16): the LLVM intrinsic returns a 16-wide
// vector with the OpSel[2] hi/lo bit selecting which half is live; from
// the FlyDSL atom perspective each lane still holds 8 distinct M
// coordinates, packed two-per-VGPR (4 VGPRs, sub-element index within VGPR).
//   M = (l/16)*8 + v*2 + s
//
// Reference space is column-major (M, N) with stride (1, M=16).
LayoutAttr getThrValLayoutCD(MLIRContext *ctx, Type elemTyAcc) {
  auto getContext = [&]() { return ctx; };

  (void)elemTyAcc;
  // Keep a single canonical 8-value lane fragment layout for all accumulator
  // types. The lowering may internally use packed/native forms, but the atom
  // interface exposes a consistent vec8 fragment to tiled_copy_C.
  return FxLayout(FxShape(FxThr(16, 2), FxVal(8)),
                  FxStride(FxThr(16, 8), FxVal(1)));
}

} // namespace rdna3

namespace mlir::fly_rocdl {

bool MmaOpRDNA3_WMMAType::isStatic() const { return true; }

Value MmaOpRDNA3_WMMAType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                              Value currentValue) const {
  if (currentValue && isa<MakeMmaAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  return MakeMmaAtomOp::create(builder, loc, MmaAtomType::get(*this));
}

Type MmaOpRDNA3_WMMAType::getValTypeA() const { return getElemTyA(); }
Type MmaOpRDNA3_WMMAType::getValTypeB() const { return getElemTyB(); }
Type MmaOpRDNA3_WMMAType::getValTypeC() const { return getElemTyAcc(); }
Type MmaOpRDNA3_WMMAType::getValTypeD() const { return getElemTyAcc(); }

Attribute MmaOpRDNA3_WMMAType::getThrLayout() const {
  return FxLayout(FxC(32), FxC(1));
}

Attribute MmaOpRDNA3_WMMAType::getShapeMNK() const {
  return IntTupleAttr::get(
      ArrayAttr::get(getContext(), {FxC(getM()), FxC(getN()), FxC(getK())}));
}

Attribute MmaOpRDNA3_WMMAType::getThrValLayoutA() const {
  return rdna3::getThrValLayoutAB(getContext());
}

Attribute MmaOpRDNA3_WMMAType::getThrValLayoutB() const {
  return rdna3::getThrValLayoutAB(getContext());
}

Attribute MmaOpRDNA3_WMMAType::getThrValLayoutC() const {
  return rdna3::getThrValLayoutCD(getContext(), getElemTyAcc());
}

LogicalResult MmaOpRDNA3_WMMAType::verify(
    function_ref<InFlightDiagnostic()> emitError, int32_t m, int32_t n,
    int32_t k, Type elemTyA, Type elemTyB, Type elemTyAcc) {
  if (m != 16 || n != 16 || k != 16) {
    return emitError()
           << "RDNA3 WMMA requires M=N=K=16, got " << m << "x" << n << "x" << k;
  }

  bool valid = false;

  if (elemTyA.isF16() && elemTyB.isF16() &&
      (elemTyAcc.isF32() || elemTyAcc.isF16()))
    valid = true;
  if (elemTyA.isBF16() && elemTyB.isBF16() &&
      (elemTyAcc.isF32() || elemTyAcc.isBF16()))
    valid = true;
  if (elemTyA.isInteger(8) && elemTyB.isInteger(8) && elemTyAcc.isInteger(32))
    valid = true;
  if (elemTyA.isInteger(4) && elemTyB.isInteger(4) && elemTyAcc.isInteger(32))
    valid = true;

  if (!valid) {
    return emitError()
           << "unsupported RDNA3 WMMA configuration: " << m << "x" << n << "x"
           << k << " with A=" << elemTyA << ", B=" << elemTyB
           << ", Acc=" << elemTyAcc
           << " (RDNA 3 / 3.5 supports only F16/BF16/IU8/IU4 K=16; "
              "FP8/FP4/scaled/sparse require gfx12+)";
  }
  return success();
}

// Vector type each lane holds for A or B operand.
//
// RDNA 3 / 3.5 WMMA at K=16:
//   F16  -> vector<8 x f16>   (8 elements per lane, 4 VGPRs)
//   BF16 -> vector<8 x bf16>  (same shape)
//   IU8  -> vector<2 x i32>   (8 i8s packed into 2 dwords per lane)
//   IU4  -> vector<2 x i32>   (16 i4s packed into 2 dwords per lane = K=16
//                              elements; M*K/32 = 256/32 = 8 elements * 4 bits
//                              = 32 bits = 1 dword. But i4 typically uses
//                              i32 packed. We follow the gfx1250 convention
//                              of vector<2xi32> for K=16 i4)
static Type getWmmaABType(MLIRContext *ctx, Type elemTy) {
  // M*K/wave_size = 16*16/32 = 8 elements per lane.
  Type i32Ty = IntegerType::get(ctx, 32);

  if (elemTy.isInteger(8)) {
    // 8 i8 values per lane = 64 bits = vector<2 x i32>.
    return VectorType::get({2}, i32Ty);
  }
  if (elemTy.isInteger(4)) {
    // 16 i4 values per lane (K=16 fully packed) = 64 bits = vector<2 x i32>.
    return VectorType::get({2}, i32Ty);
  }
  if (elemTy.isF16() || elemTy.isBF16()) {
    return VectorType::get({8}, elemTy);
  }
  return nullptr;
}

// Vector type the accumulator (C/D) holds per lane.
//
// F32 acc: vector<8 x f32>  (8 VGPRs, one f32 per VGPR)
// I32 acc: vector<8 x i32>
// F16/BF16 acc: vector<8 x elem> (op_sel selects hi/lo half of each VGPR;
//   the IR-level vector has 8 elements per lane regardless)
static int64_t getWmmaAccVecSize(int32_t /*m*/, int32_t /*k*/, Type elemTyAcc) {
  if (elemTyAcc.isF32() || elemTyAcc.isInteger(32))
    return 8;
  if (elemTyAcc.isF16() || elemTyAcc.isBF16())
    return 8;
  return 0;
}

FailureOr<Value> MmaOpRDNA3_WMMAType::emitAtomCallSSA(
    OpBuilder &builder, Location loc, Type /*resultTy*/, Type /*mmaAtomTyArg*/,
    Type /*dTyArg*/, Type /*aTyArg*/, Type /*bTyArg*/, Type /*cTyArg*/,
    Value /*atomVal*/, Value /*d*/, Value a, Value b, Value c) const {
  Type elemTyA = getElemTyA();
  Type elemTyB = getElemTyB();
  Type elemTyAcc = getElemTyAcc();
  MLIRContext *ctx = builder.getContext();

  Type abTyA = getWmmaABType(ctx, elemTyA);
  Type abTyB = getWmmaABType(ctx, elemTyB);
  if (!abTyA || !abTyB)
    return failure();

  int64_t accVecSize = getWmmaAccVecSize(getM(), getK(), elemTyAcc);
  if (accVecSize == 0)
    return failure();

  VectorType accTy = VectorType::get({accVecSize}, elemTyAcc);

  if (a.getType() != abTyA)
    a = LLVM::BitcastOp::create(builder, loc, abTyA, a);
  if (b.getType() != abTyB)
    b = LLVM::BitcastOp::create(builder, loc, abTyB, b);
  if (c.getType() != accTy)
    c = LLVM::BitcastOp::create(builder, loc, accTy, c);

  // The bf16 ROCDL ops take `AnyInteger` for A/B (and for C/D when the acc is
  // bf16), so we must bitcast bfloat vectors to i16 vectors before the call
  // and back afterwards. The f16 op takes f16 directly so no cast is needed.
  Type i16Ty = IntegerType::get(ctx, 16);

  auto castToI16Vec = [&](Value v) -> Value {
    auto vt = cast<VectorType>(v.getType());
    auto intVt = VectorType::get(vt.getShape(), i16Ty);
    return LLVM::BitcastOp::create(builder, loc, intVt, v);
  };

  // RDNA 3 / 3.5 uses `WMMA256bInsts`: the intrinsic selector expects
  // 16-wide lane operands. FlyDSL's per-lane fragment is 8 elements where
  // lane-group 0 carries K=0..7 and lane-group 1 carries K=8..15.
  //
  // Build a 16-wide lane vector as:
  //   low  half = lane-local 8 values
  //   high half = values pulled from the paired lane (lane xor 16)
  //
  // `rocdl.ds_bpermute` is 32-bit granularity, so for f16/bf16 we:
  //   vector<8x{f16|bf16}> -> bitcast vector<4xi32>
  //   bpermute each dword from lane^16
  //   bitcast back to vector<8x{f16|bf16}>
  //   concatenate [self8, pair8] to vector<16x...>
  auto expandTo16WideWmma256 = [&](Value v) -> Value {
    auto vt = cast<VectorType>(v.getType());
    if (vt.getShape().size() != 1 || vt.getShape()[0] != 8)
      return v;
    auto wideTy = VectorType::get({16}, vt.getElementType());

    Value pairedHalf = v;
    if (vt.getElementType().isF16() || vt.getElementType().isBF16()) {
      auto i32Ty = IntegerType::get(ctx, 32);
      auto i64Ty = IntegerType::get(ctx, 64);
      auto packedTy = VectorType::get({4}, i32Ty);

      Value lane = ROCDL::ThreadIdXOp::create(builder, loc, i32Ty).getResult();
      Value c31 = LLVM::ConstantOp::create(builder, loc, i32Ty,
                                           builder.getI32IntegerAttr(31));
      Value cNeg32 = LLVM::ConstantOp::create(builder, loc, i32Ty,
                                              builder.getI32IntegerAttr(-32));
      Value c16 = LLVM::ConstantOp::create(builder, loc, i32Ty,
                                           builder.getI32IntegerAttr(16));
      Value c2 = LLVM::ConstantOp::create(builder, loc, i32Ty,
                                          builder.getI32IntegerAttr(2));

      Value laneInWave = LLVM::AndOp::create(builder, loc, lane, c31);
      Value waveBase = LLVM::AndOp::create(builder, loc, lane, cNeg32);
      Value pairedLaneInWave = LLVM::XOrOp::create(builder, loc, laneInWave, c16);
      Value pairedLane = LLVM::OrOp::create(builder, loc, waveBase, pairedLaneInWave);
      Value pairedLaneByteAddr = LLVM::ShlOp::create(builder, loc, pairedLane, c2);

      Value packed = LLVM::BitcastOp::create(builder, loc, packedTy, v);
      Value pairedPacked = LLVM::UndefOp::create(builder, loc, packedTy);
      for (int i = 0; i < 4; ++i) {
        Value idx = LLVM::ConstantOp::create(builder, loc, i64Ty,
                                             builder.getI64IntegerAttr(i));
        Value laneWord = LLVM::ExtractElementOp::create(builder, loc, packed, idx);
        Value pairedWord = ROCDL::DsBpermuteOp::create(builder, loc, i32Ty,
                                                       pairedLaneByteAddr, laneWord)
                               .getResult();
        pairedPacked = LLVM::InsertElementOp::create(builder, loc, pairedPacked,
                                                     pairedWord, idx);
      }
      pairedHalf = LLVM::BitcastOp::create(builder, loc, vt, pairedPacked);
    }

    Value lowHalf = v;
    Value highHalf = pairedHalf;

    SmallVector<int32_t> concatMask = {0, 1, 2, 3, 4, 5, 6, 7,
                                       8, 9, 10, 11, 12, 13, 14, 15};
    return LLVM::ShuffleVectorOp::create(builder, loc, wideTy, lowHalf, highHalf,
                                         concatMask);
  };

  // For IU8 / IU4, each lane carries <2xi32> (64 bits). WMMA256b expects
  // a 128-bit source per operand lane, where the upper half comes from lane^16.
  auto expandPackedI32To4WideWmma256 = [&](Value v) -> Value {
    auto vt = dyn_cast<VectorType>(v.getType());
    if (!vt || vt.getShape().size() != 1 || vt.getShape()[0] != 2 ||
        !vt.getElementType().isInteger(32))
      return v;

    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto wideTy = VectorType::get({4}, i32Ty);

    Value lane = ROCDL::ThreadIdXOp::create(builder, loc, i32Ty).getResult();
    Value c31 = LLVM::ConstantOp::create(builder, loc, i32Ty,
                                         builder.getI32IntegerAttr(31));
    Value cNeg32 = LLVM::ConstantOp::create(builder, loc, i32Ty,
                                            builder.getI32IntegerAttr(-32));
    Value c16 = LLVM::ConstantOp::create(builder, loc, i32Ty,
                                         builder.getI32IntegerAttr(16));
    Value c2 = LLVM::ConstantOp::create(builder, loc, i32Ty,
                                        builder.getI32IntegerAttr(2));

    Value laneInWave = LLVM::AndOp::create(builder, loc, lane, c31);
    Value waveBase = LLVM::AndOp::create(builder, loc, lane, cNeg32);
    Value pairedLaneInWave = LLVM::XOrOp::create(builder, loc, laneInWave, c16);
    Value pairedLane = LLVM::OrOp::create(builder, loc, waveBase, pairedLaneInWave);
    Value pairedLaneByteAddr = LLVM::ShlOp::create(builder, loc, pairedLane, c2);

    Value pairedHalf = LLVM::UndefOp::create(builder, loc, vt);
    for (int i = 0; i < 2; ++i) {
      Value idx = LLVM::ConstantOp::create(builder, loc, i64Ty,
                                           builder.getI64IntegerAttr(i));
      Value laneWord = LLVM::ExtractElementOp::create(builder, loc, v, idx);
      Value pairedWord = ROCDL::DsBpermuteOp::create(builder, loc, i32Ty,
                                                     pairedLaneByteAddr, laneWord)
                             .getResult();
      pairedHalf = LLVM::InsertElementOp::create(builder, loc, pairedHalf,
                                                 pairedWord, idx);
    }

    SmallVector<int32_t> concatMask = {0, 1, 2, 3};
    return LLVM::ShuffleVectorOp::create(builder, loc, wideTy, v, pairedHalf,
                                         concatMask);
  };

  auto duplicateTo16WideSimple = [&](Value v) -> Value {
    auto vt = dyn_cast<VectorType>(v.getType());
    if (!vt || vt.getShape().size() != 1 || vt.getShape()[0] != 8)
      return v;
    auto wideTy = VectorType::get({16}, vt.getElementType());
    SmallVector<int32_t> mask = {0, 1, 2, 3, 4, 5, 6, 7,
                                 0, 1, 2, 3, 4, 5, 6, 7};
    return LLVM::ShuffleVectorOp::create(builder, loc, wideTy, v, v, mask);
  };

  // Empirically, RDNA3/3.5 `rocdl.wmma.f32.16x16x16.*` returns the per-lane
  // 8-wide accumulator in a lane-local order different from the default
  // FlyDSL f32 C-fragment order. Reorder to the fragment convention expected
  // by `getThrValLayoutCD` / tiled_copy_C.
  auto reorderAccLaneValues = [&](Value v) -> Value {
    auto vt = dyn_cast<VectorType>(v.getType());
    if (!vt || vt.getShape().size() != 1 || vt.getShape()[0] != 8)
      return v;
    bool isF32 = vt.getElementType().isF32();
    bool isI32 = vt.getElementType().isInteger(32);
    if (!isF32 && !isI32)
      return v;
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i64Ty = IntegerType::get(ctx, 64);
    auto i32VecTy = VectorType::get({8}, i32Ty);

    // First, apply the observed lane-local reorder from WMMA result encoding.
    SmallVector<int32_t> laneLocalMask = {0, 4, 1, 5, 2, 6, 3, 7};
    Value laneLocal = LLVM::ShuffleVectorOp::create(builder, loc, vt, v, v, laneLocalMask);

    // Then fix the cross-lane row bit permutation (bit0 <-> bit3) by pulling
    // companion values from lane^16 and selecting by lane-group.
    Value lane = ROCDL::ThreadIdXOp::create(builder, loc, i32Ty).getResult();
    Value c31 = LLVM::ConstantOp::create(builder, loc, i32Ty,
                                         builder.getI32IntegerAttr(31));
    Value cNeg32 = LLVM::ConstantOp::create(builder, loc, i32Ty,
                                            builder.getI32IntegerAttr(-32));
    Value c16 = LLVM::ConstantOp::create(builder, loc, i32Ty,
                                         builder.getI32IntegerAttr(16));
    Value c2 = LLVM::ConstantOp::create(builder, loc, i32Ty,
                                        builder.getI32IntegerAttr(2));
    Value c0 = LLVM::ConstantOp::create(builder, loc, i32Ty,
                                        builder.getI32IntegerAttr(0));

    Value laneInWave = LLVM::AndOp::create(builder, loc, lane, c31);
    Value waveBase = LLVM::AndOp::create(builder, loc, lane, cNeg32);
    Value pairedLaneInWave = LLVM::XOrOp::create(builder, loc, laneInWave, c16);
    Value pairedLane = LLVM::OrOp::create(builder, loc, waveBase, pairedLaneInWave);
    Value pairedLaneByteAddr = LLVM::ShlOp::create(builder, loc, pairedLane, c2);

    Value packed = isF32 ? LLVM::BitcastOp::create(builder, loc, i32VecTy, laneLocal)
                         : laneLocal;
    Value pairedPacked = LLVM::UndefOp::create(builder, loc, i32VecTy);
    for (int i = 0; i < 8; ++i) {
      Value idx = LLVM::ConstantOp::create(builder, loc, i64Ty,
                                           builder.getI64IntegerAttr(i));
      Value laneWord = LLVM::ExtractElementOp::create(builder, loc, packed, idx);
      Value pairedWord = ROCDL::DsBpermuteOp::create(builder, loc, i32Ty,
                                                     pairedLaneByteAddr, laneWord)
                             .getResult();
      pairedPacked = LLVM::InsertElementOp::create(builder, loc, pairedPacked,
                                                   pairedWord, idx);
    }
    Value pairedLaneLocal = isF32 ? LLVM::BitcastOp::create(builder, loc, vt, pairedPacked)
                                  : pairedPacked;

    SmallVector<int32_t> maskGroup0 = {0, 8, 2, 10, 4, 12, 6, 14};
    SmallVector<int32_t> maskGroup1 = {9, 1, 11, 3, 13, 5, 15, 7};
    Value outGroup0 = LLVM::ShuffleVectorOp::create(builder, loc, vt, laneLocal,
                                                    pairedLaneLocal, maskGroup0);
    Value outGroup1 = LLVM::ShuffleVectorOp::create(builder, loc, vt, laneLocal,
                                                    pairedLaneLocal, maskGroup1);

    Value laneGroupBit = LLVM::AndOp::create(builder, loc, laneInWave, c16);
    Value isUpperGroup =
        LLVM::ICmpOp::create(builder, loc, LLVM::ICmpPredicate::ne, laneGroupBit, c0);
    return LLVM::SelectOp::create(builder, loc, isUpperGroup, outGroup1, outGroup0);
  };

  // F32 <- F16/F16
  if (elemTyA.isF16() && elemTyB.isF16() && elemTyAcc.isF32()) {
    constexpr bool crossLaneA = true;
    constexpr bool crossLaneB = true;
    Value aDup = crossLaneA ? expandTo16WideWmma256(a) : duplicateTo16WideSimple(a);
    Value bDup = crossLaneB ? expandTo16WideWmma256(b) : duplicateTo16WideSimple(b);
    Value cPacked =
        reorderAccLaneValues(reorderAccLaneValues(reorderAccLaneValues(c)));
    Value res = ROCDL::wmma_f32_16x16x16_f16::create(builder, loc, accTy, aDup,
                                                     bDup, cPacked)
                    .getResult();
    return reorderAccLaneValues(res);
  }
  // F32 <- BF16/BF16 (op A/B input ty is AnyInteger -> bitcast bf16 to i16)
  if (elemTyA.isBF16() && elemTyB.isBF16() && elemTyAcc.isF32()) {
    constexpr bool crossLaneA = true;
    constexpr bool crossLaneB = true;
    Value aI = castToI16Vec(crossLaneA ? expandTo16WideWmma256(a)
                                       : duplicateTo16WideSimple(a));
    Value bI = castToI16Vec(crossLaneB ? expandTo16WideWmma256(b)
                                       : duplicateTo16WideSimple(b));
    Value cPacked =
        reorderAccLaneValues(reorderAccLaneValues(reorderAccLaneValues(c)));
    Value res =
        ROCDL::wmma_f32_16x16x16_bf16::create(builder, loc, accTy, aI, bI, cPacked)
            .getResult();
    return reorderAccLaneValues(res);
  }
  // F16 <- F16/F16.
  //
  // The native f16-acc WMMA variant uses op_sel-packed 16-wide C/D lanes, and
  // keeping that packed state across loop-carried accumulation in the current
  // vec8 fragment model is error-prone. For correctness, we use the f32-acc
  // WMMA path and truncate the final lane-local result to f16.
  if (elemTyA.isF16() && elemTyB.isF16() && elemTyAcc.isF16()) {
    Value aDup = expandTo16WideWmma256(a);
    Value bDup = expandTo16WideWmma256(b);
    auto accF32Ty = VectorType::get({accVecSize}, builder.getF32Type());
    Value cF32 = LLVM::FPExtOp::create(builder, loc, accF32Ty, c);
    Value cPacked =
        reorderAccLaneValues(reorderAccLaneValues(reorderAccLaneValues(cF32)));
    Value resF32 = ROCDL::wmma_f32_16x16x16_f16::create(
                       builder, loc, accF32Ty, aDup, bDup, cPacked)
                       .getResult();
    Value resCanonicalF32 = reorderAccLaneValues(resF32);
    return LLVM::FPTruncOp::create(builder, loc, accTy, resCanonicalF32).getResult();
  }
  // BF16 <- BF16/BF16 (op_sel = 0). Op signature is AnyInteger for both A/B
  // and C/D, so we bitcast to / from i16 vectors and extract low half.
  if (elemTyA.isBF16() && elemTyB.isBF16() && elemTyAcc.isBF16()) {
    Value aI = castToI16Vec(expandTo16WideWmma256(a));
    Value bI = castToI16Vec(expandTo16WideWmma256(b));
    Value cI = castToI16Vec(expandTo16WideWmma256(c));
    auto wideIntTy = cast<VectorType>(cI.getType());
    Value resWideI = ROCDL::wmma_bf16_16x16x16_bf16::create(
                         builder, loc, wideIntTy, aI, bI, cI, /*opsel=*/false)
                         .getResult();
    SmallVector<int32_t> mask = {0, 1, 2, 3, 4, 5, 6, 7};
    auto narrowIntTy = VectorType::get({8}, i16Ty);
    Value resI = LLVM::ShuffleVectorOp::create(builder, loc, narrowIntTy,
                                               resWideI, resWideI, mask);
    Value res = LLVM::BitcastOp::create(builder, loc, accTy, resI);
    return res;
  }
  // I32 <- IU8 (signA = signB = 0 = unsigned, clamp = 0).
  // For RDNA 3 / 3.5 WMMA256b, the upper half must come from lane^16.
  if (elemTyA.isInteger(8) && elemTyB.isInteger(8) && elemTyAcc.isInteger(32)) {
    Value aWide = expandPackedI32To4WideWmma256(a);
    Value bWide = expandPackedI32To4WideWmma256(b);
    Value cPacked =
        reorderAccLaneValues(reorderAccLaneValues(reorderAccLaneValues(c)));
    Value res = ROCDL::wmma_i32_16x16x16_iu8::create(
                    builder, loc, accTy,
                    /*signA=*/false, aWide, /*signB=*/false, bWide, cPacked,
                    /*clamp=*/false)
                    .getResult();
    return reorderAccLaneValues(res);
  }
  // I32 <- IU4. Same WMMA256b lane^16 upper-half rule as IU8.
  if (elemTyA.isInteger(4) && elemTyB.isInteger(4) && elemTyAcc.isInteger(32)) {
    Value aWide = expandPackedI32To4WideWmma256(a);
    Value bWide = expandPackedI32To4WideWmma256(b);
    Value cPacked =
        reorderAccLaneValues(reorderAccLaneValues(reorderAccLaneValues(c)));
    Value res = ROCDL::wmma_i32_16x16x16_iu4::create(
                    builder, loc, accTy,
                    /*signA=*/false, aWide, /*signB=*/false, bWide, cPacked,
                    /*clamp=*/false)
                    .getResult();
    return reorderAccLaneValues(res);
  }

  return failure();
}

LogicalResult MmaOpRDNA3_WMMAType::emitAtomCall(
    OpBuilder &builder, Location loc, Type mmaAtomTy, Type /*dMemTy*/,
    Type /*aMemTy*/, Type /*bMemTy*/, Type /*cMemTy*/, Value atomVal, Value dPtr,
    Value aPtr, Value bPtr, Value cPtr) const {
  Type elemTyA = getElemTyA();
  Type elemTyB = getElemTyB();
  Type elemTyAcc = getElemTyAcc();
  MLIRContext *ctx = builder.getContext();

  Type abTyA = getWmmaABType(ctx, elemTyA);
  Type abTyB = getWmmaABType(ctx, elemTyB);
  if (!abTyA || !abTyB)
    return failure();

  int64_t accVecSize = getWmmaAccVecSize(getM(), getK(), elemTyAcc);
  if (accVecSize == 0)
    return failure();

  VectorType accTy = VectorType::get({accVecSize}, elemTyAcc);

  Value a = LLVM::LoadOp::create(builder, loc, abTyA, aPtr);
  Value b = LLVM::LoadOp::create(builder, loc, abTyB, bPtr);
  Value c = LLVM::LoadOp::create(builder, loc, accTy, cPtr);
  auto res = emitAtomCallSSA(builder, loc, Type{}, mmaAtomTy, accTy, abTyA,
                             abTyB, accTy, atomVal, Value{}, a, b, c);
  if (failed(res))
    return failure();
  LLVM::StoreOp::create(builder, loc, *res, dPtr);
  return success();
}

} // namespace mlir::fly_rocdl
