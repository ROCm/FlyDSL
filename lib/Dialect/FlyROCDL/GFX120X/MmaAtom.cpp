// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_rocdl {

//===----------------------------------------------------------------------===//
// GFX120X (RDNA4: gfx1200, gfx1201) WMMA & SWMMAC wave32.
//
// RDNA4 sits between gfx11 and gfx1250:
//   * Dense floating-point WMMA uses 16x16x16 with the plain 3-operand
//     {a, b, c} ROCDL intrinsics (no mods/reuse operands); dense integer WMMA
//     supports IU8 16x16x16 and IU4 16x16x16 / 16x16x32.
//   * 2:4 structured sparse SWMMAC uses 16x16x32 for f16 -> f32/f16,
//     bf16 -> f32/bf16, fp8/bf8 -> f32, and IU8 -> i32, plus 16x16x32 and
//     16x16x64 for IU4 -> i32, with 4-operand {a, b, c, index} ROCDL intrinsics.
//   * On RDNA4 wave32, for a stored K dimension `kStored` (16, 32, or 64), each
//     half-wave g = lane / 16 holds one contiguous `valsPerLane = kStored / 2`
//     slice along K: M = lane % 16, K_stored = g * valsPerLane + v. Do NOT
//     delegate kStored >= 32 to gfx1250::getThrValLayoutAB, which interleaves
//     8-element blocks across half-waves (((16,2),(8,numBlocks)):((1,128),
//     (16,256))) rather than RDNA4's contiguous half-wave slice.
//   * Likewise, RDNA4 wave32 C/D fragments always map v in [0, 8) contiguously
//     to M = (lane / 16) * 8 + v, N = lane % 16 for both 32-bit (f32/i32) and
//     16-bit (f16/bf16) accumulators.
//===----------------------------------------------------------------------===//

namespace {

static LayoutAttr getThrValLayoutAB(MLIRContext *ctx, int32_t kStored) {
  auto getContext = [&]() { return ctx; };
  int32_t valsPerLane = kStored / 2;
  // RDNA4 ISA §7.12.2 / §7.12.4 wave32 A/B matrix layout:
  //   M = lane % 16,  K_stored = (lane / 16) * valsPerLane + v,  v in [0, valsPerLane)
  // Column-major (M=16, K_stored) stride (1, 16):
  //   pos = M + K_stored * 16 = (lane % 16) * 1 + (lane / 16) * (valsPerLane * 16) + v * 16
  return FxLayout(FxShape(FxThr(16, 2), FxVal(valsPerLane)),
                  FxStride(FxThr(1, valsPerLane * 16), FxVal(16)));
}

static LayoutAttr getThrValLayoutCD(MLIRContext *ctx) {
  auto getContext = [&]() { return ctx; };
  // RDNA4 wave32 C/D matrix layout (both 32-bit f32/i32 and 16-bit f16/bf16):
  //   M = (lane / 16) * 8 + v,  N = lane % 16,  v in [0, 8)
  // Column-major (M=16, N=16) stride (1, 16):
  //   pos = M + N * 16 = (lane % 16) * 16 + (lane / 16) * 8 + v
  return FxLayout(FxShape(FxThr(16, 2), FxVal(8)), FxStride(FxThr(16, 8), FxVal(1)));
}

static bool isFp8OrBf8(Type elemTy) { return isa<Float8E4M3FNType, Float8E5M2Type>(elemTy); }

static bool isIntWidth(Type t, unsigned width) {
  auto it = dyn_cast<IntegerType>(t);
  return it && it.getWidth() == width;
}

// Per-lane A/B operand register type on RDNA4 wave32 for stored K dimension
// `kStored` (`valsPerLane = kStored / 2` elements per lane):
//   fp16   kStored=16/32 -> vector<8xf16> / vector<16xf16>
//   bf16   kStored=16/32 -> vector<8xi16> / vector<16xi16>
//   fp8/bf8 kStored=16/32 -> vector<2xi32> / vector<4xi32>
//   iu8    kStored=16/32 -> vector<2xi32> / vector<4xi32>
//   iu4    kStored=16/32/64 -> i32 / vector<2xi32> / vector<4xi32>
static Type getWmmaABType(MLIRContext *ctx, Type elemTy, int32_t kStored) {
  Type i32Ty = IntegerType::get(ctx, 32);
  if (elemTy.isInteger(4)) {
    if (kStored == 16)
      return i32Ty;
    if (kStored == 32)
      return VectorType::get({2}, i32Ty);
    if (kStored == 64)
      return VectorType::get({4}, i32Ty);
    return nullptr;
  }
  if (elemTy.isInteger(8) || isFp8OrBf8(elemTy)) {
    if (kStored == 16)
      return VectorType::get({2}, i32Ty);
    if (kStored == 32)
      return VectorType::get({4}, i32Ty);
    return nullptr;
  }
  if (elemTy.isBF16() && (kStored == 16 || kStored == 32))
    return VectorType::get({kStored / 2}, IntegerType::get(ctx, 16));
  if (elemTy.isF16() && (kStored == 16 || kStored == 32))
    return VectorType::get({kStored / 2}, elemTy);
  return nullptr;
}

// Accumulator/result vector type for dense GFX120X WMMA: 8 f32/i32 slots/lane.
static Type getWmmaAccRawType(Type elemTyAcc) {
  if (elemTyAcc.isF32() || elemTyAcc.isInteger(32))
    return VectorType::get({8}, elemTyAcc);
  return nullptr;
}

// Accumulator/result vector type for GFX120X SWMMAC ROCDL intrinsics:
//   f32 / i32 / f16 -> vector<8xf32> / vector<8xi32> / vector<8xf16>
//   bf16            -> vector<8xi16> (rocdl.swmmac.bf16.16x16x32.bf16 ABI)
static Type getSwmmacAccRawType(MLIRContext *ctx, Type elemTyAcc) {
  if (elemTyAcc.isF32() || elemTyAcc.isInteger(32) || elemTyAcc.isF16())
    return VectorType::get({8}, elemTyAcc);
  if (elemTyAcc.isBF16())
    return VectorType::get({8}, IntegerType::get(ctx, 16));
  return nullptr;
}

} // namespace

//===----------------------------------------------------------------------===//
// MmaOpGFX120X_WMMA (dense)
//===----------------------------------------------------------------------===//

bool MmaOpGFX120X_WMMAType::isStatic() const { return true; }

Value MmaOpGFX120X_WMMAType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                                Value currentValue) const {
  if (currentValue && isa<MakeMmaAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  return MakeMmaAtomOp::create(builder, loc, MmaAtomType::get(*this));
}

Type MmaOpGFX120X_WMMAType::getValTypeA() const { return getElemTyA(); }
Type MmaOpGFX120X_WMMAType::getValTypeB() const { return getElemTyB(); }
Type MmaOpGFX120X_WMMAType::getValTypeC() const { return getElemTyAcc(); }
Type MmaOpGFX120X_WMMAType::getValTypeD() const { return getElemTyAcc(); }

Attribute MmaOpGFX120X_WMMAType::getThrLayout() const { return FxLayout(FxC(32), FxC(1)); }

Attribute MmaOpGFX120X_WMMAType::getShapeMNK() const {
  return IntTupleAttr::get(ArrayAttr::get(getContext(), {FxC(getM()), FxC(getN()), FxC(getK())}));
}

Attribute MmaOpGFX120X_WMMAType::getThrValLayoutA() const {
  return getThrValLayoutAB(getContext(), getK());
}

Attribute MmaOpGFX120X_WMMAType::getThrValLayoutB() const {
  return getThrValLayoutAB(getContext(), getK());
}

Attribute MmaOpGFX120X_WMMAType::getThrValLayoutC() const {
  return getThrValLayoutCD(getContext());
}

LogicalResult MmaOpGFX120X_WMMAType::verify(function_ref<InFlightDiagnostic()> emitError, int32_t m,
                                            int32_t n, int32_t k, Type elemTyA, Type elemTyB,
                                            Type elemTyAcc, bool signA, bool signB, bool clamp) {
  if (m != 16 || n != 16) {
    return emitError() << "GFX120X WMMA requires M=N=16, got " << m << "x" << n;
  }

  const bool isFp8 = isFp8OrBf8(elemTyA) && isFp8OrBf8(elemTyB) && elemTyAcc.isF32();
  const bool isFp = (elemTyA.isF16() && elemTyB.isF16() && elemTyAcc.isF32()) ||
                    (elemTyA.isBF16() && elemTyB.isBF16() && elemTyAcc.isF32()) || isFp8;
  const bool isI8 =
      isIntWidth(elemTyA, 8) && isIntWidth(elemTyB, 8) && elemTyAcc.isSignlessInteger(32);
  const bool isI4 =
      isIntWidth(elemTyA, 4) && isIntWidth(elemTyB, 4) && elemTyAcc.isSignlessInteger(32);

  if (!isFp && !isI8 && !isI4) {
    return emitError() << "unsupported GFX120X WMMA configuration: " << m << "x" << n << "x" << k
                       << " with A=" << elemTyA << ", B=" << elemTyB << ", Acc=" << elemTyAcc;
  }

  if (isFp) {
    if (k != 16) {
      return emitError() << "GFX120X WMMA floating-point forms require M=N=K=16, got " << m << "x"
                         << n << "x" << k;
    }
    if (signA || signB || clamp) {
      return emitError() << "GFX120X WMMA floating-point path does not accept signA/signB/clamp "
                            "(the ROCDL fp WMMA intrinsics have no such operands); got signA="
                         << signA << ", signB=" << signB << ", clamp=" << clamp;
    }
  }

  if (isI8 && k != 16) {
    return emitError() << "GFX120X WMMA IU8 requires K=16, got " << m << "x" << n << "x" << k;
  }
  if (isI4 && k != 16 && k != 32) {
    return emitError() << "GFX120X WMMA IU4 requires K=16 or K=32, got " << m << "x" << n << "x"
                       << k;
  }

  return success();
}

FailureOr<Value> MmaOpGFX120X_WMMAType::emitAtomCallSSA(
    OpBuilder &builder, Location loc, Type /*resultTy*/, Type /*mmaAtomTyArg*/, Type /*dTyArg*/,
    TypeRange /*aTyArg*/, TypeRange /*bTyArg*/, Type /*cTyArg*/, Value /*atomVal*/, Value /*d*/,
    ValueRange aValues, ValueRange bValues, Value c) const {
  if (aValues.size() != 1 || bValues.size() != 1) {
    emitError(loc, "this MMA atom does not support auxiliary operands");
    return failure();
  }
  Value a = aValues.front();
  Value b = bValues.front();

  int32_t m = getM();
  int32_t n = getN();
  int32_t k = getK();
  Type elemTyA = getElemTyA();
  Type elemTyB = getElemTyB();
  Type elemTyAcc = getElemTyAcc();
  MLIRContext *ctx = builder.getContext();

  if (m != 16 || n != 16)
    return failure();

  Type abTyA = getWmmaABType(ctx, elemTyA, k);
  Type abTyB = getWmmaABType(ctx, elemTyB, k);
  Type rawAccTy = getWmmaAccRawType(elemTyAcc);
  if (!abTyA || !abTyB || !rawAccTy)
    return failure();

  if (a.getType() != abTyA)
    a = LLVM::BitcastOp::create(builder, loc, abTyA, a);
  if (b.getType() != abTyB)
    b = LLVM::BitcastOp::create(builder, loc, abTyB, b);
  if (c.getType() != rawAccTy)
    c = LLVM::BitcastOp::create(builder, loc, rawAccTy, c);

  if (elemTyA.isF16() && elemTyB.isF16())
    return ROCDL::wmma_f32_16x16x16_f16::create(builder, loc, rawAccTy, a, b, c).getResult();
  if (elemTyA.isBF16() && elemTyB.isBF16())
    return ROCDL::wmma_f32_16x16x16_bf16::create(builder, loc, rawAccTy, a, b, c).getResult();
  if (isa<Float8E4M3FNType>(elemTyA) && isa<Float8E4M3FNType>(elemTyB))
    return ROCDL::wmma_f32_16x16x16_fp8_fp8::create(builder, loc, rawAccTy, a, b, c).getResult();
  if (isa<Float8E4M3FNType>(elemTyA) && isa<Float8E5M2Type>(elemTyB))
    return ROCDL::wmma_f32_16x16x16_fp8_bf8::create(builder, loc, rawAccTy, a, b, c).getResult();
  if (isa<Float8E5M2Type>(elemTyA) && isa<Float8E4M3FNType>(elemTyB))
    return ROCDL::wmma_f32_16x16x16_bf8_fp8::create(builder, loc, rawAccTy, a, b, c).getResult();
  if (isa<Float8E5M2Type>(elemTyA) && isa<Float8E5M2Type>(elemTyB))
    return ROCDL::wmma_f32_16x16x16_bf8_bf8::create(builder, loc, rawAccTy, a, b, c).getResult();

  // Integer forms share the sign and clamp attributes.
  StringRef opName;
  if (elemTyA.isInteger(8) && elemTyB.isInteger(8) && elemTyAcc.isInteger(32) && k == 16)
    opName = ROCDL::wmma_i32_16x16x16_iu8::getOperationName();
  else if (elemTyA.isInteger(4) && elemTyB.isInteger(4) && elemTyAcc.isInteger(32) && k == 16)
    opName = ROCDL::wmma_i32_16x16x16_iu4::getOperationName();
  else if (elemTyA.isInteger(4) && elemTyB.isInteger(4) && elemTyAcc.isInteger(32) && k == 32)
    opName = ROCDL::wmma_i32_16x16x32_iu4::getOperationName();

  if (!opName.empty()) {
    OperationState state(loc, opName);
    state.addTypes(rawAccTy);
    state.addOperands({a, b, c});
    state.addAttribute("signA", builder.getBoolAttr(getSignA()));
    state.addAttribute("signB", builder.getBoolAttr(getSignB()));
    state.addAttribute("clamp", builder.getBoolAttr(getClamp()));
    return builder.create(state)->getResult(0);
  }

  return failure();
}

LogicalResult MmaOpGFX120X_WMMAType::emitAtomCall(OpBuilder &builder, Location loc, Type mmaAtomTy,
                                                  Type /*dMemTy*/, TypeRange /*aMemTy*/,
                                                  TypeRange /*bMemTy*/, Type /*cMemTy*/,
                                                  Value atomVal, Value dPtr, ValueRange aPtrs,
                                                  ValueRange bPtrs, Value cPtr) const {
  if (aPtrs.size() != 1 || bPtrs.size() != 1) {
    emitError(loc, "this MMA atom does not support auxiliary operands");
    return failure();
  }
  Value aPtr = aPtrs.front();
  Value bPtr = bPtrs.front();

  MLIRContext *ctx = builder.getContext();

  Type abTyA = getWmmaABType(ctx, getElemTyA(), getK());
  Type abTyB = getWmmaABType(ctx, getElemTyB(), getK());
  Type accTy = getWmmaAccRawType(getElemTyAcc());
  if (!abTyA || !abTyB || !accTy)
    return failure();

  Value a = LLVM::LoadOp::create(builder, loc, abTyA, aPtr);
  Value b = LLVM::LoadOp::create(builder, loc, abTyB, bPtr);
  Value c = LLVM::LoadOp::create(builder, loc, accTy, cPtr);

  auto res = emitAtomCallSSA(builder, loc, Type{}, mmaAtomTy, accTy, abTyA, abTyB, accTy, atomVal,
                             Value{}, a, b, c);
  if (failed(res))
    return failure();
  LLVM::StoreOp::create(builder, loc, *res, dPtr);
  return success();
}

//===----------------------------------------------------------------------===//
// MmaOpGFX120X_SWMMAC (2:4 structured sparse)
//===----------------------------------------------------------------------===//

bool MmaOpGFX120X_SWMMACType::isStatic() const { return true; }

Value MmaOpGFX120X_SWMMACType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                                  Value currentValue) const {
  if (currentValue && isa<MakeMmaAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  return MakeMmaAtomOp::create(builder, loc, MmaAtomType::get(*this));
}

Type MmaOpGFX120X_SWMMACType::getValTypeA() const { return getElemTyA(); }
Type MmaOpGFX120X_SWMMACType::getValTypeB() const { return getElemTyB(); }
Type MmaOpGFX120X_SWMMACType::getValTypeC() const { return getElemTyAcc(); }
Type MmaOpGFX120X_SWMMACType::getValTypeD() const { return getElemTyAcc(); }

Attribute MmaOpGFX120X_SWMMACType::getThrLayout() const { return FxLayout(FxC(32), FxC(1)); }

Attribute MmaOpGFX120X_SWMMACType::getShapeMNK() const {
  return IntTupleAttr::get(ArrayAttr::get(getContext(), {FxC(getM()), FxC(getN()), FxC(getK())}));
}

// Compressed A holds K/2 non-zero elements per row (8/lane at K=32, 16/lane at
// K=64), mapping (thr=32, val=K/4) onto the physical (16, K/2) compressed tile.
// Note: generic TiledMma operand-A helpers (partition_A, make_fragment_A,
// make_tiled_copy_A) tile A by dense getShapeMNK() = (16, K), so multi-K-tile
// callers should partition compressed A and its per-lane i32 selector explicitly.
Attribute MmaOpGFX120X_SWMMACType::getThrValLayoutA() const {
  return getThrValLayoutAB(getContext(), getK() / 2);
}

// Dense B holds all K elements per column (16/lane at K=32, 32/lane at K=64).
Attribute MmaOpGFX120X_SWMMACType::getThrValLayoutB() const {
  return getThrValLayoutAB(getContext(), getK());
}

Attribute MmaOpGFX120X_SWMMACType::getThrValLayoutC() const {
  return getThrValLayoutCD(getContext());
}

LogicalResult MmaOpGFX120X_SWMMACType::verify(function_ref<InFlightDiagnostic()> emitError,
                                              int32_t m, int32_t n, int32_t k, Type elemTyA,
                                              Type elemTyB, Type elemTyAcc, bool signA, bool signB,
                                              bool clamp) {
  if (m != 16 || n != 16) {
    return emitError() << "GFX120X SWMMAC requires M=N=16, got " << m << "x" << n;
  }

  const bool isFp16 =
      elemTyA.isF16() && elemTyB.isF16() && (elemTyAcc.isF32() || elemTyAcc.isF16());
  const bool isBf16 =
      elemTyA.isBF16() && elemTyB.isBF16() && (elemTyAcc.isF32() || elemTyAcc.isBF16());
  const bool isFp8 = isFp8OrBf8(elemTyA) && isFp8OrBf8(elemTyB) && elemTyAcc.isF32();
  const bool isFp = isFp16 || isBf16 || isFp8;
  const bool isI8 =
      isIntWidth(elemTyA, 8) && isIntWidth(elemTyB, 8) && elemTyAcc.isSignlessInteger(32);
  const bool isI4 =
      isIntWidth(elemTyA, 4) && isIntWidth(elemTyB, 4) && elemTyAcc.isSignlessInteger(32);

  if (!isFp && !isI8 && !isI4) {
    return emitError() << "unsupported GFX120X SWMMAC configuration: " << m << "x" << n << "x" << k
                       << " with A=" << elemTyA << ", B=" << elemTyB << ", Acc=" << elemTyAcc;
  }

  if (isFp) {
    if (k != 32) {
      return emitError() << "GFX120X SWMMAC floating-point forms require M=N=16, K=32, got " << m
                         << "x" << n << "x" << k;
    }
    if (signA || signB || clamp) {
      return emitError() << "GFX120X SWMMAC floating-point path does not accept signA/signB/clamp "
                            "(the ROCDL fp SWMMAC intrinsics have no such operands); got signA="
                         << signA << ", signB=" << signB << ", clamp=" << clamp;
    }
  }

  if (isI8 && k != 32) {
    return emitError() << "GFX120X SWMMAC IU8 requires K=32, got " << m << "x" << n << "x" << k;
  }
  if (isI4 && k != 32 && k != 64) {
    return emitError() << "GFX120X SWMMAC IU4 requires K=32 or K=64, got " << m << "x" << n << "x"
                       << k;
  }

  return success();
}

FailureOr<Value> MmaOpGFX120X_SWMMACType::emitAtomCallSSA(
    OpBuilder &builder, Location loc, Type resultTy, Type /*mmaAtomTyArg*/, Type /*dTyArg*/,
    TypeRange /*aTyArg*/, TypeRange /*bTyArg*/, Type /*cTyArg*/, Value /*atomVal*/, Value /*d*/,
    ValueRange aValues, ValueRange bValues, Value c) const {
  if (aValues.size() != 2 || bValues.size() != 1) {
    emitError(loc, "GFX120X SWMMAC expects [a, sparse_index] for A and [b] for B");
    return failure();
  }
  Value a = aValues[0];
  Value sparseIdx = aValues[1];
  Value b = bValues.front();

  int32_t m = getM();
  int32_t n = getN();
  int32_t k = getK();
  Type elemTyA = getElemTyA();
  Type elemTyB = getElemTyB();
  Type elemTyAcc = getElemTyAcc();
  MLIRContext *ctx = builder.getContext();

  if (m != 16 || n != 16)
    return failure();

  Type abTyA = getWmmaABType(ctx, elemTyA, k / 2);
  Type abTyB = getWmmaABType(ctx, elemTyB, k);
  Type rawAccTy = getSwmmacAccRawType(ctx, elemTyAcc);
  Type i32Ty = IntegerType::get(ctx, 32);
  if (!abTyA || !abTyB || !rawAccTy)
    return failure();

  if (a.getType() != abTyA)
    a = LLVM::BitcastOp::create(builder, loc, abTyA, a);
  if (b.getType() != abTyB)
    b = LLVM::BitcastOp::create(builder, loc, abTyB, b);
  if (c.getType() != rawAccTy)
    c = LLVM::BitcastOp::create(builder, loc, rawAccTy, c);
  if (sparseIdx.getType() != i32Ty)
    sparseIdx = LLVM::BitcastOp::create(builder, loc, i32Ty, sparseIdx);

  Value res;
  if (elemTyA.isF16() && elemTyB.isF16() && elemTyAcc.isF32() && k == 32) {
    res = ROCDL::swmmac_f32_16x16x32_f16::create(builder, loc, rawAccTy, a, b, c, sparseIdx)
              .getResult();
  } else if (elemTyA.isF16() && elemTyB.isF16() && elemTyAcc.isF16() && k == 32) {
    res = ROCDL::swmmac_f16_16x16x32_f16::create(builder, loc, rawAccTy, a, b, c, sparseIdx)
              .getResult();
  } else if (elemTyA.isBF16() && elemTyB.isBF16() && elemTyAcc.isF32() && k == 32) {
    res = ROCDL::swmmac_f32_16x16x32_bf16::create(builder, loc, rawAccTy, a, b, c, sparseIdx)
              .getResult();
  } else if (elemTyA.isBF16() && elemTyB.isBF16() && elemTyAcc.isBF16() && k == 32) {
    res = ROCDL::swmmac_bf16_16x16x32_bf16::create(builder, loc, rawAccTy, a, b, c, sparseIdx)
              .getResult();
  } else if (isa<Float8E4M3FNType>(elemTyA) && isa<Float8E4M3FNType>(elemTyB) &&
             elemTyAcc.isF32() && k == 32) {
    res = ROCDL::swmmac_f32_16x16x32_fp8_fp8::create(builder, loc, rawAccTy, a, b, c, sparseIdx)
              .getResult();
  } else if (isa<Float8E4M3FNType>(elemTyA) && isa<Float8E5M2Type>(elemTyB) && elemTyAcc.isF32() &&
             k == 32) {
    res = ROCDL::swmmac_f32_16x16x32_fp8_bf8::create(builder, loc, rawAccTy, a, b, c, sparseIdx)
              .getResult();
  } else if (isa<Float8E5M2Type>(elemTyA) && isa<Float8E4M3FNType>(elemTyB) && elemTyAcc.isF32() &&
             k == 32) {
    res = ROCDL::swmmac_f32_16x16x32_bf8_fp8::create(builder, loc, rawAccTy, a, b, c, sparseIdx)
              .getResult();
  } else if (isa<Float8E5M2Type>(elemTyA) && isa<Float8E5M2Type>(elemTyB) && elemTyAcc.isF32() &&
             k == 32) {
    res = ROCDL::swmmac_f32_16x16x32_bf8_bf8::create(builder, loc, rawAccTy, a, b, c, sparseIdx)
              .getResult();
  } else {
    StringRef opName;
    if (elemTyA.isInteger(8) && elemTyB.isInteger(8) && elemTyAcc.isInteger(32) && k == 32)
      opName = ROCDL::swmmac_i32_16x16x32_iu8::getOperationName();
    else if (elemTyA.isInteger(4) && elemTyB.isInteger(4) && elemTyAcc.isInteger(32) && k == 32)
      opName = ROCDL::swmmac_i32_16x16x32_iu4::getOperationName();
    else if (elemTyA.isInteger(4) && elemTyB.isInteger(4) && elemTyAcc.isInteger(32) && k == 64)
      opName = ROCDL::swmmac_i32_16x16x64_iu4::getOperationName();

    if (opName.empty())
      return failure();

    OperationState state(loc, opName);
    state.addTypes(rawAccTy);
    state.addOperands({a, b, c, sparseIdx});
    state.addAttribute("signA", builder.getBoolAttr(getSignA()));
    state.addAttribute("signB", builder.getBoolAttr(getSignB()));
    state.addAttribute("clamp", builder.getBoolAttr(getClamp()));
    res = builder.create(state)->getResult(0);
  }

  Type targetResTy = resultTy ? resultTy : VectorType::get({8}, elemTyAcc);
  if (res.getType() != targetResTy)
    res = LLVM::BitcastOp::create(builder, loc, targetResTy, res);
  return res;
}

LogicalResult MmaOpGFX120X_SWMMACType::emitAtomCall(OpBuilder &builder, Location loc,
                                                    Type mmaAtomTy, Type /*dMemTy*/,
                                                    TypeRange /*aMemTy*/, TypeRange /*bMemTy*/,
                                                    Type /*cMemTy*/, Value atomVal, Value dPtr,
                                                    ValueRange aPtrs, ValueRange bPtrs,
                                                    Value cPtr) const {
  if (aPtrs.size() != 2 || bPtrs.size() != 1) {
    emitError(loc, "GFX120X SWMMAC expects [a, sparse_index] for A and [b] for B");
    return failure();
  }

  MLIRContext *ctx = builder.getContext();

  Type abTyA = getWmmaABType(ctx, getElemTyA(), getK() / 2);
  Type abTyB = getWmmaABType(ctx, getElemTyB(), getK());
  Type accTy = getSwmmacAccRawType(ctx, getElemTyAcc());
  Type i32Ty = IntegerType::get(ctx, 32);
  if (!abTyA || !abTyB || !accTy)
    return failure();

  Value a = LLVM::LoadOp::create(builder, loc, abTyA, aPtrs[0]);
  Value sparseIdx = LLVM::LoadOp::create(builder, loc, i32Ty, aPtrs[1]);
  Value b = LLVM::LoadOp::create(builder, loc, abTyB, bPtrs.front());
  Value c = LLVM::LoadOp::create(builder, loc, accTy, cPtr);

  SmallVector<Value, 2> aValues{a, sparseIdx};
  auto res = emitAtomCallSSA(builder, loc, accTy, mmaAtomTy, accTy, ValueRange(aValues).getTypes(),
                             abTyB, accTy, atomVal, Value{}, aValues, b, c);
  if (failed(res))
    return failure();
  LLVM::StoreOp::create(builder, loc, *res, dPtr);
  return success();
}

} // namespace mlir::fly_rocdl
