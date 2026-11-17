// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"

#include "../GFX1250/WmmaLayout.h"

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_rocdl {

//===----------------------------------------------------------------------===//
// GFX120X (RDNA4: gfx1200, gfx1201) WMMA wave32.
//
// RDNA4 sits between the two existing WMMA atoms:
//   * Floating-point instruction shapes are the gfx11 ones (16x16x16
//     fp16/bf16/fp8/bf8), and the
//     ROCDL intrinsics take the plain 3-operand {a, b, c} form — unlike
//     gfx1250, whose 16x16x32 ops carry mods/reuse operands.
//   * The register ABI is the gfx1250 "v8" one — each lane holds K/2 = 8 A/B
//     elements instead of the gfx11 16 (which broadcast across lane halves).
//     K=16 A/B and all C fragments reuse the gfx1250 helpers (same numbers as
//     numBlocks==1 / 32-bit C). IU4 K=32 uses a local A/B layout.
//===----------------------------------------------------------------------===//

namespace {

// GFX120X-local. Do NOT call gfx1250::getThrValLayoutAB(ctx, 32, i4):
// gfx1250 has no V_WMMA_I32_16X16X32_IU4 (LLVM gates the pattern on
// HasWMMA128bInsts + isGFX11PlusNot12_50), so that helper's numBlocks==2
// output for this shape is an untested extrapolation, not a second ABI.
// Paper check vs gfx1250 numBlocks==2:
//   (g,v)=(0,8): RDNA4 K=16g+v = 8;  gfx1250 = 16
//   (g,v)=(1,0): RDNA4 K=16g+v = 16; gfx1250 = 8
static LayoutAttr getThrValLayoutAB_IU4_K32(MLIRContext *ctx) {
  auto getContext = [&]() { return ctx; };
  // RDNA4 ISA §7.12.2 A-matrix 16×32 data size 4, wave32:
  //   lane = {K[4], M[3:0]}, vgpr = K[3], startPosn = K[2:0]
  //   => M = l%16,  K = (l/16)*16 + v     v ∈ [0, 16)
  // column-major (M,K) stride (1, 16):
  //   pos = M + K*16 = (l%16)*1 + (l/16)*256 + v*16
  return FxLayout(FxShape(FxThr(16, 2), FxVal(16)), FxStride(FxThr(1, 256), FxVal(16)));
}

} // namespace

bool MmaOpGFX120X_WMMAType::isStatic() const { return true; }

Type MmaOpGFX120X_WMMAType::getValTypeA() const { return getElemTyA(); }
Type MmaOpGFX120X_WMMAType::getValTypeB() const { return getElemTyB(); }
Type MmaOpGFX120X_WMMAType::getValTypeC() const { return getElemTyAcc(); }
Type MmaOpGFX120X_WMMAType::getValTypeD() const { return getElemTyAcc(); }

Attribute MmaOpGFX120X_WMMAType::getThrLayout() const { return FxLayout(FxC(32), FxC(1)); }

Attribute MmaOpGFX120X_WMMAType::getShapeMNK() const {
  return IntTupleAttr::get(ArrayAttr::get(getContext(), {FxC(getM()), FxC(getN()), FxC(getK())}));
}

// K=16 integer A/B: same numbers as gfx1250 numBlocks==1.
// Authority: RDNA4 ISA §7.12.2 A-matrix 16×16, data size 4 (IU4) and data
// size 8 (IU8) — different rows, same map M = l%16, K = (l/16)*8 + v.
Attribute MmaOpGFX120X_WMMAType::getThrValLayoutA() const {
  if (getElemTyA().isInteger(4) && getK() == 32)
    return getThrValLayoutAB_IU4_K32(getContext());
  return gfx1250::getThrValLayoutAB(getContext(), getK(), getElemTyA());
}

Attribute MmaOpGFX120X_WMMAType::getThrValLayoutB() const {
  if (getElemTyB().isInteger(4) && getK() == 32)
    return getThrValLayoutAB_IU4_K32(getContext());
  return gfx1250::getThrValLayoutAB(getContext(), getK(), getElemTyB());
}

Attribute MmaOpGFX120X_WMMAType::getThrValLayoutC() const {
  return gfx1250::getThrValLayoutCD(getContext(), getElemTyAcc());
}

static bool isFp8OrBf8(Type elemTy) { return isa<Float8E4M3FNType, Float8E5M2Type>(elemTy); }

LogicalResult MmaOpGFX120X_WMMAType::verify(function_ref<InFlightDiagnostic()> emitError, int32_t m,
                                            int32_t n, int32_t k, Type elemTyA, Type elemTyB,
                                            Type elemTyAcc, bool signA, bool signB, bool clamp) {
  if (m != 16 || n != 16) {
    return emitError() << "GFX120X WMMA requires M=N=16, got " << m << "x" << n;
  }

  const bool isFp8 = isFp8OrBf8(elemTyA) && isFp8OrBf8(elemTyB) && elemTyAcc.isF32();
  const bool isFp = (elemTyA.isF16() && elemTyB.isF16() && elemTyAcc.isF32()) ||
                    (elemTyA.isBF16() && elemTyB.isBF16() && elemTyAcc.isF32()) || isFp8;

  auto isInt = [](Type t, unsigned width) {
    auto it = dyn_cast<IntegerType>(t);
    return it && it.getWidth() == width;
  };
  const bool isI8 = isInt(elemTyA, 8) && isInt(elemTyB, 8) && elemTyAcc.isSignlessInteger(32);
  const bool isI4 = isInt(elemTyA, 4) && isInt(elemTyB, 4) && elemTyAcc.isSignlessInteger(32);

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

//===----------------------------------------------------------------------===//
// Codegen: lower the atom call to a rocdl.wmma.* intrinsic op.
//===----------------------------------------------------------------------===//

// A/B operand vector type on RDNA4 wave32 for 16x16x16: M*K/32 = 8 elements
// per lane.
//   fp16 -> vector<8xf16>
//   bf16 -> vector<8xi16>   (the bf16 WMMA intrinsic takes integer operands)
//   fp8/bf8 -> vector<2xi32>   (8 packed 8-bit values)
//   iu8 K=16 -> vector<2xi32>  (8 packed 8-bit values)
//   iu4 K=16 -> i32            (8 packed 4-bit values)
//   iu4 K=32 -> vector<2xi32>  (16 packed 4-bit values)
static Type getWmmaABType(MLIRContext *ctx, Type elemTy, int32_t k) {
  Type i32Ty = IntegerType::get(ctx, 32);
  if (elemTy.isInteger(4) && k == 16)
    return i32Ty;
  if (elemTy.isInteger(4) && k == 32)
    return VectorType::get({2}, i32Ty);
  if (elemTy.isInteger(8) && k == 16)
    return VectorType::get({2}, i32Ty);
  if (isFp8OrBf8(elemTy))
    return VectorType::get({2}, i32Ty);
  if (elemTy.isBF16())
    return VectorType::get({8}, IntegerType::get(ctx, 16));
  if (elemTy.isF16())
    return VectorType::get({8}, elemTy);
  return nullptr;
}

// Accumulator/result vector type on RDNA4 wave32: 8 f32/i32 slots per lane.
static Type getWmmaAccRawType(Type elemTyAcc) {
  if (elemTyAcc.isF32() || elemTyAcc.isInteger(32))
    return VectorType::get({8}, elemTyAcc);
  return nullptr;
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

} // namespace mlir::fly_rocdl
