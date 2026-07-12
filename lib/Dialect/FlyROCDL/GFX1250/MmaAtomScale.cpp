// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"

#include "WmmaLayout.h"

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_rocdl {

//===----------------------------------------------------------------------===//
// MmaOpGFX1250_WMMAScaleType — MX-scaled WMMA (E8M0 block scale)
//
// Wraps ROCDL::wmma_scale_f32_16x16x128_f8f6f4, the gfx1250 wave32 scaled WMMA
// for the unified f8/f6/f4 operand format. Per-operand E8M0 block scales are
// carried as atom state (ScaleA / ScaleB, i32 SGPRs) and set via SetAtomState,
// mirroring MmaOpCDNA4_MFMAScaleType.
//===----------------------------------------------------------------------===//

std::optional<unsigned> MmaOpGFX1250_WMMAScaleType::getFieldIndex(AtomStateField field) {
  switch (field) {
  case AtomStateField::ScaleA:
    return 0;
  case AtomStateField::ScaleB:
    return 1;
  default:
    return std::nullopt;
  }
}

Type MmaOpGFX1250_WMMAScaleType::getConvertedType(MLIRContext *ctx) const {
  auto i32Ty = IntegerType::get(ctx, 32);
  return LLVM::LLVMStructType::getLiteral(ctx, {i32Ty, i32Ty});
}

Value MmaOpGFX1250_WMMAScaleType::getDefaultState(OpBuilder &builder, Location loc) const {
  auto structTy = cast<LLVM::LLVMStructType>(getConvertedType(builder.getContext()));
  Value state = LLVM::UndefOp::create(builder, loc, structTy);
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  state = LLVM::InsertValueOp::create(builder, loc, state, zero,
                                      ArrayRef<int64_t>{*getFieldIndex(AtomStateField::ScaleA)});
  state = LLVM::InsertValueOp::create(builder, loc, state, zero,
                                      ArrayRef<int64_t>{*getFieldIndex(AtomStateField::ScaleB)});
  return state;
}

Value MmaOpGFX1250_WMMAScaleType::setAtomState(OpBuilder &builder, Location loc, Value atomStruct,
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
  Value scaleVal = fieldValue;
  Type srcTy = scaleVal.getType();
  Type i32Ty = IntegerType::get(builder.getContext(), 32);
  if (srcTy != i32Ty) {
    auto bitWidthOf = [](Type t) -> unsigned {
      if (auto vec = dyn_cast<VectorType>(t)) {
        Type elt = vec.getElementType();
        if (!elt.isIntOrFloat())
          return 0;
        return elt.getIntOrFloatBitWidth() * vec.getNumElements();
      }
      if (auto intTy = dyn_cast<IntegerType>(t))
        return intTy.getWidth();
      return 0;
    };
    if (bitWidthOf(srcTy) != 32)
      return nullptr;
    scaleVal = LLVM::BitcastOp::create(builder, loc, i32Ty, scaleVal);
  }
  return LLVM::InsertValueOp::create(builder, loc, atomStruct, scaleVal, ArrayRef<int64_t>{*idx});
}

Attribute MmaOpGFX1250_WMMAScaleType::getThrLayout() const { return FxLayout(FxC(32), FxC(1)); }

Attribute MmaOpGFX1250_WMMAScaleType::getShapeMNK() const {
  return IntTupleAttr::get(ArrayAttr::get(getContext(), {FxC(getM()), FxC(getN()), FxC(getK())}));
}

Type MmaOpGFX1250_WMMAScaleType::getValTypeA() const { return getElemTyA(); }
Type MmaOpGFX1250_WMMAScaleType::getValTypeB() const { return getElemTyB(); }
Type MmaOpGFX1250_WMMAScaleType::getValTypeC() const { return getElemTyAcc(); }
Type MmaOpGFX1250_WMMAScaleType::getValTypeD() const { return getElemTyAcc(); }

// PROVISIONAL layouts for the 32x16x128 f4 form. The emit path (ROCDL op
// selection + operand vector shapes) is validated by FileCheck, but these
// thr-val layouts are derived by analogy to the 16-row form and are NOT yet
// verified against hardware for make_tiled_mma partitioning. The shipping fp4
// kernels feed pre-arranged fragments through a direct mma_atom_call, which
// does not consult these layouts. Do not rely on them for tiled-MMA fragment
// construction without on-device validation.
static Attribute scaled32RowLayoutA(MLIRContext *ctx, int32_t k) {
  auto getContext = [&]() { return ctx; };
  // 32 lanes cover the 32 M rows; each lane holds the full K, reference (M,K)
  // column-major with stride (1, 32).
  return FxLayout(FxShape(FxThr(32), FxVal(k)), FxStride(FxThr(1), FxVal(32)));
}

static Attribute scaled32x16LayoutC(MLIRContext *ctx) {
  auto getContext = [&]() { return ctx; };
  // C is 32x16 f32: N = lane%16, M = (lane/16)*16 + v, 16 VGPRs per lane.
  return FxLayout(FxShape(FxThr(16, 2), FxVal(16)), FxStride(FxThr(16, 16), FxVal(1)));
}

Attribute MmaOpGFX1250_WMMAScaleType::getThrValLayoutA() const {
  if (getM() == 32)
    return scaled32RowLayoutA(getContext(), getK());
  return gfx1250::getThrValLayoutAB(getContext(), getK(), getElemTyA());
}

Attribute MmaOpGFX1250_WMMAScaleType::getThrValLayoutB() const {
  // B is always N=16 rows for both supported shapes.
  return gfx1250::getThrValLayoutAB(getContext(), getK(), getElemTyB());
}

Attribute MmaOpGFX1250_WMMAScaleType::getThrValLayoutC() const {
  if (getM() == 32)
    return scaled32x16LayoutC(getContext());
  return gfx1250::getThrValLayoutCD(getContext(), getElemTyAcc());
}

static bool isSupportedScaledElemTy(Type ty) {
  return isa<Float8E4M3FNType, Float8E5M2Type, Float6E2M3FNType, Float6E3M2FNType,
             Float4E2M1FNType>(ty);
}

LogicalResult MmaOpGFX1250_WMMAScaleType::verify(function_ref<InFlightDiagnostic()> emitError,
                                                 int32_t m, int32_t n, int32_t k, Type elemTyA,
                                                 Type elemTyB, Type elemTyAcc, int32_t opselA,
                                                 int32_t opselB) {
  bool is16 = (m == 16 && n == 16 && k == 128);
  bool is32x16 = (m == 32 && n == 16 && k == 128);
  if (!is16 && !is32x16) {
    return emitError() << "unsupported MNK for GFX1250 WMMA_Scale: " << m << "x" << n << "x" << k
                       << " (expected 16x16x128 or 32x16x128)";
  }
  if (!elemTyAcc.isF32())
    return emitError() << "elemTyAcc must be f32, got " << elemTyAcc;
  // The 32x16x128 form maps to wmma.scale.f32.32x16x128.f4, which is fp4-only.
  if (is32x16 && !(isa<Float4E2M1FNType>(elemTyA) && isa<Float4E2M1FNType>(elemTyB))) {
    return emitError() << "GFX1250 WMMA_Scale 32x16x128 requires f4E2M1FN A and B, got A="
                       << elemTyA << ", B=" << elemTyB;
  }
  if (!isSupportedScaledElemTy(elemTyA)) {
    return emitError() << "elemTyA must be one of f8E4M3FN, f8E5M2, f6E2M3FN, "
                          "f6E3M2FN, f4E2M1FN, got "
                       << elemTyA;
  }
  if (!isSupportedScaledElemTy(elemTyB)) {
    return emitError() << "elemTyB must be one of f8E4M3FN, f8E5M2, f6E2M3FN, "
                          "f6E3M2FN, f4E2M1FN, got "
                       << elemTyB;
  }
  if (opselA < 0 || opselA > 3)
    return emitError() << "opselA must be in [0, 3], got " << opselA;
  if (opselB < 0 || opselB > 3)
    return emitError() << "opselB must be in [0, 3], got " << opselB;
  return success();
}

// Element format code for the f8f6f4 unified operand (matches CDNA4 MFMA_Scale
// / the hardware V_WMMA_SCALE cbsz/blgp encoding).
static std::optional<uint32_t> wmmaScaleFmtEncode(Type elemTy) {
  if (isa<Float8E4M3FNType>(elemTy))
    return 0u;
  if (isa<Float8E5M2Type>(elemTy))
    return 1u;
  if (isa<Float6E2M3FNType>(elemTy))
    return 2u;
  if (isa<Float6E3M2FNType>(elemTy))
    return 3u;
  if (isa<Float4E2M1FNType>(elemTy))
    return 4u;
  return std::nullopt;
}

// A/B operand vector<Nxi32> container for a `rows x K` wave32 scaled WMMA
// operand: i32 words per lane = rows * K * elemBits / 1024. For 16x16x128:
//   fp8 (8-bit) -> vector<16xi32>, fp6 -> vector<12xi32>, fp4 -> vector<8xi32>.
// For the 32x16x128 f4 form: A (32 rows) -> vector<16xi32>, B (16 rows) ->
// vector<8xi32>.
static Type getScaledWmmaABType(MLIRContext *ctx, int32_t rows, int32_t k, Type elemTy) {
  if (!isSupportedScaledElemTy(elemTy))
    return nullptr;
  int64_t elemBits = elemTy.getIntOrFloatBitWidth();
  int64_t i32count = static_cast<int64_t>(rows) * k * elemBits / 1024;
  if (i32count <= 0)
    return nullptr;
  return VectorType::get({i32count}, IntegerType::get(ctx, 32));
}

FailureOr<Value> MmaOpGFX1250_WMMAScaleType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                             Type resultTy, Type mmaAtomTyArg,
                                                             Type dTyArg, Type aTyArg, Type bTyArg,
                                                             Type cTyArg, Value atomVal, Value d,
                                                             Value a, Value b, Value c) const {
  int32_t m = getM();
  int32_t n = getN();
  int32_t k = getK();
  Type elemTyA = getElemTyA();
  Type elemTyB = getElemTyB();
  MLIRContext *ctx = builder.getContext();

  Type abTyA = getScaledWmmaABType(ctx, m, k, elemTyA);
  Type abTyB = getScaledWmmaABType(ctx, n, k, elemTyB);
  if (!abTyA || !abTyB)
    return failure();

  VectorType accTy = VectorType::get({static_cast<int64_t>(m) * n / 32}, getElemTyAcc());

  if (a.getType() != abTyA)
    a = LLVM::BitcastOp::create(builder, loc, abTyA, a);
  if (b.getType() != abTyB)
    b = LLVM::BitcastOp::create(builder, loc, abTyB, b);
  if (c.getType() != accTy)
    c = LLVM::BitcastOp::create(builder, loc, accTy, c);

  Value scaleA = LLVM::ExtractValueOp::create(
      builder, loc, atomVal, ArrayRef<int64_t>{*getFieldIndex(AtomStateField::ScaleA)});
  Value scaleB = LLVM::ExtractValueOp::create(
      builder, loc, atomVal, ArrayRef<int64_t>{*getFieldIndex(AtomStateField::ScaleB)});

  // fmtScaleA / fmtScaleB default to 0 (E8M0). modC / reuse default to 0.
  if (m == 32 && n == 16 && k == 128) {
    // fp4-only form; no fmtA/fmtB operands.
    return ROCDL::wmma_scale_f32_32x16x128_f4::create(
               builder, loc, accTy, a, b, /*modC=*/(uint16_t)0, c,
               /*scaleAType=*/(uint32_t)getOpselA(), /*fmtScaleA=*/(uint32_t)0, scaleA,
               /*scaleBType=*/(uint32_t)getOpselB(), /*fmtScaleB=*/(uint32_t)0, scaleB,
               /*reuseA=*/false, /*reuseB=*/false)
        .getResult();
  }

  std::optional<uint32_t> aFmt = wmmaScaleFmtEncode(elemTyA);
  std::optional<uint32_t> bFmt = wmmaScaleFmtEncode(elemTyB);
  if (!aFmt || !bFmt)
    return failure();

  return ROCDL::wmma_scale_f32_16x16x128_f8f6f4::create(
             builder, loc, accTy, /*fmtA=*/*aFmt, a, /*fmtB=*/*bFmt, b, /*modC=*/(uint16_t)0, c,
             /*scaleAType=*/(uint32_t)getOpselA(), /*fmtScaleA=*/(uint32_t)0, scaleA,
             /*scaleBType=*/(uint32_t)getOpselB(), /*fmtScaleB=*/(uint32_t)0, scaleB,
             /*reuseA=*/false, /*reuseB=*/false)
      .getResult();
}

LogicalResult MmaOpGFX1250_WMMAScaleType::emitAtomCall(OpBuilder &builder, Location loc,
                                                       Type mmaAtomTy, Type dMemTy, Type aMemTy,
                                                       Type bMemTy, Type cMemTy, Value atomVal,
                                                       Value dPtr, Value aPtr, Value bPtr,
                                                       Value cPtr) const {
  MLIRContext *ctx = builder.getContext();
  Type abTyA = getScaledWmmaABType(ctx, getM(), getK(), getElemTyA());
  Type abTyB = getScaledWmmaABType(ctx, getN(), getK(), getElemTyB());
  if (!abTyA || !abTyB)
    return failure();

  VectorType accTy = VectorType::get({static_cast<int64_t>(getM()) * getN() / 32}, getElemTyAcc());

  Value a = LLVM::LoadOp::create(builder, loc, abTyA, aPtr);
  Value b = LLVM::LoadOp::create(builder, loc, abTyB, bPtr);
  Value c = LLVM::LoadOp::create(builder, loc, accTy, cPtr);
  auto res = emitAtomCallSSA(builder, loc, accTy, mmaAtomTy, Type{}, abTyA, abTyB, accTy, atomVal,
                             Value{}, a, b, c);
  if (failed(res))
    return failure();
  LLVM::StoreOp::create(builder, loc, *res, dPtr);
  return success();
}

} // namespace mlir::fly_rocdl
