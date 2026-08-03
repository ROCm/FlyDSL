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
//   * Instruction shapes are the gfx11 ones (16x16x16 fp16/bf16/iu8), and the
//     ROCDL intrinsics take the plain 3-operand {a, b, c} form — unlike
//     gfx1250, whose 16x16x32 ops carry mods/reuse operands.
//   * The register ABI is the gfx1250 "v8" one — each lane holds K/2 = 8 A/B
//     elements instead of the gfx11 16 (which broadcast across lane halves),
//     so the fragment layouts come from the shared gfx1250 helpers.
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

// For K=16 the shared gfx1250 helper yields shape (thr=(16,2), val=8) with
// stride (thr=(1,128), val=16) over a column-major (M,K) reference space, i.e.
//   M = lane % 16,  K = (lane / 16) * 8 + val
// which is exactly the RDNA4 v8 A/B fragment.
Attribute MmaOpGFX120X_WMMAType::getThrValLayoutA() const {
  return gfx1250::getThrValLayoutAB(getContext(), getK(), getElemTyA());
}

Attribute MmaOpGFX120X_WMMAType::getThrValLayoutB() const {
  return gfx1250::getThrValLayoutAB(getContext(), getK(), getElemTyB());
}

Attribute MmaOpGFX120X_WMMAType::getThrValLayoutC() const {
  return gfx1250::getThrValLayoutCD(getContext(), getElemTyAcc());
}

LogicalResult MmaOpGFX120X_WMMAType::verify(function_ref<InFlightDiagnostic()> emitError, int32_t m,
                                            int32_t n, int32_t k, Type elemTyA, Type elemTyB,
                                            Type elemTyAcc, bool signA, bool signB, bool clamp) {
  if (m != 16 || n != 16 || k != 16) {
    return emitError() << "GFX120X WMMA requires M=N=K=16, got " << m << "x" << n << "x" << k
                       << " (the 16x16x32 shapes are gfx1250-only; use gfx1250.wmma there)";
  }

  const bool isFp = (elemTyA.isF16() && elemTyB.isF16() && elemTyAcc.isF32()) ||
                    (elemTyA.isBF16() && elemTyB.isBF16() && elemTyAcc.isF32());

  if (!isFp) {
    return emitError() << "unsupported GFX120X WMMA configuration: " << m << "x" << n << "x" << k
                       << " with A=" << elemTyA << ", B=" << elemTyB << ", Acc=" << elemTyAcc;
  }

  // The fp16/bf16 intrinsics have no sign/clamp operands; refuse to build an
  // atom promising something codegen cannot deliver.
  if (signA || signB || clamp) {
    return emitError() << "GFX120X WMMA fp16/bf16 path does not accept signA/signB/clamp "
                          "(the ROCDL fp WMMA intrinsics have no such operands); got signA="
                       << signA << ", signB=" << signB << ", clamp=" << clamp;
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
static Type getWmmaABType(MLIRContext *ctx, Type elemTy) {
  if (elemTy.isBF16())
    return VectorType::get({8}, IntegerType::get(ctx, 16));
  if (elemTy.isF16())
    return VectorType::get({8}, elemTy);
  return nullptr;
}

// Accumulator/result vector type on RDNA4 wave32: 8 f32 slots per lane.
static Type getWmmaAccRawType(Type elemTyAcc) {
  if (elemTyAcc.isF32())
    return VectorType::get({8}, elemTyAcc);
  return nullptr;
}

FailureOr<Value> MmaOpGFX120X_WMMAType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                        Type /*resultTy*/, Type /*mmaAtomTyArg*/,
                                                        Type /*dTyArg*/, Type /*aTyArg*/,
                                                        Type /*bTyArg*/, Type /*cTyArg*/,
                                                        Value /*atomVal*/, Value /*d*/, Value a,
                                                        Value b, Value c) const {
  int32_t m = getM();
  int32_t n = getN();
  int32_t k = getK();
  Type elemTyA = getElemTyA();
  Type elemTyB = getElemTyB();
  Type elemTyAcc = getElemTyAcc();
  MLIRContext *ctx = builder.getContext();

  if (m != 16 || n != 16 || k != 16)
    return failure();

  Type abTyA = getWmmaABType(ctx, elemTyA);
  Type abTyB = getWmmaABType(ctx, elemTyB);
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

  return failure();
}

LogicalResult MmaOpGFX120X_WMMAType::emitAtomCall(OpBuilder &builder, Location loc, Type mmaAtomTy,
                                                  Type /*dMemTy*/, Type /*aMemTy*/, Type /*bMemTy*/,
                                                  Type /*cMemTy*/, Value atomVal, Value dPtr,
                                                  Value aPtr, Value bPtr, Value cPtr) const {
  MLIRContext *ctx = builder.getContext();

  Type abTyA = getWmmaABType(ctx, getElemTyA());
  Type abTyB = getWmmaABType(ctx, getElemTyB());
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
