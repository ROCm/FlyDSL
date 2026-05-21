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

namespace gfx11 {

// A/B matrix register layout for GFX11 WMMA (wave32).
//
// RDNA3 / RDNA3.5 (gfx1100..gfx1152) WMMA replicates A_frag and B_frag
// between lanes 0-15 and lanes 16-31 (the "2x replication" rule documented in
// the AMD GPUOpen WMMA-on-RDNA3 blog and the RDNA3 ISA reference).
//
// For 16x16x16 fp16/bf16:
//   - Each lane holds 16 elements packed into 8 VGPRs (32 bits each).
//   - lane%16  -> M (for A) or N (for B), i.e. the leading dim of the fragment.
//   - val ele  -> K (the contraction dim).
//   - lane/16  -> identical replica (broadcast: stride 0).
//
// Source reference (GPUOpen example, fp16 16x16x16 wave32):
//   const int lane = lIdx % 16;
//   for (int ele = 0; ele < 16; ++ele)
//     a_frag[ele] = a[16 * lane + ele];   // A[lane, ele]  (row-major)
//   for (int ele = 0; ele < 16; ++ele)
//     b_frag[ele] = b[16 * ele + lane];   // B[ele, lane]  (row-major)
//
// Note B is consumed in its transposed form (treated as N x K like A is M x K),
// so the same layout function serves both A and B. The reference space is
// column-major (X, K) with stride (1, X=16), where X is M or N respectively.
//
// Position formula (lane l in [0, 32), val v in [0, K)):
//   pos = (l % 16) * 1 + (l / 16) * 0 + v * 16
//
// 32-bit elements (f32) are not a valid A/B type on RDNA3 WMMA, so this
// helper assumes sub-32-bit (fp16, bf16, iu8, iu4). iu8/iu4 pack multiple
// values per VGPR; we still expose a flat val-shape of K elements — the
// downstream ROCDL op handles packing via a bitcast.
LayoutAttr getThrValLayoutAB(MLIRContext *ctx, int32_t K, Type /*elemTy*/) {
  auto getContext = [&]() { return ctx; };
  // shape  = (thr=(16, 2), val=K)
  // stride = (thr=(1, 0),  val=16)   ; stride 0 on lane/16 axis = broadcast
  return FxLayout(FxShape(FxThr(16, 2), FxVal(K)), FxStride(FxThr(1, 0), FxVal(16)));
}

// C/D matrix register layout for GFX11 WMMA (wave32) with 32-bit accumulator.
//
// C/D is always 16x16. 8 VGPRs/lane in wave32.
//   lane%16  -> N (column)
//   lane/16  -> row parity (even rows for lanes 0-15, odd rows for 16-31)
//   val v    -> pair index along M
//   => M = v*2 + (lane/16)   ;  N = lane%16
//
// NOTE: This differs from gfx1250, which packs each lane's 8 rows contiguously
// (M = (lane/16)*8 + v). RDNA3 instead INTERLEAVES the two lane halves along M,
// so consecutive rows alternate between lanes 0-15 and lanes 16-31.
//
// Source: GPUOpen "WMMA on RDNA3" blog, wave32 f32 accumulator example:
//   const int lane = lIdx % 32;
//   for (int ele = 0; ele < 8; ++ele) {
//     const int r = ele * 2 + (lane / 16);     // <-- interleaved row
//     c_frag[ele] = c[16 * r + (lane % 16)];   // row-major
//   }
//
// Reference space is column-major (M, N) with stride (1, M=16).
//   pos = M_coord * 1 + N_coord * 16
//       = (v*2 + l/16) * 1 + (l%16) * 16
//       = (l%16)*16 + (l/16)*1 + v*2
LayoutAttr getThrValLayoutCD_F32(MLIRContext *ctx) {
  auto getContext = [&]() { return ctx; };
  return FxLayout(FxShape(FxThr(16, 2), FxVal(8)), FxStride(FxThr(16, 1), FxVal(2)));
}

} // namespace gfx11

namespace mlir::fly_rocdl {

bool MmaOpGFX11_WMMAType::isStatic() const { return true; }

Value MmaOpGFX11_WMMAType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                              Value currentValue) const {
  if (currentValue && isa<MakeMmaAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  return MakeMmaAtomOp::create(builder, loc, MmaAtomType::get(*this));
}

Type MmaOpGFX11_WMMAType::getValTypeA() const { return getElemTyA(); }
Type MmaOpGFX11_WMMAType::getValTypeB() const { return getElemTyB(); }
Type MmaOpGFX11_WMMAType::getValTypeC() const { return getElemTyAcc(); }
Type MmaOpGFX11_WMMAType::getValTypeD() const { return getElemTyAcc(); }

Attribute MmaOpGFX11_WMMAType::getThrLayout() const { return FxLayout(FxC(32), FxC(1)); }

Attribute MmaOpGFX11_WMMAType::getShapeMNK() const {
  return IntTupleAttr::get(ArrayAttr::get(getContext(), {FxC(getM()), FxC(getN()), FxC(getK())}));
}

Attribute MmaOpGFX11_WMMAType::getThrValLayoutA() const {
  return gfx11::getThrValLayoutAB(getContext(), getK(), getElemTyA());
}

Attribute MmaOpGFX11_WMMAType::getThrValLayoutB() const {
  return gfx11::getThrValLayoutAB(getContext(), getK(), getElemTyB());
}

Attribute MmaOpGFX11_WMMAType::getThrValLayoutC() const {
  return gfx11::getThrValLayoutCD_F32(getContext());
}

LogicalResult MmaOpGFX11_WMMAType::verify(function_ref<InFlightDiagnostic()> emitError, int32_t m,
                                          int32_t n, int32_t k, Type elemTyA, Type elemTyB,
                                          Type elemTyAcc) {
  if (m != 16 || n != 16 || k != 16) {
    return emitError() << "GFX11 WMMA requires M=N=K=16, got " << m << "x" << n << "x" << k;
  }

  bool valid = false;

  // fp16/bf16 inputs, f32 accumulator. (16-bit accumulator variants exist on
  // RDNA3 but require VGPR-pair packing/expansion around OPSEL; not yet
  // implemented here.)
  if (elemTyA.isF16() && elemTyB.isF16() && elemTyAcc.isF32())
    valid = true;
  if (elemTyA.isBF16() && elemTyB.isBF16() && elemTyAcc.isF32())
    valid = true;

  // Integer inputs: REQUIRE explicit unsigned signedness (ui8/ui4).
  //
  // The atom contract is unsigned-only because emitAtomCallSSA invokes the
  // ROCDL iu8/iu4 intrinsics with signA=signB=false (unsigned interpretation
  // of the packed operands).
  auto isUI = [](Type t, unsigned width) {
    auto it = dyn_cast<IntegerType>(t);
    return it && it.getWidth() == width && it.isUnsigned();
  };

  if (isUI(elemTyA, 8) && isUI(elemTyB, 8) && elemTyAcc.isInteger(32))
    valid = true;
  if (isUI(elemTyA, 4) && isUI(elemTyB, 4) && elemTyAcc.isInteger(32))
    valid = true;

  if (!valid) {
    // Steer the caller to ui8/ui4 explicitly.
    auto looksLikeInt = [](Type t, unsigned w) {
      auto it = dyn_cast<IntegerType>(t);
      return it && it.getWidth() == w;
    };
    if ((looksLikeInt(elemTyA, 8) || looksLikeInt(elemTyA, 4)) && elemTyAcc.isInteger(32)) {
      return emitError() << "GFX11 WMMA integer inputs must be unsigned "
                            "(ui8/ui4); got A="
                         << elemTyA << ", B=" << elemTyB
                         << ". The lowered ROCDL iu8/iu4 intrinsic is invoked "
                            "with signA=signB=false, so signless/signed "
                            "operands would silently get unsigned semantics. "
                            "Signed-integer WMMA is not yet implemented.";
    }
    return emitError() << "unsupported GFX11 WMMA configuration: " << m << "x" << n << "x" << k
                       << " with A=" << elemTyA << ", B=" << elemTyB << ", Acc=" << elemTyAcc;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Codegen: lower the atom call to a rocdl.wmma.* intrinsic op.
//===----------------------------------------------------------------------===//

// A/B operand vector type on RDNA3 wave32.
//   fp16 -> vector<16xf16>
//   bf16 -> vector<16xi16>   (RDNA3 WMMA represents bf16 as i16; see upstream
//                             AMDGPUToROCDL bf16-as-i16 cast)
//   iu8  -> vector<4xi32>    (16 i8s packed across 4 i32 slots)
//   iu4  -> vector<2xi32>    (16 i4s packed across 2 i32 slots)
static Type getWmmaABType(MLIRContext *ctx, Type elemTy) {
  Type i32Ty = IntegerType::get(ctx, 32);
  Type i16Ty = IntegerType::get(ctx, 16);
  if (elemTy.isInteger(8))
    return VectorType::get({4}, i32Ty);
  if (elemTy.isInteger(4))
    return VectorType::get({2}, i32Ty);
  if (elemTy.isBF16())
    return VectorType::get({16}, i16Ty);
  // fp16
  return VectorType::get({16}, elemTy);
}

// Accumulator/result vector type on RDNA3 wave32.
//   f32  -> vector<8xf32>
//   i32  -> vector<8xi32>
// (16-bit accumulator variants intentionally unsupported; see verify().)
static Type getWmmaAccRawType(MLIRContext * /*ctx*/, Type elemTyAcc) {
  if (elemTyAcc.isF32() || elemTyAcc.isInteger(32))
    return VectorType::get({8}, elemTyAcc);
  return nullptr;
}

// Build a `rocdl.wmma.*` intrinsic via OperationState. This handles the
// heterogeneous operand layouts (some variants interleave SSA values with
// named attributes, e.g. the IU variant has signA/signB/clamp attrs).
static Value buildWmmaOp(OpBuilder &builder, Location loc, StringRef opName, Type resultTy,
                         ValueRange operands, ArrayRef<NamedAttribute> attrs) {
  OperationState state(loc, opName);
  state.addTypes(resultTy);
  state.addOperands(operands);
  state.addAttributes(attrs);
  Operation *op = builder.create(state);
  return op->getResult(0);
}

FailureOr<Value> MmaOpGFX11_WMMAType::emitAtomCallSSA(OpBuilder &builder, Location loc,
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
  Type rawAccTy = getWmmaAccRawType(ctx, elemTyAcc);
  if (!abTyA || !abTyB || !rawAccTy)
    return failure();

  if (a.getType() != abTyA)
    a = LLVM::BitcastOp::create(builder, loc, abTyA, a);
  if (b.getType() != abTyB)
    b = LLVM::BitcastOp::create(builder, loc, abTyB, b);
  if (c.getType() != rawAccTy)
    c = LLVM::BitcastOp::create(builder, loc, rawAccTy, c);

  StringRef opName;
  SmallVector<NamedAttribute, 3> attrs;
  SmallVector<Value, 3> operands;
  BoolAttr falseAttr = builder.getBoolAttr(false);

  if (elemTyA.isF16() && elemTyB.isF16() && elemTyAcc.isF32()) {
    opName = ROCDL::wmma_f32_16x16x16_f16::getOperationName();
    operands = {a, b, c};
  } else if (elemTyA.isBF16() && elemTyB.isBF16() && elemTyAcc.isF32()) {
    opName = ROCDL::wmma_f32_16x16x16_bf16::getOperationName();
    operands = {a, b, c};
  } else if (elemTyA.isInteger(8) && elemTyB.isInteger(8) && elemTyAcc.isInteger(32)) {
    // Unsigned-only by contract (see verify()). signA=signB=false matches the
    // ui8 element type enforced there. clamp=false preserves wraparound on the
    // i32 accumulator.
    opName = ROCDL::wmma_i32_16x16x16_iu8::getOperationName();
    operands = {a, b, c};
    attrs.push_back({builder.getStringAttr("signA"), falseAttr});
    attrs.push_back({builder.getStringAttr("signB"), falseAttr});
    attrs.push_back({builder.getStringAttr("clamp"), falseAttr});
  } else if (elemTyA.isInteger(4) && elemTyB.isInteger(4) && elemTyAcc.isInteger(32)) {
    // Same unsigned-only contract as iu8; see verify().
    opName = ROCDL::wmma_i32_16x16x16_iu4::getOperationName();
    operands = {a, b, c};
    attrs.push_back({builder.getStringAttr("signA"), falseAttr});
    attrs.push_back({builder.getStringAttr("signB"), falseAttr});
    attrs.push_back({builder.getStringAttr("clamp"), falseAttr});
  } else {
    return failure();
  }

  return buildWmmaOp(builder, loc, opName, rawAccTy, operands, attrs);
}

LogicalResult MmaOpGFX11_WMMAType::emitAtomCall(OpBuilder &builder, Location loc, Type mmaAtomTy,
                                                Type /*dMemTy*/, Type /*aMemTy*/, Type /*bMemTy*/,
                                                Type /*cMemTy*/, Value atomVal, Value dPtr,
                                                Value aPtr, Value bPtr, Value cPtr) const {
  Type elemTyA = getElemTyA();
  Type elemTyB = getElemTyB();
  Type elemTyAcc = getElemTyAcc();
  MLIRContext *ctx = builder.getContext();

  Type abTyA = getWmmaABType(ctx, elemTyA);
  Type abTyB = getWmmaABType(ctx, elemTyB);
  // Loads use the "user-facing" accumulator type (vector of elemTyAcc); the
  // bf16-as-i16 transformation needed by the gfx11 intrinsic happens inside
  // emitAtomCallSSA on the SSA values.
  int64_t accLen = (elemTyAcc.isF32() || elemTyAcc.isInteger(32)) ? 8 : 16;
  if (!abTyA || !abTyB)
    return failure();

  VectorType accTy = VectorType::get({accLen}, elemTyAcc);

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
