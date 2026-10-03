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
//   * Instruction shapes are the gfx11 ones (16x16x16 fp16/bf16/fp8/bf8/iu8/iu4),
//     and the ROCDL intrinsics take the plain 3-operand {a, b, c} form (plus
//     signA/signB/clamp attrs on the iu8/iu4 path) — unlike gfx1250, whose
//     16x16x32 / 16x16x64 ops carry mods/reuse operands and use K=64 for iu8.
//     AMD IU8 clamp: true saturates INT32 acc to the input element range
//     (signed [-128,127] / unsigned [0,255]); false keeps i32 wrap (default).
//     IU4 clamp likewise saturates to the i4 input range.
//   * The register ABI is the gfx1250 "v8" one — each lane holds K/2 = 8 A/B
//     elements instead of the gfx11 16 (which broadcast across lane halves),
//     so the fragment layouts come from the shared gfx1250 helpers. For iu8
//     that means vector<2xi32> A/B fragments (NOT gfx11's vector<4xi32>).
//     For iu4, gfx12 packs those 8 nibbles into a **scalar i32** A/B (NOT
//     gfx11's vector<2xi32>); see AMDGPUToROCDL wmma-gfx12.mlir / clang
//     builtins-amdgcn-gfx12-wmma-w32.cl.
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

static bool isFp8OrBf8(Type elemTy) { return isa<Float8E4M3FNType, Float8E5M2Type>(elemTy); }

LogicalResult MmaOpGFX120X_WMMAType::verify(function_ref<InFlightDiagnostic()> emitError, int32_t m,
                                            int32_t n, int32_t k, Type elemTyA, Type elemTyB,
                                            Type elemTyAcc, bool signA, bool signB, bool clamp) {
  if (m != 16 || n != 16 || k != 16) {
    return emitError() << "GFX120X WMMA requires M=N=K=16, got " << m << "x" << n << "x" << k;
  }

  // Floating-point: fp16/bf16/fp8/bf8 -> f32. Integer: iu8/iu4 -> i32. The
  // ROCDL opcode names match gfx11 (`wmma.i32.16x16x16.iu8` / `.iu4`) but the
  // RDNA4 register ABI is the gfx1250 "v8" one: 8 A/B elements per lane.
  //   iu8 -> vector<2xi32> (NOT gfx11 vector<4xi32>)
  //   iu4 -> scalar i32    (NOT gfx11 vector<2xi32>)
  // Accept any IntegerType width 8 or 4 regardless of signedness; signA/signB
  // on the intrinsic control how the packed bytes/nibbles are interpreted.
  const bool isFp8 = isFp8OrBf8(elemTyA) && isFp8OrBf8(elemTyB) && elemTyAcc.isF32();
  const bool isFp = (elemTyA.isF16() && elemTyB.isF16() && elemTyAcc.isF32()) ||
                    (elemTyA.isBF16() && elemTyB.isBF16() && elemTyAcc.isF32()) || isFp8;

  auto isInt = [](Type t, unsigned width) {
    auto it = dyn_cast<IntegerType>(t);
    return it && it.getWidth() == width;
  };
  const bool isI8 = isInt(elemTyA, 8) && isInt(elemTyB, 8) && elemTyAcc.isInteger(32);
  const bool isI4 = isInt(elemTyA, 4) && isInt(elemTyB, 4) && elemTyAcc.isInteger(32);

  if (!isFp && !isI8 && !isI4) {
    return emitError() << "unsupported GFX120X WMMA configuration: " << m << "x" << n << "x" << k
                       << " with A=" << elemTyA << ", B=" << elemTyB << ", Acc=" << elemTyAcc;
  }

  // Floating-point intrinsics have no sign/clamp operands; refuse to build an
  // atom promising something codegen cannot deliver. Integer (iu8/iu4) forwards
  // signA/signB/clamp to the ROCDL intrinsic, mirroring gfx11 / gfx1250.
  if (isFp && (signA || signB || clamp)) {
    return emitError() << "GFX120X WMMA floating-point path does not accept signA/signB/clamp "
                          "(the ROCDL fp WMMA intrinsics have no such operands); got signA="
                       << signA << ", signB=" << signB << ", clamp=" << clamp;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Codegen: lower the atom call to a rocdl.wmma.* intrinsic op.
//===----------------------------------------------------------------------===//

// A/B operand type on RDNA4 wave32 for 16x16x16: M*K/32 = 8 elements per lane.
//   fp16 -> vector<8xf16>
//   bf16 -> vector<8xi16>   (the bf16 WMMA intrinsic takes integer operands)
//   fp8/bf8/iu8 -> vector<2xi32>   (8 packed 8-bit values)
//   iu4 -> i32                 (8 packed 4-bit values; gfx12 scalar, not gfx11 v2i32)
static Type getWmmaABType(MLIRContext *ctx, Type elemTy) {
  if (elemTy.isInteger(4))
    return IntegerType::get(ctx, 32);
  if (isFp8OrBf8(elemTy) || elemTy.isInteger(8))
    return VectorType::get({2}, IntegerType::get(ctx, 32));
  if (elemTy.isBF16())
    return VectorType::get({8}, IntegerType::get(ctx, 16));
  if (elemTy.isF16())
    return VectorType::get({8}, elemTy);
  return nullptr;
}

// Accumulator/result vector type on RDNA4 wave32: 8 slots per lane (f32 or i32).
static Type getWmmaAccRawType(Type elemTyAcc) {
  if (elemTyAcc.isF32() || elemTyAcc.isInteger(32))
    return VectorType::get({8}, elemTyAcc);
  return nullptr;
}

// Build a `rocdl.wmma.*` intrinsic via OperationState. Needed for the IU
// variant, whose signA/signB/clamp attrs are not positional SSA operands.
static Value buildWmmaOp(OpBuilder &builder, Location loc, StringRef opName, Type resultTy,
                         ValueRange operands, ArrayRef<NamedAttribute> attrs) {
  OperationState state(loc, opName);
  state.addTypes(resultTy);
  state.addOperands(operands);
  state.addAttributes(attrs);
  Operation *op = builder.create(state);
  return op->getResult(0);
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
  if (isa<Float8E4M3FNType>(elemTyA) && isa<Float8E4M3FNType>(elemTyB))
    return ROCDL::wmma_f32_16x16x16_fp8_fp8::create(builder, loc, rawAccTy, a, b, c).getResult();
  if (isa<Float8E4M3FNType>(elemTyA) && isa<Float8E5M2Type>(elemTyB))
    return ROCDL::wmma_f32_16x16x16_fp8_bf8::create(builder, loc, rawAccTy, a, b, c).getResult();
  if (isa<Float8E5M2Type>(elemTyA) && isa<Float8E4M3FNType>(elemTyB))
    return ROCDL::wmma_f32_16x16x16_bf8_fp8::create(builder, loc, rawAccTy, a, b, c).getResult();
  if (isa<Float8E5M2Type>(elemTyA) && isa<Float8E5M2Type>(elemTyB))
    return ROCDL::wmma_f32_16x16x16_bf8_bf8::create(builder, loc, rawAccTy, a, b, c).getResult();

  // Integer iu8: same ROCDL op name as gfx11, but A/B are vector<2xi32> (RDNA4
  // v8 ABI) rather than gfx11's vector<4xi32>. signA/signB/clamp come from the
  // type parameters so callers control signedness / saturation.
  if (elemTyA.isInteger(8) && elemTyB.isInteger(8) && elemTyAcc.isInteger(32)) {
    SmallVector<NamedAttribute, 3> attrs;
    attrs.push_back({builder.getStringAttr("signA"), builder.getBoolAttr(getSignA())});
    attrs.push_back({builder.getStringAttr("signB"), builder.getBoolAttr(getSignB())});
    attrs.push_back({builder.getStringAttr("clamp"), builder.getBoolAttr(getClamp())});
    return buildWmmaOp(builder, loc, ROCDL::wmma_i32_16x16x16_iu8::getOperationName(), rawAccTy,
                       /*operands=*/{a, b, c}, attrs);
  }

  // Integer iu4: same ROCDL op name as gfx11, but A/B are scalar i32 (RDNA4 /
  // gfx12 packing of 8 nibbles per lane), not gfx11's vector<2xi32>.
  if (elemTyA.isInteger(4) && elemTyB.isInteger(4) && elemTyAcc.isInteger(32)) {
    SmallVector<NamedAttribute, 3> attrs;
    attrs.push_back({builder.getStringAttr("signA"), builder.getBoolAttr(getSignA())});
    attrs.push_back({builder.getStringAttr("signB"), builder.getBoolAttr(getSignB())});
    attrs.push_back({builder.getStringAttr("clamp"), builder.getBoolAttr(getClamp())});
    return buildWmmaOp(builder, loc, ROCDL::wmma_i32_16x16x16_iu4::getOperationName(), rawAccTy,
                       /*operands=*/{a, b, c}, attrs);
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
