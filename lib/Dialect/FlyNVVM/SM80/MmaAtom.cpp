// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// ThrVal layouts + lowering for PTX Multiply-and-Accumulate Instruction: mma
//   mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
// (SM80 Ampere 16x8x16 f16->f32 tensor-core instruction). Operand fragment
// ABI (per NVIDIA PTX ISA):
//   A: 4 x vector<2xf16>   B: 2 x vector<2xf16>   C/D: 4 x f32
//
// Layouts are derived from the PTX operand ABI and FlyDSL's column-major atom
// coordinate convention, then cross-checked against independent SM80 GEMM
// references. The thread axis decomposes colexicographically as lane = T + 4*G,
// so the first thread sub-mode is T = lane & 3 (size 4) and the second is
// G = lane >> 2 (size 8).
//   T = lane &  3 in [0,  4)   threadID_in_group
//   G = lane >> 2 in [0,  8)   groupID

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"
#include "flydsl/Dialect/FlyNVVM/IR/Dialect.h"

using namespace mlir;
using namespace mlir::fly;

namespace sm80 {

static LayoutAttr getThrValLayoutA(MLIRContext *ctx) {
  auto getContext = [&]() { return ctx; };
  return FxLayout(FxShape(FxThr(4, 8), FxVal(2, 2, 2)), FxStride(FxThr(32, 1), FxVal(16, 8, 128)));
}

static LayoutAttr getThrValLayoutB(MLIRContext *ctx) {
  auto getContext = [&]() { return ctx; };
  return FxLayout(FxShape(FxThr(4, 8), FxVal(2, 2)), FxStride(FxThr(16, 1), FxVal(8, 64)));
}

static LayoutAttr getThrValLayoutC(MLIRContext *ctx) {
  auto getContext = [&]() { return ctx; };
  return FxLayout(FxShape(FxThr(4, 8), FxVal(2, 2)), FxStride(FxThr(32, 1), FxVal(16, 8)));
}

} // namespace sm80

namespace mlir::fly_nvvm {

bool MmaOpSM80_MmaSyncType::isStatic() const { return true; }

Value MmaOpSM80_MmaSyncType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                                Value currentValue) const {
  if (currentValue && isa<MakeMmaAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  return MakeMmaAtomOp::create(builder, loc, MmaAtomType::get(*this));
}

Attribute MmaOpSM80_MmaSyncType::getThrLayout() const { return FxLayout(FxC(32), FxC(1)); }

Attribute MmaOpSM80_MmaSyncType::getShapeMNK() const {
  return IntTupleAttr::get(ArrayAttr::get(getContext(), {FxC(getM()), FxC(getN()), FxC(getK())}));
}

Type MmaOpSM80_MmaSyncType::getValTypeA() const { return getElemTyA(); }
Type MmaOpSM80_MmaSyncType::getValTypeB() const { return getElemTyB(); }
Type MmaOpSM80_MmaSyncType::getValTypeC() const { return getElemTyAcc(); }
Type MmaOpSM80_MmaSyncType::getValTypeD() const { return getElemTyAcc(); }

Attribute MmaOpSM80_MmaSyncType::getThrValLayoutA() const {
  return sm80::getThrValLayoutA(getContext());
}
Attribute MmaOpSM80_MmaSyncType::getThrValLayoutB() const {
  return sm80::getThrValLayoutB(getContext());
}
Attribute MmaOpSM80_MmaSyncType::getThrValLayoutC() const {
  return sm80::getThrValLayoutC(getContext());
}

LogicalResult MmaOpSM80_MmaSyncType::verify(function_ref<InFlightDiagnostic()> emitError, int32_t m,
                                            int32_t n, int32_t k, Type elemTyA, Type elemTyB,
                                            Type elemTyAcc) {
  if (m != 16 || n != 8 || k != 16)
    return emitError() << "unsupported SM80 mma.sync.aligned shape " << m << "x" << n << "x" << k
                       << ", only 16x8x16 is supported";
  if (!elemTyA.isF16() || !elemTyB.isF16())
    return emitError() << "SM80 mma.sync.aligned requires f16 inputs, got " << elemTyA << ", "
                       << elemTyB;
  if (!elemTyAcc.isF32())
    return emitError() << "SM80 mma.sync.aligned requires f32 accumulator, got " << elemTyAcc;
  return success();
}

// SSA form: operands arrive as packed register vectors
//   a: vector<8xf16>  b: vector<4xf16>  c: vector<4xf32>
// and we return the result as vector<4xf32>.
FailureOr<Value> MmaOpSM80_MmaSyncType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                        Type resultTy, Type mmaAtomTyArg,
                                                        Type dTyArg, Type aTyArg, Type bTyArg,
                                                        Type cTyArg, Value atomVal, Value d,
                                                        Value a, Value b, Value c) const {
  MLIRContext *ctx = builder.getContext();
  Type f16Ty = Float16Type::get(ctx);
  Type f32Ty = Float32Type::get(ctx);
  auto f16x2Ty = VectorType::get({2}, f16Ty);
  auto aPackTy = VectorType::get({8}, f16Ty);
  auto bPackTy = VectorType::get({4}, f16Ty);
  auto cPackTy = VectorType::get({4}, f32Ty);

  if (a.getType() != aPackTy)
    a = LLVM::BitcastOp::create(builder, loc, aPackTy, a);
  if (b.getType() != bPackTy)
    b = LLVM::BitcastOp::create(builder, loc, bPackTy, b);
  if (c.getType() != cPackTy)
    c = LLVM::BitcastOp::create(builder, loc, cPackTy, c);

  // A: vector<8xf16> -> 4 x vector<2xf16>
  SmallVector<Value> matA;
  for (int i = 0; i < 4; ++i)
    matA.push_back(LLVM::ShuffleVectorOp::create(builder, loc, f16x2Ty, a, a,
                                                 ArrayRef<int32_t>{2 * i, 2 * i + 1}));
  // B: vector<4xf16> -> 2 x vector<2xf16>
  SmallVector<Value> matB;
  for (int i = 0; i < 2; ++i)
    matB.push_back(LLVM::ShuffleVectorOp::create(builder, loc, f16x2Ty, b, b,
                                                 ArrayRef<int32_t>{2 * i, 2 * i + 1}));
  // C: vector<4xf32> -> 4 x f32
  SmallVector<Value> matC;
  for (int i = 0; i < 4; ++i) {
    Value idx = arith::ConstantIntOp::create(builder, loc, i, 32);
    matC.push_back(LLVM::ExtractElementOp::create(builder, loc, c, idx));
  }

  // Result struct of the intrinsic: !llvm.struct<(f32, f32, f32, f32)>.
  auto resStructTy = LLVM::LLVMStructType::getLiteral(ctx, {f32Ty, f32Ty, f32Ty, f32Ty});

  Value mma = NVVM::MmaOp::create(
      builder, loc, resStructTy, matA, matB, matC,
      /*shape=*/ArrayRef<int64_t>{16, 8, 16},
      /*b1Op=*/std::nullopt,
      /*intOverflow=*/std::nullopt,
      /*multiplicandPtxTypes=*/
      std::array<NVVM::MMATypes, 2>{NVVM::MMATypes::f16, NVVM::MMATypes::f16},
      /*multiplicandLayouts=*/
      std::array<NVVM::MMALayout, 2>{NVVM::MMALayout::row, NVVM::MMALayout::col});

  // Repack the 4 scalar f32 results into vector<4xf32>.
  Value res = LLVM::PoisonOp::create(builder, loc, cPackTy);
  for (int i = 0; i < 4; ++i) {
    Value el = LLVM::ExtractValueOp::create(builder, loc, mma, ArrayRef<int64_t>{i});
    Value idx = arith::ConstantIntOp::create(builder, loc, i, 32);
    res = LLVM::InsertElementOp::create(builder, loc, cPackTy, res, el, idx);
  }
  // The accumulator fragment is always 4 x f32 for this instruction; bitcast if
  // the caller asked for an equally sized but differently spelled type.
  if (resultTy && res.getType() != resultTy)
    res = LLVM::BitcastOp::create(builder, loc, resultTy, res);
  return res;
}

LogicalResult MmaOpSM80_MmaSyncType::emitAtomCall(OpBuilder &builder, Location loc, Type mmaAtomTy,
                                                  Type dMemTy, Type aMemTy, Type bMemTy,
                                                  Type cMemTy, Value atomVal, Value dPtr,
                                                  Value aPtr, Value bPtr, Value cPtr) const {
  MLIRContext *ctx = builder.getContext();
  Type f16Ty = Float16Type::get(ctx);
  Type f32Ty = Float32Type::get(ctx);
  auto aPackTy = VectorType::get({8}, f16Ty);
  auto bPackTy = VectorType::get({4}, f16Ty);
  auto cPackTy = VectorType::get({4}, f32Ty);

  Value a = LLVM::LoadOp::create(builder, loc, aPackTy, aPtr);
  Value b = LLVM::LoadOp::create(builder, loc, bPackTy, bPtr);
  Value c = LLVM::LoadOp::create(builder, loc, cPackTy, cPtr);
  auto res = emitAtomCallSSA(builder, loc, cPackTy, mmaAtomTy, Type{}, aPackTy, bPackTy, cPackTy,
                             atomVal, Value{}, a, b, c);
  if (failed(res))
    return failure();
  LLVM::StoreOp::create(builder, loc, *res, dPtr);
  return success();
}

} // namespace mlir::fly_nvvm
