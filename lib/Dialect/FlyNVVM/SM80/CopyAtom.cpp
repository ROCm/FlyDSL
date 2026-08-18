// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// NVVM copy atoms used by the SM80 GEMM path:
//   * CopyOpSM80_CpAsync   — cp.async.{ca,cg}.shared.global (global -> shared, async)
//   * CopyOpSM75_LdMatrix  — ldmatrix.sync.aligned (shared -> register, sm_75+)
//
// Thread/bit layouts are derived from the PTX cp.async / ldmatrix operand ABI
// and FlyDSL copy-atom layout invariants, then cross-checked against independent
// SM80 GEMM references.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/PointerUtils.h"
#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"
#include "flydsl/Dialect/FlyNVVM/IR/Dialect.h"

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_nvvm {

//===----------------------------------------------------------------------===//
// CopyOpSM80_CpAsync — cp.async.shared.global
//
// CuTe Copy_Traits<SM80_CP_ASYNC_CACHEGLOBAL<S,D>>: one thread, per-thread bit
// layout (1, bits). dst is shared (addrspace 3), src is global (addrspace 1).
//===----------------------------------------------------------------------===//

bool CopyOpSM80_CpAsyncType::isStatic() const { return true; }

Value CopyOpSM80_CpAsyncType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                                 Value currentValue) const {
  if (currentValue && isa<MakeCopyAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  return MakeCopyAtomOp::create(builder, loc, CopyAtomType::get(*this, getBitSize()), getBitSize());
}

Attribute CopyOpSM80_CpAsyncType::getThrLayout() const { return FxLayout(FxC(1), FxC(1)); }

Attribute CopyOpSM80_CpAsyncType::getThrBitLayoutSrc() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpSM80_CpAsyncType::getThrBitLayoutDst() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpSM80_CpAsyncType::getThrBitLayoutRef() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}

LogicalResult CopyOpSM80_CpAsyncType::verify(function_ref<InFlightDiagnostic()> emitError,
                                             int32_t bitSize) {
  if (bitSize != 32 && bitSize != 64 && bitSize != 128)
    return emitError() << "cp.async bitSize must be 32/64/128, got " << bitSize;
  return success();
}

// cp.async has no SSA result (it is a void async DMA). Use the memref path.
FailureOr<Value> CopyOpSM80_CpAsyncType::emitAtomCallSSA(OpBuilder &, Location, Type, Type, Type,
                                                         Type, Value, Value, Value) const {
  return failure();
}

FailureOr<Value> CopyOpSM80_CpAsyncType::emitAtomCallSSA(OpBuilder &, Location, Type, Type, Type,
                                                         Type, Type, Value, Value, Value,
                                                         Value) const {
  return failure();
}

LogicalResult CopyOpSM80_CpAsyncType::emitAtomCall(OpBuilder &builder, Location loc,
                                                   Type copyAtomTy, Type srcMemTy, Type dstMemTy,
                                                   Value atomVal, Value src, Value dst) const {
  MLIRContext *ctx = builder.getContext();
  int32_t sizeBytes = getBitSize() / 8;

  // cp.async is shared(3) <- global(1); cast either side if it arrived generic.
  auto globalPtrTy = LLVM::LLVMPointerType::get(ctx, /*addrspace=*/1);
  auto sharedPtrTy = LLVM::LLVMPointerType::get(ctx, /*addrspace=*/3);
  Value srcCast = src;
  if (srcCast.getType() != globalPtrTy)
    srcCast = LLVM::AddrSpaceCastOp::create(builder, loc, globalPtrTy, srcCast);
  Value dstCast = dst;
  if (dstCast.getType() != sharedPtrTy)
    dstCast = LLVM::AddrSpaceCastOp::create(builder, loc, sharedPtrTy, dstCast);

  // 16B copies use CG (bypass L1); 4/8B must use CA.
  NVVM::LoadCacheModifierKind modifier =
      sizeBytes == 16 ? NVVM::LoadCacheModifierKind::CG : NVVM::LoadCacheModifierKind::CA;

  NVVM::CpAsyncOp::create(builder, loc, dstCast, srcCast, builder.getI32IntegerAttr(sizeBytes),
                          NVVM::LoadCacheModifierKindAttr::get(ctx, modifier),
                          /*cpSize=*/Value{});
  return success();
}

LogicalResult CopyOpSM80_CpAsyncType::emitAtomCall(OpBuilder &builder, Location loc,
                                                   Type copyAtomTy, Type srcMemTy, Type dstMemTy,
                                                   Type predMemTy, Value atomVal, Value src,
                                                   Value dst, Value pred) const {
  OpBuilder::InsertionGuard guard(builder);
  auto predMemRefTy = cast<fly::MemRefType>(predMemTy);
  Value predVal = LLVM::LoadOp::create(builder, loc, predMemRefTy.getElemTy(), pred);
  auto ifOp = scf::IfOp::create(builder, loc, TypeRange{}, predVal, /*withElse=*/false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  return emitAtomCall(builder, loc, copyAtomTy, srcMemTy, dstMemTy, atomVal, src, dst);
}

//===----------------------------------------------------------------------===//
// CopyOpSM75_LdMatrix — ldmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].shared.b16
//
// Bit layouts follow the PTX ldmatrix fragment mapping:
//   num=4 non-trans -> four 32-bit registers per lane
//   num=2 non-trans -> two 32-bit registers per lane
//   num=1 non-trans -> one 32-bit register per lane
//   num=4 trans     -> eight packed 16-bit values per lane
//   num=2 trans     -> four packed 16-bit values per lane
//   num=1 trans     -> two packed 16-bit values per lane
//===----------------------------------------------------------------------===//

bool CopyOpSM75_LdMatrixType::isStatic() const { return true; }

Value CopyOpSM75_LdMatrixType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                                  Value currentValue) const {
  if (currentValue && isa<MakeCopyAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  // val bits per the b16 element granularity; CopyAtom valBits is set by the
  // Python make_copy_atom wrapper from the element type, so reuse 16 here.
  return MakeCopyAtomOp::create(builder, loc, CopyAtomType::get(*this, 16), 16);
}

Attribute CopyOpSM75_LdMatrixType::getThrLayout() const { return FxLayout(FxC(32), FxC(1)); }

// Source: (src-thr, src-val) -> shared-memory bit. From CuTe SrcLayout.
Attribute CopyOpSM75_LdMatrixType::getThrBitLayoutSrc() const {
  int32_t num = getNum();
  if (!getTrans()) {
    // SM75_U32x{1,2,4}_LDSM_N
    if (num == 1) // ((8,4),128):((128,0),1)
      return FxLayout(FxShape(FxThr(8, 4), FxC(128)), FxStride(FxThr(128, 0), FxC(1)));
    if (num == 2) // ((16,2),128):((128,0),1)
      return FxLayout(FxShape(FxThr(16, 2), FxC(128)), FxStride(FxThr(128, 0), FxC(1)));
    // num == 4: (32,128):(128,1)
    return FxLayout(FxShape(FxC(32), FxC(128)), FxStride(FxC(128), FxC(1)));
  }
  // trans variants share the same Src shapes as the N variants.
  if (num == 1)
    return FxLayout(FxShape(FxThr(8, 4), FxC(128)), FxStride(FxThr(128, 0), FxC(1)));
  if (num == 2)
    return FxLayout(FxShape(FxThr(16, 2), FxC(128)), FxStride(FxThr(128, 0), FxC(1)));
  return FxLayout(FxShape(FxC(32), FxC(128)), FxStride(FxC(128), FxC(1)));
}

// Destination: (dst-thr, dst-val) -> register bit. From CuTe DstLayout.
Attribute CopyOpSM75_LdMatrixType::getThrBitLayoutDst() const {
  int32_t num = getNum();
  if (!getTrans()) {
    // SM75_U32x{1,2,4}_LDSM_N
    if (num == 1) // (32,32):(32,1)
      return FxLayout(FxShape(FxC(32), FxC(32)), FxStride(FxC(32), FxC(1)));
    if (num == 2) // (32,(32,2)):(32,(1,1024))
      return FxLayout(FxShape(FxC(32), FxVal(32, 2)), FxStride(FxC(32), FxVal(1, 1024)));
    // num == 4: (32,(32,4)):(32,(1,1024))
    return FxLayout(FxShape(FxC(32), FxVal(32, 4)), FxStride(FxC(32), FxVal(1, 1024)));
  }
  // SM75_U16x{2,4,8}_LDSM_T
  if (num == 1) // ((4,8),(16,2)):((256,16),(1,128))
    return FxLayout(FxShape(FxThr(4, 8), FxVal(16, 2)), FxStride(FxThr(256, 16), FxVal(1, 128)));
  if (num == 2) // ((4,8),(16,2,2)):((256,16),(1,128,1024))
    return FxLayout(FxShape(FxThr(4, 8), FxVal(16, 2, 2)),
                    FxStride(FxThr(256, 16), FxVal(1, 128, 1024)));
  // num == 4: ((4,8),(16,2,4)):((256,16),(1,128,1024))
  return FxLayout(FxShape(FxThr(4, 8), FxVal(16, 2, 4)),
                  FxStride(FxThr(256, 16), FxVal(1, 128, 1024)));
}

Attribute CopyOpSM75_LdMatrixType::getThrBitLayoutRef() const { return getThrBitLayoutDst(); }

LogicalResult CopyOpSM75_LdMatrixType::verify(function_ref<InFlightDiagnostic()> emitError,
                                              int32_t num, bool trans) {
  if (num != 1 && num != 2 && num != 4)
    return emitError() << "ldmatrix num must be 1/2/4, got " << num;
  return success();
}

FailureOr<Value> CopyOpSM75_LdMatrixType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                          Type resultTy, Type copyAtomTyArg,
                                                          Type srcTyArg, Type dstTyArg,
                                                          Value atomVal, Value src,
                                                          Value dst) const {
  MLIRContext *ctx = builder.getContext();
  int32_t num = getNum();
  Type i32Ty = builder.getI32Type();

  Type ldResTy =
      num > 1 ? cast<Type>(LLVM::LLVMStructType::getLiteral(ctx, SmallVector<Type>(num, i32Ty)))
              : i32Ty;

  auto shape = NVVM::LdStMatrixShapeAttr::get(ctx, /*m=*/8, /*n=*/8);
  Value loaded = NVVM::LdMatrixOp::create(builder, loc, ldResTy, src, /*num=*/num,
                                          getTrans() ? NVVM::MMALayout::col : NVVM::MMALayout::row,
                                          shape, NVVM::LdStMatrixEltType::B16);

  // Repack the num i32 registers into the result vector type.
  if (num == 1) {
    if (resultTy && loaded.getType() != resultTy)
      loaded = LLVM::BitcastOp::create(builder, loc, resultTy, loaded);
    return loaded;
  }

  // Build vector<num x i32> from the struct, then bitcast to resultTy.
  auto i32VecTy = VectorType::get({num}, i32Ty);
  Value vec = LLVM::PoisonOp::create(builder, loc, i32VecTy);
  for (int i = 0; i < num; ++i) {
    Value el = LLVM::ExtractValueOp::create(builder, loc, loaded, ArrayRef<int64_t>{i});
    Value idx = arith::ConstantIntOp::create(builder, loc, i, 32);
    vec = LLVM::InsertElementOp::create(builder, loc, i32VecTy, vec, el, idx);
  }
  Value res = vec;

  if (resultTy && res.getType() != resultTy)
    res = LLVM::BitcastOp::create(builder, loc, resultTy, res);
  return res;
}

FailureOr<Value> CopyOpSM75_LdMatrixType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                          Type resultTy, Type copyAtomTyArg,
                                                          Type srcTyArg, Type dstTyArg,
                                                          Type predTyArg, Value atomVal, Value src,
                                                          Value dst, Value pred) const {
  assert(resultTy && "resultTy must be SSA Type");
  OpBuilder::InsertionGuard guard(builder);
  auto ifOp = scf::IfOp::create(builder, loc, resultTy, pred, /*withElseRegion=*/true);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  auto result =
      emitAtomCallSSA(builder, loc, resultTy, copyAtomTyArg, srcTyArg, dstTyArg, atomVal, src, dst);
  if (failed(result))
    return failure();
  scf::YieldOp::create(builder, loc, *result);
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  scf::YieldOp::create(builder, loc, dst);
  return ifOp.getResult(0);
}

LogicalResult CopyOpSM75_LdMatrixType::emitAtomCall(OpBuilder &builder, Location loc,
                                                    Type copyAtomTyArg, Type srcMemTyArg,
                                                    Type dstMemTyArg, Value atomVal, Value src,
                                                    Value dst) const {
  auto dstSSATy = fly::RegMem2SSAType(cast<fly::MemRefType>(dstMemTyArg), true);
  auto res = emitAtomCallSSA(builder, loc, dstSSATy, copyAtomTyArg, srcMemTyArg, Type{}, atomVal,
                             src, Value{});
  if (failed(res))
    return failure();
  LLVM::StoreOp::create(builder, loc, *res, dst);
  return success();
}

LogicalResult CopyOpSM75_LdMatrixType::emitAtomCall(OpBuilder &builder, Location loc,
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

} // namespace mlir::fly_nvvm
