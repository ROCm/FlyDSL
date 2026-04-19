// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/PointerUtils.h"
#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"
#include "flydsl/Dialect/FlyROCDL/IR/Dialect.h"

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_rocdl {

bool CopyOpCDNA4LdsReadTransposeType::isStatic() const { return true; }

Value CopyOpCDNA4LdsReadTransposeType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                                          Value currentValue) const {
  if (currentValue && isa<MakeCopyAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  return MakeCopyAtomOp::create(builder, loc, CopyAtomType::get(*this, getBitSize()), getBitSize());
}

Attribute CopyOpCDNA4LdsReadTransposeType::getThrLayout() const {
  return FxLayout(FxC(16), FxC(1));
}

Attribute CopyOpCDNA4LdsReadTransposeType::getThrBitLayoutSrc() const {
  int32_t bitSize = getBitSize();
  int32_t transGranularity = getTransGranularity();
  if (bitSize == 64 && transGranularity == 4) {
    return FxLayout(FxShape(FxC(16), FxC(64)), FxStride(FxC(64), FxC(1)));
  } else if (bitSize == 64 && transGranularity == 8) {
    return FxLayout(FxShape(FxC(16), FxC(64)), FxStride(FxC(64), FxC(1)));
  } else if (bitSize == 96 && transGranularity == 6) {
    return FxLayout(FxShape(FxC(16), FxC(96)), FxStride(FxC(96), FxC(1)));
  } else if (bitSize == 64 && transGranularity == 16) {
    return FxLayout(FxShape(FxC(16), FxC(64)), FxStride(FxC(64), FxC(1)));
  } else {
    llvm_unreachable("Invalid (bitSize, transGranularity) for LDS read transpose");
  }
}

Attribute CopyOpCDNA4LdsReadTransposeType::getThrBitLayoutDst() const {
  int32_t bitSize = getBitSize();
  int32_t transGranularity = getTransGranularity();
  if (bitSize == 64 && transGranularity == 4) {
    return FxLayout(FxShape(FxC(16), FxVal(4, 16)), FxStride(FxC(4), FxVal(1, 64)));
  } else if (bitSize == 64 && transGranularity == 8) {
    return FxLayout(FxShape(FxC(16), FxVal(8, 8)), FxStride(FxC(8), FxVal(1, 128)));
  } else if (bitSize == 96 && transGranularity == 6) {
    return FxLayout(FxShape(FxC(16), FxVal(6, 16)), FxStride(FxC(6), FxVal(1, 96)));
  } else if (bitSize == 64 && transGranularity == 16) {
    return FxLayout(FxShape(FxC(16), FxVal(16, 4)), FxStride(FxC(16), FxVal(1, 256)));
  } else {
    llvm_unreachable("Invalid (bitSize, transGranularity) for LDS read transpose");
  }
}

Attribute CopyOpCDNA4LdsReadTransposeType::getThrBitLayoutRef() const {
  return getThrBitLayoutDst();
}

FailureOr<Value> CopyOpCDNA4LdsReadTransposeType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                                  Type resultTy, Type copyAtomTyArg,
                                                                  Type srcTyArg, Type dstTyArg,
                                                                  Value atomVal, Value src,
                                                                  Value dst) const {
  int32_t bitSize = getBitSize();
  int32_t transGranularity = getTransGranularity();

  Value loaded;
  if (bitSize == 64 && transGranularity == 4) {
    auto intrTy = VectorType::get({2}, builder.getI32Type());
    loaded = ROCDL::ds_read_tr4_b64::create(builder, loc, intrTy, src);
  } else if (bitSize == 64 && transGranularity == 8) {
    auto intrTy = VectorType::get({2}, builder.getI32Type());
    loaded = ROCDL::ds_read_tr8_b64::create(builder, loc, intrTy, src);
  } else if (bitSize == 96 && transGranularity == 6) {
    auto intrTy = VectorType::get({3}, builder.getI32Type());
    loaded = ROCDL::ds_read_tr6_b96::create(builder, loc, intrTy, src);
  } else if (bitSize == 64 && transGranularity == 16) {
    auto intrTy = VectorType::get({4}, builder.getI16Type());
    loaded = ROCDL::ds_read_tr16_b64::create(builder, loc, intrTy, src);
  } else {
    return failure();
  }

  if (resultTy && loaded.getType() != resultTy)
    loaded = LLVM::BitcastOp::create(builder, loc, resultTy, loaded);

  return loaded;
}

FailureOr<Value> CopyOpCDNA4LdsReadTransposeType::emitAtomCallSSA(
    OpBuilder &builder, Location loc, Type resultTy, Type copyAtomTyArg, Type srcTyArg,
    Type dstTyArg, Type predTyArg, Value atomVal, Value src, Value dst, Value pred) const {
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

LogicalResult CopyOpCDNA4LdsReadTransposeType::emitAtomCall(OpBuilder &builder, Location loc,
                                                            Type copyAtomTyArg, Type srcMemTyArg,
                                                            Type dstMemTyArg, Value atomVal,
                                                            Value src, Value dst) const {
  auto dstSSATy = fly::RegMem2SSAType(cast<fly::MemRefType>(dstMemTyArg));
  auto res = emitAtomCallSSA(builder, loc, dstSSATy, copyAtomTyArg, srcMemTyArg, Type{}, atomVal,
                             src, Value{});
  if (failed(res))
    return failure();
  LLVM::StoreOp::create(builder, loc, *res, dst);
  return success();
}

LogicalResult CopyOpCDNA4LdsReadTransposeType::emitAtomCall(OpBuilder &builder, Location loc,
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

LogicalResult CopyOpCDNA4LdsReadTransposeType::verify(function_ref<InFlightDiagnostic()> emitError,
                                                      int32_t transGranularity, int32_t bitSize) {
  bool valid =
      (bitSize == 64 && transGranularity == 4) || (bitSize == 64 && transGranularity == 8) ||
      (bitSize == 96 && transGranularity == 6) || (bitSize == 64 && transGranularity == 16);
  if (!valid)
    return emitError() << "unsupported (bitSize, transGranularity) = (" << bitSize << ", "
                       << transGranularity << ") for LDS read transpose";
  return success();
}

} // namespace mlir::fly_rocdl
