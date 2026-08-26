// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "flydsl/Dialect/Fly/Transforms/Passes.h"

using namespace mlir;

namespace mlir {
namespace fly {
#define GEN_PASS_DEF_FLYFIXBITCASTWIDTHPASS
#include "flydsl/Dialect/Fly/Transforms/Passes.h.inc"
} // namespace fly
} // namespace mlir

namespace {

class FlyFixBitcastWidthPass
    : public mlir::fly::impl::FlyFixBitcastWidthPassBase<FlyFixBitcastWidthPass> {
public:
  using mlir::fly::impl::FlyFixBitcastWidthPassBase<
      FlyFixBitcastWidthPass>::FlyFixBitcastWidthPassBase;

  void runOnOperation() override {
    // This pass runs BEFORE the canonicalizer (between convert-fly-to-rocdl and
    // canonicalize). It inserts llvm.freeze on vector.extract_strided_slice results
    // that feed llvm.bitcast ops, preventing the canonicalizer from folding the
    // extract+bitcast chain back to the wider source value.
    //
    // Without this, the canonicalizer traces:
    //   buffer_load(i128) → bitcast(vector<8xbf16>) → extract[0:4](vector<4xbf16>)
    //     → bitcast(vector<4xi16>)
    // and folds it into:
    //   buffer_load(i128) → bitcast(vector<4xi16>)  [INVALID: 128 != 64 bits]
    auto moduleOp = getOperation();

    // Protect extract_strided_slice ops whose source traces back (through
    // bitcasts) to a wider integer type (e.g. i128 from buffer_load). Only
    // these would produce width-mismatched bitcasts after canonicalization.
    SmallVector<vector::ExtractStridedSliceOp> extractsToProtect;
    moduleOp->walk([&](vector::ExtractStridedSliceOp op) {
      auto srcVecTy = cast<VectorType>(op->getOperand(0).getType());
      auto dstVecTy = cast<VectorType>(op.getResult().getType());
      if (srcVecTy.getNumElements() <= dstVecTy.getNumElements())
        return;
      // Trace through bitcasts to find the ultimate source type.
      Value src = op->getOperand(0);
      while (auto bc = dyn_cast_or_null<LLVM::BitcastOp>(src.getDefiningOp()))
        src = bc.getArg();
      // Only protect if the ultimate source is a wider integer type.
      if (auto srcIntTy = dyn_cast<IntegerType>(src.getType())) {
        int64_t dstBits = dstVecTy.getNumElements() *
                          dstVecTy.getElementType().getIntOrFloatBitWidth();
        if (srcIntTy.getWidth() > dstBits)
          extractsToProtect.push_back(op);
      }
    });

    // Dummy to keep the toProtect interface (unused now).
    SmallVector<LLVM::BitcastOp> toProtect;

    // Also find direct width-mismatched bitcasts (not from extract chains)
    // that need to be rewritten with freeze to block canonicalization.
    SmallVector<LLVM::BitcastOp> directMismatch;
    moduleOp->walk([&](LLVM::BitcastOp op) {
      Type srcTy = op.getArg().getType();
      Type dstTy = op.getResult().getType();
      auto getBits = [](Type ty) -> int64_t {
        if (auto intTy = dyn_cast<IntegerType>(ty))
          return intTy.getWidth();
        if (auto vecTy = dyn_cast<VectorType>(ty))
          return vecTy.getNumElements() * vecTy.getElementType().getIntOrFloatBitWidth();
        return 0;
      };
      int64_t srcBits = getBits(srcTy);
      int64_t dstBits = getBits(dstTy);
      if (srcBits > 0 && dstBits > 0 && srcBits != dstBits)
        directMismatch.push_back(op);
    });

    // Replace unrealized_conversion_cast ops that bridge same-width integer
    // and float vector types (e.g. vector<8xi8> ↔ vector<8xf8E4M3FN>).
    // These arise when convert-gpu-to-rocdl inserts casts for fp8 types
    // that the reconcile-unrealized-casts pass cannot fold.
    SmallVector<UnrealizedConversionCastOp> fp8Casts;
    moduleOp->walk([&](UnrealizedConversionCastOp op) {
      if (op.getNumOperands() != 1 || op.getNumResults() != 1)
        return;
      Type srcTy = op.getOperand(0).getType();
      Type dstTy = op.getResult(0).getType();
      auto srcVec = dyn_cast<VectorType>(srcTy);
      auto dstVec = dyn_cast<VectorType>(dstTy);
      if (!srcVec || !dstVec)
        return;
      if (srcVec.getNumElements() != dstVec.getNumElements())
        return;
      unsigned srcBW = srcVec.getElementType().getIntOrFloatBitWidth();
      unsigned dstBW = dstVec.getElementType().getIntOrFloatBitWidth();
      if (srcBW == dstBW)
        fp8Casts.push_back(op);
    });
    for (auto op : fp8Casts) {
      OpBuilder b(op);
      Value bc = LLVM::BitcastOp::create(b, op.getLoc(),
                                          op.getResult(0).getType(),
                                          op.getOperand(0));
      op.getResult(0).replaceAllUsesWith(bc);
      op->erase();
    }

    if (extractsToProtect.empty() && directMismatch.empty())
      return;

    OpBuilder builder(moduleOp->getContext());

    // Insert freeze after each narrowing extract to block canonicalization
    // from folding the extract chain back to the wider source.
    for (auto extractOp : extractsToProtect) {
      builder.setInsertionPointAfter(extractOp);
      Value result = extractOp.getResult();
      Value frozen = LLVM::FreezeOp::create(builder, extractOp.getLoc(),
                                             result.getType(), result);
      result.replaceAllUsesExcept(frozen, frozen.getDefiningOp());
    }

    // For direct width-mismatched bitcasts (e.g. i128 → vector<4xbf16>),
    // insert freeze on the source to prevent canonicalization from
    // reintroducing them after other folds.
    for (LLVM::BitcastOp op : directMismatch) {
      // Skip if already protected by the extract chain fix above.
      if (op->getOperand(0).getDefiningOp() &&
          isa<LLVM::FreezeOp>(op->getOperand(0).getDefiningOp()))
        continue;

      builder.setInsertionPoint(op);
      Location loc = op.getLoc();
      Value src = op.getArg();
      Type dstTy = op.getResult().getType();

      auto getBits = [](Type ty) -> int64_t {
        if (auto intTy = dyn_cast<IntegerType>(ty))
          return intTy.getWidth();
        if (auto vecTy = dyn_cast<VectorType>(ty))
          return vecTy.getNumElements() * vecTy.getElementType().getIntOrFloatBitWidth();
        return 0;
      };
      int64_t srcBits = getBits(src.getType());
      int64_t dstBits = getBits(dstTy);

      // Bitcast to integer, truncate, then bitcast to destination.
      Type srcIntTy = IntegerType::get(builder.getContext(), srcBits);
      Type dstIntTy = IntegerType::get(builder.getContext(), dstBits);

      Value intVal = src;
      if (src.getType() != srcIntTy)
        intVal = LLVM::BitcastOp::create(builder, loc, srcIntTy, src);
      Value truncated = LLVM::TruncOp::create(builder, loc, dstIntTy, intVal);
      Value result = LLVM::BitcastOp::create(builder, loc, dstTy, truncated);

      op.getResult().replaceAllUsesWith(result);
      op->erase();
    }
  }
};

} // namespace
