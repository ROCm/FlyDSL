// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Transforms/Passes.h"
#include "flydsl/Dialect/Fly/Utils/LayoutUtils.h"
#include "flydsl/Dialect/Fly/Utils/PointerUtils.h"

using namespace mlir;
using namespace mlir::fly;

namespace mlir {
namespace fly {
#define GEN_PASS_DEF_FLYCONVERTATOMCALLTOSSAFORMPASS
#include "flydsl/Dialect/Fly/Transforms/Passes.h.inc"
} // namespace fly
} // namespace mlir

namespace {

bool isEligibleToPromote(fly::MemRefType memRefTy) {
  if (memRefTy.getAddressSpace().getValue() != AddressSpace::Register)
    return false;
  auto layoutAttr = dyn_cast<LayoutAttr>(memRefTy.getLayout());
  if (!layoutAttr)
    return false;
  LayoutBuilder<LayoutAttr> builder(memRefTy.getContext());
  auto coalesced = layoutCoalesce(builder, layoutAttr);
  if (!coalesced.isLeaf())
    return false;
  return coalesced.getStride().isLeafStaticValue(1) || coalesced.getShape().isLeafStaticValue(1);
}

class FlyConvertAtomCallToSSAFormPass
    : public mlir::fly::impl::FlyConvertAtomCallToSSAFormPassBase<FlyConvertAtomCallToSSAFormPass> {
public:
  using mlir::fly::impl::FlyConvertAtomCallToSSAFormPassBase<
      FlyConvertAtomCallToSSAFormPass>::FlyConvertAtomCallToSSAFormPassBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();

    SmallVector<CopyAtomCall> copyOpsToConvert;
    SmallVector<MmaAtomCall> mmaOpsToConvert;

    moduleOp->walk([&](CopyAtomCall op) {
      auto srcTy = cast<fly::MemRefType>(op.getSrc().getType());
      auto dstTy = cast<fly::MemRefType>(op.getDst().getType());
      if (isEligibleToPromote(srcTy) || isEligibleToPromote(dstTy))
        copyOpsToConvert.push_back(op);
    });

    moduleOp->walk([&](MmaAtomCall op) {
      auto dTy = cast<fly::MemRefType>(op.getD().getType());
      auto aTy = cast<fly::MemRefType>(op.getA().getType());
      auto bTy = cast<fly::MemRefType>(op.getB().getType());
      auto cTy = cast<fly::MemRefType>(op.getC().getType());
      if (isEligibleToPromote(dTy) || isEligibleToPromote(aTy) || isEligibleToPromote(bTy) ||
          isEligibleToPromote(cTy))
        mmaOpsToConvert.push_back(op);
    });

    OpBuilder builder(moduleOp->getContext());

    for (CopyAtomCall copyOp : copyOpsToConvert) {
      auto srcTy = cast<fly::MemRefType>(copyOp.getSrc().getType());
      auto dstTy = cast<fly::MemRefType>(copyOp.getDst().getType());
      bool srcEligible = isEligibleToPromote(srcTy);
      bool dstEligible = isEligibleToPromote(dstTy);

      builder.setInsertionPoint(copyOp);
      Location loc = copyOp.getLoc();

      Value srcVal = copyOp.getSrc();
      if (srcEligible) {
        Value srcIter = srcVal.getDefiningOp<MakeViewOp>().getIter();
        srcVal = PtrLoadOp::create(builder, loc, RegMem2SSAType(srcTy), srcIter);
      }

      Value pred = copyOp.getPred();
      if (pred) {
        auto predMemTy = cast<fly::MemRefType>(pred.getType());
        if (isEligibleToPromote(predMemTy)) {
          Value predIter = pred.getDefiningOp<MakeViewOp>().getIter();
          pred = PtrLoadOp::create(builder, loc, RegMem2SSAType(predMemTy), predIter);
        }
      }

      if (dstEligible) {
        auto ssaTy = RegMem2SSAType(dstTy);
        Value dstIter = copyOp.getDst().getDefiningOp<MakeViewOp>().getIter();
        Value oldDst = pred ? PtrLoadOp::create(builder, loc, ssaTy, dstIter) : Value{};
        auto ssaOp =
            CopyAtomCallSSA::create(builder, loc, TypeRange{ssaTy}, copyOp.getCopyAtom(), srcVal,
                                    /*dst=*/oldDst, pred);
        PtrStoreOp::create(builder, loc, ssaOp.getResult(0), dstIter);
      } else {
        CopyAtomCallSSA::create(builder, loc, TypeRange{}, copyOp.getCopyAtom(), srcVal,
                                /*dst=*/copyOp.getDst(), pred);
      }
      copyOp->erase();
    }

    for (MmaAtomCall mmaOp : mmaOpsToConvert) {
      auto dTy = cast<fly::MemRefType>(mmaOp.getD().getType());
      auto aTy = cast<fly::MemRefType>(mmaOp.getA().getType());
      auto bTy = cast<fly::MemRefType>(mmaOp.getB().getType());
      auto cTy = cast<fly::MemRefType>(mmaOp.getC().getType());
      bool dEligible = isEligibleToPromote(dTy);
      bool aEligible = isEligibleToPromote(aTy);
      bool bEligible = isEligibleToPromote(bTy);
      bool cEligible = isEligibleToPromote(cTy);

      builder.setInsertionPoint(mmaOp);
      Location loc = mmaOp.getLoc();

      Value aVal = mmaOp.getA();
      Value bVal = mmaOp.getB();
      Value cVal = mmaOp.getC();

      if (aEligible) {
        Value aIter = aVal.getDefiningOp<MakeViewOp>().getIter();
        aVal = PtrLoadOp::create(builder, loc, RegMem2SSAType(aTy), aIter).getResult();
      }
      if (bEligible) {
        Value bIter = bVal.getDefiningOp<MakeViewOp>().getIter();
        bVal = PtrLoadOp::create(builder, loc, RegMem2SSAType(bTy), bIter).getResult();
      }
      if (cEligible) {
        Value cIter = cVal.getDefiningOp<MakeViewOp>().getIter();
        cVal = PtrLoadOp::create(builder, loc, RegMem2SSAType(cTy), cIter).getResult();
      }

      if (dEligible) {
        auto ssaOp = MmaAtomCallSSA::create(builder, loc, TypeRange{RegMem2SSAType(dTy)},
                                            mmaOp.getMmaAtom(), /*d=*/nullptr, aVal, bVal, cVal);
        Value dIter = mmaOp.getD().getDefiningOp<MakeViewOp>().getIter();
        PtrStoreOp::create(builder, loc, ssaOp.getResult(0), dIter);
      } else {
        MmaAtomCallSSA::create(builder, loc, TypeRange{}, mmaOp.getMmaAtom(), /*d=*/mmaOp.getD(),
                               aVal, bVal, cVal);
      }
      mmaOp->erase();
    }
  }
};

} // namespace
