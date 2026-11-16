// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Transforms/Passes.h"
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

Value getIter(OpBuilder &builder, Location loc, Value tensor) {
  // TODO: is it possible that the tensor has a composed layout here?
  if (auto view = tensor.getDefiningOp<MakeViewOp>())
    return view.getIter();
  return GetIterOp::create(builder, loc, tensor);
}

// A copy operand may be a coordinate tensor rather than a memref (it names a
// position, not storage), and such an operand is never register-promotable.
bool isEligibleToPromote(Type ty) {
  auto memRefTy = dyn_cast<fly::MemRefType>(ty);
  if (!memRefTy)
    return false;
  if (!isGenericAddressSpace<AddressSpace::Register>(memRefTy.getAddressSpace()))
    return false;
  auto layoutAttr = dyn_cast<LayoutAttr>(memRefTy.getLayout());
  if (!layoutAttr)
    return false;
  auto coalesced = core::coalesce(layoutAttr.getRef());
  if (coalesced.isError())
    return false;
  if (!coalesced.shape().isLeaf())
    return false;
  return coalesced.stride().isOne() || coalesced.shape().isOne();
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
      // A register-space pred needs promoting on its own account: a global ->
      // shared copy has neither operand in registers, but leaving the pred
      // behind strands a register pointer that the later rmem-to-vector-SSA
      // pass cannot rewrite.
      bool predEligible = op.getPred() && isEligibleToPromote(op.getPred().getType());
      if (isEligibleToPromote(op.getSrc().getType()) ||
          isEligibleToPromote(op.getDst().getType()) || predEligible)
        copyOpsToConvert.push_back(op);
    });

    moduleOp->walk([&](MmaAtomCall op) {
      auto dTy = cast<fly::MemRefType>(op.getD().getType());
      auto cTy = cast<fly::MemRefType>(op.getC().getType());
      auto hasEligibleOperand = [](ValueRange operands) {
        return llvm::any_of(operands, [](Value operand) {
          return isEligibleToPromote(cast<fly::MemRefType>(operand.getType()));
        });
      };
      if (isEligibleToPromote(dTy) || hasEligibleOperand(op.getA()) ||
          hasEligibleOperand(op.getB()) || isEligibleToPromote(cTy))
        mmaOpsToConvert.push_back(op);
    });

    OpBuilder builder(moduleOp->getContext());

    for (CopyAtomCall copyOp : copyOpsToConvert) {
      auto srcTy = dyn_cast<fly::MemRefType>(copyOp.getSrc().getType());
      auto dstTy = dyn_cast<fly::MemRefType>(copyOp.getDst().getType());
      bool srcEligible = isEligibleToPromote(copyOp.getSrc().getType());
      bool dstEligible = isEligibleToPromote(copyOp.getDst().getType());

      builder.setInsertionPoint(copyOp);
      Location loc = copyOp.getLoc();

      Value srcVal = copyOp.getSrc();
      if (srcEligible) {
        Value srcIter = getIter(builder, loc, srcVal);
        srcVal = PtrLoadOp::create(builder, loc, RegMem2SSAType(srcTy, true), srcIter);
      }

      Value pred = copyOp.getPred();
      if (pred) {
        auto predMemTy = cast<fly::MemRefType>(pred.getType());
        if (isEligibleToPromote(predMemTy)) {
          Value predIter = getIter(builder, loc, pred);
          pred = PtrLoadOp::create(builder, loc, RegMem2SSAType(predMemTy, true), predIter);
        }
      }

      if (dstEligible) {
        auto ssaTy = RegMem2SSAType(dstTy, true);
        Value dstIter = getIter(builder, loc, copyOp.getDst());
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
      auto cTy = cast<fly::MemRefType>(mmaOp.getC().getType());
      bool dEligible = isEligibleToPromote(dTy);
      bool cEligible = isEligibleToPromote(cTy);

      builder.setInsertionPoint(mmaOp);
      Location loc = mmaOp.getLoc();

      auto promoteGroup = [&](ValueRange operands) {
        SmallVector<Value> values;
        for (Value operand : operands) {
          auto type = cast<fly::MemRefType>(operand.getType());
          if (isEligibleToPromote(type)) {
            Value iter = getIter(builder, loc, operand);
            operand = PtrLoadOp::create(builder, loc, RegMem2SSAType(type, true), iter);
          }
          values.push_back(operand);
        }
        return values;
      };
      auto aVal = promoteGroup(mmaOp.getA());
      auto bVal = promoteGroup(mmaOp.getB());
      Value cVal = mmaOp.getC();

      if (cEligible) {
        Value cIter = getIter(builder, loc, cVal);
        cVal = PtrLoadOp::create(builder, loc, RegMem2SSAType(cTy, true), cIter).getResult();
      }

      if (dEligible) {
        auto ssaOp = MmaAtomCallSSA::create(builder, loc, TypeRange{RegMem2SSAType(dTy, true)},
                                            mmaOp.getMmaAtom(), /*d=*/nullptr, aVal, bVal, cVal);
        Value dIter = getIter(builder, loc, mmaOp.getD());
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
