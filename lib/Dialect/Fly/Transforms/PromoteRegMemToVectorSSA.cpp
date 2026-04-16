// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/CSE.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Transforms/Passes.h"
#include "flydsl/Dialect/Fly/Utils/LayoutUtils.h"

#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::fly;

namespace llvm {

template <> struct DenseMapInfo<mlir::fly::MakePtrOp> : DenseMapInfo<mlir::Operation *> {
  using Base = DenseMapInfo<mlir::Operation *>;

  static mlir::fly::MakePtrOp getEmptyKey() { return mlir::fly::MakePtrOp(Base::getEmptyKey()); }

  static mlir::fly::MakePtrOp getTombstoneKey() {
    return mlir::fly::MakePtrOp(Base::getTombstoneKey());
  }
};

} // namespace llvm

namespace mlir {
namespace fly {
#define GEN_PASS_DEF_FLYPROMOTEREGMEMTOVECTORSSAPASS
#include "flydsl/Dialect/Fly/Transforms/Passes.h.inc"
} // namespace fly
} // namespace mlir

namespace {

using VectorValue = TypedValue<VectorType>;
using RegPtrValue = TypedValue<PointerType>;
using RegMem2VectorSSAMap = DenseMap<MakePtrOp, VectorValue>;

struct RegAccessInfo {
  MakePtrOp makePtrOp;
  int32_t offset;
  int32_t width;
  Type elemTy;
};

struct RegAllocaInfo {
  MakePtrOp makePtrOp;
  int32_t allocaSize;
  Type elemTy;
  VectorType vectorSSATy;
};

bool isRegValue(Value value) {
  if (auto ptrTy = dyn_cast<PointerType>(value.getType()))
    return ptrTy.getAddressSpace().getValue() == AddressSpace::Register;
  if (auto memRefTy = dyn_cast<fly::MemRefType>(value.getType()))
    return memRefTy.getAddressSpace().getValue() == AddressSpace::Register;
  return false;
}

bool isRegOperandOrResult(Operation *op) {
  for (Value result : op->getOpResults()) {
    if (isRegValue(result))
      return true;
  }
  for (Value operand : op->getOperands()) {
    if (isRegValue(operand))
      return true;
  }
  return false;
}

std::optional<std::pair<MakePtrOp, int32_t>> resolveRegOffset(Value ptr) {
  assert(isa<RegPtrValue>(ptr) && "expected register pointer");

  if (auto makePtrOp = ptr.getDefiningOp<MakePtrOp>()) {
    return std::pair<MakePtrOp, int32_t>{makePtrOp, 0};
  } else if (auto addOffsetOp = ptr.getDefiningOp<AddOffsetOp>()) {
    auto base = resolveRegOffset(addOffsetOp.getPtr());
    if (!base)
      return std::nullopt;
    IntAttr intAttr = addOffsetOp.getOffset().getType().getAttr().getLeafAsInt();
    return std::pair<MakePtrOp, int32_t>{base->first, base->second + intAttr.getValue()};
  } else {
    return std::nullopt;
  }
}

class FlyPromoteRegMemToVectorSSAPass
    : public mlir::fly::impl::FlyPromoteRegMemToVectorSSAPassBase<FlyPromoteRegMemToVectorSSAPass> {
public:
  using mlir::fly::impl::FlyPromoteRegMemToVectorSSAPassBase<
      FlyPromoteRegMemToVectorSSAPass>::FlyPromoteRegMemToVectorSSAPassBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    moduleOp->walk([&](gpu::GPUFuncOp funcOp) {
      if (failed(processFunction(funcOp)))
        signalPassFailure();
    });

    auto context = moduleOp->getContext();
    IRRewriter rewriter(context);
    DominanceInfo domInfo(getOperation());
    eliminateCommonSubExpressions(rewriter, domInfo, getOperation());
  }

private:
  DenseMap<MakePtrOp, RegAllocaInfo> regAllocaInfos;
  SmallVector<MakePtrOp> allocaOrder;

  LogicalResult processFunction(gpu::GPUFuncOp funcOp) {
    OpBuilder opBuilder(funcOp.getContext());

    regAllocaInfos.clear();
    allocaOrder.clear();

    funcOp.walk([&](MakePtrOp makePtrOp) {
      if (!isRegValue(makePtrOp))
        return;
      auto allocaSizeAttr = makePtrOp.getDictAttrs()->getAs<IntegerAttr>("allocaSize");
      if (!allocaSizeAttr || allocaSizeAttr.getInt() <= 0)
        return;

      PointerType ptrTy = cast<PointerType>(makePtrOp.getType());
      regAllocaInfos.try_emplace(
          makePtrOp,
          RegAllocaInfo{makePtrOp, static_cast<int32_t>(allocaSizeAttr.getInt()), ptrTy.getElemTy(),
                        VectorType::get({allocaSizeAttr.getInt()}, ptrTy.getElemTy())});
      allocaOrder.push_back(makePtrOp);
    });

    funcOp.walk([&](RecastIterOp recastOp) {
      if (recastOp->use_empty())
        return;
      for (Value v : {recastOp.getSrc(), recastOp.getResult()}) {
        if (!isRegValue(v))
          continue;
        auto root = resolveRegOffset(v);
        if (root)
          regAllocaInfos.erase(root->first);
      }
    });
    bool hasExcludedRoots = allocaOrder.size() != regAllocaInfos.size();
    llvm::erase_if(allocaOrder, [&](MakePtrOp r) { return !regAllocaInfos.count(r); });

    if (allocaOrder.empty())
      return success();

    Block *oldEntry = &funcOp.getBody().front();
    Block *newEntry = opBuilder.createBlock(&funcOp.getBody());
    for (BlockArgument arg : oldEntry->getArguments())
      newEntry->addArgument(arg.getType(), funcOp.getLoc());

    IRMapping mapping;
    for (auto it : llvm::zip(oldEntry->getArguments(), newEntry->getArguments()))
      mapping.map(std::get<0>(it), std::get<1>(it));

    RegMem2VectorSSAMap state;
    if (failed(rewriteBlock(oldEntry, newEntry, mapping, state))) {
      newEntry->erase();
      return failure();
    }

    oldEntry->erase();
    cleanupDeadOps(funcOp);

    if (!hasExcludedRoots) {
      bool invalid = false;
      funcOp.walk([&](Operation *op) {
        if (invalid)
          return;
        if (isRegOperandOrResult(op)) {
          op->emitOpError("register operand/result remain after rmem SSA promotion");
          invalid = true;
          return;
        }
        for (Region &region : op->getRegions()) {
          for (Block &block : region) {
            for (BlockArgument arg : block.getArguments()) {
              if (isRegValue(arg)) {
                op->emitOpError("register block arguments remain after rmem SSA promotion");
                invalid = true;
                return;
              }
            }
          }
        }
      });
      if (invalid)
        return failure();
    }
    return success();
  }

  RegAllocaInfo *getAllocaInfo(MakePtrOp makePtrOp) {
    auto it = regAllocaInfos.find(makePtrOp);
    if (it == regAllocaInfos.end())
      return nullptr;
    return &it->second;
  }

  std::optional<RegAccessInfo> getRegAccessInfo(Value ptr, Type valueType) {
    auto rootAndOffset = resolveRegOffset(ptr);
    if (!rootAndOffset)
      return std::nullopt;

    int32_t width = 0;
    if (auto vecTy = dyn_cast<VectorType>(valueType))
      width = vecTy.getNumElements();
    else if (isa<IntegerType, FloatType>(valueType))
      width = 1;
    else
      return std::nullopt;

    const RegAllocaInfo *info = getAllocaInfo(rootAndOffset->first);
    if (!info)
      return std::nullopt;

    return RegAccessInfo{rootAndOffset->first, rootAndOffset->second, width, info->elemTy};
  }

  LogicalResult collectTouchedRegAllocaInRegion(Region &region, Operation *boundaryOp,
                                                DenseSet<MakePtrOp> &touchedRoots) {
    auto tryRecord = [&](Value ptr) {
      if (!isa<RegPtrValue>(ptr))
        return;
      auto root = resolveRegOffset(ptr);
      if (root && getAllocaInfo(root->first) && !boundaryOp->isProperAncestor(root->first))
        touchedRoots.insert(root->first);
    };
    for (Block &block : region) {
      for (Operation &op : block) {
        if (auto loadOp = dyn_cast<PtrLoadOp>(&op))
          tryRecord(loadOp.getPtr());
        else if (auto storeOp = dyn_cast<PtrStoreOp>(&op))
          tryRecord(storeOp.getPtr());

        for (Region &nested : op.getRegions()) {
          if (failed(collectTouchedRegAllocaInRegion(nested, boundaryOp, touchedRoots)))
            return failure();
        }
      }
    }
    return success();
  }

  LogicalResult collectTouchedRegAlloca(Operation *boundaryOp, SmallVectorImpl<MakePtrOp> &roots) {
    DenseSet<MakePtrOp> touchedRoots;
    for (Region &region : boundaryOp->getRegions()) {
      if (failed(collectTouchedRegAllocaInRegion(region, boundaryOp, touchedRoots)))
        return failure();
    }
    for (MakePtrOp makePtrOp : allocaOrder) {
      if (touchedRoots.contains(makePtrOp))
        roots.push_back(makePtrOp);
    }
    return success();
  }

  void appendVectorSSATypes(ArrayRef<MakePtrOp> roots, SmallVectorImpl<Type> &types) {
    for (MakePtrOp makePtrOp : roots) {
      const RegAllocaInfo *info = getAllocaInfo(makePtrOp);
      assert(info && "missing register makePtrOp info");
      types.push_back(info->vectorSSATy);
    }
  }

  void appendVectorSSAValues(RegMem2VectorSSAMap &state, ArrayRef<MakePtrOp> roots,
                             SmallVectorImpl<Value> &values) {
    for (MakePtrOp makePtrOp : roots) {
      auto it = state.find(makePtrOp);
      assert(it != state.end() && "missing state for register makePtrOp");
      values.push_back(it->second);
    }
  }

  LogicalResult rewritePtrStore(PtrStoreOp storeOp, OpBuilder &builder, IRMapping &mapping,
                                RegMem2VectorSSAMap &state) {
    auto access = getRegAccessInfo(storeOp.getPtr(), storeOp.getValue().getType());
    if (!access)
      return failure();

    auto stateIt = state.find(access->makePtrOp);
    assert(stateIt != state.end() && "missing state for register ptr.store");
    VectorValue currentVec = stateIt->second;
    VectorValue updatedVec;

    auto loc = storeOp.getLoc();
    Value storedValue = mapping.lookupOrDefault(storeOp.getValue());

    if (isa<IntegerType, FloatType>(storedValue.getType())) {
      assert(access->width == 1 && "expected scalar type with width 1");
      updatedVec = vector::InsertOp::create(builder, loc, storedValue, currentVec, access->offset);
    } else {
      assert(isa<VectorType>(storedValue.getType()) && "expected vector type");
      assert(cast<VectorType>(storedValue.getType()).getNumElements() == access->width &&
             "expected vector type with same width as access");

      updatedVec = vector::InsertStridedSliceOp::create(builder, loc, storedValue, currentVec,
                                                        ArrayRef<int64_t>{access->offset},
                                                        ArrayRef<int64_t>{1});
    }
    state[access->makePtrOp] = updatedVec;
    return success();
  }

  LogicalResult rewritePtrLoad(PtrLoadOp loadOp, OpBuilder &builder, IRMapping &mapping,
                               RegMem2VectorSSAMap &state) {
    auto access = getRegAccessInfo(loadOp.getPtr(), loadOp.getResult().getType());
    if (!access)
      return failure();

    auto stateIt = state.find(access->makePtrOp);
    assert(stateIt != state.end() && "missing state for register ptr.load");
    VectorValue currentVec = stateIt->second;

    auto loc = loadOp.getLoc();
    Type resultType = loadOp.getResult().getType();
    Value extracted;

    if (isa<IntegerType, FloatType>(resultType)) {
      assert(access->width == 1 && "expected scalar type with width 1");
      extracted = vector::ExtractOp::create(builder, loc, currentVec, access->offset);
    } else {
      assert(isa<VectorType>(resultType) && "expected vector type");
      assert(cast<VectorType>(resultType).getNumElements() == access->width &&
             "expected vector type with same width as access");

      extracted = vector::ExtractStridedSliceOp::create(
          builder, loc, currentVec, ArrayRef<int64_t>{access->offset},
          ArrayRef<int64_t>{access->width}, ArrayRef<int64_t>{1});
    }
    mapping.map(loadOp, extracted);
    return success();
  }

  LogicalResult rewriteIfOp(scf::IfOp oldIf, OpBuilder &builder, IRMapping &mapping,
                            RegMem2VectorSSAMap &state) {
    SmallVector<MakePtrOp> touchedRoots;
    if (failed(collectTouchedRegAlloca(oldIf, touchedRoots)))
      return failure();

    SmallVector<Type> newResultTypes(oldIf.getResultTypes());
    appendVectorSSATypes(touchedRoots, newResultTypes);

    bool hasElse = !oldIf.getElseRegion().empty();
    bool withElse = hasElse || !touchedRoots.empty();
    auto newIf = scf::IfOp::create(builder, oldIf.getLoc(), TypeRange(newResultTypes),
                                   mapping.lookupOrDefault(oldIf.getCondition()), withElse);

    {
      // process then block
      IRMapping thenMapping = mapping;
      RegMem2VectorSSAMap thenState = state;
      Block *oldThen = &oldIf.getThenRegion().front();
      Block *newThen = &newIf.getThenRegion().front();
      if (failed(rewriteBlock(oldThen, newThen, thenMapping, thenState)))
        return failure();

      auto oldYield = cast<scf::YieldOp>(oldThen->getTerminator());
      SmallVector<Value> newYieldOperands;
      for (Value yielded : oldYield.getOperands())
        newYieldOperands.push_back(thenMapping.lookupOrDefault(yielded));
      appendVectorSSAValues(thenState, touchedRoots, newYieldOperands);

      OpBuilder thenYieldBuilder = OpBuilder::atBlockEnd(newThen);
      scf::YieldOp::create(thenYieldBuilder, oldYield.getLoc(), newYieldOperands);
    }

    if (hasElse) {
      IRMapping elseMapping = mapping;
      RegMem2VectorSSAMap elseState = state;
      Block *oldElse = &oldIf.getElseRegion().front();
      Block *newElse = &newIf.getElseRegion().front();
      if (failed(rewriteBlock(oldElse, newElse, elseMapping, elseState)))
        return failure();

      auto oldYield = cast<scf::YieldOp>(oldElse->getTerminator());
      SmallVector<Value> newYieldOperands;
      for (Value yielded : oldYield.getOperands())
        newYieldOperands.push_back(elseMapping.lookupOrDefault(yielded));
      appendVectorSSAValues(elseState, touchedRoots, newYieldOperands);

      OpBuilder elseYieldBuilder = OpBuilder::atBlockEnd(newElse);
      scf::YieldOp::create(elseYieldBuilder, oldYield.getLoc(), newYieldOperands);
    } else if (withElse) {
      Block *newElse = &newIf.getElseRegion().front();
      SmallVector<Value> elseYieldOperands;
      appendVectorSSAValues(state, touchedRoots, elseYieldOperands);
      OpBuilder elseYieldBuilder = OpBuilder::atBlockEnd(newElse);
      scf::YieldOp::create(elseYieldBuilder, oldIf.getLoc(), elseYieldOperands);
    }

    for (unsigned i = 0; i < oldIf.getNumResults(); ++i)
      mapping.map(oldIf.getResult(i), newIf.getResult(i));
    for (auto it : llvm::enumerate(touchedRoots))
      state[it.value()] = cast<VectorValue>(newIf.getResult(oldIf.getNumResults() + it.index()));

    return success();
  }

  LogicalResult rewriteForOp(scf::ForOp oldFor, OpBuilder &builder, IRMapping &mapping,
                             RegMem2VectorSSAMap &state) {
    SmallVector<MakePtrOp> touchedRoots;
    if (failed(collectTouchedRegAlloca(oldFor, touchedRoots)))
      return failure();

    SmallVector<Value> newInitArgs;
    for (Value initArg : oldFor.getInitArgs())
      newInitArgs.push_back(mapping.lookupOrDefault(initArg));
    appendVectorSSAValues(state, touchedRoots, newInitArgs);

    auto newFor = scf::ForOp::create(
        builder, oldFor.getLoc(), mapping.lookupOrDefault(oldFor.getLowerBound()),
        mapping.lookupOrDefault(oldFor.getUpperBound()), mapping.lookupOrDefault(oldFor.getStep()),
        newInitArgs, [](OpBuilder &, Location, Value, ValueRange) {}, oldFor.getUnsignedCmp());

    IRMapping bodyMapping = mapping;
    Block *oldBody = oldFor.getBody();
    Block *newBody = newFor.getBody();
    for (unsigned i = 0; i < oldBody->getNumArguments(); ++i)
      bodyMapping.map(oldBody->getArgument(i), newBody->getArgument(i));

    RegMem2VectorSSAMap bodyState = state;
    unsigned carriedArgBase = 1 + oldFor.getRegionIterArgs().size();
    for (auto it : llvm::enumerate(touchedRoots))
      bodyState[it.value()] = cast<VectorValue>(newBody->getArgument(carriedArgBase + it.index()));

    if (failed(rewriteBlock(oldBody, newBody, bodyMapping, bodyState)))
      return failure();

    auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
    SmallVector<Value> newYieldOperands;
    for (Value yielded : oldYield.getOperands())
      newYieldOperands.push_back(bodyMapping.lookupOrDefault(yielded));
    for (MakePtrOp makePtrOp : touchedRoots)
      newYieldOperands.push_back(bodyState[makePtrOp]);

    OpBuilder yieldBuilder = OpBuilder::atBlockEnd(newBody);
    scf::YieldOp::create(yieldBuilder, oldYield.getLoc(), newYieldOperands);

    for (unsigned i = 0; i < oldFor.getNumResults(); ++i)
      mapping.map(oldFor.getResult(i), newFor.getResult(i));
    for (auto it : llvm::enumerate(touchedRoots))
      state[it.value()] = cast<VectorValue>(newFor.getResult(oldFor.getNumResults() + it.index()));

    return success();
  }

  LogicalResult rewriteWhileOp(scf::WhileOp oldWhile, OpBuilder &builder, IRMapping &mapping,
                               RegMem2VectorSSAMap &state) {
    SmallVector<MakePtrOp> touchedRoots;
    if (failed(collectTouchedRegAlloca(oldWhile, touchedRoots)))
      return failure();

    SmallVector<Value> newInitArgs;
    for (Value initArg : oldWhile.getInits())
      newInitArgs.push_back(mapping.lookupOrDefault(initArg));
    appendVectorSSAValues(state, touchedRoots, newInitArgs);

    SmallVector<Type> newResultTypes(oldWhile.getResultTypes().begin(),
                                     oldWhile.getResultTypes().end());
    appendVectorSSATypes(touchedRoots, newResultTypes);

    auto newWhile =
        scf::WhileOp::create(builder, oldWhile.getLoc(), TypeRange(newResultTypes), newInitArgs);

    SmallVector<Type> beforeArgTypes;
    beforeArgTypes.reserve(newInitArgs.size());
    for (Value initArg : newInitArgs)
      beforeArgTypes.push_back(initArg.getType());
    SmallVector<Location> beforeArgLocs(beforeArgTypes.size(), oldWhile.getLoc());
    Block *newBefore =
        builder.createBlock(&newWhile.getBefore(), {}, beforeArgTypes, beforeArgLocs);

    SmallVector<Type> afterArgTypes(oldWhile.getResultTypes().begin(),
                                    oldWhile.getResultTypes().end());
    appendVectorSSATypes(touchedRoots, afterArgTypes);
    SmallVector<Location> afterArgLocs(afterArgTypes.size(), oldWhile.getLoc());
    Block *newAfter = builder.createBlock(&newWhile.getAfter(), {}, afterArgTypes, afterArgLocs);

    {
      // process before block
      IRMapping beforeMapping = mapping;
      Block *oldBefore = oldWhile.getBeforeBody();
      for (unsigned i = 0; i < oldBefore->getNumArguments(); ++i)
        beforeMapping.map(oldBefore->getArgument(i), newBefore->getArgument(i));

      RegMem2VectorSSAMap beforeState = state;
      unsigned rootArgBase = oldBefore->getNumArguments();
      for (auto it : llvm::enumerate(touchedRoots))
        beforeState[it.value()] =
            cast<VectorValue>(newBefore->getArgument(rootArgBase + it.index()));

      if (failed(rewriteBlock(oldBefore, newBefore, beforeMapping, beforeState)))
        return failure();

      auto oldCondition = cast<scf::ConditionOp>(oldBefore->getTerminator());
      SmallVector<Value> newConditionArgs;
      for (Value arg : oldCondition.getArgs())
        newConditionArgs.push_back(beforeMapping.lookupOrDefault(arg));
      for (MakePtrOp makePtrOp : touchedRoots)
        newConditionArgs.push_back(beforeState[makePtrOp]);

      OpBuilder condBuilder = OpBuilder::atBlockEnd(newBefore);
      scf::ConditionOp::create(condBuilder, oldCondition.getLoc(),
                               beforeMapping.lookupOrDefault(oldCondition.getCondition()),
                               newConditionArgs);
    }

    {
      // process after block
      IRMapping afterMapping = mapping;
      Block *oldAfter = oldWhile.getAfterBody();
      for (unsigned i = 0; i < oldAfter->getNumArguments(); ++i)
        afterMapping.map(oldAfter->getArgument(i), newAfter->getArgument(i));

      RegMem2VectorSSAMap afterState = state;
      unsigned rootArgBase = oldAfter->getNumArguments();
      for (auto it : llvm::enumerate(touchedRoots))
        afterState[it.value()] = cast<VectorValue>(newAfter->getArgument(rootArgBase + it.index()));

      if (failed(rewriteBlock(oldAfter, newAfter, afterMapping, afterState)))
        return failure();

      auto oldYield = cast<scf::YieldOp>(oldAfter->getTerminator());
      SmallVector<Value> newYieldOperands;
      for (Value yielded : oldYield.getOperands())
        newYieldOperands.push_back(afterMapping.lookupOrDefault(yielded));
      for (MakePtrOp makePtrOp : touchedRoots)
        newYieldOperands.push_back(afterState[makePtrOp]);

      OpBuilder yieldBuilder = OpBuilder::atBlockEnd(newAfter);
      scf::YieldOp::create(yieldBuilder, oldYield.getLoc(), newYieldOperands);
    }

    for (unsigned i = 0; i < oldWhile.getNumResults(); ++i)
      mapping.map(oldWhile.getResult(i), newWhile.getResult(i));
    for (auto it : llvm::enumerate(touchedRoots))
      state[it.value()] =
          cast<VectorValue>(newWhile.getResult(oldWhile.getNumResults() + it.index()));

    return success();
  }

  LogicalResult rewriteBlock(Block *oldBlock, Block *newBlock, IRMapping &mapping,
                             RegMem2VectorSSAMap &state) {
    OpBuilder builder(oldBlock->getParentOp()->getContext());

    for (Operation &op : *oldBlock) {
      builder.setInsertionPointToEnd(newBlock);

      if (isa<scf::YieldOp, scf::ConditionOp>(op))
        continue;

      if (auto makePtrOp = dyn_cast<MakePtrOp>(&op); makePtrOp && isRegValue(makePtrOp)) {
        if (auto *info = getAllocaInfo(makePtrOp)) {
          mapping.map(
              makePtrOp.getResult(),
              ub::PoisonOp::create(builder, makePtrOp.getLoc(), makePtrOp.getType()).getResult());
          state[makePtrOp] = cast<VectorValue>(
              ub::PoisonOp::create(builder, makePtrOp.getLoc(), info->vectorSSATy).getResult());
        } else {
          builder.clone(op, mapping);
        }
        continue;
      }

      if (auto storeOp = dyn_cast<PtrStoreOp>(&op); storeOp && isRegValue(storeOp.getPtr())) {
        if (getRegAccessInfo(storeOp.getPtr(), storeOp.getValue().getType())) {
          if (failed(rewritePtrStore(storeOp, builder, mapping, state)))
            return failure();
        } else {
          builder.clone(op, mapping);
        }
        continue;
      }

      if (auto loadOp = dyn_cast<PtrLoadOp>(&op); loadOp && isRegValue(loadOp.getPtr())) {
        if (getRegAccessInfo(loadOp.getPtr(), loadOp.getResult().getType())) {
          if (failed(rewritePtrLoad(loadOp, builder, mapping, state)))
            return failure();
        } else {
          builder.clone(op, mapping);
        }
        continue;
      }

      if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
        if (failed(rewriteForOp(forOp, builder, mapping, state)))
          return failure();
        continue;
      }
      if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
        if (failed(rewriteIfOp(ifOp, builder, mapping, state)))
          return failure();
        continue;
      }
      if (auto whileOp = dyn_cast<scf::WhileOp>(&op)) {
        if (failed(rewriteWhileOp(whileOp, builder, mapping, state)))
          return failure();
        continue;
      }

      if (op.getNumRegions() != 0) {
        bool hasNestedRegOperandOrResult = false;
        op.walk([&](Operation *nestedOp) {
          if (nestedOp == &op)
            return;
          if (isRegOperandOrResult(nestedOp))
            hasNestedRegOperandOrResult = true;
        });
        if (hasNestedRegOperandOrResult)
          return op.emitOpError("unsupported region op with register values during rmem SSA");
      }

      builder.clone(op, mapping);
    }

    return success();
  }

  void cleanupDeadOps(gpu::GPUFuncOp funcOp) {
    bool changed = true;
    while (changed) {
      changed = false;
      funcOp.walk([&](Block *block) {
        for (Operation &op : llvm::make_early_inc_range(llvm::reverse(*block))) {
          if (isOpTriviallyDead(&op)) {
            op.erase();
            changed = true;
          }
        }
      });
    }
  }
};

} // namespace
