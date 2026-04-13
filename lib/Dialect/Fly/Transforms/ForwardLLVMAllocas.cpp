// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "flydsl/Dialect/Fly/Utils/AllocaPromotion.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "fly-forward-llvm-allocas"

#include "flydsl/Dialect/Fly/Transforms/Passes.h"

using namespace mlir;

namespace mlir {
namespace fly {
#define GEN_PASS_DEF_FLYFORWARDLLVMALLOCASPASS
#include "flydsl/Dialect/Fly/Transforms/Passes.h.inc"
} // namespace fly
} // namespace mlir

namespace {

// ── Helpers ─────────────────────────────────────────────────────────

static int64_t getTypeSizeInBytes(Type ty) {
  if (ty.isIntOrFloat())
    return (ty.getIntOrFloatBitWidth() + 7) / 8;
  if (auto vecTy = dyn_cast<VectorType>(ty))
    return vecTy.getNumElements() * getTypeSizeInBytes(vecTy.getElementType());
  return -1;
}

/// Try to resolve a constant integer from an SSA value.
static std::optional<int64_t> getConstantIndex(Value val) {
  if (auto op = val.getDefiningOp<arith::ConstantIntOp>())
    return op.value();
  if (auto op = val.getDefiningOp<LLVM::ConstantOp>()) {
    if (auto attr = dyn_cast<IntegerAttr>(op.getValue()))
      return attr.getInt();
  }
  return std::nullopt;
}

/// Walk a GEP chain back to an AS5 LLVM::AllocaOp, accumulating a constant
/// byte offset.  Returns (alloca, byte_offset) or (nullptr, -1) on failure.
static std::pair<LLVM::AllocaOp, int64_t> resolveLLVMAllocaPtr(Value ptr) {
  int64_t byteOffset = 0;
  Value cur = ptr;

  while (auto gepOp = dyn_cast_or_null<LLVM::GEPOp>(cur.getDefiningOp())) {
    Type elemTy = gepOp.getElemType();
    int64_t elemBytes = getTypeSizeInBytes(elemTy);
    if (elemBytes <= 0)
      return {nullptr, -1};

    auto rawIndices = gepOp.getRawConstantIndices();
    if (rawIndices.size() != 1)
      return {nullptr, -1};

    int64_t idx = rawIndices[0];
    if (idx == LLVM::GEPOp::kDynamicIndex) {
      auto dynIndices = gepOp.getDynamicIndices();
      if (dynIndices.empty())
        return {nullptr, -1};
      auto maybeIdx = getConstantIndex(dynIndices[0]);
      if (!maybeIdx)
        return {nullptr, -1};
      idx = *maybeIdx;
    }

    byteOffset += idx * elemBytes;
    cur = gepOp.getBase();
  }

  auto allocaOp = dyn_cast_or_null<LLVM::AllocaOp>(cur.getDefiningOp());
  if (!allocaOp)
    return {nullptr, -1};

  auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(allocaOp.getType());
  if (!ptrTy || ptrTy.getAddressSpace() != 5)
    return {nullptr, -1};

  return {allocaOp, byteOffset};
}

using fly::AllocaKey;

// ── Intra-block store→load forwarding ───────────────────────────────

static void forwardLLVMAllocaStores(Operation *topOp) {
  int forwarded = 0;
  topOp->walk([&](Block *block) {
    llvm::MapVector<AllocaKey, Value> storeMap;

    for (auto &op : llvm::make_early_inc_range(*block)) {
      if (auto storeOp = dyn_cast<LLVM::StoreOp>(&op)) {
        auto [alloca, offset] = resolveLLVMAllocaPtr(storeOp.getAddr());
        if (alloca)
          storeMap[{alloca.getOperation(), offset}] = storeOp.getValue();
        continue;
      }
      if (auto loadOp = dyn_cast<LLVM::LoadOp>(&op)) {
        auto [alloca, offset] = resolveLLVMAllocaPtr(loadOp.getAddr());
        if (!alloca)
          continue;
        AllocaKey key = {alloca.getOperation(), offset};
        auto it = storeMap.find(key);
        if (it != storeMap.end() &&
            it->second.getType() == loadOp.getResult().getType()) {
          loadOp.getResult().replaceAllUsesWith(it->second);
          loadOp.erase();
          forwarded++;
          continue;
        }
        // Sub-element forwarding: load of scalar T from offset X may be
        // covered by a stored vector<N×T'> at offset Y where
        // sizeof(T)==sizeof(T') and Y ≤ X < Y + N*sizeof(T').
        int64_t loadBytes = getTypeSizeInBytes(loadOp.getResult().getType());
        if (loadBytes > 0) {
          Operation *allocaPtr = alloca.getOperation();
          bool found = false;
          for (auto &kv : storeMap) {
            if (kv.first.first != allocaPtr)
              continue;
            auto vecTy = dyn_cast<VectorType>(kv.second.getType());
            if (!vecTy || vecTy.getRank() != 1)
              continue;
            int64_t elemBytes =
                getTypeSizeInBytes(vecTy.getElementType());
            if (elemBytes != loadBytes)
              continue;
            int64_t storeOffset = kv.first.second;
            int64_t storeSize = vecTy.getNumElements() * elemBytes;
            if (offset < storeOffset || offset >= storeOffset + storeSize)
              continue;
            if ((offset - storeOffset) % elemBytes != 0)
              continue;
            int64_t elemIdx = (offset - storeOffset) / elemBytes;
            OpBuilder b(loadOp);
            Value idx = arith::ConstantIntOp::create(b, loadOp.getLoc(),
                                                     elemIdx, 32);
            Value elem =
                LLVM::ExtractElementOp::create(b, loadOp.getLoc(),
                                               kv.second, idx);
            // Bitcast if element type differs (e.g., bf16 → i16).
            if (elem.getType() != loadOp.getResult().getType())
              elem = LLVM::BitcastOp::create(b, loadOp.getLoc(),
                                             loadOp.getResult().getType(),
                                             elem);
            loadOp.getResult().replaceAllUsesWith(elem);
            loadOp.erase();
            forwarded++;
            found = true;
            break;
          }
          if (found)
            continue;
        }
        continue;
      }
      // Ops with nested regions (scf.for, scf.if) may write to AS5 allocas.
      // Walk the nested regions and invalidate entries for touched allocas.
      if (op.getNumRegions() > 0) {
        SmallPtrSet<Operation *, 4> clobbered;
        op.walk([&](LLVM::StoreOp innerStore) {
          auto [alloca, offset] = resolveLLVMAllocaPtr(innerStore.getAddr());
          if (alloca)
            clobbered.insert(alloca.getOperation());
        });
        if (!clobbered.empty()) {
          SmallVector<AllocaKey> toRemove;
          for (auto &kv : storeMap)
            if (clobbered.contains(kv.first.first))
              toRemove.push_back(kv.first);
          for (auto &key : toRemove)
            storeMap.erase(key);
        }
      }
    }
  });
  LLVM_DEBUG(if (forwarded > 0) llvm::dbgs()
             << "[forward-llvm-allocas] forwarded=" << forwarded << "\n");
}

// ── Cross-iteration promotion to loop-carried values ────────────────

/// Build a GEP to (allocaOp + byteOffset) using i8 element type.
static Value buildAllocaGEP(OpBuilder &builder, Location loc,
                            LLVM::AllocaOp allocaOp, int64_t byteOffset) {
  if (byteOffset == 0)
    return allocaOp.getResult();
  Value offset = arith::ConstantIntOp::create(builder, loc, byteOffset, 32);
  return LLVM::GEPOp::create(builder, loc, allocaOp.getType(),
                             builder.getI8Type(), allocaOp.getResult(),
                             ValueRange{offset});
}

static void promoteLLVMAllocaToLoopCarried(Operation *topOp) {
  fly::AllocaPromotionHooks hooks;
  hooks.resolvePtr = [](Value ptr) -> std::pair<Operation *, int64_t> {
    auto [alloca, offset] = resolveLLVMAllocaPtr(ptr);
    if (!alloca)
      return {nullptr, -1};
    return {alloca.getOperation(), offset};
  };
  hooks.tryGetStore = [](Operation &op)
      -> std::optional<std::pair<Value, Value>> {
    if (auto storeOp = dyn_cast<LLVM::StoreOp>(&op))
      return std::pair{storeOp.getAddr(), storeOp.getValue()};
    return std::nullopt;
  };
  hooks.tryGetLoad = [](Operation &op)
      -> std::optional<std::pair<Value, Value>> {
    if (auto loadOp = dyn_cast<LLVM::LoadOp>(&op))
      return std::pair{loadOp.getAddr(), loadOp.getResult()};
    return std::nullopt;
  };
  hooks.createPrologueLoad = [](OpBuilder &builder, Location loc,
                                const fly::PromotedSlot &slot) -> Value {
    auto allocaOp = cast<LLVM::AllocaOp>(slot.key.first);
    Value ptr = buildAllocaGEP(builder, loc, allocaOp, slot.key.second);
    return LLVM::LoadOp::create(builder, loc, slot.valueType, ptr);
  };
  hooks.createEpilogueStore = [](OpBuilder &builder, Location loc,
                                 Value finalVal,
                                 const fly::PromotedSlot &slot) {
    auto allocaOp = cast<LLVM::AllocaOp>(slot.key.first);
    Value ptr = buildAllocaGEP(builder, loc, allocaOp, slot.key.second);
    LLVM::StoreOp::create(builder, loc, finalVal, ptr);
  };
  fly::promoteAllocasToLoopCarried(topOp, hooks);
}

// ── Dead AS5 alloca elimination ─────────────────────────────────────

/// Check whether any user (transitively through GEPs) loads from this alloca.
static bool allocaHasLoads(LLVM::AllocaOp allocaOp) {
  SmallVector<Value, 8> worklist;
  worklist.push_back(allocaOp.getResult());

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (Operation *user : v.getUsers()) {
      if (isa<LLVM::LoadOp>(user))
        return true;
      if (isa<LLVM::GEPOp>(user))
        worklist.push_back(user->getResult(0));
    }
  }
  return false;
}

static void eliminateDeadAS5Allocas(Operation *topOp) {
  SmallVector<LLVM::AllocaOp> allocas;
  topOp->walk([&](LLVM::AllocaOp allocaOp) {
    auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(allocaOp.getType());
    if (ptrTy && ptrTy.getAddressSpace() == 5)
      allocas.push_back(allocaOp);
  });

  int eliminated = 0;
  for (auto allocaOp : allocas) {
    if (allocaHasLoads(allocaOp))
      continue;

    // Erase all stores and GEPs transitively, then the alloca itself.
    SmallVector<Operation *, 16> toErase;
    SmallVector<Value, 8> worklist;
    worklist.push_back(allocaOp.getResult());

    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      for (Operation *user : v.getUsers()) {
        if (isa<LLVM::GEPOp>(user))
          worklist.push_back(user->getResult(0));
        toErase.push_back(user);
      }
    }

    // Erase in reverse to avoid use-before-def issues.
    for (auto it = toErase.rbegin(); it != toErase.rend(); ++it) {
      if ((*it)->use_empty())
        (*it)->erase();
    }
    if (allocaOp->use_empty()) {
      allocaOp.erase();
      eliminated++;
    }
  }

  LLVM_DEBUG(if (eliminated > 0) llvm::dbgs()
             << "[eliminate-dead-as5-allocas] eliminated=" << eliminated
             << "\n");
}

// ── Pass implementation ─────────────────────────────────────────────

class FlyForwardLLVMAllocasPass
    : public fly::impl::FlyForwardLLVMAllocasPassBase<
          FlyForwardLLVMAllocasPass> {
public:
  using fly::impl::FlyForwardLLVMAllocasPassBase<
      FlyForwardLLVMAllocasPass>::FlyForwardLLVMAllocasPassBase;

  void runOnOperation() override {
    if (!enabled)
      return;

    Operation *op = getOperation();

    // Pass 1: forward intra-block stores → loads (includes sub-element).
    forwardLLVMAllocaStores(op);

    // Pass 2: promote cross-iteration alloca patterns to loop-carried.
    promoteLLVMAllocaToLoopCarried(op);

    // Pass 3: re-run forwarding for newly exposed opportunities.
    forwardLLVMAllocaStores(op);

    // Pass 4: clean up allocas with no remaining loads.
    eliminateDeadAS5Allocas(op);
  }
};

} // namespace
