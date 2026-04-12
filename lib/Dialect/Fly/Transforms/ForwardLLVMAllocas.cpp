// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

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

using AllocaKey = std::pair<Operation *, int64_t>;

// ── Intra-block store→load forwarding ───────────────────────────────

static void forwardLLVMAllocaStores(Operation *topOp) {
  int forwarded = 0;
  topOp->walk([&](Block *block) {
    DenseMap<AllocaKey, Value> storeMap;

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
  if (forwarded > 0)
    llvm::errs() << "[forward-llvm-allocas] forwarded=" << forwarded << "\n";
}

// ── Cross-iteration promotion to loop-carried values ────────────────

static void promoteLLVMAllocaToLoopCarried(Operation *topOp) {
  SmallVector<scf::ForOp> loops;
  topOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

  int promoted = 0;
  for (auto forOp : loops) {
    Block &body = forOp.getRegion().front();

    // Collect "initial loads": LLVM::LoadOps from AS5 allocas that appear
    // before any LLVM::StoreOp to the same (alloca, byte_offset).
    SmallVector<std::pair<AllocaKey, LLVM::LoadOp>> initialLoads;
    DenseSet<AllocaKey> storedKeys;

    for (auto &op : body) {
      if (auto storeOp = dyn_cast<LLVM::StoreOp>(&op)) {
        auto [alloca, offset] = resolveLLVMAllocaPtr(storeOp.getAddr());
        if (alloca)
          storedKeys.insert({alloca.getOperation(), offset});
        continue;
      }
      if (auto loadOp = dyn_cast<LLVM::LoadOp>(&op)) {
        auto [alloca, offset] = resolveLLVMAllocaPtr(loadOp.getAddr());
        if (!alloca)
          continue;
        AllocaKey key = {alloca.getOperation(), offset};
        if (!storedKeys.contains(key))
          initialLoads.push_back({key, loadOp});
        continue;
      }
    }

    if (initialLoads.empty())
      continue;

    // Find the LAST LLVM::StoreOp in the body for each (alloca, offset).
    DenseMap<AllocaKey, LLVM::StoreOp> lastStore;
    for (auto it = body.rbegin(); it != body.rend(); ++it) {
      if (auto storeOp = dyn_cast<LLVM::StoreOp>(&*it)) {
        auto [alloca, offset] = resolveLLVMAllocaPtr(storeOp.getAddr());
        if (!alloca)
          continue;
        AllocaKey key = {alloca.getOperation(), offset};
        if (!lastStore.contains(key))
          lastStore[key] = storeOp;
      }
    }

    // Filter: keep only initial loads with a matching last store and
    // compatible types.
    SmallVector<std::pair<AllocaKey, LLVM::LoadOp>> validLoads;
    for (auto &[key, loadOp] : initialLoads) {
      auto it = lastStore.find(key);
      if (it == lastStore.end())
        continue;
      if (it->second.getValue().getType() != loadOp.getResult().getType())
        continue;
      validLoads.push_back({key, loadOp});
    }

    if (validLoads.empty())
      continue;

    // De-duplicate by key.
    DenseMap<AllocaKey, SmallVector<LLVM::LoadOp>> loadsByKey;
    for (auto &[key, loadOp] : validLoads)
      loadsByKey[key].push_back(loadOp);

    SmallVector<AllocaKey> uniqueKeys;
    for (auto &[key, _] : loadsByKey)
      uniqueKeys.push_back(key);

    OpBuilder builder(forOp);
    Location loc = forOp.getLoc();

    // Collect existing init values.
    SmallVector<Value> allInitValues;
    for (Value init : forOp.getInitArgs())
      allInitValues.push_back(init);
    size_t existingIterArgs = forOp.getNumRegionIterArgs();

    // Create prologue loads (before the loop) for each promoted key.
    // Use i8-typed GEPs from the alloca base for simplicity.
    auto i8Ty = builder.getI8Type();
    for (auto &key : uniqueKeys) {
      LLVM::LoadOp firstLoad = loadsByKey[key].front();
      auto allocaOp = cast<LLVM::AllocaOp>(key.first);
      int64_t byteOffset = key.second;

      Value ptr;
      if (byteOffset == 0) {
        ptr = allocaOp.getResult();
      } else {
        Value offset =
            arith::ConstantIntOp::create(builder, loc, byteOffset, 32);
        ptr = LLVM::GEPOp::create(builder, loc, allocaOp.getType(), i8Ty,
                                  allocaOp.getResult(), ValueRange{offset});
      }
      Value initVal = LLVM::LoadOp::create(builder, loc,
                                           firstLoad.getResult().getType(),
                                           ptr);
      allInitValues.push_back(initVal);
    }

    // Create new ForOp with existing + promoted iter_args.
    auto newForOp = scf::ForOp::create(builder, loc, forOp.getLowerBound(),
                                       forOp.getUpperBound(), forOp.getStep(),
                                       allInitValues);

    Block &newBody = newForOp.getRegion().front();

    // Erase auto-generated yield in the new body (if present).
    if (newBody.mightHaveTerminator())
      newBody.getTerminator()->erase();

    // Map old induction variable and existing iter args to new ones.
    body.getArgument(0).replaceAllUsesWith(newBody.getArgument(0));
    for (size_t i = 0; i < existingIterArgs; ++i)
      body.getArgument(i + 1).replaceAllUsesWith(newBody.getArgument(i + 1));

    // Replace initial loads with new iter args.
    for (size_t i = 0; i < uniqueKeys.size(); ++i) {
      Value iterArg = newBody.getArgument(existingIterArgs + 1 + i);
      for (LLVM::LoadOp loadOp : loadsByKey[uniqueKeys[i]]) {
        loadOp.getResult().replaceAllUsesWith(iterArg);
        loadOp.erase();
        promoted++;
      }
    }

    // Get existing yield values from old terminator before erasing.
    SmallVector<Value> existingYieldValues;
    if (body.mightHaveTerminator()) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(body.getTerminator())) {
        for (Value v : yieldOp.getResults())
          existingYieldValues.push_back(v);
      }
      body.getTerminator()->erase();
    }

    // Splice remaining ops from old body to new body.
    newBody.getOperations().splice(newBody.end(), body.getOperations());

    // Build new yield: existing accumulator values + promoted alloca values.
    builder.setInsertionPointToEnd(&newBody);
    SmallVector<Value> yieldValues = existingYieldValues;
    for (auto &key : uniqueKeys)
      yieldValues.push_back(lastStore[key].getValue());
    scf::YieldOp::create(builder, loc, yieldValues);

    // Replace old ForOp results with new ones.
    for (size_t i = 0; i < forOp.getNumResults(); ++i)
      forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
    forOp.erase();

    // After the loop, store the final promoted values back to the allocas
    // so that epilogue code can read them.
    builder.setInsertionPointAfter(newForOp);
    for (size_t i = 0; i < uniqueKeys.size(); ++i) {
      auto &key = uniqueKeys[i];
      auto allocaOp = cast<LLVM::AllocaOp>(key.first);
      int64_t byteOffset = key.second;

      Value ptr;
      if (byteOffset == 0) {
        ptr = allocaOp.getResult();
      } else {
        Value offset =
            arith::ConstantIntOp::create(builder, loc, byteOffset, 32);
        ptr = LLVM::GEPOp::create(builder, loc, allocaOp.getType(), i8Ty,
                                  allocaOp.getResult(), ValueRange{offset});
      }
      LLVM::StoreOp::create(builder, loc,
                            newForOp.getResult(existingIterArgs + i), ptr);
    }
  }

  if (promoted > 0)
    llvm::errs() << "[llvm-alloca-to-loop-carried] promoted=" << promoted
                 << "\n";
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

  if (eliminated > 0)
    llvm::errs() << "[eliminate-dead-as5-allocas] eliminated=" << eliminated
                 << "\n";
}

// ── Pass implementation ─────────────────────────────────────────────

class FlyForwardLLVMAllocasPass
    : public fly::impl::FlyForwardLLVMAllocasPassBase<
          FlyForwardLLVMAllocasPass> {
public:
  void runOnOperation() override {
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
