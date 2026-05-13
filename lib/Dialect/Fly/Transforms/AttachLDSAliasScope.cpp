// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

#include "flydsl/Dialect/Fly/Transforms/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace mlir {
namespace fly {
#define GEN_PASS_DEF_FLYATTACHLDSALIASSCOPEPASS
#include "flydsl/Dialect/Fly/Transforms/Passes.h.inc"
} // namespace fly
} // namespace mlir

namespace {

// LDS address space on AMDGPU.
static constexpr unsigned kLDSAddrSpace = 3;

/// Returns true if `g` is an external `[0 x i8] addrspace(3)` global,
/// i.e. a dyn-shared LDS base. We restrict on size 0 (HSA dynamic LDS
/// convention) so we don't accidentally tag SmemAllocator-style static
/// globals (whose alias info already comes from distinct symbols).
static bool isDynSharedGlobal(LLVM::GlobalOp g) {
  if (g.getAddrSpace() != kLDSAddrSpace)
    return false;
  if (g.getLinkage() != LLVM::Linkage::External)
    return false;
  auto arrTy = dyn_cast<LLVM::LLVMArrayType>(g.getType());
  if (!arrTy)
    return false;
  return arrTy.getNumElements() == 0;
}

/// Per-SSA-value provenance maps. Absent entry == "no provenance".
/// A null mapped value means "ambiguous (mixes multiple globals)".
using PtrProvenance = llvm::DenseMap<Value, LLVM::GlobalOp>;
using IntProvenance = llvm::DenseMap<Value, LLVM::GlobalOp>;

/// True iff `op` is an `llvm.amdgcn.raw.ptr.buffer.load.lds` intrinsic.
static bool isBufferLoadLDS(LLVM::CallOp call) {
  auto callee = call.getCallee();
  if (!callee)
    return false;
  return callee->starts_with("llvm.amdgcn.raw.ptr.buffer.load.lds");
}

/// Returns the addrspace(3) pointer operand consumed by `op`, or
/// nullptr if there isn't exactly one such operand worth tagging.
static Value memoryPointerForOp(Operation *op) {
  if (auto load = dyn_cast<LLVM::LoadOp>(op)) {
    auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(load.getAddr().getType());
    if (ptrTy && ptrTy.getAddressSpace() == kLDSAddrSpace)
      return load.getAddr();
    return nullptr;
  }
  if (auto store = dyn_cast<LLVM::StoreOp>(op)) {
    auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(store.getAddr().getType());
    if (ptrTy && ptrTy.getAddressSpace() == kLDSAddrSpace)
      return store.getAddr();
    return nullptr;
  }
  if (auto call = dyn_cast<LLVM::CallOp>(op)) {
    if (!isBufferLoadLDS(call))
      return nullptr;
    // The LDS pointer is the second arg (after the buffer-desc ptr).
    if (call.getNumOperands() >= 2) {
      Value lds = call.getOperand(1);
      auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(lds.getType());
      if (ptrTy && ptrTy.getAddressSpace() == kLDSAddrSpace)
        return lds;
    }
    return nullptr;
  }
  return nullptr;
}

/// Forward dataflow that maps SSA values back to the LDS global they
/// derive from. Only tracks the patterns we care about:
///   * `LLVM::AddressOfOp(@g)` -> ptr provenance(@g)
///   * `LLVM::PtrToIntOp(p)` -> int provenance(p)
///   * `LLVM::AddOp(a, b)` / `LLVM::OrOp(a, b)` / `LLVM::SubOp(a, b)` ->
///       int provenance(a)|provenance(b) (single non-null wins;
///       conflict marks ambiguous so we don't tag downstream uses)
///   * `LLVM::IntToPtrOp(i)` -> ptr provenance(i)
///   * `LLVM::GEPOp(p)` -> ptr provenance(p)
static void computeProvenance(
    LLVM::LLVMFuncOp func,
    const llvm::DenseMap<StringRef, LLVM::GlobalOp> &nameToGlobal,
    PtrProvenance &ptrProv, IntProvenance &intProv) {
  // Combine two provenance entries. Returns (global, hasInfo) where
  // hasInfo=false means "still no info" and a null global with
  // hasInfo=true means "ambiguous".
  auto combine = [](LLVM::GlobalOp a, bool aSeen, LLVM::GlobalOp b,
                    bool bSeen) -> std::pair<LLVM::GlobalOp, bool> {
    if (!aSeen && !bSeen)
      return {nullptr, false};
    if (!aSeen)
      return {b, true};
    if (!bSeen)
      return {a, true};
    if (a == b)
      return {a, true};
    return {nullptr, true};  // ambiguous
  };

  func.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto addrOf = dyn_cast<LLVM::AddressOfOp>(op)) {
      auto it = nameToGlobal.find(addrOf.getGlobalName());
      if (it != nameToGlobal.end())
        ptrProv[addrOf.getResult()] = it->second;
      return;
    }
    if (auto p2i = dyn_cast<LLVM::PtrToIntOp>(op)) {
      auto it = ptrProv.find(p2i.getArg());
      if (it != ptrProv.end() && it->second)
        intProv[p2i.getResult()] = it->second;
      return;
    }
    auto handleAddLike = [&](Value lhs, Value rhs, Value result) {
      auto la = intProv.find(lhs);
      auto lb = intProv.find(rhs);
      bool aSeen = la != intProv.end();
      bool bSeen = lb != intProv.end();
      auto [g, hasInfo] =
          combine(aSeen ? la->second : nullptr, aSeen,
                  bSeen ? lb->second : nullptr, bSeen);
      if (hasInfo && g)
        intProv[result] = g;
    };
    if (auto add = dyn_cast<LLVM::AddOp>(op)) {
      handleAddLike(add.getLhs(), add.getRhs(), add.getResult());
      return;
    }
    if (auto orOp = dyn_cast<LLVM::OrOp>(op)) {
      handleAddLike(orOp.getLhs(), orOp.getRhs(), orOp.getResult());
      return;
    }
    if (auto sub = dyn_cast<LLVM::SubOp>(op)) {
      handleAddLike(sub.getLhs(), sub.getRhs(), sub.getResult());
      return;
    }
    if (auto i2p = dyn_cast<LLVM::IntToPtrOp>(op)) {
      auto it = intProv.find(i2p.getArg());
      if (it != intProv.end() && it->second) {
        auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(i2p.getResult().getType());
        if (ptrTy && ptrTy.getAddressSpace() == kLDSAddrSpace)
          ptrProv[i2p.getResult()] = it->second;
      }
      return;
    }
    if (auto gep = dyn_cast<LLVM::GEPOp>(op)) {
      auto it = ptrProv.find(gep.getBase());
      if (it != ptrProv.end() && it->second) {
        auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(gep.getResult().getType());
        if (ptrTy && ptrTy.getAddressSpace() == kLDSAddrSpace)
          ptrProv[gep.getResult()] = it->second;
      }
      return;
    }
  });
}

class FlyAttachLDSAliasScopePass
    : public mlir::fly::impl::FlyAttachLDSAliasScopePassBase<
          FlyAttachLDSAliasScopePass> {
public:
  using mlir::fly::impl::FlyAttachLDSAliasScopePassBase<
      FlyAttachLDSAliasScopePass>::FlyAttachLDSAliasScopePassBase;

  void runOnOperation() override {
    gpu::GPUModuleOp gpuModule = getOperation();

    // Collect dyn-shared globals in declaration order.
    SmallVector<LLVM::GlobalOp> dynGlobals;
    llvm::DenseMap<StringRef, LLVM::GlobalOp> nameToGlobal;
    for (auto g : gpuModule.getOps<LLVM::GlobalOp>()) {
      if (isDynSharedGlobal(g)) {
        dynGlobals.push_back(g);
        nameToGlobal[g.getSymName()] = g;
      }
    }
    if (dynGlobals.size() < 2)
      return;  // Single (or no) dyn-shared region: nothing to disambiguate.

    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    // One domain per gpu.module, one scope per dyn-shared global.
    auto domain = LLVM::AliasScopeDomainAttr::get(
        ctx, builder.getStringAttr("FlyDynSharedDomain"));

    llvm::DenseMap<LLVM::GlobalOp, LLVM::AliasScopeAttr> globalToScope;
    for (auto g : dynGlobals) {
      auto scope = LLVM::AliasScopeAttr::get(
          domain, builder.getStringAttr(g.getSymName()));
      globalToScope[g] = scope;
    }

    // Pre-compute the noalias-set per global = all scopes except its
    // own. This is what makes cross-global accesses no-alias.
    llvm::DenseMap<LLVM::GlobalOp, ArrayAttr> globalToNoalias;
    for (auto g : dynGlobals) {
      SmallVector<Attribute> others;
      others.reserve(dynGlobals.size() - 1);
      for (auto og : dynGlobals)
        if (og != g)
          others.push_back(globalToScope[og]);
      globalToNoalias[g] = ArrayAttr::get(ctx, others);
    }

    for (auto func : gpuModule.getOps<LLVM::LLVMFuncOp>()) {
      if (func.empty())
        continue;
      PtrProvenance ptrProv;
      IntProvenance intProv;
      computeProvenance(func, nameToGlobal, ptrProv, intProv);

      func.walk([&](Operation *op) {
        Value lds = memoryPointerForOp(op);
        if (!lds)
          return;
        auto it = ptrProv.find(lds);
        if (it == ptrProv.end() || !it->second)
          return;
        LLVM::GlobalOp g = it->second;
        auto scopeIt = globalToScope.find(g);
        auto noaliasIt = globalToNoalias.find(g);
        if (scopeIt == globalToScope.end() || noaliasIt == globalToNoalias.end())
          return;
        auto scopeAttr = ArrayAttr::get(ctx, {scopeIt->second});
        op->setAttr("alias_scopes", scopeAttr);
        op->setAttr("noalias_scopes", noaliasIt->second);
      });
    }
  }
};

} // namespace
