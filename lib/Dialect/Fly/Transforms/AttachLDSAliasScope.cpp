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

/// Per-SSA-value provenance, encoded as a tri-state DenseMap:
///   - absent entry      => unknown / not derived from any tracked global
///   - mapped to G       => derived from exactly the LDS global G
///   - mapped to nullptr => *known* to mix two or more globals (ambiguous);
///                          downstream uses that consume this value must
///                          also be marked ambiguous so the pass never
///                          tags an access with a single scope when its
///                          true scope set is larger.
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
/// derive from. Only the canonical pointer-arithmetic chain is tracked
/// so that we never tag an access whose pointer might really span more
/// than one global:
///   * `LLVM::AddressOfOp(@g)` -> ptr provenance(@g)
///   * `LLVM::PtrToIntOp(p)`   -> int provenance(p)
///   * `LLVM::AddOp(a, b)`     -> int provenance(a) iff *exactly one*
///                                 operand carries provenance; if both
///                                 carry provenance, the result mixes
///                                 globals and is recorded as ambiguous
///   * `LLVM::IntToPtrOp(i)`   -> ptr provenance(i)
///   * `LLVM::GEPOp(p)`        -> ptr provenance(p)
///
/// `or`/`sub`/`xor`/`and`/`shl`/`shr`/`bitcast` and any other op are
/// treated as provenance-destroying. The dataflow is intentionally
/// fail-safe: when in doubt, drop the tag rather than emit one that
/// could wrongly tell LLVM "no alias" about pointers that really do
/// alias at runtime.
static void computeProvenance(
    LLVM::LLVMFuncOp func,
    const llvm::DenseMap<StringRef, LLVM::GlobalOp> &nameToGlobal,
    PtrProvenance &ptrProv, IntProvenance &intProv) {
  // Tri-state DenseMap merge. Mirrors the encoding documented on
  // `IntProvenance` / `PtrProvenance`:
  //   - absent entry  => unknown
  //   - present, G    => provenance(G)
  //   - present, null => ambiguous
  //
  // Returns (resultProvenance, hasInfo). When hasInfo is false the
  // caller stores nothing (keeps the value unknown); when hasInfo is
  // true and resultProvenance is null the caller stores a sentinel
  // entry so subsequent uses also propagate as ambiguous.
  auto combine = [](LLVM::GlobalOp a, bool aSeen, LLVM::GlobalOp b,
                    bool bSeen) -> std::pair<LLVM::GlobalOp, bool> {
    if (!aSeen && !bSeen)
      return {nullptr, false};
    if (!aSeen)
      return {b, true};
    if (!bSeen)
      return {a, true};
    // Both operands have known provenance entries.
    // - either is ambiguous (null) -> ambiguous
    // - same non-null global       -> *still* ambiguous, because adding
    //   a pointer-derived int to itself doesn't represent any single
    //   well-formed pointer
    // - different non-null globals -> ambiguous
    if (!a || !b || a != b)
      return {nullptr, true};
    return {nullptr, true};  // see comment above (a == b case)
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
      if (it != ptrProv.end())
        intProv[p2i.getResult()] = it->second;  // may store ambiguous
      return;
    }
    if (auto add = dyn_cast<LLVM::AddOp>(op)) {
      auto la = intProv.find(add.getLhs());
      auto lb = intProv.find(add.getRhs());
      bool aSeen = la != intProv.end();
      bool bSeen = lb != intProv.end();
      auto [g, hasInfo] = combine(aSeen ? la->second : nullptr, aSeen,
                                  bSeen ? lb->second : nullptr, bSeen);
      if (hasInfo)
        intProv[add.getResult()] = g;  // g may be null = ambiguous sentinel
      return;
    }
    if (auto i2p = dyn_cast<LLVM::IntToPtrOp>(op)) {
      auto it = intProv.find(i2p.getArg());
      if (it != intProv.end()) {
        auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(i2p.getResult().getType());
        if (ptrTy && ptrTy.getAddressSpace() == kLDSAddrSpace)
          ptrProv[i2p.getResult()] = it->second;  // propagate ambiguous too
      }
      return;
    }
    if (auto gep = dyn_cast<LLVM::GEPOp>(op)) {
      auto it = ptrProv.find(gep.getBase());
      if (it != ptrProv.end()) {
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
