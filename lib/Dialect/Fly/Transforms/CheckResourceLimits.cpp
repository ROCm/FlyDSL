// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Reject kernels whose statically known resource usage exceeds the limits that their
// gpu.module targets state through GPUTargetInfoAttrInterface.
//
// The pass knows no resource. Resources are attributes (KernelResourceAttrInterface), ops
// report what they allocate and free (ResourceUsageOpInterface) and targets report limits,
// so a backend adds a resource of its own without touching this pass.

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::fly;

namespace mlir {
namespace fly {
#define GEN_PASS_DEF_FLYCHECKRESOURCELIMITSPASS
#include "flydsl/Dialect/Fly/Transforms/Passes.h.inc"
} // namespace fly
} // namespace mlir

namespace {

/// Amount used per resource attribute, in first-use order.
using ResourceAmounts = llvm::MapVector<Attribute, int64_t>;

/// The most of each resource that the allocations in `kernel` hold at once. As in a linear
/// scan, ops are taken in the order they appear: an allocation is held from its op to the
/// op freeing it, or to the end of the kernel, so allocations never freed add up wherever
/// they are. A resource with a free that names no earlier allocation is left out, since
/// what it frees is unknown.
ResourceAmounts collectPeakUsage(gpu::GPUFuncOp kernel) {
  ResourceAmounts held, peak;
  llvm::DenseMap<Value, ResourceUsage> allocations;
  llvm::SmallDenseSet<Attribute> unmatched;
  SmallVector<ResourceUsage> usage;
  kernel.walk<WalkOrder::PreOrder>([&](ResourceUsageOpInterface op) {
    usage.clear();
    op.getResourceUsage(usage);
    for (const ResourceUsage &entry : usage) {
      if (entry.kind == ResourceUsage::Allocate) {
        int64_t &amount = held[entry.resource];
        amount += entry.amount;
        peak[entry.resource] = std::max(peak[entry.resource], amount);
        if (entry.allocation)
          allocations[entry.allocation] = entry;
        continue;
      }
      auto allocation = allocations.find(entry.allocation);
      if (allocation == allocations.end() || allocation->second.resource != entry.resource) {
        unmatched.insert(entry.resource);
        continue;
      }
      held[entry.resource] -= allocation->second.amount;
      allocations.erase(allocation);
    }
  });
  for (Attribute resource : unmatched)
    peak.erase(resource);
  return peak;
}

/// Report every resource `kernel` uses beyond a limit of `target`.
LogicalResult checkLimits(gpu::GPUFuncOp kernel, Attribute target, const ResourceAmounts &usage) {
  auto info = dyn_cast<GPUTargetInfoAttrInterface>(target);
  if (!info)
    return success();
  LogicalResult result = success();
  for (const auto &[resource, used] : usage) {
    std::optional<int64_t> limit = info.getResourceLimit(resource);
    if (!limit || used <= *limit)
      continue;
    InFlightDiagnostic diag = kernel.emitError()
                              << "kernel '" << kernel.getName() << "' uses " << used << " ";
    if (auto described = dyn_cast<KernelResourceAttrInterface>(resource))
      diag << described.getDescription();
    else
      diag << resource;
    diag << ", exceeding the limit of " << *limit << " of target " << target;
    result = failure();
  }
  return result;
}

class FlyCheckResourceLimitsPass
    : public mlir::fly::impl::FlyCheckResourceLimitsPassBase<FlyCheckResourceLimitsPass> {
public:
  using mlir::fly::impl::FlyCheckResourceLimitsPassBase<
      FlyCheckResourceLimitsPass>::FlyCheckResourceLimitsPassBase;

  void runOnOperation() override {
    ArrayAttr targets = getOperation().getTargetsAttr();
    if (!targets)
      return;
    bool failed = false;
    for (auto kernel : getOperation().getOps<gpu::GPUFuncOp>()) {
      if (!kernel.isKernel())
        continue;
      ResourceAmounts usage = collectPeakUsage(kernel);
      for (Attribute target : targets)
        failed |= mlir::failed(checkLimits(kernel, target, usage));
    }
    if (failed)
      signalPassFailure();
  }
};

} // namespace
