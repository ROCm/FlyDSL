// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
//
// Inject amdgpu-cluster-dims into llvm.func passthrough. Run inside gpu.module() AFTER
// convert-gpu-to-rocdl.
//
// The upstream ROCDL dialect does not translate `rocdl.cluster_dims` to the LLVM IR
// function attribute `amdgpu-cluster-dims`. This pass bridges the gap by converting the
// discardable attribute that `GPUFuncOpLowering` copied from gpu.func into an LLVM
// passthrough entry that the LLVM IR emitter honours.
//
// It rewrites an attribute on ops the GPU-to-ROCDL conversion already produced, so it is a
// transform on this dialect's own annotation rather than part of any Fly conversion.

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

#include "flydsl/Dialect/FlyROCDL/Transforms/Passes.h"

using namespace mlir;

namespace mlir {
namespace fly_rocdl {
#define GEN_PASS_DEF_FLYROCDLCLUSTERATTRPASS
#include "flydsl/Dialect/FlyROCDL/Transforms/Passes.h.inc"
} // namespace fly_rocdl
} // namespace mlir

namespace {

class FlyROCDLClusterAttrPass
    : public mlir::fly_rocdl::impl::FlyROCDLClusterAttrPassBase<FlyROCDLClusterAttrPass> {
public:
  using mlir::fly_rocdl::impl::FlyROCDLClusterAttrPassBase<
      FlyROCDLClusterAttrPass>::FlyROCDLClusterAttrPassBase;

  void runOnOperation() override {
    getOperation()->walk([&](LLVM::LLVMFuncOp func) {
      auto clusterAttr = func->getAttrOfType<StringAttr>("rocdl.cluster_dims");
      if (!clusterAttr)
        return;

      MLIRContext *ctx = func.getContext();

      // Build the new passthrough entry: ["amdgpu-cluster-dims", "2,2,1"].
      auto key = StringAttr::get(ctx, "amdgpu-cluster-dims");
      auto entry = ArrayAttr::get(ctx, {key, clusterAttr});

      // Append to existing passthrough list (if any).
      SmallVector<Attribute, 4> passthroughAttrs;
      if (auto existing = func.getPassthroughAttr())
        passthroughAttrs.append(existing.begin(), existing.end());
      passthroughAttrs.push_back(entry);

      func.setPassthroughAttr(ArrayAttr::get(ctx, passthroughAttrs));
      func->removeAttr("rocdl.cluster_dims");
    });
  }
};

} // namespace
