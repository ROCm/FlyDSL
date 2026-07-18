// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
//
// Test fixture for tests/kernels/test_llvm_pass_plugin_e2e.py (compiled at test
// time, not by CMake).
//
// An LLVM new-PM IR pass plugin registering "flydsl-print-tid".  At the entry of
// every amdgpu_kernel it emits FlyDSL's exact hostcall device-printf sequence
// (__ockl_printf_begin / append_string_n / append_args) printing threadIdx.x.
// Using the same ockl ABI as fx.printf means the ROCm runtime FlyDSL already
// sets up services it, and ockl is linked during the O=0 re-codegen.  (The C
// printf + amdgpu-printf-runtime-binding route instead emits the buffered
// __printf_alloc path, which FlyDSL's runtime does not service.)

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
using namespace llvm;
namespace {
struct PrintTidPass : PassInfoMixin<PrintTidPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &) {
    LLVMContext &C = M.getContext();
    auto *i64 = Type::getInt64Ty(C);
    auto *i32 = Type::getInt32Ty(C);
    auto *ptr = PointerType::get(C, 0);
    FunctionCallee beginF =
        M.getOrInsertFunction("__ockl_printf_begin", FunctionType::get(i64, {i64}, false));
    FunctionCallee strF = M.getOrInsertFunction(
        "__ockl_printf_append_string_n", FunctionType::get(i64, {i64, ptr, i64, i32}, false));
    FunctionCallee argsF = M.getOrInsertFunction(
        "__ockl_printf_append_args",
        FunctionType::get(i64, {i64, i32, i64, i64, i64, i64, i64, i64, i64, i32}, false));
    Function *widx = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_workitem_id_x);
    bool changed = false;
    for (Function &F : M) {
      if (F.isDeclaration() || F.getCallingConv() != CallingConv::AMDGPU_KERNEL)
        continue;
      IRBuilder<> B(&*F.getEntryBlock().getFirstInsertionPt());
      Constant *str = ConstantDataArray::getString(C, "flydsl-pass: threadIdx.x=%d\n", true);
      // Format string must live in addrspace 0 (matches the ockl append ABI).
      auto *gv = new GlobalVariable(M, str->getType(), true, GlobalValue::InternalLinkage, str,
                                    "flydsl_tid_fmt", nullptr, GlobalValue::NotThreadLocal, 0);
      uint64_t len = cast<ArrayType>(str->getType())->getNumElements();
      Value *tid = B.CreateZExt(B.CreateCall(widx, {}), i64);
      Value *z = ConstantInt::get(i64, 0);
      Value *h0 = B.CreateCall(beginF, {z});
      Value *h1 =
          B.CreateCall(strF, {h0, gv, ConstantInt::get(i64, len), ConstantInt::get(i32, 0)});
      B.CreateCall(argsF,
                   {h1, ConstantInt::get(i32, 1), tid, z, z, z, z, z, z, ConstantInt::get(i32, 1)});
      changed = true;
    }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};
} // namespace
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "FlydslPrintTid", LLVM_VERSION_STRING, [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef N, ModulePassManager &MPM, ArrayRef<PassBuilder::PipelineElement>) {
                  if (N == "flydsl-print-tid") {
                    MPM.addPass(PrintTidPass());
                    return true;
                  }
                  return false;
                });
          }};
}
