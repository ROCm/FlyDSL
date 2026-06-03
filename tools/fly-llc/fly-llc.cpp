// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
//
// fly-llc: minimal LLVM IR -> object emitter that allows injecting custom
// MachineFunction (MIR) passes into the standard codegen pipeline, at the
// pre-emit slot.  This is the piece MLIR's `gpu-module-to-binary` does not
// expose: it mirrors `CodeGenTargetMachineImpl::addPassesToEmitFile`, but adds
// named legacy MIR passes (loaded from `--load` plugins) after
// `addMachinePasses()` and before the asm printer.
//
// Usage:
//   fly-llc <input.ll> -o <out.o> -mtriple=... -mcpu=... \
//           [--load=lib.so ...] [--pre-emit-pass=name ...]

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include <dlfcn.h>

using namespace llvm;

static cl::opt<std::string> InputFile(cl::Positional, cl::Required, cl::desc("<input .ll>"));
static cl::opt<std::string> OutputFile("o", cl::Required, cl::desc("output object file"));
static cl::opt<std::string> MTriple("mtriple", cl::init("amdgcn-amd-amdhsa"),
                                    cl::desc("target triple"));
static cl::opt<std::string> MCPU("mcpu", cl::init(""), cl::desc("target cpu (e.g. gfx942)"));
static cl::list<std::string> LoadLib("load",
                                     cl::desc("dlopen a legacy MIR pass plugin .so (repeatable)"));
static cl::list<std::string>
    PreEmitPass("pre-emit-pass", cl::desc("named MIR pass to insert pre-emit (repeatable)"));

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();
  cl::ParseCommandLineOptions(argc, argv, "fly-llc: IR -> object with injectable MIR passes\n");

  // Legacy pass plugins self-register into the global PassRegistry on load.
  for (auto &lib : LoadLib)
    if (!dlopen(lib.c_str(), RTLD_NOW | RTLD_GLOBAL)) {
      errs() << "fly-llc: dlopen failed: " << dlerror() << "\n";
      return 1;
    }

  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIRFile(InputFile, Err, Ctx);
  if (!M) {
    Err.print(argv[0], errs());
    return 1;
  }

  Triple TT(MTriple);
  std::string E;
  const Target *T = TargetRegistry::lookupTarget(TT, E);
  if (!T) {
    errs() << "fly-llc: " << E << "\n";
    return 1;
  }
  TargetOptions O;
  std::unique_ptr<TargetMachine> TM(
      T->createTargetMachine(TT, MCPU, "", O, Reloc::PIC_, std::nullopt, CodeGenOptLevel::Default));
  M->setDataLayout(TM->createDataLayout());
  M->setTargetTriple(TT);

  // Replicate addPassesToEmitFile so a custom MIR pass can be injected after
  // the standard machine passes and before the asm printer.
  auto &CG = static_cast<CodeGenTargetMachineImpl &>(*TM);
  legacy::PassManager PM;
  auto *MMIWP = new MachineModuleInfoWrapperPass(&CG);
  TargetPassConfig *PC = CG.createPassConfig(PM);
  PC->setDisableVerify(true);
  PM.add(PC);
  PM.add(MMIWP);
  TargetLibraryInfoImpl TLII(TT);
  PM.add(new TargetLibraryInfoWrapperPass(TLII));
  if (PC->addISelPasses()) {
    errs() << "fly-llc: addISelPasses failed\n";
    return 1;
  }
  PC->addMachinePasses();
  for (auto &name : PreEmitPass) {
    const PassInfo *PI = PassRegistry::getPassRegistry()->getPassInfo(StringRef(name));
    if (!PI || !PI->getNormalCtor()) {
      errs() << "fly-llc: unknown MIR pass: " << name << "\n";
      return 1;
    }
    PM.add(PI->getNormalCtor()());
  }
  PC->setInitialized();

  std::error_code EC;
  raw_fd_ostream Out(OutputFile, EC, sys::fs::OF_None);
  if (EC) {
    errs() << "fly-llc: " << EC.message() << "\n";
    return 1;
  }
  if (CG.addAsmPrinter(PM, Out, nullptr, CodeGenFileType::ObjectFile,
                       MMIWP->getMMI().getContext())) {
    errs() << "fly-llc: addAsmPrinter failed (object emission unsupported)\n";
    return 1;
  }
  PM.add(createFreeMachineFunctionPass());
  PM.run(*M);
  Out.flush();
  return 0;
}
